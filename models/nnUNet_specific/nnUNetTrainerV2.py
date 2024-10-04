#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from collections import OrderedDict
from typing import Tuple
from multiprocessing import Pool
from nnunet.postprocessing.connected_components import determine_postprocessing
from batchgenerators.augmentations.utils import pad_nd_image
import shutil
from time import sleep
import traceback

import scipy
from scipy.spatial import cKDTree
from pathlib import Path
import random



import numpy as np
import torch
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.configuration import default_num_threads

from nnunet.evaluation.evaluator import aggregate_scores
from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax


 
class nnUNetTrainerV2(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 0
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        
        self.splits_file = "splits_final.pkl"

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)

            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params['patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

        
    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def validate_MICCAI(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        
        ds = self.network.do_ds
        self.network.do_ds = False

        current_mode = self.network.training
        self.network.eval()
        
        # TODO ? 
        self.splits_file = "test_data.pkl"

        self.dataset_val = None
        step_size = .5
        print(f"In validate {step_size}")
        
        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            print("in init")
            self.print_to_log_file(f"Data test split: {self.splits_file}" , also_print_to_console=True)
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(default_num_threads)
        M = 24
        save_M = M
        TTA = False # Test time augmentations
        save = True
        fixed_r = True
        compare_reg = True
        extended_metrics = True
        TTA_pred = []
        TTA_metrics = []
        TTA_metrics_reg = []
        rotations = get_euler_angles(M)
        
        # zones :
        #  0 all sphere
        #  1 upper hemisphere (UH) z>0, 
        #  Halves
        #   11 right UH z>0 - y>0
        #   12 left UH z>0 - y<0
        #  Quarters
        #   21 front right quarter UH (QUH) z>0 - y>0 - x>0
        #   22 front left QHU z>0 - y<0 - x>0
        #   23 back right QUH z>0 - y>0 - x<0
        #   24 back left QHU z>0 - y<0 - x<0
        #  Cones
        #   4 values > .5 -> angle of 120 (60deg each)
        # For lower hemisphere, same but times -1
        if M != 24:
            zone = 4
            z_rot_as_zone = False
            n_gamma=4
            desired_z_rot = 1 # 4
            # nbr of iterations i = 0 , 6 * z'' rot

            nbr_iter = 3 # 2
            cone_half_angle = 30 #60
            
            # 13 points
            zone = 4
            z_rot_as_zone = False
            n_gamma=4
            desired_z_rot = 1
            # nbr of iterations
            nbr_iter = 2
            cone_half_angle = 60
            rotations = get_euler_angles_new(nbr_iter=nbr_iter, n_gamma=n_gamma, zone=zone, z_rot_as_zone=z_rot_as_zone, desired_z_rot=desired_z_rot, cone_half_angle=cone_half_angle)
            save_M = str(len(rotations)) 
            if zone and cone_half_angle != 90:
                save_M = save_M + '_cone_HA_' + str(cone_half_angle)
            if desired_z_rot != n_gamma:
                save_M = save_M + '_nbr_Z''_' + str(desired_z_rot)
             
            
        results = []
        dice_score = []
        metrics_score_p = []
        metrics_score = []
        metrics_score_reg_p = []
        metrics_score_reg = [] # compare the data with the non rotated input prediction
        self.print_to_log_file(f"M {save_M} Step size {step_size} Extended Metrics {extended_metrics} M {save_M}" , also_print_to_console=False)
        for k in self.dataset_val.keys():
            properties = load_pickle(self.dataset[k]['properties_file'])
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            no_rot_pred = None
            self.print_to_log_file(f"Image: {k}" , also_print_to_console=False)
            for id_rot, rot in enumerate(rotations):
                
                #fname2 = fname + f'_{rot[0]}z_{rot[1]}y_{rot[2]}z'
                if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                        (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                    data = np.load(self.dataset[k]['data_file'])['data']
                    print(f'id rot {id_rot} rotations {rot}')
                    print(f'shapes {k, data[-1].shape}') # data shape (img, GT), 3D volume
                    data[-1][data[-1] == -1] = 0
                    data_p, sc = pad_nd_image(data.copy(), np.array([40,40,40]), mode = 'constant', return_slicer = True, kwargs = {'constant_values': 0})

                    if np.sum(rot):
                        inv_rot = [-rot[i] for i in range(2, -1, -1)]
                        a = data.copy().sum()
                        r_pred, trg = euler_rotation_3d(data[:-1], rot, lbl=data[-1], mode = 'constant', crop_img=False,order=0, reshape_rot=True)
                        p_pred, p_trg = euler_rotation_3d(data_p[:-1], rot,lbl=data_p[-1], mode = 'constant', crop_img=False,order=0, reshape_rot=True)
                        
                        data = np.concatenate([r_pred, np.expand_dims(trg,axis=0)],axis=0) 
                    
                        data_p = np.concatenate([p_pred, np.expand_dims(p_trg,axis=0)],axis=0) 
                        
                                            
                    if type(data[-1]) is np.ndarray:
                        gt = torch.clone(torch.from_numpy(data[-1]))
                        gt_p = torch.clone(torch.from_numpy(data_p[-1]))
                    else:
                        gt = torch.clone(data[-1])
                        gt_p = torch.clone(data_p[-1])

                    softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data[:-1],
                                                                                        do_mirroring=do_mirroring,
                                                                                        mirror_axes=mirror_axes,
                                                                                        use_sliding_window=use_sliding_window,
                                                                                        step_size=step_size,
                                                                                        use_gaussian=use_gaussian,
                                                                                        all_in_gpu=all_in_gpu,
                                                                                        mixed_precision=self.fp16,
                                                                                        verbose=False)[1]
                    softmax_pred_p = self.predict_preprocessed_data_return_seg_and_softmax(data_p[:-1],
                                                                                        do_mirroring=do_mirroring,
                                                                                        mirror_axes=mirror_axes,
                                                                                        use_sliding_window=use_sliding_window,
                                                                                        step_size=step_size,
                                                                                        use_gaussian=use_gaussian,
                                                                                        all_in_gpu=all_in_gpu,
                                                                                        mixed_precision=self.fp16,
                                                                                        verbose=False)[1]
                    
                    print(f'\nSUMS: \n{data[-1].sum(), data_p[-1].sum()}\n {softmax_pred[0].sum(),softmax_pred[1].sum(),softmax_pred[2].sum()} \n{softmax_pred_p[0].sum(),softmax_pred_p[1].sum(),softmax_pred_p[2].sum()}\n')
                    #print(f'DICE DICE \n {data[:-1].shape, softmax_pred.shape}')
                    softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])
                    softmax_pred_p = softmax_pred_p.transpose([0] + [i + 1 for i in self.transpose_backward])
                    #print(f'{softmax_pred.shape} \n')
                    if TTA: 
                        # TODO append the reversed rotated softmax_pred
                        inv_rot = [-rot[i] for i in range(2, -1, -1)]
                        tmp = np.copy(softmax_pred)
                        TTA_pred.append(euler_rotation_3d(tmp, inv_rot, mode = 'constant', crop_img=False,order=0, reshape_rot=True))
                    
                    
                    if type(softmax_pred) is np.ndarray:
                        pred = torch.clone(torch.from_numpy(softmax_pred))
                        pred_p = torch.clone(torch.from_numpy(softmax_pred_p))
                    else:
                        pred = torch.clone(softmax_pred)
                        pred_p = torch.clone(softmax_pred_p)

                    key = f'{rot[0]}z_{rot[1]}y_{rot[2]}z'
                    
                    if not id_rot and (compare_reg or TTA):
                        tmp2 = torch.clone(pred)
                        no_rot_pred = [tmp2, torch.clone(gt)]
                        no_rot_pred_gt = no_rot_pred.copy()

                        # add the gt for TTA
                        tmp = torch.clone(pred)
                        metrics_reg = compute_metrice_pc(no_rot_pred_gt[0], tmp, extended_metrics, fixed_r) 
                        metrics_score_reg.append({key :metrics_reg})
                        
                        no_rot_pred_p = [torch.clone(pred_p), torch.clone(gt_p)]
                        no_rot_pred_gt = no_rot_pred_p.copy()

                        # add the gt for TTA
                        tmp = torch.clone(pred_p)
                        metrics_reg_p = compute_metrice_pc(no_rot_pred_gt[0], tmp, extended_metrics,fixed_r) 
                        metrics_score_reg_p.append({key :metrics_reg_p})
                        print(f"compute reg")
                    elif compare_reg:   
                                   
                        no_rot_pred_gt = no_rot_pred.copy()  
                        rotated_base_pred = euler_rotation_3d(no_rot_pred_gt[0], rot, mode = 'constant', crop_img=False,order=0, reshape_rot=True)
                        tmp = torch.clone(pred)
                        metrics_reg = compute_metrice_pc(rotated_base_pred,tmp, extended_metrics, fixed_r)
                        metrics_score_reg.append({key :metrics_reg})
                        print(f"compute reg")
                                   
                        no_rot_pred_gt_p = no_rot_pred_p.copy()  
                        rotated_base_pred_p = euler_rotation_3d(no_rot_pred_gt_p[0], rot, mode = 'constant', crop_img=False,order=0, reshape_rot=True)
                        tmp = torch.clone(pred_p)
                        metrics_reg_p = compute_metrice_pc(rotated_base_pred_p,tmp, extended_metrics, fixed_r)
                        #print(f"compute reg {metrics_reg['RMSE'], rotated_base_pred.shape, tmp.shape}\n\n")

                        metrics_score_reg_p.append({key :metrics_reg_p})
                    tmp = torch.clone(pred)
                    gt_tmp = torch.clone(gt)
                    metrics = compute_metrice_pc(gt_tmp, tmp, extended_metrics, fixed_r) 
                    metrics_score.append({key :metrics})
                    
                    # Padded
                    tmp = torch.clone(pred_p)
                    gt_tmp = torch.clone(gt_p)
                    metrics_p = compute_metrice_pc(gt_tmp, tmp, extended_metrics, fixed_r) 
                    metrics_score_p.append({key :metrics_p})
                    
                    self.print_to_log_file(f"Rotation {key}:" , also_print_to_console=False)
                    self.print_to_log_file(f"Reg RMSE {metrics_reg['RMSE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"Reg DICE {metrics_reg['DICE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"Perf RMSE {metrics['RMSE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"Perf DICE {metrics['DICE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"P Reg RMSE {metrics_reg_p['RMSE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"P Reg DICE {metrics_reg_p['DICE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"P Perf RMSE {metrics_p['RMSE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"P Perf DICE {metrics_p['DICE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"========================================\n" , also_print_to_console=False)
                    
                    #rmse = compute_rmse_pc(gt, pred)
                    # print(f'RMSE {rmse}')
                    #rmse_score.append({key :rmse})
                    
                    #pred = torch.argmax(pred, axis=0)
                    #dice_score.append({key :compute_dice_pc(gt, pred)})

                    if np.sum(rot):
                        continue
                    if save_softmax:
                        softmax_fname = join(output_folder, fname + f'_{rot[0]}z_{rot[1]}y_{rot[2]}z' + ".npz")
                    else:
                        softmax_fname = None

                    """There is a problem with python process communication that prevents us from communicating objects
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(join(output_folder, fname + ".npy"), softmax_pred)
                        softmax_pred = join(output_folder, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                            ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                            properties, interpolation_order, self.regions_class_order,
                                                            None, None,
                                                            softmax_fname, None, force_separate_z,
                                                            interpolation_order_z),
                                                            )
                                                            )
                                )
                    

                
                pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

            # TODO stack pred and take mean of them, compute perf metrics a reg (?)
            if TTA:
                TTA_pred = torch.mean(torch.from_numpy(np.stack(TTA_pred)), dim=0)
                no_rot_pred_gt = no_rot_pred.copy()
                tmp = torch.clone(TTA_pred)
                TTA_metrics_id = compute_metrice_pc(no_rot_pred_gt[1], tmp, extended_metrics, fixed_r) 
                TTA_metrics.append(TTA_metrics_id)
                if compare_reg:
                    tmp = torch.clone(TTA_pred)
                    no_rot_pred_gt = no_rot_pred.copy()
                    TTA_metrics_reg_id = compute_metrice_pc(no_rot_pred_gt[0],  tmp, extended_metrics, fixed_r)
                    TTA_metrics_reg.append(TTA_metrics_reg_id)
                TTA_pred = []
        #json_path_dice = join(output_folder[:-15], f'DICE_M{M}_rotated_results.json')
        #out_file = open(json_path_dice, "w") 
        #json.dump(dice_score, out_file) 
        #out_file.close()
        
        #json_path_rmse = join(output_folder[:-15], f'RMSE_M{M}_rotated_results.json')
        #out_file = open(json_path_rmse, "w") 
        #json.dump(rmse_score, out_file) 
        #out_file.close()
        if save:
            if TTA:
                
                #son_path_metrics = join(output_folder[:-15], f'TTA_metrics_M{M}_rotated_results.json')
                #if step_size != 0.5:
                #    json_path_metrics = join(output_folder[:-15], f'TTA_metrics_M{M}_rotated_results_ss{step_size}.json')
                
                base_str = f'TTA_metrics_M{save_M}_rotated_results'
                if step_size != 0.5:
                    base_str = base_str + f'_ss{step_size}'
                if extended_metrics:
                    base_str = base_str + '_EM'
                base_str = base_str + '.json'
                
                json_path_metrics = join(output_folder[:-15], base_str)
                out_file = open(json_path_metrics, "w") 
                json.dump(TTA_metrics, out_file) 
                out_file.close()
                if compare_reg:
                    #json_path_metrics = join(output_folder[:-15], f'TTA_regulization_metrics_M{M}_rotated_results.json')
                    #if step_size != 0.5:
                    #    json_path_metrics = join(output_folder[:-15], f'TTA_regulization_metrics_M{M}_rotated_results_ss{step_size}.json')
                    
                    base_str = f'TTA_regulization_metrics_M{save_M}_rotated_results'
                    if step_size != 0.5:
                        base_str = base_str + f'_ss{step_size}'
                    if extended_metrics:
                        base_str = base_str + '_EM'
                    base_str = base_str + '.json'
                    json_path_metrics = join(output_folder[:-15], base_str)
                    
                    out_file = open(json_path_metrics, "w") 
                    json.dump(TTA_metrics_reg, out_file) 
                    out_file.close() 
            
            if fixed_r:
                base_str = f'fixed_nDSC_true_metrics_M{save_M}_rotated_results'
            else:
                 base_str = f'nDSC_true_metrics_M{save_M}_rotated_results'
            if step_size != 0.5:
                base_str = base_str + f'_ss{step_size}'
            if extended_metrics:
                base_str = base_str + '_EM'
            base_str = base_str + '.json'
            json_path_metrics = join(output_folder[:-15], base_str)   
                             
            #json_path_metrics = join(output_folder[:-15], f'true_metrics_M{M}_rotated_results.json')
            #if step_size != 0.5:
            #    json_path_metrics = join(output_folder[:-15], f'true_metrics_M{M}_rotated_results_ss{step_size}.json')
            out_file = open(json_path_metrics, "w") 
            json.dump(metrics_score, out_file) 
            out_file.close()
            if fixed_r:
                base_str = f'fixed_nDSC_true_metrics_padded_M{save_M}_rotated_results'
            else:
                base_str = f'nDSC_true_metrics_padded_M{save_M}_rotated_results'
            if step_size != 0.5:
                base_str = base_str + f'_ss{step_size}'
            if extended_metrics:
                base_str = base_str + '_EM'
            base_str = base_str + '.json'
            json_path_metrics = join(output_folder[:-15], base_str)   
                             
            out_file = open(json_path_metrics, "w") 
            json.dump(metrics_score_p, out_file) 
            out_file.close()
        
            if compare_reg:   
                if fixed_r:
                    base_str = f'fixed_nDSC_true_regulization_metrics_M{save_M}_rotated_results'
                else:
                    base_str = f'nDSC_true_regulization_metrics_M{save_M}_rotated_results'
                if step_size != 0.5:
                    base_str = base_str + f'_ss{step_size}'
                if extended_metrics:
                    base_str = base_str + '_EM'
                base_str = base_str + '.json'
                json_path_metrics = join(output_folder[:-15], base_str)   
                #json_path_metrics = join(output_folder[:-15], f'true_regulization_metrics_M{M}_rotated_results.json')
                #if step_size != 0.5:
                #    json_path_metrics = join(output_folder[:-15], f'true_regulization_metrics_M{M}_rotated_results_ss{step_size}.json')               
                out_file = open(json_path_metrics, "w") 
                json.dump(metrics_score_reg, out_file) 
                out_file.close() 

                if fixed_r:
                    base_str = f'fixed_nDSC_true_regulization_metrics_paddded_M{save_M}_rotated_results'
                else:
                    base_str = f'nDSC_true_regulization_metrics_M{save_M}_rotated_results'
                if step_size != 0.5:
                    base_str = base_str + f'_ss{step_size}'
                if extended_metrics:
                    base_str = base_str + '_EM'
                base_str = base_str + '.json'
                json_path_metrics = join(output_folder[:-15], base_str)          
                out_file = open(json_path_metrics, "w") 
                json.dump(metrics_score_reg_p, out_file) 
                out_file.close() 
            
        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError:
                    print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                    traceback.print_exc()
                    attempts += 1
                    sleep(1)
            if not success:
                raise OSError(f"Something went wrong while copying nifti files to {gt_nifti_folder}. See above for the trace.")

        self.network.train(current_mode)
        
        self.network.do_ds = ds


    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        
        ds = self.network.do_ds
        self.network.do_ds = False

        current_mode = self.network.training
        self.network.eval()
        
        # TODO ? 
        self.splits_file = "test_data.pkl"
        fold = 0
        self.dataset_val = None
        step_size = .5
        print(f"In validate {step_size}")
        
        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            print("in init")
            self.print_to_log_file(f"Data test split: {self.splits_file}" , also_print_to_console=True)
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(default_num_threads)
        M = 24
        save_M = M
        TTA = False # Test time augmentations
        save = False
        save_pred = True
        save_GT = True
        
        fixed_r = True
        compare_reg = True
        not_workshop_test = False
        extended_metrics = True
        TTA_pred = []
        TTA_metrics = []
        TTA_metrics_reg = []
        rotations = get_euler_angles(M)
        
        # zones :
        #  0 all sphere
        #  1 upper hemisphere (UH) z>0, 
        #  Halves
        #   11 right UH z>0 - y>0
        #   12 left UH z>0 - y<0
        #  Quarters
        #   21 front right quarter UH (QUH) z>0 - y>0 - x>0
        #   22 front left QHU z>0 - y<0 - x>0
        #   23 back right QUH z>0 - y>0 - x<0
        #   24 back left QHU z>0 - y<0 - x<0
        #  Cones
        #   4 values > .5 -> angle of 120 (60deg each)
        # For lower hemisphere, same but times -1
        if M != 24:
            zone = 4
            z_rot_as_zone = False
            n_gamma=4
            desired_z_rot = 1 # 4
            # nbr of iterations i = 0 , 6 * z'' rot

            nbr_iter = 3 # 2
            cone_half_angle = 30 #60
            
            # 13 points
            zone = 4
            z_rot_as_zone = False
            n_gamma=4
            desired_z_rot = 1
            # nbr of iterations
            nbr_iter = 2
            cone_half_angle = 60
            rotations = get_euler_angles_new(nbr_iter=nbr_iter, n_gamma=n_gamma, zone=zone, z_rot_as_zone=z_rot_as_zone, desired_z_rot=desired_z_rot, cone_half_angle=cone_half_angle)
            save_M = str(len(rotations)) 
            if zone and cone_half_angle != 90:
                save_M = save_M + '_cone_HA_' + str(cone_half_angle)
            if desired_z_rot != n_gamma:
                save_M = save_M + '_nbr_Z''_' + str(desired_z_rot)
             
        
        results = []
        dice_score = []
        metrics_score_p = []
        metrics_score = []
        metrics_score_reg_p = []
        metrics_score_reg = [] # compare the data with the non rotated input prediction
        self.print_to_log_file(f"M {save_M} Step size {step_size} Extended Metrics {extended_metrics} M {save_M}" , also_print_to_console=False)
        for k in self.dataset_val.keys():
            properties = load_pickle(self.dataset[k]['properties_file'])
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            no_rot_pred = None
            self.print_to_log_file(f"Image: {k}" , also_print_to_console=False)
            for id_rot, rot in enumerate(rotations):
                
                if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                        (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                    data = np.load(self.dataset[k]['data_file'])['data']
                    print("=====")
                    print(f'id rot {id_rot} rotations {rot}')
                    print(f'shapes {k, data[-1].shape}') # data shape (img, GT), 3D volume
                    data[-1][data[-1] == -1] = 0
                    data_p, sc = pad_nd_image(data.copy(), np.array([40,40,40]), mode = 'constant', return_slicer = True, kwargs = {'constant_values': 0})

                    if np.sum(rot):
                        inv_rot = [-rot[i] for i in range(2, -1, -1)]
                        # TODO 
                        a = data.copy().sum()
                        r_pred, trg = euler_rotation_3d(data[:-1], rot, lbl=data[-1], mode = 'constant', crop_img=False,order=0, reshape_rot=True)
                        p_pred, p_trg = euler_rotation_3d(data_p[:-1], rot,lbl=data_p[-1], mode = 'constant', crop_img=False,order=0, reshape_rot=True)
                        
                        data = np.concatenate([r_pred, np.expand_dims(trg,axis=0)],axis=0) 
                    
                        data_p = np.concatenate([p_pred, np.expand_dims(p_trg,axis=0)],axis=0) 
                                                                    
                    if type(data[-1]) is np.ndarray:
                        gt = torch.clone(torch.from_numpy(data[-1]))
                        gt_p = torch.clone(torch.from_numpy(data_p[-1]))
                    else:
                        gt = torch.clone(data[-1])
                        gt_p = torch.clone(data_p[-1])

                    # TODO rotate here
                    softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data[:-1],
                                                                                        do_mirroring=do_mirroring,
                                                                                        mirror_axes=mirror_axes,
                                                                                        use_sliding_window=use_sliding_window,
                                                                                        step_size=step_size,
                                                                                        use_gaussian=use_gaussian,
                                                                                        all_in_gpu=all_in_gpu,
                                                                                        mixed_precision=self.fp16,
                                                                                        verbose=False)[1]
                    softmax_pred_p = self.predict_preprocessed_data_return_seg_and_softmax(data_p[:-1],
                                                                                        do_mirroring=do_mirroring,
                                                                                        mirror_axes=mirror_axes,
                                                                                        use_sliding_window=use_sliding_window,
                                                                                        step_size=step_size,
                                                                                        use_gaussian=use_gaussian,
                                                                                        all_in_gpu=all_in_gpu,
                                                                                        mixed_precision=self.fp16,
                                                                                        verbose=False)[1]
                    
                    
                    key = f'_{rot[0]:.2f}z_{rot[1]:.2f}y_{rot[2]:.2f}z'
                    print(f'oh ah {self.transpose_backward}')
                    if save_pred:
                        if M == 24:
                            path = output_folder[:-14] + 'rotated_RA_folder/'
                        else:
                            path = output_folder[:-14] + 'rotated_folder/'
                        pred_p_path = path + fname + key + '_padded'
                        pred_path = path + fname  + key
                        
                        np.savez_compressed(pred_path, softmax_pred)
                        np.savez_compressed(pred_p_path, softmax_pred_p)
                        
                        if save_GT:
                            if M == 24:
                                path = output_folder[:-21] + 'gt_niftis_RA_rotated/'
                            else:
                                path = output_folder[:-21] + 'gt_niftis_rotated/'
                            
                            pred_p_path = path + fname + key + '_padded'
                            pred_path = path + fname  + key
                            np.savez_compressed(pred_path, gt)
                            np.savez_compressed(pred_p_path, gt_p)                    
                        #print(f'saved to {pred_path} and {pred_p_path}')

                    print(f'\nSUMS: \n{data[-1].sum(), data_p[-1].sum()}\n {softmax_pred[0].sum(),softmax_pred[1].sum(),softmax_pred[2].sum()} \n{softmax_pred_p[0].sum(),softmax_pred_p[1].sum(),softmax_pred_p[2].sum()}\n')
                    #print(f'DICE DICE \n {data[:-1].shape, softmax_pred.shape}')
                    softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])
                    softmax_pred_p = softmax_pred_p.transpose([0] + [i + 1 for i in self.transpose_backward])
                    
                    #print(f'{softmax_pred.shape} \n')
                    if TTA: 
                        # TODO append the reversed rotated softmax_pred
                        inv_rot = [-rot[i] for i in range(2, -1, -1)]
                        tmp = np.copy(softmax_pred)
                        TTA_pred.append(euler_rotation_3d(tmp, inv_rot, mode = 'constant', crop_img=False,order=0, reshape_rot=True))
                    
                    
                    if type(softmax_pred) is np.ndarray:
                        pred = torch.clone(torch.from_numpy(softmax_pred))
                        pred_p = torch.clone(torch.from_numpy(softmax_pred_p))
                    else:
                        pred = torch.clone(softmax_pred)
                        pred_p = torch.clone(softmax_pred_p)

                    key = f'{rot[0]}z_{rot[1]}y_{rot[2]}z'
                    
                    if not id_rot and (compare_reg or TTA):
                        tmp2 = torch.clone(pred)
                        no_rot_pred = [tmp2, torch.clone(gt)]
                        no_rot_pred_gt = no_rot_pred.copy()

                        # add the gt for TTA
                        tmp = torch.clone(pred)
                        metrics_reg = compute_metrice_pc(no_rot_pred_gt[0], tmp, extended_metrics, fixed_r) 
                        metrics_score_reg.append({key :metrics_reg})
                        
                        no_rot_pred_p = [torch.clone(pred_p), torch.clone(gt_p)]
                        no_rot_pred_gt = no_rot_pred_p.copy()

                        # add the gt for TTA
                        tmp = torch.clone(pred_p)
                        metrics_reg_p = compute_metrice_pc(no_rot_pred_gt[0], tmp, extended_metrics,fixed_r) 
                        metrics_score_reg_p.append({key :metrics_reg_p})
                        print(f"compute reg")
                    elif compare_reg:   
                                   
                        no_rot_pred_gt = no_rot_pred.copy()  
                        rotated_base_pred = euler_rotation_3d(no_rot_pred_gt[0], rot, mode = 'constant', crop_img=False,order=0, reshape_rot=True)
                        tmp = torch.clone(pred)
                        metrics_reg = compute_metrice_pc(rotated_base_pred,tmp, extended_metrics, fixed_r)
                        metrics_score_reg.append({key :metrics_reg})
                        print(f"compute reg")
                                   
                        no_rot_pred_gt_p = no_rot_pred_p.copy()  
                        rotated_base_pred_p = euler_rotation_3d(no_rot_pred_gt_p[0], rot, mode = 'constant', crop_img=False,order=0, reshape_rot=True)
                        tmp = torch.clone(pred_p)
                        metrics_reg_p = compute_metrice_pc(rotated_base_pred_p,tmp, extended_metrics, fixed_r)
                        #print(f"compute reg {metrics_reg['RMSE'], rotated_base_pred.shape, tmp.shape}\n\n")

                        metrics_score_reg_p.append({key :metrics_reg_p})
                    tmp = torch.clone(pred)
                    gt_tmp = torch.clone(gt)
                    metrics = compute_metrice_pc(gt_tmp, tmp, extended_metrics, fixed_r) 
                    metrics_score.append({key :metrics})
                    
                    # Padded
                    tmp = torch.clone(pred_p)
                    gt_tmp = torch.clone(gt_p)
                    metrics_p = compute_metrice_pc(gt_tmp, tmp, extended_metrics, fixed_r) 
                    metrics_score_p.append({key :metrics_p})
                    
                    self.print_to_log_file(f"Rotation {key}:" , also_print_to_console=False)
                    self.print_to_log_file(f"Reg RMSE {metrics_reg['RMSE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"Reg DICE {metrics_reg['DICE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"Perf RMSE {metrics['RMSE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"Perf DICE {metrics['DICE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"P Reg RMSE {metrics_reg_p['RMSE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"P Reg DICE {metrics_reg_p['DICE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"P Perf RMSE {metrics_p['RMSE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"P Perf DICE {metrics_p['DICE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"========================================\n" , also_print_to_console=False)
                    
                    #rmse = compute_rmse_pc(gt, pred)
                    # print(f'RMSE {rmse}')
                    #rmse_score.append({key :rmse})
                    
                    #pred = torch.argmax(pred, axis=0)
                    #dice_score.append({key :compute_dice_pc(gt, pred)})

                    if np.sum(rot):
                        continue
                    if save_softmax:
                        softmax_fname = join(output_folder, fname + f'_{rot[0]}z_{rot[1]}y_{rot[2]}z' + ".npz")
                    else:
                        softmax_fname = None

                    """There is a problem with python process communication that prevents us from communicating objects
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(join(output_folder, fname + ".npy"), softmax_pred)
                        softmax_pred = join(output_folder, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                            ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                            properties, interpolation_order, self.regions_class_order,
                                                            None, None,
                                                            softmax_fname, None, force_separate_z,
                                                            interpolation_order_z),
                                                            )
                                                            )
                                )
                    

                
                pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

            # TODO stack pred and take mean of them, compute perf metrics a reg (?)
            if TTA:
                TTA_pred = torch.mean(torch.from_numpy(np.stack(TTA_pred)), dim=0)
                no_rot_pred_gt = no_rot_pred.copy()
                tmp = torch.clone(TTA_pred)
                TTA_metrics_id = compute_metrice_pc(no_rot_pred_gt[1], tmp, extended_metrics, fixed_r) 
                TTA_metrics.append(TTA_metrics_id)
                if compare_reg:
                    tmp = torch.clone(TTA_pred)
                    no_rot_pred_gt = no_rot_pred.copy()
                    TTA_metrics_reg_id = compute_metrice_pc(no_rot_pred_gt[0],  tmp, extended_metrics, fixed_r)
                    TTA_metrics_reg.append(TTA_metrics_reg_id)
                TTA_pred = []
        #json_path_dice = join(output_folder[:-15], f'DICE_M{M}_rotated_results.json')
        #out_file = open(json_path_dice, "w") 
        #json.dump(dice_score, out_file) 
        #out_file.close()
        
        #json_path_rmse = join(output_folder[:-15], f'RMSE_M{M}_rotated_results.json')
        #out_file = open(json_path_rmse, "w") 
        #json.dump(rmse_score, out_file) 
        #out_file.close()
        if save:
            if TTA:
                
                #son_path_metrics = join(output_folder[:-15], f'TTA_metrics_M{M}_rotated_results.json')
                #if step_size != 0.5:
                #    json_path_metrics = join(output_folder[:-15], f'TTA_metrics_M{M}_rotated_results_ss{step_size}.json')
                
                base_str = f'REDO_workshop_TTA_metrics_M{save_M}_rotated_results'
                if step_size != 0.5:
                    base_str = base_str + f'_ss{step_size}'
                if extended_metrics:
                    base_str = base_str + '_EM'
                base_str = base_str + '.json'
                
                json_path_metrics = join(output_folder[:-15], base_str)
                out_file = open(json_path_metrics, "w") 
                json.dump(TTA_metrics, out_file) 
                out_file.close()
                if compare_reg:
                    #json_path_metrics = join(output_folder[:-15], f'TTA_regulization_metrics_M{M}_rotated_results.json')
                    #if step_size != 0.5:
                    #    json_path_metrics = join(output_folder[:-15], f'TTA_regulization_metrics_M{M}_rotated_results_ss{step_size}.json')
                    
                    base_str = f'REDO_workshop_TTA_regulization_metrics_M{save_M}_rotated_results'
                    if step_size != 0.5:
                        base_str = base_str + f'_ss{step_size}'
                    if extended_metrics:
                        base_str = base_str + '_EM'
                    base_str = base_str + '.json'
                    json_path_metrics = join(output_folder[:-15], base_str)
                    
                    out_file = open(json_path_metrics, "w") 
                    json.dump(TTA_metrics_reg, out_file) 
                    out_file.close() 
            
            if fixed_r:
                base_str = f'REDO_workshop_fixed_nDSC_true_metrics_M{save_M}_rotated_results'
            else:
                 base_str = f'REDO_workshop_nDSC_true_metrics_M{save_M}_rotated_results'
            if step_size != 0.5:
                base_str = base_str + f'_ss{step_size}'
            if extended_metrics:
                base_str = base_str + '_EM'
            base_str = base_str + '.json'
            json_path_metrics = join(output_folder[:-15], base_str)   
                             
            #json_path_metrics = join(output_folder[:-15], f'true_metrics_M{M}_rotated_results.json')
            #if step_size != 0.5:
            #    json_path_metrics = join(output_folder[:-15], f'true_metrics_M{M}_rotated_results_ss{step_size}.json')
            out_file = open(json_path_metrics, "w") 
            json.dump(metrics_score, out_file) 
            out_file.close()
            if fixed_r:
                base_str = f'REDO_workshop_fixed_nDSC_true_metrics_padded_M{save_M}_rotated_results'
            else:
                base_str = f'REDO_workshop_nDSC_true_metrics_padded_M{save_M}_rotated_results'
            if step_size != 0.5:
                base_str = base_str + f'_ss{step_size}'
            if extended_metrics:
                base_str = base_str + '_EM'
            base_str = base_str + '.json'
            json_path_metrics = join(output_folder[:-15], base_str)   
                             
            out_file = open(json_path_metrics, "w") 
            json.dump(metrics_score_p, out_file) 
            out_file.close()
        
            if compare_reg:   
                if fixed_r:
                    base_str = f'REDO_workshop_fixed_nDSC_true_regulization_metrics_M{save_M}_rotated_results'
                else:
                    base_str = f'REDO_workshop_nDSC_true_regulization_metrics_M{save_M}_rotated_results'
                if step_size != 0.5:
                    base_str = base_str + f'_ss{step_size}'
                if extended_metrics:
                    base_str = base_str + '_EM'
                base_str = base_str + '.json'
                json_path_metrics = join(output_folder[:-15], base_str)   
                #json_path_metrics = join(output_folder[:-15], f'true_regulization_metrics_M{M}_rotated_results.json')
                #if step_size != 0.5:
                #    json_path_metrics = join(output_folder[:-15], f'true_regulization_metrics_M{M}_rotated_results_ss{step_size}.json')               
                out_file = open(json_path_metrics, "w") 
                json.dump(metrics_score_reg, out_file) 
                out_file.close() 

                if fixed_r:
                    base_str = f'REDO_workshop_fixed_nDSC_true_regulization_metrics_paddded_M{save_M}_rotated_results'
                else:
                    base_str = f'REDO_workshop_nDSC_true_regulization_metrics_M{save_M}_rotated_results'
                if step_size != 0.5:
                    base_str = base_str + f'_ss{step_size}'
                if extended_metrics:
                    base_str = base_str + '_EM'
                base_str = base_str + '.json'
                json_path_metrics = join(output_folder[:-15], base_str)          
                out_file = open(json_path_metrics, "w") 
                json.dump(metrics_score_reg_p, out_file) 
                out_file.close() 
                
        if not_workshop_test:
            _ = [i.get() for i in results]
            self.print_to_log_file("finished prediction")

            # evaluate raw predictions
            self.print_to_log_file("evaluation of raw predictions")
            task = self.dataset_directory.split("/")[-1]
            job_name = self.experiment_name
            _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                                json_output_file=join(output_folder, "summary.json"),
                                json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                                json_author="Fabian",
                                json_task=task, num_threads=default_num_threads)

            if run_postprocessing_on_folds:
                # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
                # except the largest connected component for each class. To see if this improves results, we do this for all
                # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
                # have this applied during inference as well
                self.print_to_log_file("determining postprocessing")
                determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                        final_subf_name=validation_folder_name + "_postprocessed", debug=debug)
                # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
                # They are always in that folder, even if no postprocessing as applied!

            # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
            # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
            # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
            # be used later
            gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
            maybe_mkdir_p(gt_nifti_folder)
            for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
                success = False
                attempts = 0
                while not success and attempts < 10:
                    try:
                        shutil.copy(f, gt_nifti_folder)
                        success = True
                    except OSError:
                        print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                        traceback.print_exc()
                        attempts += 1
                        sleep(1)
                if not success:
                    raise OSError(f"Something went wrong while copying nifti files to {gt_nifti_folder}. See above for the trace.")

        self.network.train(current_mode)
        
        self.network.do_ds = ds

    
    
    
    def validate_old_3101(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        
        ds = self.network.do_ds
        self.network.do_ds = False

        current_mode = self.network.training
        self.network.eval()
        
        # TODO ? 
        self.splits_file = "test_data.pkl"
        #self.splits_file = "test_data_25.pkl"
        self.dataset_val = None
        step_size = .5
        print(f"In validate {step_size}")
        
        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            print("in init")
            self.print_to_log_file(f"Data test split: {self.splits_file}" , also_print_to_console=True)
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(default_num_threads)
        M = 24
        TTA = True # Test time augmentations
        save = True
        compare_reg = True
        TTA_pred = []
        TTA_metrics = []
        TTA_metrics_reg = []
        rotations = get_euler_angles(M)
        results = []
        dice_score = []
        rmse_score = []
        metrics_score = []
        metrics_score_reg_inv = []
        metrics_score_reg = [] # compare the data with the non rotated input prediction
        self.print_to_log_file(f"Step size {step_size}" , also_print_to_console=False)
        for k in self.dataset_val.keys():
            properties = load_pickle(self.dataset[k]['properties_file'])
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            no_rot_pred = None
            self.print_to_log_file(f"Image: {k}" , also_print_to_console=False)
            for id_rot, rot in enumerate(rotations):
                
                #fname = fname + f'_{rot[0]}z_{rot[1]}y_{rot[2]}z'
                if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                        (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                    data = np.load(self.dataset[k]['data_file'])['data']
                    print(f'id rot {id_rot} rotations {rot}')
                    print(f'shapes {k, data[-1].shape}') # data shape (img, GT), 3D volume
                    data[-1][data[-1] == -1] = 0
                    #data = pad_nd_image(data, np.array([40,40,40]), mode = 'constant', kwargs = {'constant_values': 0})

                    if np.sum(rot):
                        r_pred, trg = euler_rotation_3d(data[:-1], rot, lbl=data[-1], mode = 'constant', crop_img=False,order=0, reshape_rot=True)
                        print(f'YES YES {r_pred.shape, trg.shape}')
                        data = np.concatenate([r_pred, np.expand_dims(trg,axis=0)],axis=0) 
                    if type(data[-1]) is np.ndarray:
                        gt = torch.clone(torch.from_numpy(data[-1]))
                    else:
                        gt = torch.clone(data[-1])
                        
                    do_mirroring = False

                    # TODO rotate here
                    softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data[:-1],
                                                                                        do_mirroring=do_mirroring,
                                                                                        mirror_axes=mirror_axes,
                                                                                        use_sliding_window=use_sliding_window,
                                                                                        step_size=step_size,
                                                                                        use_gaussian=use_gaussian,
                                                                                        all_in_gpu=all_in_gpu,
                                                                                        mixed_precision=self.fp16)[1]
                    
                    softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])
                    if TTA: 
                        # TODO append the reversed rotated softmax_pred
                        inv_rot = [-rot[i] for i in range(2, -1, -1)]
                        tmp = np.copy(softmax_pred)
                        TTA_pred.append(euler_rotation_3d(tmp, inv_rot, mode = 'constant', crop_img=False,order=0, reshape_rot=True))
                    
                    
                    if type(softmax_pred) is np.ndarray:
                        pred = torch.clone(torch.from_numpy(softmax_pred))
                    else:
                        pred = torch.clone(softmax_pred)
                        
                   
                        

                
                    key = f'{rot[0]}z_{rot[1]}y_{rot[2]}z'
                    
                    if not id_rot and (compare_reg or TTA):
                        tmp2 = torch.clone(pred)
                        no_rot_pred = [tmp2, torch.clone(gt)]
                        no_rot_pred_gt = no_rot_pred.copy()
                        # add the gt for TTA
                        print("compute reg metrics")
                        tmp = torch.clone(pred)
                        metrics_reg = compute_metrice_pc(no_rot_pred_gt[0], tmp) 
                        metrics_score_reg.append({key :metrics_reg})
                        metrics_score_reg_inv.append({key :metrics_reg})
                    elif compare_reg:              
                        no_rot_pred_gt = no_rot_pred.copy()
                        rotated_base_pred = euler_rotation_3d(no_rot_pred_gt[0], rot, mode = 'constant', crop_img=False,order=0, reshape_rot=True)

                        print("compute reg metrics")
                        tmp = torch.clone(pred)
                            
                        metrics_reg = compute_metrice_pc(rotated_base_pred,tmp)
                        metrics_score_reg.append({key :metrics_reg})
                    
                    print("compute performance metrics")
                    tmp = torch.clone(pred)
                    gt_tmp = torch.clone(gt)
                    metrics = compute_metrice_pc(gt_tmp, tmp) 
                    metrics_score.append({key :metrics})
                    
                    self.print_to_log_file(f"Rotation {key}" , also_print_to_console=False)
                    self.print_to_log_file(f"Reg RMSE {metrics_reg['RMSE']}" , also_print_to_console=False)
                    self.print_to_log_file(f"Reg DICE {metrics_reg['DICE']}" , also_print_to_console=False)
                    self.print_to_log_file(f"Perf RMSE {metrics['RMSE']}" , also_print_to_console=False)
                    self.print_to_log_file(f"Perf DICE {metrics['DICE']}\n" , also_print_to_console=False)
                    
                    
                    #rmse = compute_rmse_pc(gt, pred)
                    # print(f'RMSE {rmse}')
                    #rmse_score.append({key :rmse})
                    
                    #pred = torch.argmax(pred, axis=0)
                    #dice_score.append({key :compute_dice_pc(gt, pred)})

                    if np.sum(rot):
                        continue
                    if save_softmax:
                        softmax_fname = join(output_folder, fname + f'_{rot[0]}z_{rot[1]}y_{rot[2]}z' + ".npz")
                    else:
                        softmax_fname = None

                    """There is a problem with python process communication that prevents us from communicating objects
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(join(output_folder, fname + ".npy"), softmax_pred)
                        softmax_pred = join(output_folder, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                            ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                            properties, interpolation_order, self.regions_class_order,
                                                            None, None,
                                                            softmax_fname, None, force_separate_z,
                                                            interpolation_order_z),
                                                            )
                                                            )
                                )
                    

                
                pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

            # TODO stack pred and take mean of them, compute perf metrics a reg (?)
            if TTA:
                TTA_pred = torch.mean(torch.from_numpy(np.stack(TTA_pred)), dim=0)
                no_rot_pred_gt = no_rot_pred.copy()
                tmp = torch.clone(TTA_pred)
                TTA_metrics_id = compute_metrice_pc(no_rot_pred_gt[1], tmp) 
                TTA_metrics.append(TTA_metrics_id)
                if compare_reg:
                    tmp = torch.clone(TTA_pred)
                    no_rot_pred_gt = no_rot_pred.copy()
                    TTA_metrics_reg_id = compute_metrice_pc(no_rot_pred_gt[0],  tmp)
                    TTA_metrics_reg.append(TTA_metrics_reg_id)
                TTA_pred = []
        #json_path_dice = join(output_folder[:-15], f'DICE_M{M}_rotated_results.json')
        #out_file = open(json_path_dice, "w") 
        #json.dump(dice_score, out_file) 
        #out_file.close()
        
        #json_path_rmse = join(output_folder[:-15], f'RMSE_M{M}_rotated_results.json')
        #out_file = open(json_path_rmse, "w") 
        #json.dump(rmse_score, out_file) 
        #out_file.close()
        if save:
            if TTA:
                json_path_metrics = join(output_folder[:-15], f'TTA_metrics_M{M}_rotated_results.json')
                if step_size != 0.5:
                    json_path_metrics = join(output_folder[:-15], f'TTA_metrics_M{M}_rotated_results_ss{step_size}.json')
                out_file = open(json_path_metrics, "w") 
                json.dump(TTA_metrics, out_file) 
                out_file.close()
                if compare_reg:
                    json_path_metrics = join(output_folder[:-15], f'TTA_regulization_metrics_M{M}_rotated_results.json')
                    if step_size != 0.5:
                        json_path_metrics = join(output_folder[:-15], f'TTA_regulization_metrics_M{M}_rotated_results_ss{step_size}.json')
                    out_file = open(json_path_metrics, "w") 
                    json.dump(TTA_metrics_reg, out_file) 
                    out_file.close() 
                    
            json_path_metrics = join(output_folder[:-15], f'true_metrics_M{M}_rotated_results.json')
            if step_size != 0.5:
                json_path_metrics = join(output_folder[:-15], f'true_metrics_M{M}_rotated_results_ss{step_size}.json')
            out_file = open(json_path_metrics, "w") 
            json.dump(metrics_score, out_file) 
            out_file.close()
        
            if compare_reg:    
                json_path_metrics = join(output_folder[:-15], f'true_regulization_metrics_M{M}_rotated_results.json')
                if step_size != 0.5:
                    json_path_metrics = join(output_folder[:-15], f'true_regulization_metrics_M{M}_rotated_results_ss{step_size}.json')               
                out_file = open(json_path_metrics, "w") 
                json.dump(metrics_score_reg, out_file) 
                out_file.close() 
            
            
        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError:
                    print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                    traceback.print_exc()
                    attempts += 1
                    sleep(1)
            if not success:
                raise OSError(f"Something went wrong while copying nifti files to {gt_nifti_folder}. See above for the trace.")

        self.network.train(current_mode)
        
        self.network.do_ds = ds

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, self.splits_file)

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]
            
    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            print("Setting 3D augmentation")
            self.data_aug_params = default_3D_augmentation_params

            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
 

class nnUNetTrainerV2_RA_augment(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 100
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        self.prob_RA_augment = 1
        self.splits_file = "splits_final.pkl"        
                
    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        #"rotation_p_per_axis": 1,
        #"p_rot": 0.2,
        if self.threeD:
            print("Setting 3D augmentation")
            self.data_aug_params = default_3D_augmentation_params
            if self.prob_RA_augment:
                min_angle = 180.
                self.data_aug_params['p_rot'] = self.prob_RA_augment
            else:
                min_angle = 30.
                self.data_aug_params['p_rot'] = .2

            self.data_aug_params['rotation_x'] = (-min_angle / 360 * 2. * np.pi, min_angle / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-min_angle / 360 * 2. * np.pi, min_angle / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-min_angle / 360 * 2. * np.pi, min_angle / 360 * 2. * np.pi)

            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2


    def run_iteration_old(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        
        M = 24
        rotations = get_euler_angles(M)
        rot_id = random.choice([i for i in range(len(rotations))])

        rot = rotations[rot_id]

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if (random.random() <= self.prob_RA_augment) and self.prob_RA_augment and rot_id:
            if type(target) == list:
                lbl_trg = target.copy()
            data, target = euler_rotation_3d_RA(data, rot, lbl=lbl_trg, mode = 'constant', crop_img=False,order=0)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

class nnUNetTrainerV2_RA_augment_RA_quart(nnUNetTrainerV2_RA_augment):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 100
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        self.prob_RA_augment = .25
        self.splits_file = "splits_final.pkl"
# RUNNED
class nnUNetTrainerV2_Adam_drop(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 70
        self.initial_lr = 1e-3
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True


    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0.25, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        self.lr_scheduler = None

class newds_156_nnUNetTrainerV2(nnUNetTrainerV2_Adam_drop):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 100
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        training_sample = 156 # 26, 52, 104, 156
        
        self.splits_file = f"size_{training_sample}_splits_final.pkl"

   

class newds_156_nnUNetTrainerV2_RA_augment(nnUNetTrainerV2_RA_augment):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 100
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        training_sample = 156 # 26, 52, 104, 156
        self.prob_RA_augment = 1.
        
        self.splits_file = f"size_{training_sample}_splits_final.pkl"
 

 
class newds_156_nnUNetTrainerV2_RA_augment_again(nnUNetTrainerV2_RA_augment):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 100
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        training_sample = 156 # 26, 52, 104, 156
        self.prob_RA_augment = 1.
        
        self.splits_file = f"size_{training_sample}_splits_final.pkl"
  

class newds_156_nnUNetTrainerV2_RA_augment_SC(nnUNetTrainerV2_RA_augment):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 100
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        training_sample = 156 # 26, 52, 104, 156
        self.prob_RA_augment = 0.
        
        self.splits_file = f"size_{training_sample}_splits_final.pkl"

 
class nnUNetTrainerV2_ATM(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 0
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        self.r_arr = 0.001
        self.splits_file = "splits_final.pkl"

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)

            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params['patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)
    
    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        
        ds = self.network.do_ds
        self.network.do_ds = False

        current_mode = self.network.training
        self.network.eval()
        
        
        # TODO ? 
        self.splits_file = 'clean_test_data_27.pkl'
        self.splits_file = 'clean_test_data_22.pkl'
        self.splits_file = 'clean_test_data_15_22.pkl'
        self.splits_file = 'clean_test_data_endquart.pkl'
        self.splits_file = 'clean_test_data_10_15.pkl'
        self.splits_file = 'clean_test_data_sort_firsthalf.pkl'
        self.splits_file = 'clean_test_data_sort_endhalf.pkl'
        #self.splits_file = "test_data.pkl"

        self.dataset_val = None
        step_size = 1.
        print(f"In validate {step_size}")
        
        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            print("in init")
            self.print_to_log_file(f"Data test split: {self.splits_file}" , also_print_to_console=True)
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(default_num_threads)
        M = 24
        save_M = M
        TTA = False # Test time augmentations
        save = True
        compare_reg = True
        extended_metrics = True
        TTA_pred = []
        TTA_metrics = []
        TTA_metrics_reg = []
        rotations = get_euler_angles(M)
        
        r_fix = self.r_arr
        # zones :
        #  0 all sphere
        #  1 upper hemisphere (UH) z>0, 
        #  Halves
        #   11 right UH z>0 - y>0
        #   12 left UH z>0 - y<0
        #  Quarters
        #   21 front right quarter UH (QUH) z>0 - y>0 - x>0
        #   22 front left QHU z>0 - y<0 - x>0
        #   23 back right QUH z>0 - y>0 - x<0
        #   24 back left QHU z>0 - y<0 - x<0
        #  Cones
        #   4 values > .5 -> angle of 120 (60deg each)
        # For lower hemisphere, same but times -1
        if False:
            zone = 4
            z_rot_as_zone = False
            n_gamma=4
            desired_z_rot = 4
            # nbr of iterations i = 0 , 6 * z'' rot
            i = 2
            zone = 4
            z_rot_as_zone = True
            n_gamma=4
            desired_z_rot = 1
            # nbr of iterations
            nbr_iter = 2
            cone_half_angle = 60
            #M = int((4**(2+i)+8)*(desired_z_rot/n_gamma))
            #rotations = get_euler_angles_new(M=M, n_gamma=n_gamma, zone=zone, z_rot_as_zone=z_rot_as_zone)
            #save_M = len(rotations)
            rotations = get_euler_angles_new(nbr_iter=nbr_iter, n_gamma=n_gamma, zone=zone, z_rot_as_zone=z_rot_as_zone, desired_z_rot=desired_z_rot, cone_half_angle=cone_half_angle)
            save_M = str(len(rotations)) 
            if zone and cone_half_angle != 90:
                save_M = save_M + '_cone_HA_' + str(cone_half_angle)
            if desired_z_rot != n_gamma:
                save_M = save_M + '_nbr_Z''_' + str(desired_z_rot)
            
            
        results = []
        dice_score = []
        metrics_score_p = []
        metrics_score = []
        metrics_score_reg_p = []
        metrics_score_reg = [] # compare the data with the non rotated input prediction
        self.print_to_log_file(f"M {save_M} Step size {step_size} Extended Metrics {extended_metrics} M {save_M}" , also_print_to_console=False)
        for k in self.dataset_val.keys():
            properties = load_pickle(self.dataset[k]['properties_file'])
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            no_rot_pred = None
            self.print_to_log_file(f"Image: {k}" , also_print_to_console=False)
            for id_rot, rot in enumerate(rotations):
                
                if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                        (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                    data = np.load(self.dataset[k]['data_file'])['data']
                    print(f'id rot {id_rot} rotations {rot}')
                    print(f'shapes {k, data[-1].shape}') # data shape (img, GT), 3D volume
                    data[-1][data[-1] == -1] = 0
                    data_p, sc = pad_nd_image(data.copy(), np.array([80,80,80]), mode = 'constant', return_slicer = True, kwargs = {'constant_values': 0})

                    if np.sum(rot):
                        inv_rot = [-rot[i] for i in range(2, -1, -1)]
                        # TODO 
                        a = data.copy().sum()
                        r_pred, trg = euler_rotation_3d(data[:-1], rot, lbl=data[-1], mode = 'constant', crop_img=False,order=0, reshape_rot=True)
                        p_pred, p_trg = euler_rotation_3d(data_p[:-1], rot,lbl=data_p[-1], mode = 'constant', crop_img=False,order=0, reshape_rot=True)
                        
                        data = np.concatenate([r_pred, np.expand_dims(trg,axis=0)],axis=0) 
                    
                        data_p = np.concatenate([p_pred, np.expand_dims(p_trg,axis=0)],axis=0) 
                        
                                            
                    if type(data[-1]) is np.ndarray:
                        gt = torch.clone(torch.from_numpy(data[-1]))
                        gt_p = torch.clone(torch.from_numpy(data_p[-1]))
                    else:
                        gt = torch.clone(data[-1])
                        gt_p = torch.clone(data_p[-1])

                    # TODO rotate here
                    softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data[:-1],
                                                                                        do_mirroring=do_mirroring,
                                                                                        mirror_axes=mirror_axes,
                                                                                        use_sliding_window=use_sliding_window,
                                                                                        step_size=step_size,
                                                                                        use_gaussian=use_gaussian,
                                                                                        all_in_gpu=all_in_gpu,
                                                                                        mixed_precision=self.fp16,
                                                                                        verbose=False)[1]
                    softmax_pred_p = self.predict_preprocessed_data_return_seg_and_softmax(data_p[:-1],
                                                                                        do_mirroring=do_mirroring,
                                                                                        mirror_axes=mirror_axes,
                                                                                        use_sliding_window=use_sliding_window,
                                                                                        step_size=step_size,
                                                                                        use_gaussian=use_gaussian,
                                                                                        all_in_gpu=all_in_gpu,
                                                                                        mixed_precision=self.fp16,
                                                                                        verbose=False)[1]
                    
                    #print(f'\nSUMS: \n{data[-1].sum(), data_p[-1].sum()}\n {softmax_pred[0].sum(),softmax_pred[1].sum(),softmax_pred[2].sum()} \n{softmax_pred_p[0].sum(),softmax_pred_p[1].sum(),softmax_pred_p[2].sum()}\n')
                    #print(f'DICE DICE \n {data[:-1].shape, softmax_pred.shape}')
                    softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])
                    softmax_pred_p = softmax_pred_p.transpose([0] + [i + 1 for i in self.transpose_backward])
                    #print(f'{softmax_pred.shape} \n')
                    if TTA: 
                        # TODO append the reversed rotated softmax_pred
                        inv_rot = [-rot[i] for i in range(2, -1, -1)]
                        tmp = np.copy(softmax_pred)
                        TTA_pred.append(euler_rotation_3d(tmp, inv_rot, mode = 'constant', crop_img=False,order=0, reshape_rot=True))
                    
                    
                    if type(softmax_pred) is np.ndarray:
                        pred = torch.clone(torch.from_numpy(softmax_pred))
                        pred_p = torch.clone(torch.from_numpy(softmax_pred_p))
                    else:
                        pred = torch.clone(softmax_pred)
                        pred_p = torch.clone(softmax_pred_p)

                    key = f'{rot[0]}z_{rot[1]}y_{rot[2]}z'
                    
                    if not id_rot and (compare_reg or TTA):
                        tmp2 = torch.clone(pred)
                        no_rot_pred = [tmp2, torch.clone(gt)]
                        no_rot_pred_gt = no_rot_pred.copy()

                        # add the gt for TTA
                        tmp = torch.clone(pred)
                        metrics_reg = compute_metrice_pc(no_rot_pred_gt[0], tmp, extended_metrics, r_fix) 
                        metrics_score_reg.append({key :metrics_reg})
                        
                        no_rot_pred_p = [torch.clone(pred_p), torch.clone(gt_p)]
                        no_rot_pred_gt = no_rot_pred_p.copy()

                        # add the gt for TTA
                        tmp = torch.clone(pred_p)
                        metrics_reg_p = compute_metrice_pc(no_rot_pred_gt[0], tmp, extended_metrics, r_fix) 
                        metrics_score_reg_p.append({key :metrics_reg_p})
                        print(f"compute reg")
                    elif compare_reg:   
                                   
                        no_rot_pred_gt = no_rot_pred.copy()  
                        rotated_base_pred = euler_rotation_3d(no_rot_pred_gt[0], rot, mode = 'constant', crop_img=False,order=0, reshape_rot=True)
                        tmp = torch.clone(pred)
                        metrics_reg = compute_metrice_pc(rotated_base_pred,tmp, extended_metrics, r_fix)
                        metrics_score_reg.append({key :metrics_reg})
                        print(f"compute reg")
                                   
                        no_rot_pred_gt_p = no_rot_pred_p.copy()  
                        rotated_base_pred_p = euler_rotation_3d(no_rot_pred_gt_p[0], rot, mode = 'constant', crop_img=False,order=0, reshape_rot=True)
                        tmp = torch.clone(pred_p)
                        metrics_reg_p = compute_metrice_pc(rotated_base_pred_p,tmp, extended_metrics, r_fix)
                        #print(f"compute reg {metrics_reg['RMSE'], rotated_base_pred.shape, tmp.shape}\n\n")

                        metrics_score_reg_p.append({key :metrics_reg_p})
                    tmp = torch.clone(pred)
                    gt_tmp = torch.clone(gt)
                    metrics = compute_metrice_pc(gt_tmp, tmp, extended_metrics, r_fix) 
                    metrics_score.append({key :metrics})
                    
                    # Padded
                    tmp = torch.clone(pred_p)
                    gt_tmp = torch.clone(gt_p)
                    metrics_p = compute_metrice_pc(gt_tmp, tmp, extended_metrics, r_fix) 
                    metrics_score_p.append({key :metrics_p})
                    
                    self.print_to_log_file(f"Rotation {key}:" , also_print_to_console=False)
                    self.print_to_log_file(f"Reg RMSE {metrics_reg['RMSE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"Reg DICE {metrics_reg['DICE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"Perf RMSE {metrics['RMSE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"Perf DICE {metrics['DICE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"P Reg RMSE {metrics_reg_p['RMSE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"P Reg DICE {metrics_reg_p['DICE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"P Perf RMSE {metrics_p['RMSE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"P Perf DICE {metrics_p['DICE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"========================================\n" , also_print_to_console=False)
                    
                    #rmse = compute_rmse_pc(gt, pred)
                    # print(f'RMSE {rmse}')
                    #rmse_score.append({key :rmse})
                    
                    #pred = torch.argmax(pred, axis=0)
                    #dice_score.append({key :compute_dice_pc(gt, pred)})

                    if np.sum(rot):
                        continue
                    if save_softmax:
                        softmax_fname = join(output_folder, fname + f'_{rot[0]}z_{rot[1]}y_{rot[2]}z' + ".npz")
                    else:
                        softmax_fname = None

                    """There is a problem with python process communication that prevents us from communicating objects
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(join(output_folder, fname + ".npy"), softmax_pred)
                        softmax_pred = join(output_folder, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                            ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                            properties, interpolation_order, self.regions_class_order,
                                                            None, None,
                                                            softmax_fname, None, force_separate_z,
                                                            interpolation_order_z),
                                                            )
                                                            )
                                )
                    

                
                pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

            # TODO stack pred and take mean of them, compute perf metrics a reg (?)
            if TTA:
                TTA_pred = torch.mean(torch.from_numpy(np.stack(TTA_pred)), dim=0)
                no_rot_pred_gt = no_rot_pred.copy()
                tmp = torch.clone(TTA_pred)
                TTA_metrics_id = compute_metrice_pc(no_rot_pred_gt[1], tmp, extended_metrics, r_fix) 
                TTA_metrics.append(TTA_metrics_id)
                if compare_reg:
                    tmp = torch.clone(TTA_pred)
                    no_rot_pred_gt = no_rot_pred.copy()
                    TTA_metrics_reg_id = compute_metrice_pc(no_rot_pred_gt[0],  tmp, extended_metrics, r_fix)
                    TTA_metrics_reg.append(TTA_metrics_reg_id)
                TTA_pred = []
        #json_path_dice = join(output_folder[:-15], f'DICE_M{M}_rotated_results.json')
        #out_file = open(json_path_dice, "w") 
        #json.dump(dice_score, out_file) 
        #out_file.close()
        
        #json_path_rmse = join(output_folder[:-15], f'RMSE_M{M}_rotated_results.json')
        #out_file = open(json_path_rmse, "w") 
        #json.dump(rmse_score, out_file) 
        #out_file.close()
        if save:
            if TTA:
                
                #son_path_metrics = join(output_folder[:-15], f'TTA_metrics_M{M}_rotated_results.json')
                #if step_size != 0.5:
                #    json_path_metrics = join(output_folder[:-15], f'TTA_metrics_M{M}_rotated_results_ss{step_size}.json')
                
                base_str = f'TTA_metrics_M{save_M}_rotated_results'
                if step_size != 0.5:
                    base_str = base_str + f'_ss{step_size}'
                if extended_metrics:
                    base_str = base_str + '_EM'
                base_str = base_str + '.json'
                
                json_path_metrics = join(output_folder[:-15], base_str)
                out_file = open(json_path_metrics, "w") 
                json.dump(TTA_metrics, out_file) 
                out_file.close()
                if compare_reg:
                    #json_path_metrics = join(output_folder[:-15], f'TTA_regulization_metrics_M{M}_rotated_results.json')
                    #if step_size != 0.5:
                    #    json_path_metrics = join(output_folder[:-15], f'TTA_regulization_metrics_M{M}_rotated_results_ss{step_size}.json')
                    
                    base_str = f'TTA_regulization_metrics_M{save_M}_rotated_results'
                    if step_size != 0.5:
                        base_str = base_str + f'_ss{step_size}'
                    if extended_metrics:
                        base_str = base_str + '_EM'
                    base_str = base_str + '.json'
                    json_path_metrics = join(output_folder[:-15], base_str)
                    
                    out_file = open(json_path_metrics, "w") 
                    json.dump(TTA_metrics_reg, out_file) 
                    out_file.close() 
                    
            base_str = f'true_metrics_M{save_M}_rotated_results'
            if step_size != 0.5:
                base_str = base_str + f'_ss{step_size}'
            if extended_metrics:
                base_str = base_str + '_EM'
            base_str = base_str + '.json'
            json_path_metrics = join(output_folder[:-15], base_str)   
                             
            #json_path_metrics = join(output_folder[:-15], f'true_metrics_M{M}_rotated_results.json')
            #if step_size != 0.5:
            #    json_path_metrics = join(output_folder[:-15], f'true_metrics_M{M}_rotated_results_ss{step_size}.json')
            out_file = open(json_path_metrics, "w") 
            json.dump(metrics_score, out_file) 
            out_file.close()
            
            base_str = f'true_metrics_padded_M{save_M}_rotated_results'
            if step_size != 0.5:
                base_str = base_str + f'_ss{step_size}'
            if extended_metrics:
                base_str = base_str + '_EM'
            base_str = base_str + '.json'
            json_path_metrics = join(output_folder[:-15], base_str)   
                             
            out_file = open(json_path_metrics, "w") 
            json.dump(metrics_score_p, out_file) 
            out_file.close()
        
            if compare_reg:    
                base_str = f'true_regulization_metrics_M{save_M}_rotated_results'
                if step_size != 0.5:
                    base_str = base_str + f'_ss{step_size}'
                if extended_metrics:
                    base_str = base_str + '_EM'
                base_str = base_str + '.json'
                json_path_metrics = join(output_folder[:-15], base_str)   
                #json_path_metrics = join(output_folder[:-15], f'true_regulization_metrics_M{M}_rotated_results.json')
                #if step_size != 0.5:
                #    json_path_metrics = join(output_folder[:-15], f'true_regulization_metrics_M{M}_rotated_results_ss{step_size}.json')               
                out_file = open(json_path_metrics, "w") 
                json.dump(metrics_score_reg, out_file) 
                out_file.close() 

                base_str = f'true_regulization_metrics_paddded_M{save_M}_rotated_results'
                if step_size != 0.5:
                    base_str = base_str + f'_ss{step_size}'
                if extended_metrics:
                    base_str = base_str + '_EM'
                base_str = base_str + '.json'
                json_path_metrics = join(output_folder[:-15], base_str)          
                out_file = open(json_path_metrics, "w") 
                json.dump(metrics_score_reg_p, out_file) 
                out_file.close() 
            
        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError:
                    print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                    traceback.print_exc()
                    attempts += 1
                    sleep(1)
            if not success:
                raise OSError(f"Something went wrong while copying nifti files to {gt_nifti_folder}. See above for the trace.")

        self.network.train(current_mode)
        
        self.network.do_ds = ds

    
    def validate_old(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        
        ds = self.network.do_ds
        self.network.do_ds = False

        current_mode = self.network.training
        self.network.eval()
        
        # TODO ? 
        self.splits_file = "test_data.pkl"
        self.splits_file = "test_data_25.pkl"
        self.dataset_val = None
        step_size = 1.
        print(f"In validate {step_size}")
        
        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            print("in init")
            self.print_to_log_file(f"Data test split: {self.splits_file}" , also_print_to_console=True)
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = join(self.output_folder, validation_folder_name)
        maybe_mkdir_p(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, join(output_folder, "validation_args.json"))

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(default_num_threads)
        M = 24
        TTA = True # Test time augmentations
        save = True
        compare_reg = True
        extended_metrics = True
        TTA_pred = []
        TTA_metrics = []
        TTA_metrics_reg = []
        rotations = get_euler_angles(M)
        results = []
        dice_score = []
        rmse_score = []
        metrics_score = []
        metrics_score_reg_inv = []
        metrics_score_reg = [] # compare the data with the non rotated input prediction
        self.print_to_log_file(f"Extended Metrics: {extended_metrics}" , also_print_to_console=True)
        for k in self.dataset_val.keys():
            properties = load_pickle(self.dataset[k]['properties_file'])
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            no_rot_pred = None
            self.print_to_log_file(f"Image: {k} \n Step size {step_size}" , also_print_to_console=False)
            for id_rot, rot in enumerate(rotations):
                
                #fname = fname + f'_{rot[0]}z_{rot[1]}y_{rot[2]}z'
                if overwrite or (not isfile(join(output_folder, fname + ".nii.gz"))) or \
                        (save_softmax and not isfile(join(output_folder, fname + ".npz"))):
                    data = np.load(self.dataset[k]['data_file'])['data']
                    print(f'id rot {id_rot} rotations {rot}')
                    print(f'shapes {k, data[-1].shape}') # data shape (img, GT), 3D volume
                    data[-1][data[-1] == -1] = 0
                    #data = pad_nd_image(data, np.array([40,40,40]), mode = 'constant', kwargs = {'constant_values': 0})

                    if np.sum(rot):
                        r_pred, trg = euler_rotation_3d(data[:-1], rot, lbl=data[-1], mode = 'constant', crop_img=False,order=0, reshape_rot=True)
                        print(f'YES YES {r_pred.shape, trg.shape}')
                        data = np.concatenate([r_pred, np.expand_dims(trg,axis=0)],axis=0) 
                    if type(data[-1]) is np.ndarray:
                        gt = torch.clone(torch.from_numpy(data[-1]))
                    else:
                        gt = torch.clone(data[-1])

                    # TODO rotate here
                    softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data[:-1],
                                                                                        do_mirroring=do_mirroring,
                                                                                        mirror_axes=mirror_axes,
                                                                                        use_sliding_window=use_sliding_window,
                                                                                        step_size=step_size,
                                                                                        use_gaussian=use_gaussian,
                                                                                        all_in_gpu=all_in_gpu,
                                                                                        mixed_precision=self.fp16)[1]

                    softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])
                    if TTA: 
                        # TODO append the reversed rotated softmax_pred
                        inv_rot = [-rot[i] for i in range(2, -1, -1)]
                        tmp = np.copy(softmax_pred)
                        TTA_pred.append(euler_rotation_3d(tmp, inv_rot, mode = 'constant', crop_img=False,order=0, reshape_rot=True))
                    
                    
                    if type(softmax_pred) is np.ndarray:
                        pred = torch.clone(torch.from_numpy(softmax_pred))
                    else:
                        pred = torch.clone(softmax_pred)
                        
                   
                        

                
                    key = f'{rot[0]}z_{rot[1]}y_{rot[2]}z'
                    
                    if not id_rot and (compare_reg or TTA):
                        tmp2 = torch.clone(pred)
                        no_rot_pred = [tmp2, torch.clone(gt)]
                        no_rot_pred_gt = no_rot_pred.copy()
                        # add the gt for TTA
                        print("compute reg metrics")
                        tmp = torch.clone(pred)
                        metrics_reg = compute_metrice_pc(no_rot_pred_gt[0], tmp, extended_metrics) 
                        print(metrics_reg['RMSE'])
                        metrics_score_reg.append({key :metrics_reg})
                        metrics_score_reg_inv.append({key :metrics_reg})
                    elif compare_reg:              
                        no_rot_pred_gt = no_rot_pred.copy()

                        rotated_base_pred = euler_rotation_3d(no_rot_pred_gt[0], rot, mode = 'constant', crop_img=False,order=0, reshape_rot=True)

                        print("compute reg metrics")
                        tmp = torch.clone(pred)
                            
                        metrics_reg = compute_metrice_pc(rotated_base_pred,tmp, extended_metrics)
                        metrics_score_reg.append({key :metrics_reg})
                    
                    print("compute performance metrics")
                    tmp = torch.clone(pred)
                    gt_tmp = torch.clone(gt)
                    metrics = compute_metrice_pc(gt_tmp, tmp, extended_metrics) 
                    metrics_score.append({key :metrics})
                    
                    self.print_to_log_file(f"Rotation {key}:" , also_print_to_console=False)
                    self.print_to_log_file(f"Reg RMSE {metrics_reg['RMSE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"Reg DICE {metrics_reg['DICE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"Perf RMSE {metrics['RMSE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"Perf DICE {metrics['DICE']}:" , also_print_to_console=False)
                    self.print_to_log_file(f"========================================\n" , also_print_to_console=False)
                    
                    #rmse = compute_rmse_pc(gt, pred)
                    # print(f'RMSE {rmse}')
                    #rmse_score.append({key :rmse})
                    
                    #pred = torch.argmax(pred, axis=0)
                    #dice_score.append({key :compute_dice_pc(gt, pred)})

                    if np.sum(rot):
                        continue
                    if save_softmax:
                        softmax_fname = join(output_folder, fname + f'_{rot[0]}z_{rot[1]}y_{rot[2]}z' + ".npz")
                    else:
                        softmax_fname = None

                    """There is a problem with python process communication that prevents us from communicating objects
                    larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                    communicated by the multiprocessing.Pipe object then the placeholder (I think) does not allow for long
                    enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                    patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                    then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                    filename or np.ndarray and will handle this automatically"""
                    if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                        np.save(join(output_folder, fname + ".npy"), softmax_pred)
                        softmax_pred = join(output_folder, fname + ".npy")

                    results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                            ((softmax_pred, join(output_folder, fname + ".nii.gz"),
                                                            properties, interpolation_order, self.regions_class_order,
                                                            None, None,
                                                            softmax_fname, None, force_separate_z,
                                                            interpolation_order_z),
                                                            )
                                                            )
                                )
                    

                
                pred_gt_tuples.append([join(output_folder, fname + ".nii.gz"),
                                   join(self.gt_niftis_folder, fname + ".nii.gz")])

            # TODO stack pred and take mean of them, compute perf metrics a reg (?)
            if TTA:
                TTA_pred = torch.mean(torch.from_numpy(np.stack(TTA_pred)), dim=0)
                no_rot_pred_gt = no_rot_pred.copy()
                tmp = torch.clone(TTA_pred)
                TTA_metrics_id = compute_metrice_pc(no_rot_pred_gt[1], tmp, extended_metrics) 
                TTA_metrics.append(TTA_metrics_id)
                if compare_reg:
                    tmp = torch.clone(TTA_pred)
                    no_rot_pred_gt = no_rot_pred.copy()
                    TTA_metrics_reg_id = compute_metrice_pc(no_rot_pred_gt[0],  tmp, extended_metrics)
                    TTA_metrics_reg.append(TTA_metrics_reg_id)
                TTA_pred = []
        #json_path_dice = join(output_folder[:-15], f'DICE_M{M}_rotated_results.json')
        #out_file = open(json_path_dice, "w") 
        #json.dump(dice_score, out_file) 
        #out_file.close()
        
        #json_path_rmse = join(output_folder[:-15], f'RMSE_M{M}_rotated_results.json')
        #out_file = open(json_path_rmse, "w") 
        #json.dump(rmse_score, out_file) 
        #out_file.close()
        if save:
            if TTA:
                
                #son_path_metrics = join(output_folder[:-15], f'TTA_metrics_M{M}_rotated_results.json')
                #if step_size != 0.5:
                #    json_path_metrics = join(output_folder[:-15], f'TTA_metrics_M{M}_rotated_results_ss{step_size}.json')
                
                base_str = f'TTA_metrics_M{M}_rotated_results'
                if step_size != 0.5:
                    base_str = base_str + f'_ss{step_size}'
                if extended_metrics:
                    base_str = base_str + '_EM'
                base_str = base_str + '.json'
                
                json_path_metrics = join(output_folder[:-15], base_str)
                out_file = open(json_path_metrics, "w") 
                json.dump(TTA_metrics, out_file) 
                out_file.close()
                if compare_reg:
                    #json_path_metrics = join(output_folder[:-15], f'TTA_regulization_metrics_M{M}_rotated_results.json')
                    #if step_size != 0.5:
                    #    json_path_metrics = join(output_folder[:-15], f'TTA_regulization_metrics_M{M}_rotated_results_ss{step_size}.json')
                    
                    base_str = f'TTA_regulization_metrics_M{M}_rotated_results'
                    if step_size != 0.5:
                        base_str = base_str + f'_ss{step_size}'
                    if extended_metrics:
                        base_str = base_str + '_EM'
                    base_str = base_str + '.json'
                    json_path_metrics = join(output_folder[:-15], base_str)
                    
                    out_file = open(json_path_metrics, "w") 
                    json.dump(TTA_metrics_reg, out_file) 
                    out_file.close() 
                    
            base_str = f'true_metrics_M{M}_rotated_results'
            if step_size != 0.5:
                base_str = base_str + f'_ss{step_size}'
            if extended_metrics:
                base_str = base_str + '_EM'
            base_str = base_str + '.json'
            json_path_metrics = join(output_folder[:-15], base_str)   
                             
            #json_path_metrics = join(output_folder[:-15], f'true_metrics_M{M}_rotated_results.json')
            #if step_size != 0.5:
            #    json_path_metrics = join(output_folder[:-15], f'true_metrics_M{M}_rotated_results_ss{step_size}.json')
            out_file = open(json_path_metrics, "w") 
            json.dump(metrics_score, out_file) 
            out_file.close()
        
            if compare_reg:    
                base_str = f'true_regulization_metrics_M{M}_rotated_results'
                if step_size != 0.5:
                    base_str = base_str + f'_ss{step_size}'
                if extended_metrics:
                    base_str = base_str + '_EM'
                base_str = base_str + '.json'
                json_path_metrics = join(output_folder[:-15], base_str)   
                #json_path_metrics = join(output_folder[:-15], f'true_regulization_metrics_M{M}_rotated_results.json')
                #if step_size != 0.5:
                #    json_path_metrics = join(output_folder[:-15], f'true_regulization_metrics_M{M}_rotated_results_ss{step_size}.json')               
                out_file = open(json_path_metrics, "w") 
                json.dump(metrics_score_reg, out_file) 
                out_file.close() 
            
            
        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=join(output_folder, "summary.json"),
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=default_num_threads)

        if run_postprocessing_on_folds:
            # in the old nnunet we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = join(self.output_folder_base, "gt_niftis")
        maybe_mkdir_p(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError:
                    print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                    traceback.print_exc()
                    attempts += 1
                    sleep(1)
            if not success:
                raise OSError(f"Something went wrong while copying nifti files to {gt_nifti_folder}. See above for the trace.")

        self.network.train(current_mode)
        
        self.network.do_ds = ds

    

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, self.splits_file)

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()

        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]
            
    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            print("Setting 3D augmentation")
            self.data_aug_params = default_3D_augmentation_params

            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
 
 
class nnUNetTrainerV2_ATM_RA_drop(nnUNetTrainerV2_ATM):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 100
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        self.prob_RA_augment = 1. # base 30
        self.patch_size = np.array([80,80,80])
            
    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """

        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        self.patch_size = np.array([80,80,80])
        if self.threeD:
            print("Setting 3D augmentation")
            
            self.data_aug_params = default_3D_augmentation_params
            if self.prob_RA_augment:
                min_angle = 180.
                self.data_aug_params['p_rot'] = self.prob_RA_augment
            else:
                min_angle = 30.
                self.data_aug_params['p_rot'] = .2
                
            self.data_aug_params['patch_size'] = self.patch_size
            
            self.data_aug_params['rotation_x'] = (-min_angle / 360 * 2. * np.pi, min_angle / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-min_angle / 360 * 2. * np.pi, min_angle / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-min_angle / 360 * 2. * np.pi, min_angle / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        else:
            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = self.patch_size

        self.data_aug_params["num_cached_per_thread"] = 2

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0.25, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


class newds_156_nnUNetTrainerV2_RA_ATM_augment_SC_100(nnUNetTrainerV2_ATM_RA_drop):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 100
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        training_sample = 156 # 26, 52, 104, 156
        self.patch_size = np.array([80,80,80])
        self.prob_RA_augment = 0.
        
        self.splits_file = f"size_{training_sample}_splits_final.pkl"


class newds_156_nnUNetTrainerV2_RA_ATM_augment_100e(nnUNetTrainerV2_ATM_RA_drop):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 100
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        training_sample = 156 # 26, 52, 104, 156
        self.patch_size = np.array([80,80,80])
        self.prob_RA_augment = 1.
        
        self.splits_file = f"size_{training_sample}_splits_final.pkl"
   

def euler_rotation_3d(img, angle, lbl=None, mode = 'nearest', denoise=False, denoise_eps=0.0001, crop_img=True,order=0, reshape_rot=False):
    """

    Arguments:
    angle: angles to rotate according to euler zyz

    Returns:
    rotated 3D image
    """
    transform_str = f'3D_rotation_Axis_{angle}'
    input_type = type(img)

    if input_type == torch.Tensor:
        img = img.numpy()
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
        
    #img_rot = np.zeros(img.shape)
    # rotate 1,2 rotate facing 1 according to [1,2,3]
    # rotate 0,2 rotate facing 2 according to [1,2,3]
    # rotate 0,1 rotate facing 3 according to [1,2,3]
    nbr_c = img.shape[0]
    
    if lbl is not None:
        target_type = type(lbl)
        if target_type is list:
            target_type = type(lbl[0])

    # rotate along z-axis
    if angle[0]:
        tmp = []
        for c in range(nbr_c):
            tmp.append(scipy.ndimage.rotate(img[c], -angle[0], mode=mode, axes=(0,1), reshape=reshape_rot, order=order,cval=0))
        img = tmp.copy()
        if lbl is not None:
            lbl = scipy.ndimage.rotate(lbl, -angle[0], mode=mode, axes=(0,1), reshape=reshape_rot, order=order)
            lbl = np.where(lbl<denoise_eps,0.,lbl)


    # rotate along y-axis
    if angle[1]:
        tmp = []
        for c in range(nbr_c):
            tmp.append(scipy.ndimage.rotate(img[c], angle[1], mode=mode, axes=(1, 2), reshape=reshape_rot, order=order,cval=0))
        img = tmp.copy()
        if lbl is not None:
            lbl = scipy.ndimage.rotate(lbl, angle[1], mode=mode, axes=(1,2), reshape=reshape_rot, order=order)
            lbl = np.where(lbl<denoise_eps,0.,lbl)


    # rotate along z-axis
    if angle[2]:
        tmp = []
        for c in range(nbr_c):
            tmp.append(scipy.ndimage.rotate(img[c], -angle[2], mode=mode, axes=(0,1), reshape=reshape_rot, order=order,cval=0))
        img = tmp.copy()
        
        if lbl is not None:
            lbl = scipy.ndimage.rotate(lbl, -angle[2], mode=mode, axes=(0,1), reshape=reshape_rot, order=order)
            lbl = np.where(lbl<denoise_eps,0.,lbl)

    img_rot = np.stack(img, axis=0)
    if denoise:
      img_rot = np.where(img_rot<denoise_eps,0.,img_rot)
    
    if input_type == torch.Tensor:
        img_rot = torch.from_numpy(img_rot)
    if lbl is None:
        return img_rot
    else:
        return img_rot, lbl

def euler_rotation_3d_RA(img, angle, lbl=None, mode = 'constant', denoise=False, denoise_eps=0.0001, crop_img=False,order=0):
    """

    Arguments:
    angle: angles to rotate according to euler zyz

    Returns:
    rotated 3D image
    """
    transform_str = f'3D_rotation_Axis_{angle}'
    input_type = type(img)

    if input_type == torch.Tensor:
        img = img.numpy()
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    
    img_rot = np.zeros(img.shape)
    # rotate 1,2 rotate facing 1 according to [1,2,3]
    # rotate 0,2 rotate facing 2 according to [1,2,3]
    # rotate 0,1 rotate facing 3 according to [1,2,3]
    
    if lbl is not None:
        target_type = type(lbl)
        if target_type is list:
            target_type = type(lbl[0])
            if target_type == torch.Tensor:
                lbl = [lbl[id].numpy() if len(lbl[id].shape) >= 4 else np.expand_dims(lbl[id].numpy(), axis=0) for id in range(len(lbl))]
        elif target_type == torch.Tensor:
            lbl = lbl.numpy()
            if len(lbl.shape) == 3:
                lbl = np.expand_dims(lbl, axis=0)
        elif len(lbl.shape) == 3:
            lbl = np.expand_dims(lbl, axis=0)

    # rotate along z-axis
    if angle[0]:
        for c in range(img.shape[0]):
            if len(img[c].shape) > 3:
                for i in range(img[c].shape[0]):
                    img[c,i] = scipy.ndimage.rotate(img[c,i], -angle[0], mode=mode, axes=(0,1), reshape=False, order=order)
            else:
                img[c] = scipy.ndimage.rotate(img[c], -angle[0], mode=mode, axes=(0,1), reshape=False, order=order)
        if lbl is not None:
            if type(lbl) is list:
                for id in range(len(lbl)):
                    for c in range(lbl[id].shape[0]):
                        lbl[id][lbl[id] == -1] = 0
                        if len(lbl[id][c].shape) > 3:
                            for i in range(lbl[id][c].shape[0]):
                                lbl[id][c,i] = scipy.ndimage.rotate(lbl[id][c,i], -angle[0], mode=mode, axes=(0,1), reshape=False, order=order)
                        else:
                            lbl[id][c] = scipy.ndimage.rotate(lbl[id][c], -angle[0], mode=mode, axes=(0,1), reshape=False, order=order)
                    lbl[id] = np.where(lbl[id]<denoise_eps,0.,lbl[id])
            else:
                for c in range(lbl.shape[0]):
                    lbl[lbl == -1] = 0
                    if len(lbl[c].shape) > 3:
                        for i in range(lbl.shape[0]):
                            lbl[c,i] = scipy.ndimage.rotate(lbl[c,i], -angle[0], mode=mode, axes=(0,1), reshape=False, order=order)
                    else:
                        lbl[c] = scipy.ndimage.rotate(lbl[c], -angle[0], mode=mode, axes=(0,1), reshape=False, order=order)

                lbl = np.where(lbl<denoise_eps,0.,lbl)

    # rotate along y-axis
    if angle[1]:
        for c in range(img.shape[0]):
            if len(img[c].shape) > 3:
                for i in range(img[c].shape[0]):
                    img[c,i] = scipy.ndimage.rotate(img[c,i], angle[1], mode=mode, axes=(1,2), reshape=False, order=order)
            else:
                img[c] = scipy.ndimage.rotate(img[c], angle[1], mode=mode, axes=(1,2), reshape=False, order=order)
        if lbl is not None:
            if type(lbl) is list:
                for id in range(len(lbl)):
                    for c in range(lbl[id].shape[0]):
                        lbl[id][lbl[id] == -1] = 0
                        if len(lbl[id][c].shape) > 3:
                            for i in range(lbl[id][c].shape[0]):
                                lbl[id][c,i] = scipy.ndimage.rotate(lbl[id][c,i], angle[1], mode=mode, axes=(1,2), reshape=False, order=order)
                        else:
                            lbl[id][c] = scipy.ndimage.rotate(lbl[id][c], angle[1], mode=mode, axes=(1,2), reshape=False, order=order)
                    lbl[id] = np.where(lbl[id]<denoise_eps,0.,lbl[id])
            else:
                for c in range(lbl.shape[0]):
                    lbl[lbl == -1] = 0
                    if len(lbl[c].shape) > 3 :
                        for i in range(lbl[c].shape[0]):
                            lbl[c,i] = scipy.ndimage.rotate(lbl[c,i], angle[1], mode=mode, axes=(1,2), reshape=False, order=order)
                    else:
                        lbl[c] = scipy.ndimage.rotate(lbl[c], angle[1], mode=mode, axes=(1,2), reshape=False, order=order)
                lbl = np.where(lbl<denoise_eps,0.,lbl)

    # rotate along z-axis
    if angle[2]:
        for c in range(img.shape[0]):
            if len(img[c].shape) > 3:
                for i in range(img[c].shape[0]):
                    img[c,i] = scipy.ndimage.rotate(img[c,i], -angle[2], mode=mode, axes=(0,1), reshape=False, order=order)
            else:
                img[c] = scipy.ndimage.rotate(img[c], -angle[2], mode=mode, axes=(0,1), reshape=False, order=order)
        if lbl is not None:
            if type(lbl) is list:
                for id in range(len(lbl)):
                    for c in range(lbl[id].shape[0]):
                        lbl[id][lbl[id] == -1] = 0
                        
                        if len(lbl[id][c].shape)>3:
                            for i in range(lbl[id][c].shape[0]):
                                lbl[id][c,i] = scipy.ndimage.rotate(lbl[id][c,i], -angle[2], mode=mode, axes=(0,1), reshape=False, order=order)
                        else:
                            lbl[id][c] = scipy.ndimage.rotate(lbl[id][c], -angle[2], mode=mode, axes=(0,1), reshape=False, order=order)
                    lbl[id] = np.where(lbl[id]<denoise_eps,0.,lbl[id])
            else:
                for c in range(lbl.shape[0]):
                    lbl[lbl == -1] = 0
                    if len(lbl[c].shape)>3:
                        for i in range(lbl[c].shape[0]):
                            lbl[c,i] = scipy.ndimage.rotate(lbl[c,i], -angle[2], mode=mode, axes=(0,1), reshape=False, order=order)
                    else:
                        lbl[c] = scipy.ndimage.rotate(lbl[c], -angle[2], mode=mode, axes=(0,1), reshape=False, order=order)
                lbl = np.where(lbl<denoise_eps,0.,lbl)

    if denoise:
      img_rot = np.where(img<denoise_eps,0.,img)
    else:
      img_rot = img

    if crop_img:
      center = int(img_rot.shape[1]/2)
      new_half_img = int(np.cos(np.pi/4)*(center))
      center = [center-1, center-1, center-1]
      if len(img_rot.shape) == 4:
          img_rot = img_rot[:,center[0]-new_half_img:center[0]+new_half_img,
                          center[1]-new_half_img:center[1]+new_half_img,
                          center[2]-new_half_img:center[2]+new_half_img]
      else:
        print("Error. Please provide a correct format for label file which is either Dim1xDim2xDim3 or batch_sizexDim1xDim2xDim3")
      if lbl is not None:

        if type(lbl) is list:
            for id in range(len(lbl)):
                if len(lbl[id].shape) == 3:
                    lbl[id] = lbl[id][center[0]-new_half_img:center[0]+new_half_img,
                                    center[1]-new_half_img:center[1]+new_half_img,
                                    center[2]-new_half_img:center[2]+new_half_img]
                elif len(lbl[id].shape) == 4:
                    lbl[id] = lbl[id][:,center[0]-new_half_img:center[0]+new_half_img,
                                    center[1]-new_half_img:center[1]+new_half_img,
                                    center[2]-new_half_img:center[2]+new_half_img]
                else:
                    print("Error. Please provide a correct format for label file which is either Dim1xDim2xDim3 or batch_sizexDim1xDim2xDim3")
        else:
            lbl = lbl[center[0]-new_half_img:center[0]+new_half_img,
                            center[1]-new_half_img:center[1]+new_half_img,
                            center[2]-new_half_img:center[2]+new_half_img]
        
            if len(lbl.shape) == 3:
                lbl = lbl[center[0]-new_half_img:center[0]+new_half_img,
                          center[1]-new_half_img:center[1]+new_half_img,
                          center[2]-new_half_img:center[2]+new_half_img]
            elif len(lbl.shape) == 3:
                lbl = lbl[:,center[0]-new_half_img:center[0]+new_half_img,
                          center[1]-new_half_img:center[1]+new_half_img,
                          center[2]-new_half_img:center[2]+new_half_img]
            else:
                print("Error. Please provide a correct format for label file which is either Dim1xDim2xDim3 or batch_sizexDim1xDim2xDim3")
    if input_type == torch.Tensor:
        img_rot = torch.from_numpy(img_rot)
    if lbl is None:
        return img_rot
    else:
        if type(lbl) is list:
            lbl = [torch.from_numpy(lbl[id]) if target_type == torch.Tensor else lbl[id] for id in range(len(lbl))]
        return img_rot, lbl


def change_vars(MG):
    """
    MG: np array of shape (3,...) containing 3D cartesian coordinates.
    returns spherical coordinates theta and phi (could return rho if needed)
    """
    rho = np.sqrt(np.sum(np.square(MG), axis=0))
    phi = np.squeeze(np.arctan2(MG[1, ...], MG[0, ...])) 
    theta = np.squeeze(np.arccos(MG[2, ...] / rho))
    # The center value is Nan due to the 0/0. So make it 0.
    theta[np.isnan(theta)] = 0
    rho = np.squeeze(rho)

    return theta, phi

def rev_chane_vars(phi, theta, rho=1):
    
    x = rho*np.sin(phi)*np.cos(theta)
    y = rho*np.sin(phi)*np.sin(theta)
    z = rho*np.cos(phi)
    DP = np.stack([x,y,z],axis=1)
    return DP
    
def arsNorm(A):
    # vectorized norm() function
    rez = A[:, 0] ** 2 + A[:, 1] ** 2 + A[:, 2] ** 2
    rez = np.sqrt(rez)
    return rez


def arsUnit(A, radius):
    # vectorized unit() functon
    normOfA = arsNorm(A)
    rez = A / np.stack((normOfA, normOfA, normOfA), 1)
    rez = rez * radius
    return rez

def sphereTriangulation(M, n_gamma):
    """
    Defines points on the sphere that we use for alpha (z) and beta (y') Euler angles sampling. We can have 24 points (numIterations=0), 72 (numIterations=1), 384 (numIterations=2) etc.
    Copied from the matlab function https://ch.mathworks.com/matlabcentral/fileexchange/38909-parametrized-uniform-triangulation-of-3d-circle-sphere
    M is the number total of orientation, i.e. number of points on the sphere * number of angles for the gamma angle (n_gamma).

    """
    
    F0 = 8 # number of faces of a octahedra
    V0 = 6 # number of vertices
    
    # the number of point after each iterations is given by V+n = 1.5*F0/3(4**n-1)+ V0 
    # numIter0 = int(np.log(((M/n_gamma)-V0)*(2/F0)+1)/(np.log(4)))
    # numIter1 = int(np.ceil(np.clip((np.log(M/(24))/np.log(n_gamma)+1)-1, a_min=0, a_max=None)))
    
    numIter = int(np.ceil(np.log(M-8)/np.log(n_gamma)-2))

    # function returns stlPoints fromat and ABC format if its needed,if not - just delete it and adapt to your needs
    radius = 1
    
    
    # basic Octahedron reff:http://en.wikipedia.org/wiki/Octahedron
    # ( ?1, 0, 0 )
    # ( 0, ?1, 0 )
    # ( 0, 0, ?1 )
    A = np.asarray([1, 0, 0]) * radius
    B = np.asarray([0, 1, 0]) * radius
    C = np.asarray([0, 0, 1]) * radius
    # from +-ABC create initial triangles which define oxahedron
    triangles = np.asarray([A, B, C,
                            A, B, -C,
                            # -x, +y, +-Z quadrant
                            -A, B, C,
                            -A, B, -C,
                            # -x, -y, +-Z quadrant
                            -A, -B, C,
                            -A, -B, -C,
                            # +x, -y, +-Z quadrant
                            A, -B, C,
                            A, -B, -C])  # -----STL-similar format
    # for simplicity lets break into ABC points...
    #triangles = np.unique(triangles,axis=0)
    selector = np.arange(0, len(triangles[:, 1]) - 2, 3)
    Apoints = triangles[selector, :]
    Bpoints = triangles[selector + 1, :]
    Cpoints = triangles[selector + 2, :]
    # in every of numIterations
    for iteration in range(numIter):
        # devide every of triangle on three new
        #        ^ C
        #       / \
        # AC/2 /_4_\CB/2
        #     /\ 3 /\
        #    / 1\ /2 \
        # A /____V____\B           1st              2nd              3rd               4th
        #        AB/2
        # new triangleSteck is [ A AB/2 AC/2;     AB/2 B CB/2;     AC/2 AB/2 CB/2    AC/2 CB/2 C]
        AB_2 = (Apoints + Bpoints) / 2
        # do normalization of vector
        AB_2 = arsUnit(AB_2, radius)  # same for next 2 lines
        AC_2 = (Apoints + Cpoints) / 2
        AC_2 = arsUnit(AC_2, radius)
        CB_2 = (Cpoints + Bpoints) / 2
        CB_2 = arsUnit(CB_2, radius)
        Apoints = np.concatenate((Apoints,  # A point from 1st triangle
                                  AB_2,  # A point from 2nd triangle
                                  AC_2,  # A point from 3rd triangle
                                  AC_2))  # A point from 4th triangle..same for B and C
        Bpoints = np.concatenate((AB_2, Bpoints, AB_2, CB_2))
        Cpoints = np.concatenate((AC_2, CB_2, CB_2, Cpoints))
    # now tur points back to STL-like format....
    numPoints = np.shape(Apoints)[0]

    selector = np.arange(numPoints)
    selector = np.stack((selector, selector + numPoints, selector + 2 * numPoints))

    selector = np.swapaxes(selector, 0, 1)
    selector = np.concatenate(selector)
    stlPoints = np.concatenate((Apoints, Bpoints, Cpoints))
    stlPoints = stlPoints[selector, :]

    return stlPoints, Apoints, Bpoints, Cpoints


def get_euler_angles(M, n_gamma=4):
    '''
    Returns the zyz Euler angles with shape (M, 3) for the defined number of orientations M.
    (intrinsic Euler angles in the zyz convention)
    '''
    if M == 1:
        zyz = np.array([[0, 0, 0]])
    elif M == 2:
        zyz = np.array([[0, 0, 0], [180, 0, 0]])
    elif M == 4:  # Implement Klein's four group see Worrall and Brostow 2018
        zyz = np.array([[0, 0, 0], [180, 0, 0], [0, 180, 0], [180, 180, 0]])
    elif M == 8:  # Test of theta and phi.
        zyz = np.array(
            [[0, 0, 0], [0, 45, 315], [0, 90, 270], [0, 135, 225], [0, 180, 180], [0, 225, 135], [0, 270, 90],
             [0, 315, 45]])
    elif M == 24:  # as represented in Worrall and Brostow 2018, derived from the Caley's table
        # For intrinsic Euler angles (each row is for one of the six points on the sphere (theta,phi angles))
        zyz = np.array([[0, 0, 0], [0, 0, 90], [0, 0, 180], [0, 0, 270],
                        [0, 90, 0], [0, 90, 90], [0, 90, 180], [0, 90, 270],
                        [0, 180, 0], [0, 180, 90], [0, 180, 180], [0, 180, 270],
                        [0, 270, 0], [0, 270, 90], [0, 270, 180], [0, 270, 270],
                        [90, 90, 0], [90, 90, 90], [90, 90, 180], [90, 90, 270],
                        [90, 270, 0], [90, 270, 90], [90, 270, 180], [90, 270, 270]
                        ])

    else:
        # Parametrized uniform triangulation of 3D circle/sphere:

        i = np.log(M-8)/np.log(n_gamma)-2
        # No need for stlPoints AND A, B, C
        stlPoints, _, _, _ = sphereTriangulation(M,n_gamma)
        # Then do spherical coordinates to get the alpha and beta angles uniformly sampled on the sphere.

        # Then do spherical coordinates to get the alpha and beta angles uniformly sampled on the sphere.
        alpha, beta = change_vars(np.swapaxes(stlPoints,0,1)) # The Euler angles alpha and beta are respectively theta and phi in spherical coord.



        n_gamma = min(n_gamma, max(M//np.unique(stlPoints, axis=0).shape[0],1))


        step_gamma = 2*np.pi/n_gamma
        gamma2 = np.tile(np.linspace(0,2*np.pi-step_gamma,n_gamma),alpha.shape[0])
        alpha2 = np.repeat(alpha,n_gamma)
        beta2 = np.repeat(beta,n_gamma)
        zyz2 = np.stack((alpha2,beta2,gamma2),axis=1)*180/np.pi
        zyz2[zyz2<0] = zyz2[zyz2<0]+360
        zyz2 = np.where(np.unique(zyz2,axis=0)==360., 0., np.unique(zyz2,axis=0))
        zyz = np.unique(zyz2,axis=0)
    

    return zyz.tolist()

def get_euler_angles_new_old(M, zone= 0,n_gamma=4, z_rot_as_zone=True):
    '''
    Returns the zyz Euler angles with shape (M, 3) for the defined number of orientations M.
    (intrinsic Euler angles in the zyz convention)
    '''
    pi = np.pi
    if M == 1:
        zyz = np.array([[0, 0, 0]])
    elif M == 2:
        zyz = np.array([[0, 0, 0], [180, 0, 0]])
    elif M == 4:  # Implement Klein's four group see Worrall and Brostow 2018
        zyz = np.array([[0, 0, 0], [180, 0, 0], [0, 180, 0], [180, 180, 0]])
    elif M == 8:  # Test of theta and phi.
        zyz = np.array(
            [[0, 0, 0], [0, 45, 315], [0, 90, 270], [0, 135, 225], [0, 180, 180], [0, 225, 135], [0, 270, 90],
             [0, 315, 45]])
    elif M == 24:  # as represented in Worrall and Brostow 2018, derived from the Caley's table
        # For intrinsic Euler angles (each row is for one of the six points on the sphere (theta,phi angles))
        zyz = np.array([[0, 0, 0], [0, 0, 90], [0, 0, 180], [0, 0, 270],
                        [0, 90, 0], [0, 90, 90], [0, 90, 180], [0, 90, 270],
                        [0, 180, 0], [0, 180, 90], [0, 180, 180], [0, 180, 270],
                        [0, 270, 0], [0, 270, 90], [0, 270, 180], [0, 270, 270],
                        [90, 90, 0], [90, 90, 90], [90, 90, 180], [90, 90, 270],
                        [90, 270, 0], [90, 270, 90], [90, 270, 180], [90, 270, 270]
                        ])

    else:
        # Parametrized uniform triangulation of 3D circle/sphere:

        i = np.log(M-8)/np.log(n_gamma)-2
        # No need for stlPoints AND A, B, C
        stlPoints, _, _, _ = sphereTriangulation(M,n_gamma)
        gamma_ROM = 2*pi
        #  0 all sphere
        #  1 upper hemisphere (UH) z>0, 
        #  11 right UH z>0 - y>0, 12 left UH z>0 - y<0
        #  21 front right quarter UH (QUH) z>0 - y>0 - x>0, 
        #  22 front left QHU z>0 - y<0 - x>0, 
        #  23 back right QUH z>0 - y>0 - x<0, 
        #  24 back left QHU z>0 - y<0 - x<0
        #
        #
        #  -1 lower hemisphere z < 0
        #  -11 right LH z<0 - y>0 ...
        if zone > 0:
            gamma_ROM /= 2
            if zone == 4:
                lim = np.cos(pi/3)
            else:
                lim = 0
            stlPoints = stlPoints[stlPoints[:,2]>= lim]
            if zone%2 and zone > 10: #11, 21 or 23
                gamma_ROM /= 2
                stlPoints = stlPoints[stlPoints[:,1]>= 0]
                if zone == 21:
                    gamma_ROM /= 2
                    
                    stlPoints = stlPoints[stlPoints[:,0]>= 0]
                elif zone == 23:
                    gamma_ROM /= 2
                    stlPoints = stlPoints[stlPoints[:,0]<= 0]
                    
            elif not zone%2 and zone > 10: #12, 22 or 24
                gamma_ROM /= 2
                stlPoints = stlPoints[stlPoints[:,1]<= 0]
                if zone == 22:
                    gamma_ROM /= 2
                    stlPoints = stlPoints[stlPoints[:,0]>= 0]
                elif zone == 24:
                    gamma_ROM /= 2
                    stlPoints = stlPoints[stlPoints[:,0]<= 0]
                    
        elif zone < 0:
            gamma_ROM /= 2
            if zone == 4:
                lim = np.cos(pi/3)
            else:
                lim = 0
            stlPoints = stlPoints[stlPoints[:,2]<= lim]
            if abs(zone)%2 and abs(zone) > 10: #-11, -21 or -23
                gamma_ROM /= 2
                stlPoints = stlPoints[stlPoints[:,1]>= 0]
                if zone == -21:
                    gamma_ROM /= 2
                    stlPoints = stlPoints[stlPoints[:,0]>= 0]
                elif zone == -23:
                    gamma_ROM /= 2
                    stlPoints = stlPoints[stlPoints[:,0]<= 0]
                    
            elif not abs(zone)%2 and abs(zone) > 10: #-12, -22 or -24
                gamma_ROM /= 2
                stlPoints = stlPoints[stlPoints[:,1]<= 0]
                if zone == -22:
                    gamma_ROM /= 2
                    stlPoints = stlPoints[stlPoints[:,0]>= 0]
                elif zone == -24:
                    gamma_ROM /= 2
                    stlPoints = stlPoints[stlPoints[:,0]<= 0]
        # Then do spherical coordinates to get the alpha and beta angles uniformly sampled on the sphere.

        # Then do spherical coordinates to get the alpha and beta angles uniformly sampled on the sphere.
        alpha, beta = change_vars(np.swapaxes(stlPoints,0,1)) # The Euler angles alpha and beta are respectively theta and phi in spherical coord.

        if not z_rot_as_zone:
            gamma_ROM = 2*pi
        n_gamma = min(n_gamma, max(M//np.unique(stlPoints, axis=0).shape[0],1))

        
        step_gamma = gamma_ROM/n_gamma

        gamma2 = np.tile(np.linspace(0,gamma_ROM-step_gamma,n_gamma),alpha.shape[0])
        alpha2 = np.repeat(alpha,n_gamma)
        beta2 = np.repeat(beta,n_gamma)
        zyz2 = np.stack((alpha2,beta2,gamma2),axis=1)*180/pi
        zyz2[zyz2<0] = zyz2[zyz2<0]+360
        zyz2 = np.where(np.unique(zyz2,axis=0)==360., 0., np.unique(zyz2,axis=0))
        zyz = np.unique(zyz2,axis=0)
    

    return zyz.tolist()

def get_euler_angles_new(nbr_iter, zone= 0,n_gamma=4, z_rot_as_zone=True, desired_z_rot=4, cone_half_angle=60):
    '''
    Returns the zyz Euler angles with shape (M, 3) for the defined number of orientations M.
    (intrinsic Euler angles in the zyz convention)
    '''
    pi = np.pi
    
    M = int((4**(2+nbr_iter)+8)*(int(zone!=0)+int(zone==0)*(desired_z_rot/n_gamma)))
    print(f"You asked for {M} rotations with {desired_z_rot} rotation(s) along z axis")
    M_inf = int((4**(1+nbr_iter)+8))
    if M<M_inf and (desired_z_rot/n_gamma) < 1:
        M = int((4**(2+nbr_iter)+8))
    if M == 1:
        zyz = np.array([[0, 0, 0]])
    elif M == 2:
        zyz = np.array([[0, 0, 0], [180, 0, 0]])
    elif M == 4:  # Implement Klein's four group see Worrall and Brostow 2018
        zyz = np.array([[0, 0, 0], [180, 0, 0], [0, 180, 0], [180, 180, 0]])
    elif M == 8:  # Test of theta and phi.
        zyz = np.array(
            [[0, 0, 0], [0, 45, 315], [0, 90, 270], [0, 135, 225], [0, 180, 180], [0, 225, 135], [0, 270, 90],
             [0, 315, 45]])
    elif M == 124:  # as represented in Worrall and Brostow 2018, derived from the Caley's table
        # For intrinsic Euler angles (each row is for one of the six points on the sphere (theta,phi angles))
        zyz = np.array([[0, 0, 0], [0, 0, 90], [0, 0, 180], [0, 0, 270],
                        [0, 90, 0], [0, 90, 90], [0, 90, 180], [0, 90, 270],
                        [0, 180, 0], [0, 180, 90], [0, 180, 180], [0, 180, 270],
                        [0, 270, 0], [0, 270, 90], [0, 270, 180], [0, 270, 270],
                        [90, 90, 0], [90, 90, 90], [90, 90, 180], [90, 90, 270],
                        [90, 270, 0], [90, 270, 90], [90, 270, 180], [90, 270, 270]
                        ])

    else:
        # Parametrized uniform triangulation of 3D circle/sphere:

        i = np.log(M-8)/np.log(n_gamma)-2
        # No need for stlPoints AND A, B, C
        stlPoints, _, _, _ = sphereTriangulation(M,n_gamma)
        gamma_ROM = 2*pi
        #  0 all sphere
        #  1 upper hemisphere (UH) z>0, 
        #  11 right UH z>0 - y>0, 12 left UH z>0 - y<0
        #  21 front right quarter UH (QUH) z>0 - y>0 - x>0, 
        #  22 front left QHU z>0 - y<0 - x>0, 
        #  23 back right QUH z>0 - y>0 - x<0, 
        #  24 back left QHU z>0 - y<0 - x<0
        #
        #
        #  -1 lower hemisphere z < 0
        #  -11 right LH z<0 - y>0 ...
        if zone > 0:
            gamma_ROM /= 2
            if zone == 4:
                lim= np.cos(cone_half_angle/180*np.pi)
            else:
                lim = 0
            stlPoints = stlPoints[stlPoints[:,2]>= lim]
            
            if zone%2 and zone > 10: #11, 21 or 23
                gamma_ROM /= 2
                stlPoints = stlPoints[stlPoints[:,1]>= 0]
                if zone == 21:
                    gamma_ROM /= 2
                    
                    stlPoints = stlPoints[stlPoints[:,0]>= 0]
                elif zone == 23:
                    gamma_ROM /= 2
                    stlPoints = stlPoints[stlPoints[:,0]<= 0]
                    
            elif not zone%2 and zone > 10: #12, 22 or 24
                gamma_ROM /= 2
                stlPoints = stlPoints[stlPoints[:,1]<= -lim]
                if zone == 22:
                    gamma_ROM /= 2
                    stlPoints = stlPoints[stlPoints[:,0]>= 0]
                elif zone == 24:
                    gamma_ROM /= 2
                    stlPoints = stlPoints[stlPoints[:,0]<= 0]
                    
        elif zone < 0:
            gamma_ROM /= 2
            if abs(zone) == 4:
                lim = np.sign(zone)*np.cos(cone_half_angle/180*np.pi)
            else:
                lim = 0
            stlPoints = stlPoints[stlPoints[:,2]<= lim]
            if abs(zone)%2 and abs(zone) > 10: #-11, -21 or -23
                gamma_ROM /= 2
                stlPoints = stlPoints[stlPoints[:,1]>= 0]
                if zone == -21:
                    gamma_ROM /= 2
                    stlPoints = stlPoints[stlPoints[:,0]>= 0]
                elif zone == -23:
                    gamma_ROM /= 2
                    stlPoints = stlPoints[stlPoints[:,0]<= 0]
                    
            elif not abs(zone)%2 and abs(zone) > 10: #-12, -22 or -24
                gamma_ROM /= 2
                stlPoints = stlPoints[stlPoints[:,1]<= 0]
                if zone == -22:
                    gamma_ROM /= 2
                    stlPoints = stlPoints[stlPoints[:,0]>= 0]
                elif zone == -24:
                    gamma_ROM /= 2
                    stlPoints = stlPoints[stlPoints[:,0]<= 0]
        # Then do spherical coordinates to get the alpha and beta angles uniformly sampled on the sphere.

        # Then do spherical coordinates to get the alpha and beta angles uniformly sampled on the sphere.
        alpha, beta = change_vars(np.swapaxes(stlPoints,0,1)) # The Euler angles alpha and beta are respectively theta and phi in spherical coord.

        if not z_rot_as_zone:
            gamma_ROM = 2*pi
        #n_gamma = min(n_gamma, max(M//np.unique(stlPoints, axis=0).shape[0],1))
        n_gamma = desired_z_rot
        step_gamma = gamma_ROM/n_gamma

        gamma2 = np.tile(np.linspace(0,gamma_ROM-step_gamma,n_gamma),alpha.shape[0])
        alpha2 = np.repeat(alpha,n_gamma)
        beta2 = np.repeat(beta,n_gamma)
        zyz2 = np.stack((alpha2,beta2,gamma2),axis=1)*180/pi
        zyz2[zyz2<0] = zyz2[zyz2<0]+360
        zyz2 = np.where(np.unique(zyz2,axis=0)==360., 0., np.unique(zyz2,axis=0))
        zyz = np.unique(zyz2,axis=0)
    

    return zyz

def compute_dice_pc(gt, pred):
    dice = {}
    for c in torch.unique(gt):
        c = int(c.item())
        volume_sum = (gt==c).int().sum().item() + (pred==c).int().sum().item()
        if volume_sum == 0:
            dice[c] = -1.0
            print("vol zero")
        else:
            volume_intersect = ((gt==c) & (pred==c)).int().sum().item()
            dice_c = 2*volume_intersect / volume_sum 
            dice[c] = dice_c
            
    return dice

def compute_rmse_pc(gt, pred):
    # pred is C 3D volumes, gt is 3D volume
    
    rmse = {}
    for c in torch.unique(gt):
        c = int(c.item())            
        mse_c = np.square(np.subtract((gt==c).float(), pred[c])).mean() 
        rmse_c = np.sqrt(mse_c)
        rmse[c] = float(rmse_c)
    
    return rmse

def hausdorff_distance(image0, image1):
    """Code copied from 
    https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/set_metrics.py#L7-L54
    for compatibility reason with python 3.6
    """

    a_points = np.nonzero(image0)
    b_points = np.nonzero(image1)

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf
    
    return max(max(cKDTree(a_points).query(b_points, k=1)[0]),
               max(cKDTree(b_points).query(a_points, k=1)[0]))
    


def nDSC_metric(y_pred: np.ndarray, y: np.ndarray, r: float = 0.001, fixed_r = False):
    
    #if np.sum(y_pred) + np.sum(y) > 0:
    if not fixed_r:
        nbr_voxel = 1
        for i in range(len(y.shape)):
            nbr_voxel *= y.shape[i]
        r = y.sum().item()/nbr_voxel
    
    y_pred = y_pred.int()
    y = y.int()
    if (y_pred.sum().item() + y.sum().item()) > 0:
        scaling_factor = 1.0 if y.sum().item() == 0 else (1 - r) * (y.sum().item()) / (r * (len(y.flatten()) - (y.sum().item())))
        tp = torch.sum(y_pred[y == 1])
        fp = torch.sum(y_pred[y == 0])
        fn = torch.sum(y[y_pred == 0])
        fp_scaled = scaling_factor * fp
        return (2 * tp / (fp_scaled + 2 * tp + fn)).item(), r
    return 1.0, r


def compute_metrice_pc(gt, soft_pred,extended_metrics=False, fixed_r = False):
    rmse = {}
    dice = {}
    ndice = {}
    jacc = {}
    TFP = {}
    HD = {}
    soft_gt = None
    if len(gt.shape) > 3:
        soft_gt = torch.clone(gt)
        gt = torch.argmax(gt, axis=0)
    pred = torch.argmax(soft_pred, axis=0)
    r_arr = np.array([0.94711684, 0.02748482, 0.02535005])
    for c in torch.unique(gt):

        c = int(c.item())         
        c_pred = torch.clone(pred==c)
        c_gt = torch.clone(gt==c) 
        if soft_gt is not None:  
            mse_c = np.square(np.subtract(soft_gt[c], soft_pred[c])).mean()
        else:
            mse_c = np.square(np.subtract((c_gt).float(), soft_pred[c])).mean() 
        rmse_c = np.sqrt(mse_c)
        rmse[c] = float(rmse_c)
    
        # Compute dice & jaccard
        
        volume_sum = (c_gt).int().sum().item() + (c_pred).int().sum().item()
        volume_intersect = ((c_gt) & (c_pred)).int().sum().item()
        
        # Compute dice
        if volume_sum == 0:
            dice[c] = 1.0 # before 31.01. 11h18 -1
        else:
            dice[c] = 2*volume_intersect / volume_sum 
        if extended_metrics:
            
            ndice[c], r = nDSC_metric(c_pred, c_gt, r_arr[c], fixed_r=fixed_r)#.item()
            HD[c] = hausdorff_distance(c_pred, c_gt)
        # Compute jaccard
        if (volume_sum - volume_intersect):
            jacc[c] = volume_intersect / (volume_sum - volume_intersect) 
        else:
            jacc[c] = 1.0 # before 31.01. 11h18 -1
            
        
        
        # compute TP, TN, FP, FN
        TFP[c]  = {'TP': torch.sum(torch.logical_and(pred==c, gt==c)).item(),
                'TN': torch.sum(torch.logical_and(pred!=c, gt!=c)).item(),
                'FP': torch.sum(torch.logical_and(pred==c, gt!=c)).item(),
                'FN': torch.sum(torch.logical_and(pred!=c, gt==c)).item(),
                'r': r
                }
    if extended_metrics:
        return {'RMSE': rmse, 'DICE': dice, 'JACC': jacc, 'TFP': TFP, 'nDICE': ndice, 'HD': HD} # before 31.01. no ndice, no HD
    else:
        return {'RMSE': rmse, 'DICE': dice, 'JACC': jacc, 'TFP': TFP} # before 31.01. no ndice, no HD
        
    
 