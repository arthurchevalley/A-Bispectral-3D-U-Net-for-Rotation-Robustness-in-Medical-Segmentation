import os 
from os.path import *

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2, nnUNetTrainerV2_ATM_RA_drop
from nnunet.utilities.nd_softmax import softmax_helper

from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.network_architecture.custom_modules.bispectral_unet import *


#########################################################################################  
# Bispectral 
class newds_156_bispectral_nnUNetTrainerV2_Adam_16_drop_k3_d3_even_noproj(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 100
        self.initial_lr = 1e-3 #1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        
        self.predict = False
        training_sample = 156 # 26, 52, 104, 156
        
        self.splits_file = f"size_{training_sample}_splits_final.pkl"
        
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
        conv_op_type = 'bispectral' # standard, bispectral, spectral
        dropout_op = nn.Dropout3d
        norm_op = nn.BatchNorm3d #nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0.25, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        max_degree = 3
        num_conv_per_stage = 2
        self.base_num_features = 16
        indices = ((0, 0, 0), (0, 1, 1), (0, 2, 2), (1, 1, 0), (1, 1, 2), (1, 2, 1), (1, 2, 3)) # max degree 3
        #indices = ((0, 0, 0), (0, 1, 1), (0, 2, 2), (1, 1, 0), (1, 1, 2), (1, 2, 1), (1, 2, 3), (2, 2, 0),(2, 2, 2), (2, 2, 4)) # max degree 4
        #self.indices = None

        self.network = Spectral_UNet_noproj(input_channels = self.num_input_channels, 
                                     base_num_features = self.base_num_features, 
                                     num_classes = self.num_classes,
                                     num_pool = len(self.net_num_pool_op_kernel_sizes),
                                     max_degree = max_degree,
                                     indices=indices,
                                     num_conv_per_stage = self.conv_per_stage, 
                                     feat_map_mul_on_downscale = num_conv_per_stage, 
                                     conv_op_type = conv_op_type, 
                                     norm_op = norm_op, norm_op_kwargs = norm_op_kwargs, 
                                     dropout_op = dropout_op, dropout_op_kwargs = dropout_op_kwargs,
                                     nonlin = net_nonlin, nonlin_kwargs = net_nonlin_kwargs, 
                                     deep_supervision = True, 
                                     dropout_in_localization = False, 
                                     # weightInitializer = InitWeights_He(1e-2),
                                     final_nonlin=lambda x: x,
                                     pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes, 
                                     conv_kernel_sizes = self.net_conv_kernel_sizes, 
                                     upscale_logits = False, 
                                     convolutional_pooling = False, 
                                     linear_upsampling = True)
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
        self.print_to_log_file("Used indices: ", indices)

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        self.lr_scheduler = None
     
    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """
        
        self.patch_size = np.array([40,40,40])
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:

            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['patch_size'] = np.array([40,40,40])
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




class newds_156_bispectral_ATM_nnUNetTrainerV2_Adam_8_drop_k3_d3_even(nnUNetTrainerV2_ATM_RA_drop):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.max_num_epochs = 50
        self.initial_lr = 1e-3 #1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True
        
        self.predict = False
        training_sample = 156 # 26, 52, 104, 156
        self.patch_size = np.array([80,80,80])
        self.prob_RA_augment = 0.
        
        self.splits_file = f"size_{training_sample}_splits_final.pkl"
        
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
        conv_op_type = 'bispectral' # standard, bispectral, spectral
        dropout_op = nn.Dropout3d
        norm_op = nn.BatchNorm3d #nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0.25, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        max_degree = 3
        num_conv_per_stage = 2
        self.base_num_features = 8
        indices = ((0, 0, 0), (0, 1, 1), (0, 2, 2), (1, 1, 0), (1, 1, 2), (1, 2, 1), (1, 2, 3)) # max degree 3
        #indices = ((0, 0, 0), (0, 1, 1), (0, 2, 2), (1, 1, 0), (1, 1, 2), (1, 2, 1), (1, 2, 3), (2, 2, 0),(2, 2, 2), (2, 2, 4)) # max degree 4
        #self.indices = None

        self.network = Spectral_UNet(input_channels = self.num_input_channels, 
                                     base_num_features = self.base_num_features, 
                                     num_classes = self.num_classes,
                                     num_pool = len(self.net_num_pool_op_kernel_sizes),
                                     max_degree = max_degree,
                                     indices=indices,
                                     num_conv_per_stage = self.conv_per_stage, 
                                     feat_map_mul_on_downscale = num_conv_per_stage, 
                                     conv_op_type = conv_op_type, 
                                     norm_op = norm_op, norm_op_kwargs = norm_op_kwargs, 
                                     dropout_op = dropout_op, dropout_op_kwargs = dropout_op_kwargs,
                                     nonlin = net_nonlin, nonlin_kwargs = net_nonlin_kwargs, 
                                     deep_supervision = True, 
                                     dropout_in_localization = False, 
                                     # weightInitializer = InitWeights_He(1e-2),
                                     final_nonlin=lambda x: x,
                                     pool_op_kernel_sizes = self.net_num_pool_op_kernel_sizes, 
                                     conv_kernel_sizes = self.net_conv_kernel_sizes, 
                                     upscale_logits = False, 
                                     convolutional_pooling = False, 
                                     linear_upsampling = True)
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
        
        self.print_to_log_file("Used indices: ", indices)

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay)
        self.lr_scheduler = None
