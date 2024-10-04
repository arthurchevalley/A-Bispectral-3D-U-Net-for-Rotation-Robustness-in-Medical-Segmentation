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


from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional
from typing import Tuple


from nnunet.utilities.random_stuff import no_op
from torch.cuda.amp import autocast

from .bispectrum_layer import *

class SpectralStackedConvLayers(nn.Module):
    def __init__(self, 
                 input_feature_channels, 
                 output_feature_channels, 
                 num_convs,
                 max_degree = 3,
                 conv_op=BSHConv3D, 
                 indices = None,
                 conv_kwargs=None,
                 norm_op=torch.nn.BatchNorm3d,
                 norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, 
                 dropout_op_kwargs=None,
                 nonlin=nn.ReLU, 
                 nonlin_kwargs=None, 
                 first_stride=None):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 'valid', 'dilation': 1, 
                           'use_bias': True, 'bias_init': 'zeros',
                           'activation':  ("Linear", {"inplace": True}), 
                           'initializer': 'glorot_uniform',
                           'proj': True, 
                           'proj_activation': ("RELU", {"inplace": True}), 
                           'proj_initializer': 'glorot_uniform',
                           'radial_profile_type': 'radial'}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(SpectralStackedConvLayers, self).__init__()

        modules = []
        for i in range(num_convs):
            if conv_op == nn.Conv3d:
                modules.append(self.conv_op(in_channels=input_feature_channels, 
                                            out_channels=output_feature_channels,
                                            kernel_size=conv_kwargs['kernel_size'],
                                            stride=conv_kwargs['stride'],
                                            padding=conv_kwargs['padding'],
                                            ))
            else:
                modules.append(self.conv_op(in_channels=input_feature_channels, 
                                            out_channels=output_feature_channels,
                                            kernel_size=conv_kwargs['kernel_size'],
                                            max_degree=max_degree,
                                            strides=conv_kwargs['stride'],
                                            padding=conv_kwargs['padding'],
                                            activation=conv_kwargs['activation'],
                                            kernel_initializer=conv_kwargs['initializer'],
                                            use_bias=conv_kwargs['use_bias'],
                                            bias_initializer=conv_kwargs['bias_init'],
                                            radial_profile_type=conv_kwargs['radial_profile_type'],
                                            proj_activation=conv_kwargs['proj_activation'],
                                            proj_initializer=conv_kwargs['proj_initializer'],
                                            project=conv_kwargs['proj'],
                                            indices = indices
                                            ))
                
            if norm_op is not None:
                modules.append(norm_op(output_feature_channels, **self.norm_op_kwargs))
            if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs['p'] > 0:
                modules.append(dropout_op(**self.dropout_op_kwargs))
            modules.append(nonlin(**self.nonlin_kwargs))
            input_feature_channels = output_feature_channels

        self.blocks = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.blocks(x)



def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class Spectral_UNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320
    
    BASE_NUM_FEATURES_bispectral = 12
    MAX_NUMPOOL_bispectral = 999
    MAX_NUM_FILTERS_bispectral = 320
    
    BASE_NUM_FEATURES_spectral = 8
    MAX_NUMPOOL_spectral = 999
    MAX_NUM_FILTERS_spectral = 320
    
    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440
    
    CONV = {
        "standard": nn.Conv3d,
        "bispectral": BSHConv3D,
        "spectral": SSHConv3D,
    }

    def __init__(self, 
                 input_channels, 
                 base_num_features, 
                 num_classes, 
                 num_pool, 
                 max_degree = 3,
                 kernel_size = 3,
                 indices = None,
                 num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, 
                 conv_op_type='bispectral',
                 conv_op_type_decode = None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.ReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, linear_upsampling=False,
                 max_num_features=None,
                 seg_output_use_bias=False,
                 residual = False):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Spectral_UNet, self).__init__()
        self.linear_upsampling = linear_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'kernel_size': kernel_size, 'stride': 1, 'padding': 'same',
                            'dilation': 1, 
                           'use_bias': True, 'bias_init': 'zeros',
                           'activation': ("Linear", {"inplace": True}), 
                           'initializer':  'glorot_uniform',#'glorot_adapted',
                           'proj': True, 'proj_activation': ("RELU", {"inplace": True}), 'proj_initializer':'glorot_uniform',#'glorot_adapted',
                           'radial_profile_type': 'radial'}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = self.CONV[conv_op_type]
        
        self.residual = residual
        
        if conv_op_type_decode is None:
            conv_op_type_decode = conv_op_type
        self.conv_op_decode = self.CONV[conv_op_type_decode]
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if self.conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(kernel_size, kernel_size, kernel_size)] * (num_pool + 1)
                
        elif self.conv_op == BSHConv3D or self.conv_op == SSHConv3D:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(kernel_size, kernel_size, kernel_size)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(self.conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            if self.conv_op == BSHConv3D:
                self.max_num_features = self.MAX_NUM_FILTERS_bispectral
            if self.conv_op == SSHConv3D:
                self.max_num_features = self.MAX_NUM_FILTERS_spectral
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.res_conv_blocks_context = []
        self.res_conv_blocks_localization = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):

            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            #self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            #self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions

            self.conv_blocks_context.append(SpectralStackedConvLayers(input_features, output_features, num_conv_per_stage,max_degree,
                                                              self.conv_op, indices,
                                                              self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride))
            if residual:
                self.res_conv_blocks_context.append(self.CONV['standard'](in_channels = input_features, 
                                                                        out_channels = output_features,
                                                                        kernel_size = 1, 
                                                                        stride = 1
                                                                        ))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.linear_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        #self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]

        self.conv_blocks_context.append(nn.Sequential(
            SpectralStackedConvLayers(input_features, output_features, num_conv_per_stage-1,max_degree,
                                        self.conv_op, indices, 
                                        self.conv_kwargs, self.norm_op,
                                        self.norm_op_kwargs, self.dropout_op,
                                        self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                        first_stride),
            SpectralStackedConvLayers(output_features, final_num_features, 1,max_degree,
                                        self.conv_op, indices, 
                                        self.conv_kwargs, self.norm_op,
                                        self.norm_op_kwargs, self.dropout_op,
                                        self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                        first_stride)))
        if self.residual:
            self.res_conv_blocks_context.append(self.CONV['standard'](in_channels = input_features, 
                                                                      out_channels = final_num_features,
                                                                      kernel_size = 1, 
                                                                      stride = 1
                                                                      ))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):

            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2
            
            

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.linear_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.linear_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(LinearUpsampling3D(size=pool_op_kernel_sizes[-(u + 1)]))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            #self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                SpectralStackedConvLayers(nfeatures_from_down + nfeatures_from_skip, nfeatures_from_skip, num_conv_per_stage-1,max_degree,
                                            self.conv_op_decode, indices,
                                            self.conv_kwargs, self.norm_op,
                                            self.norm_op_kwargs, self.dropout_op,
                                            self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                            first_stride),
                SpectralStackedConvLayers(nfeatures_from_skip, final_num_features, 1,max_degree,
                                            self.conv_op_decode, indices, 
                                            self.conv_kwargs, self.norm_op,
                                            self.norm_op_kwargs, self.dropout_op,
                                            self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                            first_stride)))
            if self.residual:
                self.res_conv_blocks_localization.append(self.CONV['standard'](in_channels = nfeatures_from_down + nfeatures_from_skip, 
                                                                               out_channels = final_num_features,
                                                                               kernel_size = 1, 
                                                                               stride = 1
                                                                               ))

        for ds in range(len(self.conv_blocks_localization)):
            
            self.seg_outputs.append(self.CONV['standard'](in_channels = self.conv_blocks_localization[ds][-1].output_channels, 
                                                 out_channels = num_classes,
                                                 kernel_size = 1, 
                                                 stride = 1, 
                                                 bias = seg_output_use_bias
                                                ))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        if self.residual:
            self.res_conv_blocks_localization = nn.ModuleList(self.res_conv_blocks_localization)
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        if self.residual:
            self.res_conv_blocks_context = nn.ModuleList(self.res_conv_blocks_context)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)
            

    def forward(self, x):

        skips = []
        seg_outputs = []
        #print(f'In {x.shape}')
        for d in range(len(self.conv_blocks_context) - 1):

            if self.residual:
                x_res = self.res_conv_blocks_context[d](x)
            x = self.conv_blocks_context[d](x)
            
            skips.append(x)
            if self.residual:
                #print(f'res co\n {x_res.shape, torch.mean(x_res).item(), torch.max(x_res).item(), torch.min(x_res).item()}')
                #print(f'end\n {x.shape, torch.mean(x).item(), torch.max(x).item(), torch.min(x).item()}')
                x = x + x_res
            if not self.convolutional_pooling:
                x = self.td[d](x)
            #print(f'Down {x.shape}')

        x = self.conv_blocks_context[-1](x)
        #print(f'Bottleneck {x.shape}')
        for u in range(len(self.tu)):
            
            x = self.tu[u](x)
            #print(f'UP {x.shape}')

            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            if self.residual:
                x_res = self.res_conv_blocks_localization[u](x)
            x = self.conv_blocks_localization[u](x)
            if self.residual:
                x = x + x_res
            #print(f'UP 2 {x.shape}')
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
        
    def predict_3D(self, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                use_sliding_window: bool = False,
                step_size: float = 0.5, patch_size: Tuple[int, ...] = None, regions_class_order: Tuple[int, ...] = None,
                use_gaussian: bool = False, pad_border_mode: str = "constant",
                pad_kwargs: dict = None, all_in_gpu: bool = False,
                verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use this function to predict a 3D image. It does not matter whether the network is a 2D or 3D U-Net, it will
        detect that automatically and run the appropriate code.

        When running predictions, you need to specify whether you want to run fully convolutional of sliding window
        based inference. We very strongly recommend you use sliding window with the default settings.

        It is the responsibility of the user to make sure the network is in the proper mode (eval for inference!). If
        the network is not in eval mode it will print a warning.

        :param x: Your input data. Must be a nd.ndarray of shape (c, x, y, z).
        :param do_mirroring: If True, use test time data augmentation in the form of mirroring
        :param mirror_axes: Determines which axes to use for mirroing. Per default, mirroring is done along all three
        axes
        :param use_sliding_window: if True, run sliding window prediction. Heavily recommended! This is also the default
        :param step_size: When running sliding window prediction, the step size determines the distance between adjacent
        predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given
        as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between
        predictions. step_size cannot be larger than 1!
        :param patch_size: The patch size that was used for training the network. Do not use different patch sizes here,
        this will either crash or give potentially less accurate segmentations
        :param regions_class_order: Fabian only
        :param use_gaussian: (Only applies to sliding window prediction) If True, uses a Gaussian importance weighting
         to weigh predictions closer to the center of the current patch higher than those at the borders. The reason
         behind this is that the segmentation accuracy decreases towards the borders. Default (and recommended): True
        :param pad_border_mode: leave this alone
        :param pad_kwargs: leave this alone
        :param all_in_gpu: experimental. You probably want to leave this as is it
        :param verbose: Do you want a wall of text? If yes then set this to True
        :param mixed_precision: if True, will run inference in mixed precision with autocast()
        :return:
        """
        torch.cuda.empty_cache()

        assert step_size <= 1, 'step_size must be smaller than 1. Otherwise there will be a gap between consecutive ' \
                               'predictions'

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            if self.conv_op == nn.Conv2d:
                if max(mirror_axes) > 1:
                    raise ValueError("mirror axes. duh")
            if self.conv_op in [nn.Conv3d, BSHConv3D, SSHConv3D, SHConv3D, SHConv3DRadial]:
                if max(mirror_axes) > 2:
                    raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if self.conv_op in [nn.Conv3d, BSHConv3D, SSHConv3D, SHConv3D, SHConv3DRadial]:
                    if use_sliding_window:
                        res = self._internal_predict_3D_3Dconv_tiled(x, step_size, do_mirroring, mirror_axes, patch_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                     verbose=verbose)
                    else:
                        do_mirroring = False
                        res = self._internal_predict_3D_3Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs=pad_kwargs, verbose=verbose)
                elif self.conv_op == nn.Conv2d:
                    if use_sliding_window:
                        res = self._internal_predict_3D_2Dconv_tiled(x, patch_size, do_mirroring, mirror_axes, step_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs, all_in_gpu, False)
                    else:
                        res = self._internal_predict_3D_2Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs, all_in_gpu, False)
                else:
                    raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")

        return res


    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and linear_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp
    
class Spectral_UNet_noproj(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320
    
    BASE_NUM_FEATURES_bispectral = 12
    MAX_NUMPOOL_bispectral = 999
    MAX_NUM_FILTERS_bispectral = 320
    
    BASE_NUM_FEATURES_spectral = 8
    MAX_NUMPOOL_spectral = 999
    MAX_NUM_FILTERS_spectral = 320
    
    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440
    
    CONV = {
        "standard": nn.Conv3d,
        "bispectral": BSHConv3D,
        "spectral": SSHConv3D,
    }

    def __init__(self, 
                 input_channels, 
                 base_num_features, 
                 num_classes, 
                 num_pool, 
                 max_degree = 3,
                 kernel_size = 3,
                 indices = None,
                 num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, 
                 conv_op_type='bispectral',
                 conv_op_type_decode = None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.ReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, linear_upsampling=False,
                 max_num_features=None,
                 seg_output_use_bias=False,
                 residual = False):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Spectral_UNet_noproj, self).__init__()
        self.linear_upsampling = linear_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'kernel_size': kernel_size, 'stride': 1, 'padding': 'same',
                            'dilation': 1, 
                           'use_bias': True, 'bias_init': 'zeros',
                           'activation': ("Linear", {"inplace": True}), 
                           'initializer':  'glorot_uniform',#'glorot_adapted',
                           'proj': True, 
                           'proj_activation': None,#("RELU", {"inplace": True}), 
                           'proj_initializer':'glorot_uniform',#'glorot_adapted',
                           'radial_profile_type': 'radial'}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = self.CONV[conv_op_type]
        
        self.residual = residual
        
        if conv_op_type_decode is None:
            conv_op_type_decode = conv_op_type
        self.conv_op_decode = self.CONV[conv_op_type_decode]
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if self.conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(kernel_size, kernel_size, kernel_size)] * (num_pool + 1)
                
        elif self.conv_op == BSHConv3D or self.conv_op == SSHConv3D:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(kernel_size, kernel_size, kernel_size)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(self.conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            if self.conv_op == BSHConv3D:
                self.max_num_features = self.MAX_NUM_FILTERS_bispectral
            if self.conv_op == SSHConv3D:
                self.max_num_features = self.MAX_NUM_FILTERS_spectral
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.res_conv_blocks_context = []
        self.res_conv_blocks_localization = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):

            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            #self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            #self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions

            self.conv_blocks_context.append(SpectralStackedConvLayers(input_features, output_features, num_conv_per_stage,max_degree,
                                                              self.conv_op, indices,
                                                              self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride))
            if residual:
                self.res_conv_blocks_context.append(self.CONV['standard'](in_channels = input_features, 
                                                                        out_channels = output_features,
                                                                        kernel_size = 1, 
                                                                        stride = 1
                                                                        ))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.linear_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        #self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]

        self.conv_blocks_context.append(nn.Sequential(
            SpectralStackedConvLayers(input_features, output_features, num_conv_per_stage-1,max_degree,
                                        self.conv_op, indices, 
                                        self.conv_kwargs, self.norm_op,
                                        self.norm_op_kwargs, self.dropout_op,
                                        self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                        first_stride),
            SpectralStackedConvLayers(output_features, final_num_features, 1,max_degree,
                                        self.conv_op, indices, 
                                        self.conv_kwargs, self.norm_op,
                                        self.norm_op_kwargs, self.dropout_op,
                                        self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                        first_stride)))
        if self.residual:
            self.res_conv_blocks_context.append(self.CONV['standard'](in_channels = input_features, 
                                                                      out_channels = final_num_features,
                                                                      kernel_size = 1, 
                                                                      stride = 1
                                                                      ))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):

            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2
            
            

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.linear_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.linear_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(LinearUpsampling3D(size=pool_op_kernel_sizes[-(u + 1)]))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            #self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                SpectralStackedConvLayers(nfeatures_from_down + nfeatures_from_skip, nfeatures_from_skip, num_conv_per_stage-1,max_degree,
                                            self.conv_op_decode, indices,
                                            self.conv_kwargs, self.norm_op,
                                            self.norm_op_kwargs, self.dropout_op,
                                            self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                            first_stride),
                SpectralStackedConvLayers(nfeatures_from_skip, final_num_features, 1,max_degree,
                                            self.conv_op_decode, indices, 
                                            self.conv_kwargs, self.norm_op,
                                            self.norm_op_kwargs, self.dropout_op,
                                            self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                            first_stride)))
            if self.residual:
                self.res_conv_blocks_localization.append(self.CONV['standard'](in_channels = nfeatures_from_down + nfeatures_from_skip, 
                                                                               out_channels = final_num_features,
                                                                               kernel_size = 1, 
                                                                               stride = 1
                                                                               ))

        for ds in range(len(self.conv_blocks_localization)):
            
            self.seg_outputs.append(self.CONV['standard'](in_channels = self.conv_blocks_localization[ds][-1].output_channels, 
                                                 out_channels = num_classes,
                                                 kernel_size = 1, 
                                                 stride = 1, 
                                                 bias = seg_output_use_bias
                                                ))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        if self.residual:
            self.res_conv_blocks_localization = nn.ModuleList(self.res_conv_blocks_localization)
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        if self.residual:
            self.res_conv_blocks_context = nn.ModuleList(self.res_conv_blocks_context)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)
            

    def forward(self, x):

        skips = []
        seg_outputs = []
        #print(f'In {x.shape}')
        for d in range(len(self.conv_blocks_context) - 1):

            if self.residual:
                x_res = self.res_conv_blocks_context[d](x)
            x = self.conv_blocks_context[d](x)
            
            skips.append(x)
            if self.residual:
                #print(f'res co\n {x_res.shape, torch.mean(x_res).item(), torch.max(x_res).item(), torch.min(x_res).item()}')
                #print(f'end\n {x.shape, torch.mean(x).item(), torch.max(x).item(), torch.min(x).item()}')
                x = x + x_res
            if not self.convolutional_pooling:
                x = self.td[d](x)
            #print(f'Down {x.shape}')

        x = self.conv_blocks_context[-1](x)
        #print(f'Bottleneck {x.shape}')
        for u in range(len(self.tu)):
            
            x = self.tu[u](x)
            #print(f'UP {x.shape}')

            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            if self.residual:
                x_res = self.res_conv_blocks_localization[u](x)
            x = self.conv_blocks_localization[u](x)
            if self.residual:
                x = x + x_res
            #print(f'UP 2 {x.shape}')
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
        
    def predict_3D(self, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                use_sliding_window: bool = False,
                step_size: float = 0.5, patch_size: Tuple[int, ...] = None, regions_class_order: Tuple[int, ...] = None,
                use_gaussian: bool = False, pad_border_mode: str = "constant",
                pad_kwargs: dict = None, all_in_gpu: bool = False,
                verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use this function to predict a 3D image. It does not matter whether the network is a 2D or 3D U-Net, it will
        detect that automatically and run the appropriate code.

        When running predictions, you need to specify whether you want to run fully convolutional of sliding window
        based inference. We very strongly recommend you use sliding window with the default settings.

        It is the responsibility of the user to make sure the network is in the proper mode (eval for inference!). If
        the network is not in eval mode it will print a warning.

        :param x: Your input data. Must be a nd.ndarray of shape (c, x, y, z).
        :param do_mirroring: If True, use test time data augmentation in the form of mirroring
        :param mirror_axes: Determines which axes to use for mirroing. Per default, mirroring is done along all three
        axes
        :param use_sliding_window: if True, run sliding window prediction. Heavily recommended! This is also the default
        :param step_size: When running sliding window prediction, the step size determines the distance between adjacent
        predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given
        as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between
        predictions. step_size cannot be larger than 1!
        :param patch_size: The patch size that was used for training the network. Do not use different patch sizes here,
        this will either crash or give potentially less accurate segmentations
        :param regions_class_order: Fabian only
        :param use_gaussian: (Only applies to sliding window prediction) If True, uses a Gaussian importance weighting
         to weigh predictions closer to the center of the current patch higher than those at the borders. The reason
         behind this is that the segmentation accuracy decreases towards the borders. Default (and recommended): True
        :param pad_border_mode: leave this alone
        :param pad_kwargs: leave this alone
        :param all_in_gpu: experimental. You probably want to leave this as is it
        :param verbose: Do you want a wall of text? If yes then set this to True
        :param mixed_precision: if True, will run inference in mixed precision with autocast()
        :return:
        """
        torch.cuda.empty_cache()

        assert step_size <= 1, 'step_size must be smaller than 1. Otherwise there will be a gap between consecutive ' \
                               'predictions'

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            if self.conv_op == nn.Conv2d:
                if max(mirror_axes) > 1:
                    raise ValueError("mirror axes. duh")
            if self.conv_op in [nn.Conv3d, BSHConv3D, SSHConv3D, SHConv3D, SHConv3DRadial]:
                if max(mirror_axes) > 2:
                    raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if self.conv_op in [nn.Conv3d, BSHConv3D, SSHConv3D, SHConv3D, SHConv3DRadial]:
                    if use_sliding_window:
                        res = self._internal_predict_3D_3Dconv_tiled(x, step_size, do_mirroring, mirror_axes, patch_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                     verbose=verbose)
                    else:
                        do_mirroring = False
                        res = self._internal_predict_3D_3Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs=pad_kwargs, verbose=verbose)
                elif self.conv_op == nn.Conv2d:
                    if use_sliding_window:
                        res = self._internal_predict_3D_2Dconv_tiled(x, patch_size, do_mirroring, mirror_axes, step_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs, all_in_gpu, False)
                    else:
                        res = self._internal_predict_3D_2Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs, all_in_gpu, False)
                else:
                    raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")

        return res


    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and linear_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp
    
    

class TEST_Spectral_UNet(Spectral_UNet):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320
    
    BASE_NUM_FEATURES_bispectral = 12
    MAX_NUMPOOL_bispectral = 999
    MAX_NUM_FILTERS_bispectral = 320
    
    BASE_NUM_FEATURES_spectral = 8
    MAX_NUMPOOL_spectral = 999
    MAX_NUM_FILTERS_spectral = 320
    
    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440
    
    CONV = {
        "standard": nn.Conv3d,
        "bispectral": BSHConv3D,
        "spectral": SSHConv3D,
    }

    def __init__(self, 
                 input_channels, 
                 base_num_features, 
                 num_classes, 
                 num_pool, 
                 max_degree = 3,
                 indices=None,
                 num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, 
                 conv_op_type='bispectral',
                 conv_op_type_decode = None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.ReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, linear_upsampling=False,
                 max_num_features=None,
                 seg_output_use_bias=False):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Spectral_UNet, self).__init__()
        self.linear_upsampling = linear_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 'same',
                            'dilation': 1, 
                           'use_bias': True, 'bias_init': 'zeros',
                           'activation': ("Linear", {"inplace": True}), 
                           'initializer':  'glorot_uniform',#'glorot_adapted',
                           'proj': True, 'proj_activation': ("RELU", {"inplace": True}), 'proj_initializer':'glorot_uniform',#'glorot_adapted',
                           'radial_profile_type': 'radial'}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = self.CONV[conv_op_type]
        if conv_op_type_decode is None:
            conv_op_type_decode = conv_op_type
        self.conv_op_decode = self.CONV[conv_op_type_decode]
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.residual = False
        if self.conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
                
        elif self.conv_op == BSHConv3D or self.conv_op == SSHConv3D:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(self.conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            if self.conv_op == BSHConv3D:
                self.max_num_features = self.MAX_NUM_FILTERS_bispectral
            if self.conv_op == SSHConv3D:
                self.max_num_features = self.MAX_NUM_FILTERS_spectral
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.res_conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels

        for d in range(num_pool):

            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            #self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            #self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions

            self.conv_blocks_context.append(SpectralStackedConvLayers(input_features, output_features, num_conv_per_stage,max_degree,
                                                              self.conv_op,indices, 
                                                              self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride))
        
            self.res_conv_blocks_context.append(self.CONV['standard'](in_channels = input_features, 
                                                                      out_channels = output_features,
                                                                      kernel_size = 1, 
                                                                      stride = 1
                                                                      ))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.linear_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        #self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]

        self.conv_blocks_context.append(nn.Sequential(
            SpectralStackedConvLayers(input_features, output_features, num_conv_per_stage-1,max_degree,
                                        self.conv_op, indices,
                                        self.conv_kwargs, self.norm_op,
                                        self.norm_op_kwargs, self.dropout_op,
                                        self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                        first_stride),
            SpectralStackedConvLayers(output_features, final_num_features, 1,max_degree,
                                        self.conv_op, indices,
                                        self.conv_kwargs, self.norm_op,
                                        self.norm_op_kwargs, self.dropout_op,
                                        self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                        first_stride)))
        self.res_conv_blocks_context.append(self.CONV['standard'](in_channels = input_features, 
                                                                  out_channels = final_num_features,
                                                                  kernel_size = 1, 
                                                                  stride = 1
                                                                  ))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):

            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2
            
            

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.linear_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.linear_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(LinearUpsampling3D(size=pool_op_kernel_sizes[-(u + 1)]))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            #self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                SpectralStackedConvLayers(nfeatures_from_down + nfeatures_from_skip, nfeatures_from_skip, num_conv_per_stage-1,max_degree,
                                            self.conv_op_decode, indices,
                                            self.conv_kwargs, self.norm_op,
                                            self.norm_op_kwargs, self.dropout_op,
                                            self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                            first_stride),
                SpectralStackedConvLayers(nfeatures_from_skip, final_num_features, 1,max_degree,
                                            self.conv_op_decode, indices,
                                            self.conv_kwargs, self.norm_op,
                                            self.norm_op_kwargs, self.dropout_op,
                                            self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                            first_stride)))

        for ds in range(len(self.conv_blocks_localization)):
            
            self.seg_outputs.append(self.CONV['standard'](in_channels = self.conv_blocks_localization[ds][-1].output_channels, 
                                                 out_channels = num_classes,
                                                 kernel_size = 1, 
                                                 stride = 1, 
                                                 bias = seg_output_use_bias
                                                ))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.res_conv_blocks_context = nn.ModuleList(self.res_conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)
            

    def forward(self, x):

        skips = []
        seg_outputs = []
        print(f'In {x.shape}')
        for d in range(len(self.conv_blocks_context) - 1):

            x_res = self.res_conv_blocks_context[d](x)
            x = self.conv_blocks_context[d](x)
            #print(f'res co\n {x_res.shape, torch.mean(x_res).item(), torch.max(x_res).item(), torch.min(x_res).item()}')
            #print(f'end\n {x.shape, torch.mean(x).item(), torch.max(x).item(), torch.min(x).item()}')
            if self.residual:
                x = x + x_res
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
            print(f'Down {x.shape}')

        x = self.conv_blocks_context[-1](x)
        print(f'Bottleneck {x.shape}')
        for u in range(len(self.tu)):
            
            x = self.tu[u](x)
            print(f'UP {x.shape}')

            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)

            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            print(f'UP 2 {x.shape}')

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
        
