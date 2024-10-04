import numpy as np
import torch
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2

from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset

from nnunet.utilities.nd_softmax import softmax_helper

from sklearn.model_selection import KFold

from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast

from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2

from torch import nn
import os 

from os.path import *


import math
from typing import Callable
import logging
from scipy import special as sp
from sympy.physics.quantum.cg import CG
from sympy import Ynm, Symbol, lambdify

from copy import deepcopy
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)


def get_lri_conv3d(*args, kind="bispectrum", **kwargs):
    if kind == "bispectrum":
        return BSHConv3D(*args, **kwargs)
    elif kind == "spectrum":
        return SSHConv3D(*args, **kwargs)
    else:
        raise ValueError(f"The kind {kind} is not supported")

def create_activation_layer(name):
    if name[0].lower() == "" or name[0].lower() == "linear":
      return torch.nn.Identity()
    elif name[0].lower() == 'relu':
      return nn.ReLU(inplace = name[1]['inplace'])
    elif name[0].lower() == 'leaky_relu':
      return nn.LeakyReLU(inplace = name[1]['inplace'])
    elif name[0].lower() == 'gelu':
      return nn.GELU(inplace = name[1]['inplace'])
    elif name[0].lower() == 'sigmoid':
      return nn.Sigmoid(inplace = name[1]['inplace'])
    elif name[0].lower() == 'tanh':
      return nn.Tanh(inplace = name[1]['inplace'])
    else:
        assert NotImplementedError(f"Activation funtion {name[0]} not implemented yet")

def init_weight(name):
    if "zeros" in name.lower():
        init_fn = nn.init.zeros_    
    elif "ones" in name.lower():
        init_fn = nn.init.ones_  
    elif "glorot_adapted" in name.lower() or "xavier_uniform" in name.lower():
        init_fn = nn.init.xavier_uniform_   
    elif "xavier_normal" in name.lower():
        init_fn = nn.init.xavier_normal_   
    else:
        assert NotImplementedError(f"Initialisation {name} not implemented yet")
    
    
    return init_fn
    
def create_conv_layer(in_channels, out_channels, kernel_size, padding='same', strides = 1, activation = None, kernel_init=None):
    layer = nn.Sequential()
    conv_layer = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=strides),

    if kernel_init is not None:
        init_fn = init_weight(kernel_init)
        init_fn(conv_layer.weight)
    layer.add_module(
        "3dConv",
        conv_layer
    )
    if activation is not None:
        layer.add_module(
            "activation",
            create_activation_layer(activation))
        
    return layer

class BSHConv3D(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 max_degree=3,
                 strides=1,
                 padding='SAME',
                 kernel_initializer="glorot_uniform",
                 use_bias=True,
                 bias_initializer="zeros",
                 radial_profile_type="radial",
                 activation=("Linear", {"inplace": True}),
                 proj_activation=("RELU", {"inplace": True}),
                 proj_initializer="glorot_uniform",
                 project=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.max_degree = max_degree
        self._indices = None
        # self._indices = (
        #     # (0, 0, 0),
        #     (0, 1, 1),
        #     (0, 2, 2),
        #     (1, 1, 2),
        #     (1, 2, 1),
        #     (1, 2, 3),
        # )
        self.output_bispectrum_channels = self._output_bispectrum_channels()

        self._indices_inverse = None
        self.activation = create_activation_layer(activation)
        self.clebschgordan_matrix = self._compute_clebschgordan_matrix()

        self.conv_sh = SHConv3D.get(name=radial_profile_type)(
            in_channels,
            out_channels,
            kernel_size,
            max_degree=max_degree,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            **kwargs)

        if use_bias:
            bias = torch.empty((self.output_bispectrum_channels) * self.out_channels)
            init_fn = init_weight(bias_initializer)
            init_fn(bias)
            self.bias = torch.nn.Parameter(
                data=bias,
                requires_grad=True)

        else:
            self.bias = None

        if project:
            self.proj_conv = create_conv_layer(in_channels=in_channels, 
                                               out_channels=out_channels, 
                                               kernel_size = 1, 
                                               padding = 'same',
                                               activation = proj_activation, 
                                               kernel_init=proj_initializer)

        else:
            self.proj_conv = None

    def _output_bispectrum_channels(self):
        return len(self.indices)
        # return self.max_degree + 1
        # n_outputs = 0
        # for n1 in range(0, math.floor(self.max_degree / 2) + 1):
        #     for n2 in range(n1, math.ceil(self.max_degree / 2) + 1):
        #         for i in range(np.abs(n1 - n2), n1 + n2 + 1):
        #             n_outputs += 1
        # return n_outputs

    def _compute_clebschgordan_matrix(self):
        cg_mat = {}  # the set of cg matrices
        for n1 in range(self.max_degree + 1):
            for n2 in range(self.max_degree + 1):
                cg_mat[(n1, n2)] = torch.tensor(
                    compute_clebschgordan_matrix(n1, n2),
                    dtype=torch.cdouble,
                )
        return cg_mat

    @property
    def indices(self):
        if self._indices is None:
            self._indices = []
            for n1 in range(0, math.floor(self.max_degree / 2) + 1):
                for n2 in range(n1, math.ceil(self.max_degree / 2) + 1):
                    for i in range(np.abs(n1 - n2), n1 + n2 + 1):
                        self._indices.append((n1, n2, i))
        return self._indices

    @property
    def indices_inverse(self):
        if self._indices_inverse is None:
            self._indices_inverse = {}
            for k, (n1, n2, i) in enumerate(self.indices):
                self._indices_inverse[(n1, n2, i)] = k
        return self._indices_inverse

    def get_bisp_feature_maps(self, sh_feature_maps):
        _, depth, height, width, filters, n_harmonics = sh_feature_maps.get_shape(
        ).as_list()
        batch_size = sh_feature_maps.shape[0]
        sh_feature_maps = torch.reshape(sh_feature_maps, [-1, n_harmonics])

        bispectrum_coeffs = []
        for n1, n2, i in self.indices:
            kronecker_product = []
            f_n1 = self._get_fn(sh_feature_maps, n1)
            for m1 in range(2 * n1 + 1):
                kronecker_product.append(
                    torch.unsqueeze(f_n1[..., m1], -1) *
                    self._get_fn(sh_feature_maps, n2))
            kronecker_product = torch.cat(kronecker_product, axis=-1)
            kronp_clebshgordan = torch.matmul(kronecker_product,
                                           self.clebschgordan_matrix[(n1, n2)])

            n_p = i**2 - (n1 - n2)**2
            Fi = torch.conj(self._get_fn(sh_feature_maps, i))

            if (n1 + n2 + i) % 2 == 0:
                bispectrum_coeffs.append(
                    torch.real(
                        torch.sum(
                            kronp_clebshgordan[:, n_p:n_p + 2 * i + 1] * Fi,
                            -1)))
            else:
                bispectrum_coeffs.append(
                    torch.imag(
                        torch.sum(
                            kronp_clebshgordan[:, n_p:n_p + 2 * i + 1] * Fi,
                            -1)))
        bispectrum_coeffs = torch.stack(bispectrum_coeffs, -1)
        return torch.reshape(bispectrum_coeffs, [
            batch_size,
            depth,
            height,
            width,
            filters * self.output_bispectrum_channels,
        ])

    def _get_spectrum_feature_maps(
        self,
        x,
    ):
        # Iterating over a symbolic `tf.Tensor` is not allowed: AutoGraph did convert this function
        #batch_size = tf.shape(x)[0]
        #depth = tf.shape(x)[1]
        #height = tf.shape(x)[2]
        #width = tf.shape(x)[3]
        #filters = tf.shape(x)[4]
        batch_size, depth, height, width, filters = x.shape
        spect_feature_maps = []
        for n in range(self.max_degree + 1):
            spect_feature_maps.append(1 / (2 * n + 1) * torch.sum(
                self._get_fn(torch.real(x), n)**2 +
                self._get_fn(torch.imag(x), n)**2, -1))
        spect_feature_maps = torch.stack(spect_feature_maps, -1)
        return torch.reshape(spect_feature_maps, [
            batch_size,
            depth,
            height,
            width,
            filters * (self.max_degree + 1),
        ])

    def _get_fn(self, x, n):
        return x[..., n * n:n * n + 2 * n + 1]

    def call(self, inputs, training=None):
        real_x, imag_x = self.conv_sh(inputs, training=training)
        # x = self.get_bisp_feature_maps(
        #     torch.complex(tf.cast(real_x, tf.float32),
        #                tf.cast(imag_x, tf.float32)))
        x = torch.complex(real_x, imag_x)
        x = self.get_bisp_feature_maps(x)
        # x = self._get_spectrum_feature_maps(x)
        x = torch.sign(x) * torch.log(1 + torch.abs(x))
        # x = torch.cat([real_x[..., 0], x], -1)
        # x = tf.cast(x, inputs.dtype)
        if self.bias is not None:
            x = x + self.bias
        x = self.activation(x)
        if self.proj_conv is not None:
            x = self.proj_conv(x)
        return x


class SSHConv3D(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 max_degree=3,
                 strides=1,
                 padding='SAME',
                 kernel_initializer="glorot_uniform",
                 use_bias=True,
                 bias_initializer="zeros",
                 radial_profile_type="radial",
                 activation=("Linear", {"inplace": True}),
                 proj_activation=("RELU", {"inplace": True}),
                 proj_initializer="glorot_uniform",
                 project=True,
                 **kwargs):
        super().__init__(**kwargs)
        logger.info(f"Initializing SSHConv3D layer with filters: {out_channels}")
        self.filters = out_channels
        self.max_degree = max_degree
        self._indices = None
        self._indices_inverse = None
        self.activation = create_activation_layer(activation)

        self.conv_sh = SHConv3D.get(name=radial_profile_type)(
            out_channels,
            kernel_size,
            max_degree=max_degree,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            **kwargs)

        self.n_radial_profiles = self.conv_sh.n_radial_profiles
        self.n_harmonics = self.conv_sh.n_harmonics

        if use_bias:
            bias = torch.empty((self.max_degree + 1) * self.out_channels)
            init_fn = init_weight(bias_initializer)
            init_fn(bias)
            self.bias = torch.nn.Parameter(
                data=bias,
                requires_grad=True)

        else:
            self.bias = None

        if project:
            self.proj_conv = create_conv_layer(in_channels=in_channels, 
                                               out_channels=out_channels, 
                                               kernel_size = 1, 
                                               padding = 'same',
                                               activation = proj_activation, 
                                               kernel_init=proj_initializer)
        else:
            self.proj_conv = None
        logger.info(
            f"Initializing SSHConv3D layer with filters: {out_channels} - done")

    @property
    def indices(self):
        if self._indices is None:
            self._indices = list(range(self.max_degree + 1))
        return self._indices

    @property
    def indices_inverse(self):
        if self._indices_inverse is None:
            self._indices_inverse = list(range(self.max_degree + 1))
        return self._indices_inverse

    def _get_fn(self, x, n):
        return x[..., n * n:n * n + 2 * n + 1]

    def _get_spectrum_feature_maps(
        self,
        real_sh_feature_maps,
        imag_sh_feature_maps,
    ):
        # Iterating over a symbolic `tf.Tensor` is not allowed: AutoGraph did convert this function        
        #batch_size = tf.shape(x)[0]
        #depth = tf.shape(x)[1]
        #height = tf.shape(x)[2]
        #width = tf.shape(x)[3]
        #filters = tf.shape(x)[4]
        batch_size, depth, height, width, filters = real_sh_feature_maps.shape

        spect_feature_maps = []
        for n in range(self.max_degree + 1):
            spect_feature_maps.append(1 / (2 * n + 1) * torch.sum(
                self._get_fn(real_sh_feature_maps, n)**2 +
                self._get_fn(imag_sh_feature_maps, n)**2, -1))
        spect_feature_maps = torch.stack(spect_feature_maps, -1)
        return torch.reshape(spect_feature_maps, [
            batch_size,
            depth,
            height,
            width,
            filters * (self.max_degree + 1),
        ])

    def call(self, inputs):
        real_x, imag_x = self.conv_sh(inputs)

        x = self._get_spectrum_feature_maps(real_x, imag_x)

        if self.bias is not None:
            x += self.bias

        x = torch.sign(x) * torch.log(1 + torch.abs(x))
        if self.proj_conv is not None:
            x = self.proj_conv(x)
        return x


class SHConv3D(torch.nn.Module):
    _registry = {}  # class var that store the different daughter

    def __init_subclass__(cls, name, **kwargs):
        cls.name = name
        SHConv3D._registry[name] = cls
        super().__init_subclass__(**kwargs)

    @classmethod
    def get(cls, name: str):
        return SHConv3D._registry[name]

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 max_degree=3,
                 strides=1,
                 padding='valid',
                 kernel_initializer="glorot_adapted",
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = out_channels
        self.max_degree = max_degree
        self.n_harmonics = (max_degree + 1)**2
        self.kernel_size = np.max(kernel_size)
        self.strides = strides if type(strides) is not int else 5 * (strides, )
        self.padding = padding.upper()
        self.sh_indices = list(self._sh_indices())
        self.atoms = self._atoms()
        self.n_radial_profiles = self.atoms.shape[-2]
        self.kernel_initializer = kernel_initializer
        if padding.lower() == "same":
            self.conv_central_pixel = create_conv_layer(in_channels, out_channels, kernel_size=1, padding='same', strides=strides, activation = None, kernel_init=kernel_initializer)

        else:
            # crop = self.kernel_size // 2
            self.conv_central_pixel = create_conv_layer(in_channels, out_channels, kernel_size=1, padding='valid', strides=strides, activation = None, kernel_init=kernel_initializer)


    def call(self, inputs, training=None):
        filters = self.atoms
        channels = inputs.shape[-1]
        filters = torch.reshape(filters, (
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
            1,
            self.n_radial_profiles * self.n_harmonics,
        ))

        real_filters = torch.real(filters).type(inputs.dtype)
        imag_filters = torch.imag(filters).type(inputs.dtype)
        reals = list()
        imags = list()
        xs =torch.unbind(inputs, axis=-1)
        for x in xs:
            x = torch.unsqueeze(x, -1)
            reals.append(
               torch.nn.functional.conv3d(
                    x, real_filters, stride=self.strides, padding=self.padding) 
                )
            
                
            imags.append(
                torch.nn.functional.conv3d(
                    x, imag_filters, stride=self.strides, padding=self.padding) )

        real_feature_maps = torch.stack(reals, axis=4)
        imag_feature_maps = torch.stack(imags, axis=4)

        # tf is too dumb for tf.shape(...)[:3]
        # batch_size = tf.shape(real_feature_maps)[0]
        # depth = tf.shape(real_feature_maps)[1]
        # height = tf.shape(real_feature_maps)[2]
        # width = tf.shape(real_feature_maps)[3]
        batch_size, depth, height, width = real_feature_maps.shape
        
        real_feature_maps = torch.reshape(real_feature_maps, (
            batch_size,
            depth,
            height,
            width,
            channels,
            1,
            self.n_radial_profiles,
            self.n_harmonics,
        ))
        imag_feature_maps = torch.reshape(imag_feature_maps, (
            batch_size,
            depth,
            height,
            width,
            channels,
            1,
            self.n_radial_profiles,
            self.n_harmonics,
        ))
        w = torch.repeat_interleave(
            self.w,
            torch.tensor([2 * k + 1 for k in range(self.max_degree + 1)]),
            axis=-1,
        )
        # real_feature_maps = torch.sum(w * real_feature_maps, axis=(4, 6))
        real_feature_maps =torch.unbind(
            torch.sum(w * real_feature_maps, axis=(4, 6)),
            axis=-1,
        )
        real_feature_maps[0] = real_feature_maps[0] + self.conv_central_pixel(
            inputs)
        real_feature_maps = torch.stack(real_feature_maps, axis=-1)

        imag_feature_maps = torch.sum(w * imag_feature_maps, axis=(4, 6))

        return real_feature_maps, imag_feature_maps

    def _atoms(self):
        raise NotImplementedError("It is an abstrac class")

    def _get_spherical_coordinates(self):
        x_grid = np.linspace(-(self.kernel_size // 2), self.kernel_size // 2,
                             self.kernel_size)
        x, y, z = np.meshgrid(x_grid, x_grid, x_grid, indexing='xy')
        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)
        theta = np.arccos(np.divide(z, r, out=np.zeros_like(r), where=r != 0))
        return r, theta, phi

    def _compute_spherical_harmonics(self, theta, phi):
        sh = np.zeros((self.kernel_size, self.kernel_size, self.kernel_size,
                       self.n_harmonics),
                      dtype=np.complex64)
        for n in range(self.max_degree + 1):
            for m in range(-n, n + 1):
                sh[..., self.ravel_sh_index(n, m)] = spherical_harmonics(
                    m, n, theta, phi)
        return sh

    def ravel_sh_index(self, n, m):
        if np.abs(m) > n:
            raise ValueError("m must be in [-n, n]")
        return n**2 + m + n

    def _sh_indices(self):
        for n in range(self.max_degree + 1):
            for m in range(-n, n + 1):
                yield (n, m)

    def unravel_sh_index(self, index):
        return self.sh_indices[index]


class SHConv3DRadial(SHConv3D, name="radial"):

    def __init__(self,
                 filters,
                 kernel_size,
                 max_degree=3,
                 strides=1,
                 padding='valid',
                 radial_function=None,
                 **kwargs):

        self.radial_function = SHConv3DRadial._get_radial_function(
            radial_function)
        # number of radial profiles used to build the filters, w/o the central one
        self.n_radial_profiles = np.max(kernel_size) // 2
        # number of atoms used to build the filters, w/o the central one
        self.n_atoms = (max_degree + 1)**2 * self.n_radial_profiles
        super().__init__(
            filters,
            kernel_size,
            max_degree=max_degree,
            strides=strides,
            padding=padding,
            **kwargs,
        )

    @staticmethod
    def _get_radial_function(input):
        if input is None:
            return lambda r, i: tri(r - i)
        if input == "triangle":
            return lambda r, i: tri(r - i)
        if input == "gaussian":
            return lambda r, i: np.exp(-0.5 * ((i - r) / 0.5)**2)
        if isinstance(input, Callable):
            return input

        raise ValueError("Unknown radial function")

    def _atoms(self):
        r, theta, phi = self._get_spherical_coordinates()
        kernel_profiles = self._compute_kernel_profiles(r)
        spherical_harmonics = self._compute_spherical_harmonics(theta, phi)

        norm = np.sqrt(np.sum(kernel_profiles**2, axis=(0, 1, 2)))
        kernel_profiles = kernel_profiles / norm
        kernel_profiles = kernel_profiles[:, :, :, :, np.newaxis]

        spherical_harmonics = spherical_harmonics[:, :, :, np.newaxis, :]
        atoms = kernel_profiles * spherical_harmonics

        # norm = np.sqrt(np.sum(np.conj(atoms) * atoms, axis=(0, 1, 2)))
        # norm[norm == 0] = 1
        # atoms = atoms / norm

        return torch.tensor(atoms, dtype=torch.cfloat)

    def _compute_kernel_profiles(self, radius):
        n_profiles = self.kernel_size // 2
        kernel_profiles = np.zeros(
            (
                self.kernel_size,
                self.kernel_size,
                self.kernel_size,
                n_profiles,
            ),
            dtype=np.float32,
        )
        r0s = np.arange(1, n_profiles + 1)
        for i, r0 in enumerate(r0s):
            kernel_profiles[:, :, :, i] = self.radial_function(radius, r0)
        return kernel_profiles


class ResidualLayer3D(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding='valid', activation='relu', use_batch_norm=True, conv_type='standard', **kwargs):
        super().__init__()
        self.filters = out_channels
        self.strides = kwargs.get("strides", 1)
        self.conv = create_conv_layer(in_channels, out_channels, kernel_size, padding=padding, strides=self.strides, activation = activation)
        self.c_in = in_channels
        self.activation = activation
        self.bn_1 = torch.nn.BatchNorm3d(in_channels)
        self.bn_2 = torch.nn.BatchNorm3d(out_channels)
        self.proj = create_conv_layer(out_channels, out_channels, kernel_size=1, padding=padding, strides=self.strides, activation = activation)


    def call(self, x, training=None):
        if self.proj:
            return self.bn_1(self.conv(x)) + self.bn_2(self.proj(x))
        else:
            return self.bn_1(self.conv(x)) + x

class LinearUpsampling3D(torch.nn.Module):

    def __init__(self, size=(2, 2, 2), **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.kernel = self._get_kernel(size)

    @staticmethod
    def _get_kernel(size):
        k1 = tri(np.linspace(-1, 1, 2 * size[0] + 1))
        k1 = k1[1:-1]
        k2 = tri(np.linspace(-1, 1, 2 * size[1] + 1))
        k2 = k2[1:-1]
        k3 = tri(np.linspace(-1, 1, 2 * size[2] + 1))
        k3 = k3[1:-1]
        k = np.tensordot(k1, k2, axes=0)
        k = np.tensordot(k, k3, axes=0)
        k = np.reshape(k, k.shape + (
            1,
            1,
        ))
        return torch.tensor(k, dtype=torch.float32)

    def call(self, inputs, training=None):
        xs =torch.unbind(inputs, axis=-1)
        out = []
        kernel = self.kernel.type(inputs.dtype)
        for x in xs:
            x = torch.unsqueeze(x, axis=-1)
            x = conv3d_transpose(x, kernel, self.size, padding="SAME")
            out.append(x)
        return torch.cat(out, axis=-1)


def conv3d_complex(input, filters, strides, **kwargs):
    filters_expanded = torch.cat(
        [
            torch.real(filters),
            torch.imag(filters),
        ],
        axis=-1,
    )

    if type(strides) is int:
        strides = 5 * (strides, )

    return torch.nn.functional.conv3d(input, filters_expanded, stride=strides, **kwargs)


def conv3d_transpose_complex(input, filters, strides, **kwargs):
    out_channels = filters.shape[-1]
    filters_expanded = torch.cat(
        [
            torch.real(filters),
            torch.imag(filters),
        ],
        axis=3,
    )

    output = conv3d_transpose(input, filters_expanded, strides, **kwargs)
    return torch.complex(output[..., :out_channels], output[..., out_channels:])


def conv3d_transpose(input, filters, strides, **kwargs):
    filter_depth, filter_height, filter_width, _, out_channels = filters.get_shape(
    ).as_list()
    batch_size = input.shape[0]
    in_depth = input.shape[1]
    in_height = input.shape[2]
    in_width = input.shape[3]
    if type(strides) is int:
        stride_d = strides
        stride_h = strides
        stride_w = strides
    elif len(strides) == 3:
        stride_d, stride_h, stride_w = strides

    padding = kwargs.get("padding", "SAME")
    if padding == 'VALID':
        output_size_d = (in_depth - 1) * stride_d + filter_depth
        output_size_h = (in_height - 1) * stride_h + filter_height
        output_size_w = (in_width - 1) * stride_w + filter_width
        padding_value = (0,0,0)
    elif padding == 'SAME':
        output_size_d = in_depth * stride_d
        output_size_h = in_height * stride_h
        output_size_w = in_width * stride_w
        pad_d = 0.5 * in_depth * (stride_d - 1) + filter_depth - stride_d
        pad_h = 0.5 * in_height * (stride_h - 1) + filter_height - stride_h
        pad_w = 0.5 * in_width * (stride_w - 1) + filter_width - stride_w
        padding_value = (pad_d, pad_h, pad_w)
    else:
        raise ValueError("unknown padding")
    output_shape = (batch_size, output_size_d, output_size_h, output_size_w,
                    out_channels)

    return torch.nn.functional.conv_transpose3d(
        input, 
        torch.transpose(filters, (0, 1, 2, 4, 3)),
        padding=padding_value, 
        stride = strides, 
        **kwargs)


def is_approx_equal(x, y, epsilon=1e-3):
    return np.abs(x - y) / (np.sqrt(np.abs(x) * np.abs(y)) + epsilon) < epsilon


def tri(x):
    return np.where(np.abs(x) <= 1, np.where(x < 0, x + 1, 1 - x), 0)


def limit_glorot(c_in, c_out):
    return np.sqrt(6 / (c_in + c_out))


def legendre(n, X):
    '''
    Legendre polynomial used to define the SHs for degree n
    '''
    res = np.zeros(((n + 1, ) + (X.shape)))
    for m in range(n + 1):
        res[m] = sp.lpmv(m, n, X)
    return res


def spherical_harmonics_old(m, n, p_legendre, phi):
    '''
    Returns the SH of degree n, order m
    '''
    P_n_m = np.squeeze(p_legendre[np.abs(m)])
    sign = (-1)**((m + np.abs(m)) / 2)
    # Normalization constant
    A = sign * np.sqrt(
        (2 * n + 1) / (4 * np.pi) * np.math.factorial(n - np.abs(m)) /
        np.math.factorial(n + np.abs(m)))
    # Spherical harmonics
    sh = A * np.exp(1j * m * phi) * P_n_m
    # Normalize the SH to unit norm
    sh /= np.sqrt(np.sum(sh * np.conj(sh)))
    return sh.astype(np.complex64)


def spherical_harmonics(m, n, theta, phi):
    '''
    Returns the SH of degree n, order m
    '''
    theta_s = Symbol('theta')
    phi_s = Symbol('phi')
    ynm = Ynm(n, m, theta_s, phi_s).expand(func=True)
    f = lambdify([theta_s, phi_s], ynm, 'numpy')
    return f(theta, phi).astype(np.complex64)


def compute_clebschgordan_matrix(k, l):
    '''
    Computes the matrix that block-diagonilizes the Kronecker product of
    Wigned D matrices of degree k and l respectively
    Output size (2k+1)(2l+1)x(2k+1)(2l+1)
    '''
    c_kl = np.zeros([(2 * k + 1) * (2 * l + 1), (2 * k + 1) * (2 * l + 1)])

    n_off = 0
    for J in range(abs(k - l), k + l + 1):
        m_off = 0
        for m1_i in range(2 * k + 1):
            m1 = m1_i - k
            for m2_i in range(2 * l + 1):
                m2 = m2_i - l
                for n_i in range(2 * J + 1):
                    n = n_i - J
                    if m1 + m2 == n:
                        c_kl[m_off + m2_i,
                             n_off + n_i] = CG(k, m1, l, m2, J,
                                               m1 + m2).doit()
            m_off = m_off + 2 * l + 1
        n_off = n_off + 2 * J + 1

    return c_kl


def degree_to_indices_range(n):
    return range(n * n, n * n + 2 * n + 1)


def degree_to_indices_slice(n):
    return slice(n * n, n * n + 2 * n + 1)


class SpectralStackedConvLayers(nn.Module):
    def __init__(self, 
                 input_feature_channels, 
                 output_feature_channels, 
                 num_convs,
                 max_degree = 3,
                 conv_op=BSHConv3D, 
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
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 'same', 'dilation': 1, 
                           'use_bias': True, 'bias_init': 'zeros',
                           'activation': ("Linear", {"inplace": True}), 
                           'initializer': 'glorot_uniform',
                           'proj': True, 'proj_activation': ("RELU", {"inplace": True}), 'proj_initializer': 'glorot_uniform',
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
        for _ in range(num_convs):
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
                                        project=conv_kwargs['proj']
                                        ))
            
            if norm_op is not None:
                modules.append(norm_op(output_feature_channels))
            modules.append(nonlin)

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
    
    BASE_NUM_FEATURES_bispectral = 8
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
                 num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, 
                 conv_op_type='bispectral',
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

        self.conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 'same', 'dilation': 1, 
                           'use_bias': True, 'bias_init': 'zeros',
                           'activation': ("Linear", {"inplace": True}), 
                           'initializer': 'glorot_uniform',
                           'proj': True, 'proj_activation': ("RELU", {"inplace": True}), 'proj_initializer': 'glorot_uniform',
                           'radial_profile_type': 'radial'}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = self.CONV[conv_op_type]
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

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(SpectralStackedConvLayers(input_features, output_features, num_conv_per_stage,max_degree,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride))
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
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            SpectralStackedConvLayers(input_features, output_features, num_conv_per_stage-1,max_degree,
                                        self.conv_op, self.conv_kwargs, self.norm_op,
                                        self.norm_op_kwargs, self.dropout_op,
                                        self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                        first_stride),
            SpectralStackedConvLayers(output_features, final_num_features, 1,max_degree,
                                        self.conv_op, self.conv_kwargs, self.norm_op,
                                        self.norm_op_kwargs, self.dropout_op,
                                        self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                        first_stride)))

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
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                SpectralStackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage-1,max_degree,
                                            self.conv_op, self.conv_kwargs, self.norm_op,
                                            self.norm_op_kwargs, self.dropout_op,
                                            self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                            first_stride),
                SpectralStackedConvLayers(nfeatures_from_skip, final_num_features, 1,max_degree,
                                            self.conv_op, self.conv_kwargs, self.norm_op,
                                            self.norm_op_kwargs, self.dropout_op,
                                            self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                            first_stride)))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(self.conv_op(in_channels = self.conv_blocks_localization[ds][-1].output_channels, 
                                                 out_channels = num_classes,
                                                 max_degree = max_degree,
                                                 kernel_size = 1, 
                                                 strides = 1, 
                                                 padding = 'valid', 
                                                 use_bias = seg_output_use_bias,
                                                 activation=None,
                                                 kernel_initializer=self.conv_kwargs['initializer'],
                                                 bias_initializer=self.conv_kwargs['bias_init'],
                                                 radial_profile_type=self.conv_kwargs['radial_profile_type'],
                                                 proj_activation=self.conv_kwargs['proj_activation'],
                                                 proj_initializer=self.conv_kwargs['proj_initializer'],
                                                 project=self.conv_kwargs['proj']
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
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

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
