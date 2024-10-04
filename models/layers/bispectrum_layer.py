import math
from typing import Callable
import logging
import torch
from torch import nn
import numpy as np
from scipy import special as sp
from sympy.physics.quantum.cg import CG
from sympy import Ynm, Symbol, lambdify
import time
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

def init_weight(name, limit=None):
    if "zeros" in name.lower():
        init_fn = nn.init.zeros_    
    elif "ones" in name.lower():
        init_fn = nn.init.ones_  
    elif "xavier_uniform" in name.lower() or "glorot_uniform" in name.lower():
        init_fn = nn.init.xavier_uniform_   
    elif "glorot_adapted" in name.lower():
        init_fn = nn.init.uniform_
    elif "xavier_normal" in name.lower():
        init_fn = nn.init.xavier_normal_   
    else:
        assert NotImplementedError(f"Initialisation {name} not implemented yet")
    
    return init_fn
    
def create_conv_layer(in_channels, out_channels, kernel_size, padding='same', strides = 1, activation = None, kernel_init=None):
    layer = nn.Sequential()
    conv_layer = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=strides)
    if kernel_init is not None:
        init_fn = init_weight(kernel_init)
        if kernel_init == 'glorot_adapted':
            limit = np.sqrt(6 / (in_channels + out_channels))
            init_fn(conv_layer.weight, a=-limit, b=limit)
        else:
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
                 padding='same',
                 kernel_initializer="glorot_uniform",
                 use_bias=True,
                 bias_initializer="zeros",
                 radial_profile_type="radial",
                 activation=("Linear", {"inplace": True}),
                 proj_activation=("RELU", {"inplace": True}),
                 proj_initializer="glorot_uniform",
                 project=True,
                 indices=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.max_degree = max_degree
        #self._indices = ((0, 0, 0),(0, 1, 1),(0, 2, 2),(1, 1, 2),(1, 2, 1),(1, 2, 3),)
        #self._indices = ((0, 0, 0), (0, 1, 1), (0, 2, 2), (1, 1, 0), (1, 1, 2), (1, 2, 1), (1, 2, 3)) # max degree 3
        #self._indices = ((0, 0, 0), (0, 1, 1), (0, 2, 2), (1, 1, 0), (1, 1, 2), (1, 2, 1), (1, 2, 3), (2, 2, 0),(2, 2, 2), (2, 2, 4)) # max degree 4
        # self._indices = None
        self._indices = indices

        self.device = "cpu"
        self.output_bispectrum_channels = self._output_bispectrum_channels()

        self._indices_inverse = None
        if activation is None:
            activation=("Linear", {"inplace": True})
        self.activation = create_activation_layer(activation)
        self.clebschgordan_matrix = self._compute_clebschgordan_matrix()

        self.conv_sh = SHConv3D.get(name=radial_profile_type)(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            max_degree=max_degree,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            **kwargs)

        
        if use_bias:
            bias = torch.empty((self.output_bispectrum_channels * self.out_channels,1,1,1))
            init_fn = init_weight(bias_initializer)
            init_fn(bias)
            self.bias = torch.nn.Parameter(
                data=bias,
                requires_grad=True)

        else:
            self.bias = None

        if project:
            self.proj_conv = create_conv_layer(in_channels=self.output_bispectrum_channels * self.out_channels, 
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
                )#.type(torch.complex32)
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
        _, depth, height, width, filters, n_harmonics = sh_feature_maps.shape

        batch_size = sh_feature_maps.shape[0]
        sh_feature_maps = torch.reshape(sh_feature_maps, [-1, n_harmonics])

        bispectrum_coeffs = []

        for n1, n2, i in self.indices:
            #print(n1,n2,i)
            kronecker_product = []
            #f_n1 = self._get_fn(sh_feature_maps, n1)
            for m1 in range(2 * n1 + 1):
                kronecker_product.append((
                    torch.unsqueeze(self._get_fn(sh_feature_maps, n1)[...,m1],-1) * #f_n1[..., m1], -1) *
                    self._get_fn(sh_feature_maps, n2)))#.to(self.device))

            kronecker_product = torch.cat(kronecker_product, axis=-1).type(self.clebschgordan_matrix[(n1, n2)].dtype).cuda()#.to(self.device)#
            
            kronp_clebshgordan = torch.matmul(kronecker_product, self.clebschgordan_matrix[(n1, n2)].cuda())
            
            del kronecker_product
            torch.cuda.empty_cache()
            n_p = i**2 - (n1 - n2)**2
            #Fi = torch.conj(self._get_fn(sh_feature_maps, i)).cuda()#.to(self.device)


            if (n1 + n2 + i) % 2 == 0:
                bispectrum_coeffs.append(
                    torch.real(
                        torch.sum(
                            kronp_clebshgordan[:, n_p:n_p + 2 * i + 1] * torch.conj(self._get_fn(sh_feature_maps, i)).cuda(),#Fi,
                            -1)))
            else:
                bispectrum_coeffs.append(
                    torch.imag(
                        torch.sum(
                            kronp_clebshgordan[:, n_p:n_p + 2 * i + 1] * torch.conj(self._get_fn(sh_feature_maps, i)).cuda(),#Fi,
                            -1)))
            del kronp_clebshgordan
            torch.cuda.empty_cache()
            #print(f'bispec coef {bispectrum_coeffs[-1].mean().item(), bispectrum_coeffs[-1].min().item(), bispectrum_coeffs[-1].max().item()}')
        bispectrum_coeffs = torch.stack(bispectrum_coeffs, -1).type(torch.float32)
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

    def forward(self, inputs, training=None):

        real_x, imag_x = self.conv_sh(inputs, training=training)

        x = torch.complex(real_x, imag_x)#.type(torch.complex32)
        del real_x, imag_x, inputs
        torch.cuda.empty_cache() 

        x = self.get_bisp_feature_maps(x).cuda()
        # x = self._get_spectrum_feature_maps(x)
        x = torch.sign(x) * torch.log(1 + torch.abs(x))
        x = torch.moveaxis(x, -1,1)
        # x = torch.cat([real_x[..., 0], x], -1)
        # x = tf.cast(x, inputs.dtype)
        
        if self.bias is not None:
            x = x + self.bias
        x = self.activation(x).type(torch.float32)
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
                 indices=None,
                 **kwargs):
        super().__init__(**kwargs)
        # logger.info(f"Initializing SSHConv3D layer with filters: {out_channels}")
        self.filters = out_channels
        self.max_degree = max_degree
        self._indices = indices
        self._indices_inverse = indices
        self.activation = create_activation_layer(activation)

        self.conv_sh = SHConv3D.get(name=radial_profile_type)(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            max_degree=max_degree,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_initializer,
            **kwargs)

        self.n_radial_profiles = self.conv_sh.n_radial_profiles
        self.n_harmonics = self.conv_sh.n_harmonics

        if use_bias:
            bias = torch.empty((self.max_degree + 1) * self.filters,1,1,1)
            init_fn = init_weight(bias_initializer)
            init_fn(bias)
            self.bias = torch.nn.Parameter(
                data=bias,
                requires_grad=True)

        else:
            self.bias = None

        if project:
            self.proj_conv = create_conv_layer(in_channels=(self.max_degree + 1) * self.filters, 
                                               out_channels=out_channels, 
                                               kernel_size = 1, 
                                               padding = 'same',
                                               activation = proj_activation, 
                                               kernel_init=proj_initializer)
        else:
            self.proj_conv = None
        # logger.info(f"Initializing SSHConv3D layer with filters: {out_channels} - done")

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

        batch_size, depth, height, width, filters, _ = real_sh_feature_maps.shape

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

    def forward(self, inputs):
        real_x, imag_x = self.conv_sh(inputs)
        
        x = self._get_spectrum_feature_maps(real_x, imag_x)
        x = torch.moveaxis(x, -1,1)
        
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
        self.strides = strides if type(strides) is not int else 3 * (strides, )
        self.padding = padding
        self.sh_indices = list(self._sh_indices())
        self.atoms = self._atoms()
        self.n_radial_profiles = self.atoms.shape[-2]
        w = torch.empty(
            1,  # batch size
            1,  # depth
            1,  # height
            1,  # width
            in_channels,  # input channels
            self.filters,  # output channels
            self.n_radial_profiles,
            self.max_degree + 1,
            )
        
        init_fn = init_weight(kernel_initializer)
        
        if kernel_initializer == 'glorot_adapted' or True:
            limit = np.sqrt(6 / (in_channels + out_channels))
            torch.nn.init.uniform_(w, a=-limit, b=limit)
            #init_fn(w, a=-limit, b=limit)
        else:
            init_fn(w)

        self.w = nn.Parameter(w, requires_grad=True)

        self.repeat_w_vector = torch.tensor([2 * k + 1 for k in range(max_degree + 1)]) 
        self.kernel_initializer = kernel_initializer
        self.conv_central_pixel = create_conv_layer(in_channels, out_channels, kernel_size=1, padding=padding.lower(), strides=strides, activation = None, kernel_init=kernel_initializer)


    def forward(self, inputs, training=None):

        filters = self.atoms
        #channels = inputs.shape[1]
        filters = torch.movedim(
            torch.reshape(filters, (
                self.kernel_size,
                self.kernel_size,
                self.kernel_size,
                1,  
                self.n_radial_profiles * self.n_harmonics,
                )), 
            (3,4), (1,0))

        #real_filters = torch.real(filters).type(inputs.dtype).cuda()
        #imag_filters = torch.imag(filters).type(inputs.dtype).cuda()
        
        #real_filters, imag_filters = real_filters.cuda(), imag_filters.cuda()

        real_feature_maps = list()
        imag_feature_maps = list()
        xs =list(torch.unbind(inputs, axis=1))
        padding_mult = int((self.kernel_size - 1)/2)
        for x in xs:
            padding = int(self.padding.lower() == 'same')*padding_mult
            x = torch.unsqueeze(x, 1)

            real_feature_maps.append(torch.moveaxis(
               torch.nn.functional.conv3d(
                    x, torch.real(filters).type(inputs.dtype).cuda(), stride=self.strides, padding=padding),1,-1)
            )
            
                
            imag_feature_maps.append(torch.moveaxis(
                torch.nn.functional.conv3d(
                    x, torch.imag(filters).type(inputs.dtype).cuda(), stride=self.strides, padding=padding),1, -1) )

        del filters
        real_feature_maps = torch.stack(real_feature_maps, axis=4)#.to('cpu')
        imag_feature_maps = torch.stack(imag_feature_maps, axis=4)#.to('cpu')
        #del reals#, real_filters
        #del imags#, imag_filters
        torch.cuda.empty_cache() 

        
        
        #batch_size = real_feature_maps.shape[0]
        #depth = real_feature_maps.shape[1]
        #height = real_feature_maps.shape[2] 
        #width = real_feature_maps.shape[3]
        
        real_feature_maps = torch.reshape(real_feature_maps, (
            real_feature_maps.shape[0],
            real_feature_maps.shape[1],
            real_feature_maps.shape[2],
            real_feature_maps.shape[3],
            inputs.shape[1],
            1,
            self.n_radial_profiles,
            self.n_harmonics,
        ))
        imag_feature_maps = torch.reshape(imag_feature_maps, (
            real_feature_maps.shape[0],
            real_feature_maps.shape[1],
            real_feature_maps.shape[2],
            real_feature_maps.shape[3],
            inputs.shape[1],
            1,
            self.n_radial_profiles,
            self.n_harmonics,
        ))

        
        w = torch.repeat_interleave(
            self.w,#.to('cpu'),
            self.repeat_w_vector.cuda(),
            axis=-1,
        )
        

        # real_feature_maps = torch.sum(w * real_feature_maps, axis=(4, 6))
        real_feature_maps =list(torch.unbind(
            torch.sum(w * real_feature_maps, axis=(4, 6)).cuda(),
            axis=-1,
        ))
        
        imag_feature_maps = torch.sum(w * imag_feature_maps, axis=(4, 6)).cuda()#torch.moveaxis(torch.sum(w * imag_feature_maps, axis=(4, 6)), -1, 1)
        del w        
        torch.cuda.empty_cache() 
        
        central_pixel = self.conv_central_pixel(inputs)
        if self.padding.lower() == 'valid':
            crop = self.kernel_size//2
            shapes = central_pixel.shape[2:]
            central_pixel = central_pixel[...,crop:shapes[0]-crop, crop:shapes[0]-crop, crop:shapes[0]-crop]

        real_feature_maps[0] = real_feature_maps[0] + torch.moveaxis(central_pixel, 1,-1)
        real_feature_maps = torch.stack(real_feature_maps, axis=-1)

        
        del central_pixel
        torch.cuda.empty_cache() 

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
                 in_channels,
                 out_channels,
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
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
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


        # sh normalised
        spherical_harmonics = spherical_harmonics[:, :, :, np.newaxis, :]
        atoms = kernel_profiles * spherical_harmonics

        #norm = np.sqrt(np.sum(np.conj(atoms) * atoms, axis=(0, 1, 2)))

        #norm[norm == 0] = 1
        #atoms = atoms / norm

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


class LinearUpsampling3D(torch.nn.Module):

    def __init__(self, size=(2, 2, 2), **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.kernel = self._get_kernel(size)

    @staticmethod
    def _get_kernel(size):
        k1 = tri(np.linspace(-1, 1, 2 * size[0] + 2)) #1))
        k1 = k1[1:-1]
        k2 = tri(np.linspace(-1, 1, 2 * size[1] + 2)) #1))
        k2 = k2[1:-1]
        k3 = tri(np.linspace(-1, 1, 2 * size[2] + 2)) #1))
        k3 = k3[1:-1]
        k = np.tensordot(k1, k2, axes=0)
        k = np.tensordot(k, k3, axes=0)
        k = np.reshape(k, k.shape + (
            1,
            1,
        ))
        return torch.tensor(k, dtype=torch.float32)

    def forward(self, inputs, training=None):

        xs = list(torch.unbind(inputs, axis=1))
        out = []
        kernel = self.kernel.type(inputs.dtype)

        for x in xs:
            x = torch.unsqueeze(x, axis=1)
            x = conv3d_transpose(x, kernel, self.size, padding="SAME")
            out.append(x)
        c = torch.cat(out, axis=1)

        return c



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
    filter_depth, filter_height, filter_width, _, out_channels = filters.shape

    batch_size = input.shape[0]

    in_depth = input.shape[2]
    in_height = input.shape[3]
    in_width = input.shape[4]
    if type(strides) is int:
        stride_d = strides
        stride_h = strides
        stride_w = strides
    elif len(strides) == 3:
        stride_d, stride_h, stride_w = strides

    padding = kwargs.get("padding", "SAME")
    if padding == 'SAME':
        output_size_d = (in_depth - 1) * stride_d + filter_depth
        output_size_h = (in_height - 1) * stride_h + filter_height
        output_size_w = (in_width - 1) * stride_w + filter_width
        padding_value = (1,1,1)
    elif padding == 'VALID':
        output_size_d = in_depth * stride_d
        output_size_h = in_height * stride_h
        output_size_w = in_width * stride_w
        pad_d = 0.5 * (in_depth * (stride_d - 1) + filter_depth - stride_d)
        pad_h = 0.5 * (in_height * (stride_h - 1) + filter_height - stride_h)
        pad_w = 0.5 * (in_width * (stride_w - 1) + filter_width - stride_w)
        padding_value = (int(pad_d), int(pad_h), int(pad_w))
    else:
        raise ValueError("unknown padding")
    output_shape = (batch_size, output_size_d, output_size_h, output_size_w,
                    out_channels)
    p_filters = torch.permute(filters, (3,4,0, 1, 2)).cuda()

    return torch.nn.functional.conv_transpose3d(
        input, 
        p_filters,
        padding=padding_value, 
        stride = strides)


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


def spherical_harmonics(m, n, theta, phi):
    '''
    Returns the SH of degree n, order m
    '''
    theta_s = Symbol('theta')
    phi_s = Symbol('phi')
    ynm = Ynm(n, m, theta_s, phi_s).expand(func=True)
    f = lambdify([theta_s, phi_s], ynm, 'numpy')
    sh = f(theta, phi).astype(np.complex64)
    return sh.astype(np.complex64)


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

