import torch
import torch.nn as nn

import models
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp
from systems.utils import update_module_step


@models.register('rignerf-volume-radiance')
class RignerfVolumeRadiance(nn.Module): # used by rignerf
    def __init__(self, config):
        super(RignerfVolumeRadiance, self).__init__()
        self.config = config # texture config
        self.n_dir_dims = self.config.get('n_dir_dims', 3) # direction dimension
        self.n_output_dims = 3 # RGB
        dir_encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
        self.n_input_dims = self.config.input_feature_dim + dir_encoding.n_output_dims + self.config.fea_3dmm_dim # grid output dimension + direction encoding dimension
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    
        self.dir_encoding = dir_encoding
        self.network = network
    
    def forward(self, features, dirs, param_3dmm, *args):
        # the line below will print: texture forward with features shape:  torch.Size([255932, 16])  dirs shape:  torch.Size([255932, 3])  param_3dmm shape:  torch.Size([1, 56])
        # print("texture forward with features shape: ",features.shape," dirs shape: ",dirs.shape," param_3dmm shape: ",param_3dmm.shape)
        # expand param_3dmm to match the shape of features
        param_3dmm = param_3dmm.expand(features.shape[0], param_3dmm.shape[-1])
        dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
        dirs_embd = self.dir_encoding(dirs.view(-1, self.n_dir_dims))
        network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd, param_3dmm.view(-1,param_3dmm.shape[-1])] , dim=-1)
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def update_step(self, epoch, global_step):
        update_module_step(self.dir_encoding, epoch, global_step)

    def regularizations(self, out):
        return {}
