import torch
import torch.nn as nn

import models
from models.utils import get_activation, scale_anything
from models.network_utils import get_encoding, get_mlp
from systems.utils import update_module_step
from scipy import spatial
import numpy as np
from nerfacc import ContractionType

def contract_to_unisphere(x, radius, contraction_type):
    if contraction_type == ContractionType.AABB:
        x = scale_anything(x, (-radius, radius), (0, 1))
    elif contraction_type == ContractionType.UN_BOUNDED_SPHERE: # true for rignerf
        x = scale_anything(x, (-radius, radius), (0, 1)) # scale from [-radius, radius] to [0, 1]
        x = x.clone() * 2 - 1  # aabb is at [-1, 1]
        mag = x.clone().norm(dim=-1, keepdim=True) # magnitude of each point
        mask = mag.squeeze(-1) > 1 # index of points outside the sphere
        x[mask] = (2 - 1 / mag[mask]) * (x.clone()[mask] / mag[mask]) # contract points outside the sphere to the sphere
        x = x.clone() / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        raise NotImplementedError
    return x



@models.register('rignerf-deformation')
class RignerfDeformation(nn.Module): # used by rignerf
    def __init__(self, config):
        super(RignerfDeformation, self).__init__()
        self.config = config # texture config
        self.n_point_dims = self.config.get('n_point_dim', 3) # input point dimension
        self.n_output_dims = 4 # delta x, delta y, delta z, occlusion
        point_encoding = get_encoding(self.n_point_dims, self.config.point_encoding_config)
        deformation_3dmm_encoding = get_encoding(self.n_point_dims, self.config.deformation_encoding_config)
        self.n_input_dims = point_encoding.n_output_dims + deformation_3dmm_encoding.n_output_dims # grid output dimension + direction encoding dimension
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    
        self.point_encoding = point_encoding
        self.deformation_3dmm_encoding = deformation_3dmm_encoding
        self.network = network
        self.contraction_type = None
        self.radius = self.config.radius 
        self.method = self.config.method
        assert self.method in ['trilinear', 'nearest']
    
    def forward(self, points, mesh_canonical, mesh_deformed,device):
        # calculate 3DMM deformation
        # np.save("./test_tmp_saved_files/deformation_mesh_canonical.npy",mesh_canonical.detach().cpu().numpy())
        # np.save("./test_tmp_saved_files/deformation_mesh_deformed.npy",mesh_deformed.detach().cpu().numpy())
        # np.save("./test_tmp_saved_files/deformation_point.npy",points.detach().cpu().numpy())
        contract_points = contract_to_unisphere(points.clone(), self.radius, self.contraction_type)
        contract_mesh_canonical = contract_to_unisphere(mesh_canonical, self.radius, self.contraction_type)
        contract_mesh_deformed = contract_to_unisphere(mesh_deformed, self.radius, self.contraction_type)
        # mesh_scale = (mesh_scale  + self.radius) / (self.radius*2)
        mesh_scale = (contract_mesh_deformed - contract_mesh_deformed.mean(dim=0, keepdim=True)).norm(p=2, dim=-1).max()
        mesh_scale = mesh_scale.detach().cpu().numpy()
        # np.save("./test_tmp_saved_files/deformation_mesh_canonical_contract.npy",contract_mesh_canonical.detach().cpu().numpy())
        # np.save("./test_tmp_saved_files/deformation_mesh_deformed_contract.npy",contract_mesh_deformed.detach().cpu().numpy())
        # np.save("./test_tmp_saved_files/deformation_point_contract.npy",contract_points.detach().cpu().numpy())
        # print("deformation mesh_scale",mesh_scale) (kuoyuan_home_glass:0.04-0.05)
        contract_mesh_deformed_kdtree = spatial.KDTree(contract_mesh_deformed.detach().cpu().numpy())
        if self.method == 'nearest':
            distance_to_mesh, deform_mesh_idx = contract_mesh_deformed_kdtree.query(contract_points.clone().detach().cpu().numpy())
            deform_mesh_points = contract_mesh_deformed[deform_mesh_idx].detach().cpu().numpy()
            canonical_mesh_points = contract_mesh_canonical[deform_mesh_idx].detach().cpu().numpy()
            # apply binary mask to remove points that are too far away
            mask = torch.ones_like(contract_points)
            factor = self.config.get('factor', 0.8)
            mask[distance_to_mesh > factor*mesh_scale] = 0
            deform = (canonical_mesh_points - deform_mesh_points)/(np.exp(distance_to_mesh).reshape(-1,1)) 
            deform = torch.from_numpy(deform).float().to(device)
        elif self.method == 'trilinear':
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        # go through MLP to get residual deformation
        encoded_deform = self.deformation_3dmm_encoding(deform)
        encoded_point = self.point_encoding(contract_points)
        network_inp = torch.cat([encoded_point, encoded_deform], dim=-1)
        network_out = self.network(network_inp)
        network_deform_out, network_occlusion_out = network_out.split([3, 1], dim=-1)
        # map network_occlusion_out to [0, 1]
        network_occlusion_out = torch.sigmoid(network_occlusion_out)
        all_deform = network_deform_out + deform
        filtered_all_deform = all_deform * mask
        deformed_points = contract_points + filtered_all_deform
        return deformed_points, network_occlusion_out


    def update_step(self, epoch, global_step):
        update_module_step(self.deformation_3dmm_encoding, epoch, global_step)

    def regularizations(self, out):
        return {}
