import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss
import random

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR

import numpy as np

"""
When call trainer.fit(system, datamodule=dm)
Step 1: call on_train_batch_start() in systems/base.py, which calls self.preprocess_data(batch, 'train')
Step 2: call training_step to get the loss, which calls self.forward(batch)
Check https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks for the whole training loop
"""


@systems.register('rignerf-system')
class RignerfSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self): # called by __init__
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * self.config.model.num_samples_per_ray # number of points
        self.train_num_rays = self.config.model.train_num_rays # how many rays to sample from each image

    def forward(self, batch):
        return self.model(batch['rays'],batch['mesh_canonical'],batch['mesh_deformed'],batch['param_3dmm'])
                        
    def preprocess_data(self, batch, stage):
        # load batch data
        if 'index' in batch: # validation / testing
            index = batch['index']
        else: 
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,), device=self.dataset.all_images.device)
        index_cpu = index.cpu()
        # print("preprocess_data index shape", index.shape) # [1]
        # print("preprocess_data index", index) # [one_index]
        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]
            # print("preprocess_data c2w", c2w.shape) #  [1, 3, 4]
            # if the iteration is over a preset number, sample focus on the face area
            if self.global_step > self.config.model.face_focus_after:
                face_area = 1 -  self.dataset.all_face_masks[index_cpu]
                face_area = face_area.bool()
                sample_ray_weight = torch.ones_like(face_area, dtype=torch.float32)
                sample_ray_weight[face_area] *= self.config.model.bias_factor
                sample_ray_weight = sample_ray_weight / torch.sum(sample_ray_weight)
                # sample train_num_rays rays based on the 2D weight map
                ray_index = torch.multinomial(sample_ray_weight.view(-1), self.train_num_rays, replacement=True)
                x = ray_index % self.dataset.w
                y = ray_index // self.dataset.w
                x = x.to(self.dataset.all_images.device)
                y = y.to(self.dataset.all_images.device)
                # np.save(f"./test_tmp_saved_files/x_index_face.npy", x.cpu().numpy())
                # np.save(f"./test_tmp_saved_files/y_index_face.npy", y.cpu().numpy())
            else:
                # sample random rays
                x = torch.randint(
                    0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
                )
                y = torch.randint(
                    0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
                )
                # save to npy
                # np.save(f"./test_tmp_saved_files/x_index.npy", x.cpu().numpy())
                # np.save(f"./test_tmp_saved_files/y_index.npy", y.cpu().numpy())
            
            # print("self.dataset.directions shape", self.dataset.directions.shape) # (H, W, 3)
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            # directions are rays in camera space
            rays_o, rays_d = get_rays(directions, c2w) 
            # np.save(f"./test_tmp_saved_files/rays_o.npy", rays_o.cpu().numpy())
            # np.save(f"./test_tmp_saved_files/rays_d.npy", rays_d.cpu().numpy())
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1])
        else: # validation / testing
            c2w = self.dataset.all_c2w[index][0]
            # print("validation/testing preprocess_data c2w", c2w.shape) #  [3, 4]
            # print("validation/testing preprocess_data directions shape", self.dataset.directions.shape) # (H, W, 3)
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions # all directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0]
            rays_o, rays_d = get_rays(directions, c2w)
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1])
        
        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1) # concate rays_d and rays_o
        
        # get mesh_canonical, mesh_deformed, param_3dmm
        param_3dmm = self.dataset.all_param_3DMM[index].view(-1, self.dataset.all_param_3DMM.shape[-1])
        mesh_deformed = self.dataset.all_mesh_deformed[index_cpu].view(-1, self.dataset.all_mesh_deformed.shape[-1])
        mesh_canonical = self.dataset.all_mesh_canonical[index_cpu].view(-1, self.dataset.all_mesh_canonical.shape[-1])

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random': # true
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        batch.update({
            'rays': rays,
            'rgb': rgb,
            'mesh_deformed': mesh_deformed,
            'mesh_canonical': mesh_canonical,
            'param_3dmm': param_3dmm
        })
    
    def training_step(self, batch, batch_idx):
        # the complete training step, return loss
        out = self(batch) # use model to render batch['rays']
        # the line below will print: training_step out dict dict_keys(['comp_rgb', 'opacity', 'depth', 'rays_valid', 'num_samples', 'weights', 'points', 'intervals', 'ray_indices'])
        # print("training_step out dict", out.keys())
        loss = 0.

        # gradually increase the number of rays to sample
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples'].sum().item()))
            # print("training_step train_num_rays", train_num_rays) 131      
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
        
        loss_rgb = F.smooth_l1_loss(out['comp_rgb'][out['rays_valid'][...,0]], batch['rgb'][out['rays_valid'][...,0]])
        # print("training_step loss_rgb", loss_rgb)
        self.log('train/loss_rgb', loss_rgb)
        loss += loss_rgb * self.C(self.config.system.loss.lambda_rgb)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss, but still slows down training by ~30%
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)
        losses_model_reg = self.model.regularizations(out)
        # print("training_step losses_model_reg", losses_model_reg) empty dict
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))
        
        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        print("validation_step called with batch_idx",batch_idx)
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
        ]) # save image grid
        return {
            'psnr': psnr,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        print("validation_epoch_end called")
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)         

    def test_step(self, batch, batch_idx):  
        print("test_step called with batch_idx",batch_idx)
        out = self(batch)
        psnr = self.criterions['psnr'](out['comp_rgb'].to(batch['rgb']), batch['rgb'])
        W, H = self.dataset.img_wh
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'grayscale', 'img': out['opacity'].view(H, W), 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
        ])
        return {
            'psnr': psnr,
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    

            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=20
            )
            
            self.export()

    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh
        )    