import os
import math
import numpy as np
import imageio 
from PIL import Image
import json

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF
from datasets.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from models.ray_utils import get_ray_directions

import pytorch_lightning as pl

import datasets
from utils.misc import get_rank
from scipy import spatial
import cv2



def create_test_meshes(n_test_traj_steps,norm_inv_trans, norm_scale,root_dir):
    # create 3dmm 
    exp_pose_reenact_path = os.path.join(root_dir, 'exp_pose_dict_test.json')
    exp_pose_reenact = json.load(open(exp_pose_reenact_path))
    all_3dmm_params = []
    reenact_mesh_path = os.path.join(root_dir, 'mesh_reenact')
    num_test_meshes = len(os.listdir(reenact_mesh_path))
    # calculate how many steps per mesh
    n_steps_per_mesh = math.ceil(n_test_traj_steps / num_test_meshes)
    # create test meshes
    all_mesh_reenact = []
    for mesh_counter in range(num_test_meshes):
        mesh_file = os.path.join(reenact_mesh_path, '{:06d}_mesh_colmap_coordinate.npy'.format(mesh_counter))
        mesh = torch.from_numpy(np.load(mesh_file)).float()
        mesh_normalized = (norm_inv_trans @ torch.cat([mesh, torch.ones_like(mesh[:,0:1])], dim=-1)[...,None])[:,:3,0]
        mesh_normalized = mesh_normalized / norm_scale
        all_mesh_reenact.append(torch.stack([mesh_normalized] * n_steps_per_mesh, dim=0))
        # load 3dmm params
        param_3DMM_exp = exp_pose_reenact['{:06d}'.format(mesh_counter)]['exp'][0]
        param_3DMM_pose = exp_pose_reenact['{:06d}'.format(mesh_counter)]['pose'][0]
        param_3DMM_i = np.concatenate([param_3DMM_exp, param_3DMM_pose])
        param_3DMM_i = param_3DMM_i.astype(np.float32)
        param_3DMM_i = torch.from_numpy(param_3DMM_i)
        all_3dmm_params.append(torch.stack([param_3DMM_i] * n_steps_per_mesh, dim=0))
    all_mesh_reenact = torch.cat(all_mesh_reenact, dim=0)
    all_mesh_reenact = all_mesh_reenact[:n_test_traj_steps]
    all_3dmm_params = torch.cat(all_3dmm_params, dim=0)
    all_3dmm_params = all_3dmm_params[:n_test_traj_steps]

    return all_mesh_reenact, all_3dmm_params
    

def get_center(pts):
    center = pts.mean(0)
    dis = (pts - center[None,:]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    center = pts[valid].mean(0)
    return center

def create_spheric_poses(cameras, n_steps=100):
    # center = torch.as_tensor(cameras[...,3].mean(0), dtype=cameras.dtype, device=cameras.device)
    # r = (cameras[...,3] - center).norm(p=2, dim=-1).mean()
    
    # up = torch.as_tensor([0., 0., 1.], dtype=center.dtype, device=center.device)

    # all_c2w = []
    # for theta in torch.linspace(0, 2 * math.pi, n_steps):
    #     cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
    #     l = F.normalize(center - cam_pos, p=2, dim=0)
    #     s = F.normalize(l.cross(up), p=2, dim=0)
    #     u = F.normalize(s.cross(l), p=2, dim=0)
    #     c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1)
    #     all_c2w.append(c2w)

    # all_c2w = torch.stack(all_c2w, dim=0)
    
    # sample 

    # get camera from index 100 and jump 10, get n_steps in total
    all_c2w = []
    for i in range(100, 100 + n_steps * 10, 10):
        c2w = cameras[i]
        all_c2w.append(c2w)
    all_c2w = torch.stack(all_c2w, dim=0)
    return all_c2w
    


def normalize_poses(poses, pts,all_mesh_canonical, all_mesh_deformed, up_est_method, center_est_method):
    """
    Args:
        poses: all poses, shape (N, 3, 4)
        pts: all colmap sparse points, shape (N, 3)
        all_mesh_canonical: all canonical meshes, shape (N, V, 3)
        all_mesh_deformed: all deformed meshes, shape (N, V, 3)
        up_est_method: 'ground' for colmap config
        center_est_method: 'lookat' for colmap config
    """

    if center_est_method == 'camera':
        # estimation scene center as the average of all camera positions
        center = poses[...,3].mean(0)
    elif center_est_method == 'lookat': # true for rignerf config
        # estimation scene center as the average of the intersection of selected pairs of camera rays
        cams_ori = poses[...,3]
        cams_dir = poses[:,:3,:3] @ torch.as_tensor([0.,0.,-1.]) # middle point of the camera rays
        cams_dir = F.normalize(cams_dir, dim=-1)
        A = torch.stack([cams_dir, -cams_dir.roll(1,0)], dim=-1) 
        b = -cams_ori + cams_ori.roll(1,0)
        t = torch.linalg.lstsq(A, b).solution
        center = (torch.stack([cams_dir, cams_dir.roll(1,0)], dim=-1) * t[:,None,:] + torch.stack([cams_ori, cams_ori.roll(1,0)], dim=-1)).mean((0,2))
    elif center_est_method == 'point':
        # first estimation scene center as the average of all camera positions
        # later we'll use the center of all points bounded by the cameras as the final scene center
        center = poses[...,3].mean(0)
    else:
        raise NotImplementedError(f'Unknown center estimation method: {center_est_method}')
    
    if up_est_method == 'ground': # true for rignerf config
        # estimate up direction as the normal of the estimated ground plane
        # use RANSAC to estimate the ground plane in the point cloud
        import pyransac3d as pyrsc
        ground = pyrsc.Plane()
        plane_eq, inliers = ground.fit(pts.numpy(), thresh=0.01) # TODO: determine thresh based on scene scale
        plane_eq = torch.as_tensor(plane_eq) # A, B, C, D in Ax + By + Cz + D = 0
        z = F.normalize(plane_eq[:3], dim=-1) # plane normal as up direction
        signed_distance = (torch.cat([pts, torch.ones_like(pts[...,0:1])], dim=-1) * plane_eq).sum(-1)
        if signed_distance.mean() < 0:
            z = -z # flip the direction if points lie under the plane
    elif up_est_method == 'camera':
        # estimate up direction as the average of all camera up directions
        z = F.normalize((poses[...,3] - center).mean(0), dim=0)
    else:
        raise NotImplementedError(f'Unknown up estimation method: {up_est_method}')
    
    # new axis
    y_ = torch.as_tensor([z[1], -z[0], 0.])
    x = F.normalize(y_.cross(z), dim=0)
    y = z.cross(x)


    if center_est_method == 'point':
        # rotation
        Rc = torch.stack([x, y, z], dim=1)
        R = Rc.T
        poses_homo = torch.cat([poses, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([R, torch.as_tensor([[0.,0.,0.]]).T], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3]
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]

        # translation and scaling
        poses_min, poses_max = poses_norm[...,3].min(0)[0], poses_norm[...,3].max(0)[0]
        pts_fg = pts[(poses_min[0] < pts[:,0]) & (pts[:,0] < poses_max[0]) & (poses_min[1] < pts[:,1]) & (pts[:,1] < poses_max[1])]
        center = get_center(pts_fg)
        tc = center.reshape(3, 1)
        t = -tc
        poses_homo = torch.cat([poses_norm, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses_norm.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([torch.eye(3), t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3]
        scale = poses_norm[...,3].norm(p=2, dim=-1).min()
        poses_norm[...,3] /= scale
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
        pts = pts / scale
    else: # true for rignerf
        # rotation and translation
        Rc = torch.stack([x, y, z], dim=1)
        tc = center.reshape(3, 1)
        R, t = Rc.T, -Rc.T @ tc

        # apply the transformation to the poses
        poses_homo = torch.cat([where, torch.as_tensor([[[0.,0.,0.,1.]]]).expand(poses.shape[0], -1, -1)], dim=1)
        inv_trans = torch.cat([torch.cat([R, t], dim=1), torch.as_tensor([[0.,0.,0.,1.]])], dim=0)
        poses_norm = (inv_trans @ poses_homo)[:,:3] # (N_images, 4, 4)

        # scaling
        scale = poses_norm[...,3].norm(p=2, dim=-1).min()
        poses_norm[...,3] /= scale

        # apply the transformation to the point cloud
        pts = (inv_trans @ torch.cat([pts, torch.ones_like(pts[:,0:1])], dim=-1)[...,None])[:,:3,0]
        pts = pts / scale

        # apply the transformation to each canonical and deformed mesh
        new_all_mesh_canonical = []
        for mesh_canonical in all_mesh_canonical:
            new_mesh_canonical = (inv_trans @ torch.cat([mesh_canonical, torch.ones_like(mesh_canonical[:,0:1])], dim=-1)[...,None])[:,:3,0]
            new_all_mesh_canonical.append(new_mesh_canonical / scale)
        new_all_mesh_canonical = torch.stack(new_all_mesh_canonical, dim=0)

        new_all_mesh_deformed = []
        for mesh_deformed in all_mesh_deformed:
            new_mesh_deformed = (inv_trans @ torch.cat([mesh_deformed, torch.ones_like(mesh_deformed[:,0:1])], dim=-1)[...,None])[:,:3,0]
            new_all_mesh_deformed.append(new_mesh_deformed / scale)
        new_all_mesh_deformed = torch.stack(new_all_mesh_deformed, dim=0)
        
    
    return poses_norm, pts, new_all_mesh_canonical, new_all_mesh_deformed, inv_trans, scale


class RignerfNormalizeDatasetBase():
    # the data only has to be processed once
    initialized = False
    properties = {}

    def setup(self, config, split): # called when initializing the child dataset
        self.config = config
        self.split = split # 'train', 'val_split', 'test_split', 'test'
        self.rank = get_rank() # for distributed training

        if not RignerfNormalizeDatasetBase.initialized:
            # load camera data (W,H,focal,cx,cy) from .bin file
            camdata = read_cameras_binary(os.path.join(self.config.root_dir, 'colmap/sparse/0/cameras.bin'))
            H = int(camdata[1].height)
            W = int(camdata[1].width)
            img_wh = torch.as_tensor([W,H])

            if camdata[1].model == 'SIMPLE_RADIAL':
                fx = fy = camdata[1].params[0]  # focal length
                cx = camdata[1].params[1]  # principal point
                cy = camdata[1].params[2]
            elif camdata[1].model in ['PINHOLE', 'OPENCV']:
                fx = camdata[1].params[0] 
                fy = camdata[1].params[1] 
                cx = camdata[1].params[2] 
                cy = camdata[1].params[3] 
            else:
                raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
            
            directions = get_ray_directions(W, H, fx, fy, cx, cy).to(self.rank)
            
            # load image data (pose(qvec, tvec), name (used for loading images and masks))
            imdata = read_images_binary(os.path.join(self.config.root_dir, 'colmap/sparse/0/images.bin'))
            
            mesh_canonical_path = os.path.join(self.config.root_dir, 'mesh_canonical')
            mesh_deformed_path = os.path.join(self.config.root_dir, 'mesh_deformed')
            param_3DMM_path = os.path.join(self.config.root_dir,'exp_pose_dict.json')
            mask_path = os.path.join(self.config.root_dir, 'rgb/face_masks')
            param_3DMM = json.load(open(param_3DMM_path))

            all_c2w, all_images, all_mesh_deformed, all_mesh_canonical,all_param_3DMM, all_face_masks = [], [], [], [], [], []

            for i, d in enumerate(imdata.values()):
                # d.name is px_00xxxx.png
                image_name = d.name.split('.')[0] # px_00xxxx
                if not image_name.startswith(tuple(self.config.section.strip().split(','))):
                    continue
                
                R = d.qvec2rotmat()
                t = d.tvec.reshape(3, 1)
                c2w = torch.from_numpy(np.concatenate([R.T, -R.T@t], axis=1)).float()
                c2w[:,1:3] *= -1. # COLMAP => OpenGL # flip the second and third row to match OpenGLs coordinate system in our world coordinate model. 
                all_c2w.append(c2w)
                
                if self.split in ['train', 'val','test']: # train and val dataset
                    if self.config.img_downscale:
                        img_path = os.path.join(self.config.root_dir, 'rgb/ds', d.name)
                    else:
                        raise ValueError("Only support downscale for now!")
                        img_path = os.path.join(self.config.root_dir, 'rgb/1x', d.name)
                    img = Image.open(img_path)
                    img = TF.to_tensor(img).permute(1, 2, 0)[...,:3]
                    all_images.append(img)

                    # load mesh canonical
                    mesh_canonical_path_i = os.path.join(mesh_canonical_path, f'canonical_mesh_colmap_coordinate.npy')
                    mesh_canonical = np.load(mesh_canonical_path_i)
                    mesh_canonical = mesh_canonical.astype(np.float32)
                    mesh_canonical = torch.from_numpy(mesh_canonical)
                    all_mesh_canonical.append(mesh_canonical)

                     # load mesh deformed
                    mesh_deformed_path_i = os.path.join(mesh_deformed_path, f'{image_name}_mesh_colmap_coordinate.npy')
                    mesh_deformed = np.load(mesh_deformed_path_i)
                    mesh_deformed = mesh_deformed.astype(np.float32)
                    mesh_deformed = torch.from_numpy(mesh_deformed)
                    all_mesh_deformed.append(mesh_deformed)

                    # load param_3DMM
                    param_3DMM_exp = param_3DMM[image_name]['exp'][0]
                    param_3DMM_pose = param_3DMM[image_name]['pose'][0]
                    param_3DMM_i = np.concatenate([param_3DMM_exp, param_3DMM_pose])
                    param_3DMM_i = param_3DMM_i.astype(np.float32)
                    param_3DMM_i = torch.from_numpy(param_3DMM_i)
                    all_param_3DMM.append(param_3DMM_i)

                    # load mask to guide the face_focus sampling
                    mask_path_i = os.path.join(mask_path,f'{image_name}.png.png')
                    mask = cv2.imread(mask_path_i, cv2.IMREAD_GRAYSCALE)
                    mask[mask>0] = 1
                    mask = torch.from_numpy(mask.astype(np.float32))
                    all_face_masks.append(mask)
                    
            all_c2w = torch.stack(all_c2w, dim=0)   
            all_mesh_canonical = torch.stack(all_mesh_canonical, dim=0) 
            all_mesh_deformed = torch.stack(all_mesh_deformed, dim=0)
            all_param_3DMM = torch.stack(all_param_3DMM, dim=0)
            all_images = torch.stack(all_images, dim=0)
            all_face_masks = torch.stack(all_face_masks, dim=0)
            
            # load sparse 3D points
            pts3d = read_points3d_binary(os.path.join(self.config.root_dir, 'colmap/sparse/0/points3D.bin'))
            pts3d = [pts3d[k].xyz for k in pts3d]
            # remove points that are too far away
            pts3d = torch.from_numpy(np.array([i for i in pts3d if i[2]<self.config.far_scene and i[2]>self.config.near_scene])).float()
            
            # normalize poses and 3D points
            all_c2w, pts3d,all_mesh_canonical,all_mesh_deformed, norm_inv_trans, norm_scale = normalize_poses(all_c2w, pts3d,all_mesh_canonical, all_mesh_deformed,up_est_method=self.config.up_est_method, center_est_method=self.config.center_est_method)
            RignerfNormalizeDatasetBase.properties = {
                'w': W, 
                'h': H,
                'img_wh': img_wh, 
                'directions': directions, # ([H, W, 3])
                'pts3d': pts3d, # ([N_point, 3])                 
                'all_c2w':all_c2w, # ([N, 3, 4])
                'all_images':all_images, # ([N, H, W, 3])
                'all_mesh_canonical':all_mesh_canonical, # ([N, N_vertex, 3])
                'all_mesh_deformed':all_mesh_deformed, # ([N, N_vertex, 3])
                'all_param_3DMM':all_param_3DMM, # ([N, 56])
                'all_face_masks':all_face_masks, # ([N, H, W])
                'norm_inv_trans': norm_inv_trans,
                'norm_scale': norm_scale
            }
            RignerfNormalizeDatasetBase.initialized = True
            
        for k, v in RignerfNormalizeDatasetBase.properties.items():
            setattr(self, k, v)

        print('RignerfNormalizeDatasetBase initialized with properties: w: {}, h: {}, /\n \
                  img_wh: {}, directions shape: {}, pts3d shape: {}, all_c2w shape: {}, /\n \
                  all_images shape: {}, all_mesh_canonical : {}, all_mesh_deformed length: {}, /\n \
                  all_param_3DMM shape: {}, all_face_masks shape: {}'.format(self.w, self.h, self.img_wh, 
            self.directions.shape, self.pts3d.shape, self.all_c2w.shape, self.all_images.shape, 
            self.all_mesh_canonical.shape, self.all_mesh_deformed.shape, self.all_param_3DMM.shape,
            self.all_face_masks.shape))

        if self.split == 'test':
            # for test, we need to generate the novel poses and use test meshes
            self.all_c2w = create_spheric_poses(self.all_c2w, n_steps=self.config.n_test_traj_steps)
            self.all_images = torch.zeros((self.config.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32)
            self.all_mesh_deformed, self.all_param_3DMM = create_test_meshes(self.config.n_test_traj_steps,self.norm_inv_trans, self.norm_scale,self.config.root_dir)
            self.all_mesh_canonical = torch.stack([self.all_mesh_canonical[0]]*self.config.n_test_traj_steps, dim=0)
            self.all_face_masks = torch.ones((self.config.n_test_traj_steps, self.h, self.w), dtype=torch.float32)
            print('RignerfNormalizeDatasetBase test initialized with: all_c2w shape: {}, all_images shape: {}, /\n \
                all_param_3DMM shape : {}, all_mesh_deformed shape: {}, /\n \
                all_mesh_canonical shape: {}, all_face_masks shape: {}'.format(self.all_c2w.shape, self.all_images.shape,
                self.all_param_3DMM.shape, self.all_mesh_deformed.shape, self.all_mesh_canonical.shape, self.all_face_masks.shape))
        
        self.all_c2w, self.all_images, self.all_param_3DMM= \
            self.all_c2w.float().to(self.rank), \
            self.all_images.float().to(self.rank), \
            self.all_param_3DMM.float().to(self.rank)

class RignerfNormalizeDataset(Dataset, RignerfNormalizeDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class RignerfNormalizeIterableDataset(IterableDataset, RignerfNormalizeDatasetBase): # training dataset by batch
    def __init__(self, config, split):
        self.setup(config, split) # obtain dataset (poses, images, masks) from colmap files (.bin files)

    def __iter__(self):
        # we don't use the dataloader to load data, but the on_train_batch_start in systems/base.py
        while True:
            yield {}


@datasets.register('rignerf')
class RignerfDataModule(pl.LightningDataModule):
    def __init__(self, config): # config: dataset config in yaml file
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']: # train
            self.train_dataset = RignerfNormalizeIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']: # val
            self.val_dataset = RignerfNormalizeDataset(self.config, self.config.get('val_split', 'train'))
        if stage in [None, 'test']: # test
            self.test_dataset = RignerfNormalizeDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']: # predict
            self.predict_dataset = RignerfNormalizeDataset(self.config, 'train')         

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, # self.train/val/test/predict_dataset in setup()
            num_workers=os.cpu_count(), 
            batch_size=batch_size, # 1
            pin_memory=True,
            sampler=sampler # None
        )
    
    def random_loader(self, dataset, batch_size):
        sampler = torch.utils.data.RandomSampler(dataset)
        return DataLoader(
            dataset, # self.train/val/test/predict_dataset in setup()
            num_workers=os.cpu_count(), 
            batch_size=batch_size, # 1
            pin_memory=True,
            sampler=sampler # None
        )
    
    def train_dataloader(self): # load train data (used when calling trainer.fit())
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self): # load validation data (used when calling trainer.validate())
        return self.random_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self): # load test data (used when calling trainer.test())
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self): # load predict data (used when calling trainer.predict())
        return self.general_loader(self.predict_dataset, batch_size=1)       
