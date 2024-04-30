from itertools import count
import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F


trans_t = lambda t : torch.Tensor([ # translate along the z-axis
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([ # w.r.t, x-axis
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()
 
rot_theta = lambda th : torch.Tensor([ # w.r.t, y-axis
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()

def rodrigues_mat_to_rot(R):
  eps =1e-16
  trc = np.trace(R)
  trc2 = (trc - 1.)/ 2.
  #sinacostrc2 = np.sqrt(1 - trc2 * trc2)
  s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
  if (1 - trc2 * trc2) >= eps:
    tHeta = np.arccos(trc2)
    tHetaf = tHeta / (2 * (np.sin(tHeta)))
  else:
    tHeta = np.real(np.arccos(trc2))
    tHetaf = 0.5 / (1 - tHeta / 6)
  omega = tHetaf * s
  return omega

def rodrigues_rot_to_mat(r):
  wx,wy,wz = r
  theta = np.sqrt(wx * wx + wy * wy + wz * wz)
  a = np.cos(theta)
  b = (1 - np.cos(theta)) / (theta*theta)
  c = np.sin(theta) / theta
  R = np.zeros([3,3])
  R[0, 0] = a + b * (wx * wx)
  R[0, 1] = b * wx * wy - c * wz
  R[0, 2] = b * wx * wz + c * wy
  R[1, 0] = b * wx * wy + c * wz
  R[1, 1] = a + b * (wy * wy)
  R[1, 2] = b * wy * wz - c * wx
  R[2, 0] = b * wx * wz - c * wy
  R[2, 1] = b * wz * wy + c * wx
  R[2, 2] = a + b * (wz * wz)
  return R


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_nerfies_vali_data(basedir, section, downscale = True): 
    # Based on the dataset.json file, determine which images are train and test
    dataset_json = os.path.join(basedir, 'dataset.json')
    if downscale:
        rgb_path = os.path.join(basedir, f'rgb/ds')
    else:
        rgb_path = os.path.join(basedir, f'rgb/1x')
    camera_path = os.path.join(basedir, 'camera')
    mesh_canonical_path = os.path.join(basedir, 'mesh_canonical')
    mesh_deformed_path = os.path.join(basedir, 'mesh_deformed')
    metadata_path = os.path.join(basedir, 'metadata.json')
    scene_path = os.path.join(basedir, 'scene.json')
    param_3DMM_path = os.path.join(basedir,'exp_pose_dict.json' )

    # load dataset and metadata
    dataset = json.load(open(dataset_json))
    metadatas = json.load(open(metadata_path))
    param_3DMM = json.load(open(param_3DMM_path))
    val_ids_all = dataset['val_ids']
    val_ids = []
    for id in val_ids_all:
        if id.startswith(tuple(section.strip().split(','))):
            val_ids.append(id)

    all_imgs = []
    all_poses = []
    all_warp = []
    all_appearance = []
    all_mesh_canonical = []
    all_mesh_deformed = []
    all_param_3DMM = []

    for dataID in val_ids:
        # Load the rgb image
        img_path = os.path.join(rgb_path, f'{dataID}.png')
        img = imageio.imread(img_path)
        img = (np.array(img) / 255.).astype(np.float32)
        all_imgs.append(img)

        # load camera pose (4x4 matric)
        camera_path_i = os.path.join(camera_path, f'{dataID}.json')
        camera = json.load(open(camera_path_i))
        camera_orientation = np.asarray(camera['orientation']).T
        camera_position = np.asarray(camera['position'])
        camera_pose = np.vstack([np.hstack([camera_orientation, camera_position.reshape(3,1)]), np.asarray([0,0,0,1])])
        camera_pose = camera_pose.astype(np.float32)
        all_poses.append(camera_pose)

        # Load the warp and appearance code 
        # Use 0 for validation
        all_warp.append(0)
        all_appearance.append(0)

        # load mesh canonical
        mesh_canonical_path_i = os.path.join(mesh_canonical_path, 'canonical_mesh_colmap_coordinate.npy')
        mesh_canonical = np.load(mesh_canonical_path_i)
        mesh_canonical = mesh_canonical.astype(np.float32)
        all_mesh_canonical.append(mesh_canonical)

        # load mesh deformed
        mesh_deformed_path_i = os.path.join(mesh_deformed_path, f'{dataID}_mesh_colmap_coordinate.npy')
        mesh_deformed = np.load(mesh_deformed_path_i)
        mesh_deformed = mesh_deformed.astype(np.float32)
        all_mesh_deformed.append(mesh_deformed)
        
        # load param_3DMM
        param_3DMM_exp = param_3DMM[dataID]['exp'][0]
        param_3DMM_pose = param_3DMM[dataID]['pose'][0]
        param_3DMM_i = np.concatenate([param_3DMM_exp, param_3DMM_pose])
        param_3DMM_i = param_3DMM_i.astype(np.float32)
        all_param_3DMM.append(param_3DMM_i)


    assert len(all_imgs) == len(all_poses) == len(all_warp) == len(all_appearance) == len(all_mesh_canonical) == len(all_mesh_deformed)

    # Load the camera focal length
    camera_one = json.load(open(os.path.join(camera_path, 'p2_000001.json')))
    camera_focal = camera_one['focal_length'] 

    # Load the scene near and far
    scene = json.load(open(scene_path))
    near, far = scene['near'], scene['far']
    
    # Load the image size
    H, W = all_imgs[0].shape[:2] #height and width
        
    # convert to numpy array
    all_imgs = np.asarray(all_imgs)
    all_poses = np.asarray(all_poses)
    all_warp = np.asarray(all_warp)
    all_appearance = np.asarray(all_appearance)
    all_mesh_canonical = np.asarray(all_mesh_canonical)
    all_mesh_deformed = np.asarray(all_mesh_deformed)
    all_param_3DMM = np.asarray(all_param_3DMM)

    return all_imgs, all_poses, all_warp, all_appearance, all_mesh_canonical, all_mesh_deformed,all_param_3DMM, near, far, [H, W, camera_focal]


def load_nerfies_data(basedir, section, downscale = True): 
    # Based on the dataset.json file, determine which images are train and test
    dataset_json = os.path.join(basedir, 'dataset.json')
    if downscale:
        rgb_path = os.path.join(basedir, f'rgb/ds')
    else:
        rgb_path = os.path.join(basedir, f'rgb/1x')
    camera_path = os.path.join(basedir, 'camera')
    mesh_canonical_path = os.path.join(basedir, 'mesh_canonical')
    mesh_deformed_path = os.path.join(basedir, 'mesh_deformed')
    metadata_path = os.path.join(basedir, 'metadata.json')
    scene_path = os.path.join(basedir, 'scene.json')
    param_3DMM_path = os.path.join(basedir,'exp_pose_dict.json' )
    
    
    # load dataset and metadata
    dataset = json.load(open(dataset_json))
    metadatas = json.load(open(metadata_path))
    param_3DMM = json.load(open(param_3DMM_path))
    train_ids_all = dataset['train_ids']
    val_ids_all = dataset['val_ids']
    train_ids = []
    val_ids = []
    for id in train_ids_all:
        if id.startswith(tuple(section.strip().split(','))):
            train_ids.append(id)
    for id in val_ids_all:
        if id.startswith(tuple(section.strip().split(','))):
            val_ids.append(id)

    all_imgs = []
    all_poses = []
    all_warp = []
    all_appearance = []
    all_mesh_canonical = []
    all_mesh_deformed = []
    all_param_3DMM = []
    counts = [0]
    for part in [train_ids, val_ids]:
        for dataID in part:
            # Load the rgb image
            img_path = os.path.join(rgb_path, f'{dataID}.png')
            img = imageio.imread(img_path)
            img = (np.array(img) / 255.).astype(np.float32)
            all_imgs.append(img)

            # load camera pose (4x4 matric)
            camera_path_i = os.path.join(camera_path, f'{dataID}.json')
            camera = json.load(open(camera_path_i))
            camera_orientation = np.asarray(camera['orientation']).T
            camera_position = np.asarray(camera['position'])
            camera_pose = np.vstack([np.hstack([camera_orientation, camera_position.reshape(3,1)]), np.asarray([0,0,0,1])])
            camera_pose = camera_pose.astype(np.float32)
            all_poses.append(camera_pose)

            # Load the warp and appearance code 
            all_warp.append(metadatas[dataID]['warp_id'])
            all_appearance.append(metadatas[dataID]['appearance_id'])

            # load mesh canonical
            mesh_canonical_path_i = os.path.join(mesh_canonical_path, f'canonical_mesh_colmap_coordinate.npy')
            mesh_canonical = np.load(mesh_canonical_path_i)
            mesh_canonical = mesh_canonical.astype(np.float32)
            all_mesh_canonical.append(mesh_canonical)

            # load mesh deformed
            mesh_deformed_path_i = os.path.join(mesh_deformed_path, f'{dataID}_mesh_colmap_coordinate.npy')
            mesh_deformed = np.load(mesh_deformed_path_i)
            mesh_deformed = mesh_deformed.astype(np.float32)
            all_mesh_deformed.append(mesh_deformed)
            
            # load param_3DMM
            param_3DMM_exp = param_3DMM[dataID]['exp'][0]
            param_3DMM_pose = param_3DMM[dataID]['pose'][0]
            param_3DMM_i = np.concatenate([param_3DMM_exp, param_3DMM_pose])
            param_3DMM_i = param_3DMM_i.astype(np.float32)
            all_param_3DMM.append(param_3DMM_i)
        counts.append(counts[-1] + len(part))
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(counts)-1)]

    assert len(all_imgs) == len(all_poses) == len(all_warp) == len(all_appearance) == len(all_mesh_canonical) == len(all_mesh_deformed)

    # Load the camera focal length
    camera_one = json.load(open(os.path.join(camera_path, 'p2_000001.json')))
    camera_focal = camera_one['focal_length'] 

    # Load the scene near and far
    scene = json.load(open(scene_path))
    near, far = scene['near'], scene['far']
    
    # Load the image size
    H, W = all_imgs[0].shape[:2] #height and width
        
    # convert to numpy array
    all_imgs = np.asarray(all_imgs)
    all_poses = np.asarray(all_poses)
    all_warp = np.asarray(all_warp)
    all_appearance = np.asarray(all_appearance)
    all_mesh_canonical = np.asarray(all_mesh_canonical)
    all_mesh_deformed = np.asarray(all_mesh_deformed)
    all_param_3DMM = np.asarray(all_param_3DMM)

    return all_imgs, all_poses, all_warp, all_appearance, all_mesh_canonical, all_mesh_deformed,all_param_3DMM, near, far, [H, W, camera_focal], i_split

