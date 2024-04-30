from run_dnerf import *
import matplotlib.pyplot as plt
import torch
from run_dnerf_helpers import to8b
from load_nerfies import load_nerfies_data, load_nerfies_vali_data
import numpy as np
import json
import cv2
from tqdm import tqdm
import math
# set cuda
torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get config file
config_file = "configs/kuoyuan_home_glass.txt"
root_dir = "./kuoyuan_home_glass_renders"
parser = config_parser()
args = parser.parse_args(f'--config {config_file}')

images, poses, warps, appearances, mesh_canonicals, mesh_deformeds,param_3DMM, near, far, hwf, i_split = load_nerfies_data(args.datadir, args.section, downscale = args.downscale)
frame_num = max(appearances)+1

render_train = False
render_vali = False
render_test_one = False
render_test_series = True
print("render_train: ", render_train, "render_vali: ", render_vali, "render_test_one: ", render_test_one, "render_test_series: ", render_test_series)
# load the nerf model
_,render_kwargs_test,_,_,_ =  create_nerf(args,frame_num=frame_num)
bds_dict = {'near' : near,'far' : far}
render_kwargs_test.update(bds_dict)

if render_train:
    train_output_path = os.path.join(root_dir,"./train_images")
    if not os.path.exists(train_output_path):
        os.makedirs(train_output_path)
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    i_train, i_val = i_split
    
    # move data to device
    N_rand = args.N_rand
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    warps = torch.Tensor(warps).to(device)
    appearances = torch.Tensor(appearances).to(device)
    param_3DMM = torch.Tensor(param_3DMM).to(device)
    mesh_canonicals = torch.Tensor(mesh_canonicals).to(device)
    mesh_deformeds = torch.Tensor(mesh_deformeds).to(device)

    for i in tqdm(range(0,len(images),len(images)//12)): # len(images) is 131
        img_i = np.random.choice(i_train)
        target = images[img_i]
        pose = poses[img_i, :3, :4]
        warp = warps[img_i]
        appearance = appearances[img_i]
        param_3DMM_i = param_3DMM[img_i]
        mesh_canonical_i = mesh_canonicals[img_i]
        mesh_deformed_i = mesh_deformeds[img_i]

        with torch.no_grad():
            # batchify rays to avoid OOM issue
            rays_o, rays_d = get_rays(H, W, focal, pose, device)  # rays origin: (H, W, 3), rays direction: (H, W, 3)
            rgb_all = []
            disp_all = []
            acc_all = []
            extras_all = []

            # iteratively render the image by heights
            for i in range(0, H, 2):
                rays_o_chunk = rays_o[i:i+2]
                rays_d_chunk = rays_d[i:i+2]
                batch_rays = torch.stack([rays_o_chunk, rays_d_chunk], dim=0)
                rgb, _, _, _,_ = render(H, W, focal, chunk=args.chunk, rays=batch_rays, 
                                        frame_warp=warp, frame_appearance = appearance, 
                                        param_3DMM = param_3DMM_i, mesh_canonical = mesh_canonical_i, 
                                        mesh_deformed = mesh_deformed_i, **render_kwargs_test)
                rgb_all.append(rgb)

            rgb = torch.cat(rgb_all, dim=0)
            
            # Display and save the gt
            target = target.detach().cpu().numpy()
            target = to8b(target)
            cv2.imwrite(os.path.join(train_output_path, f'train_{img_i}_gt.png'),  cv2.cvtColor(target, cv2.COLOR_RGB2BGR))
            
            # plt.figure(2, figsize=(20,6))
            # plt.imshow(target)
            # plt.show()
            
            # Display the image
            # Convert to numpy first
            rgb = rgb.detach().cpu().numpy()
            rgb = to8b(rgb)
            # save the image
            cv2.imwrite(os.path.join(train_output_path, f'train_{img_i}.png'),  cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            # plt.figure(2, figsize=(20,6))
            # plt.imshow(rgb)
            # plt.show()

if render_vali:
    vali_output_path = os.path.join(root_dir,"./vali_images")
    if not os.path.exists(vali_output_path):
        os.makedirs(vali_output_path)

    images, poses, warps, appearances, mesh_canonicals, mesh_deformeds,param_3DMM, near, far, hwf = load_nerfies_vali_data(args.datadir,args.section,downscale = args.downscale)
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # move data to device
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    warps = torch.Tensor(warps).to(device)
    appearances = torch.Tensor(appearances).to(device)
    param_3DMM = torch.Tensor(param_3DMM).to(device)
    mesh_canonicals = torch.Tensor(mesh_canonicals).to(device)
    mesh_deformeds = torch.Tensor(mesh_deformeds).to(device)

    for img_i in tqdm(range(0,len(images),len(images)//10)): # len(images) is 131
        target = images[img_i]
        pose = poses[img_i, :3, :4]
        warp = warps[img_i]
        appearance = appearances[img_i]
        param_3DMM_i = param_3DMM[img_i]
        mesh_canonical_i = mesh_canonicals[img_i]
        mesh_deformed_i = mesh_deformeds[img_i]

        with torch.no_grad():
            # batchify rays to avoid OOM issue
            rays_o, rays_d = get_rays(H, W, focal, pose, device)  # rays origin: (H, W, 3), rays direction: (H, W, 3)
            rgb_all = []
            disp_all = []
            acc_all = []
            extras_all = []

            # iteratively render the image by heights
            for i in range(0, H, 2):
                rays_o_chunk = rays_o[i:i+2]
                rays_d_chunk = rays_d[i:i+2]
                batch_rays = torch.stack([rays_o_chunk, rays_d_chunk], dim=0)
                rgb, _, _, _,_ = render(H, W, focal, chunk=args.chunk, rays=batch_rays, 
                                        frame_warp=warp, frame_appearance = appearance, 
                                        param_3DMM = param_3DMM_i, mesh_canonical = mesh_canonical_i, 
                                        mesh_deformed = mesh_deformed_i, **render_kwargs_test)
                rgb_all.append(rgb)

            rgb = torch.cat(rgb_all, dim=0)
            
            # Display and save the gt
            target = target.detach().cpu().numpy()
            target = to8b(target)
            cv2.imwrite(os.path.join(vali_output_path, f'vali_{img_i}_gt.png'),  cv2.cvtColor(target, cv2.COLOR_RGB2BGR))
            
            # plt.figure(2, figsize=(20,6))
            # plt.imshow(target)
            # plt.show()
            
            # Display the image
            # Convert to numpy first
            rgb = rgb.detach().cpu().numpy()
            rgb = to8b(rgb)
            # save the image
            cv2.imwrite(os.path.join(vali_output_path, f'vali_{img_i}.png'),  cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            # plt.figure(2, figsize=(20,6))
            # plt.imshow(rgb)
            # plt.show()
if render_test_one:
    # test: new viewpoint and moving face
    mesh_id = 'p3_000728'
    # load the render range
    test_output_path =  os.path.join(root_dir,f"./test_output_{mesh_id}")
    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)
        
    # Load the mesh and parameter of the mesh
    basedir = args.datadir
    # Load the image size
    H,W = 512,512
    mesh_canonical_path = os.path.join(basedir, 'mesh_canonical/canonical_mesh_colmap_coordinate.npy')
    mesh_deformed_path = os.path.join(basedir, f'mesh_deformed/{mesh_id}_mesh_colmap_coordinate.npy')
    param_3DMM_path = os.path.join(basedir, 'exp_pose_dict.json')

    # Based on the dataset.json file, determine which images are train and test
    camera_path = os.path.join(basedir, 'test_camera/orbit-mild')
    scene_path = os.path.join(basedir, 'scene.json')
    # load 3DMM parameters
    param_3DMM = json.load(open(param_3DMM_path))
    test_cameras = os.listdir(camera_path)
    test_cameras = sorted(test_cameras, key=lambda x: int(x.split('.')[0]))
    warp_code, appearance_code = 0,0
    all_poses = []
    all_warp = []
    all_appearance = []
    all_mesh_canonical = []
    all_mesh_deformed = []
    all_param_3DMM = []

    for test_camera in test_cameras:
        # load camera pose (4x4 matric)
        test_camera_path = os.path.join(camera_path, test_camera)
        camera = json.load(open(test_camera_path))
        camera_orientation = np.asarray(camera['orientation']).T
        camera_position = np.asarray(camera['position'])
        camera_pose = np.vstack([np.hstack([camera_orientation, camera_position.reshape(3,1)]), np.asarray([0,0,0,1])])
        camera_pose = camera_pose.astype(np.float32)
        all_poses.append(camera_pose)

        # Load the warp and appearance code 
        # Use 0 for validation
        all_warp.append(warp_code)
        all_appearance.append(appearance_code)

        # load mesh canonical
        mesh_canonical = np.load(mesh_canonical_path)
        mesh_canonical = mesh_canonical.astype(np.float32)
        all_mesh_canonical.append(mesh_canonical)

        # load mesh deformed
        mesh_deformed = np.load(mesh_deformed_path)
        mesh_deformed = mesh_deformed.astype(np.float32)
        all_mesh_deformed.append(mesh_deformed)
        
        # load param_3DMM
        param_3DMM_exp = param_3DMM[mesh_id]['exp'][0]
        param_3DMM_pose = param_3DMM[mesh_id]['pose'][0]
        param_3DMM_i = np.concatenate([param_3DMM_exp, param_3DMM_pose])
        param_3DMM_i = param_3DMM_i.astype(np.float32)
        all_param_3DMM.append(param_3DMM_i)


    assert len(all_poses) == len(all_warp) == len(all_appearance) == len(all_mesh_canonical) == len(all_mesh_deformed) == len(all_param_3DMM)

    # Load the camera focal length
    camera_one = json.load(open(os.path.join(camera_path, '000000.json')))
    camera_focal = camera_one['focal_length'] 
        
    # convert to numpy array
    all_poses = np.asarray(all_poses)
    all_warp = np.asarray(all_warp)
    all_appearance = np.asarray(all_appearance)
    all_mesh_canonical = np.asarray(all_mesh_canonical)
    all_mesh_deformed = np.asarray(all_mesh_deformed)
    all_param_3DMM = np.asarray(all_param_3DMM)

    # Move data to device
    all_poses = torch.from_numpy(all_poses).to(device)
    all_warp = torch.from_numpy(all_warp).to(device)
    all_appearance = torch.from_numpy(all_appearance).to(device)
    all_mesh_canonical = torch.from_numpy(all_mesh_canonical).to(device)
    all_mesh_deformed = torch.from_numpy(all_mesh_deformed).to(device)
    all_param_3DMM = torch.from_numpy(all_param_3DMM).to(device)

    all_rendered_imgs = []
    rendered_images_file = os.listdir(test_output_path)
    rendered_images = sorted([int(i.strip().split('.')[0].split('_')[1]) for i in rendered_images_file])
    # load rendered images
    for rendered_id in rendered_images:
        rendered_image = cv2.imread(os.path.join(test_output_path,f'test_{rendered_id}.png'))
        rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
        all_rendered_imgs.append(rendered_image)

    for i_img in tqdm(range(0,len(all_poses))):
        if i_img in rendered_images:
            continue
        pose = all_poses[i_img,:3, :4]
        warp = all_warp[i_img]
        appearance = all_appearance[i_img]
        mesh_canonical_i = all_mesh_canonical[i_img]
        mesh_deformed_i = all_mesh_deformed[i_img]
        param_3DMM_i = all_param_3DMM[i_img]

        with torch.no_grad():
            rays_o, rays_d = get_rays(H, W, camera_focal, pose, device)
            rgb_all = []

            # iteratively render the image by heights
            for i in range(0, H,2):
                rays_o_chunk = rays_o[i:i+2]
                rays_d_chunk = rays_d[i:i+2]
                batch_rays = torch.stack([rays_o_chunk, rays_d_chunk], dim=0)
                rgb,_,_,_,_ = render(H, W, camera_focal, chunk=args.chunk, rays=batch_rays, 
                                        frame_warp=warp, frame_appearance = appearance, 
                                        param_3DMM = param_3DMM_i, mesh_canonical = mesh_canonical_i, 
                                        mesh_deformed = mesh_deformed_i, **render_kwargs_test)
                rgb_all.append(rgb)


            rgb = torch.cat(rgb_all, dim=0)

            # Display the image
            # Convert to numpy first
            rgb = rgb.detach().cpu().numpy()
            rgb = to8b(rgb)
            # save image
            cv2.imwrite(os.path.join(test_output_path, f'test_{i_img}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            # plt.figure(2, figsize=(20,6))
            # plt.imshow(rgb)
            # plt.show()
            # add to all images
            all_rendered_imgs.append(rgb)

    fps = 30
    video_out = cv2.VideoWriter(os.path.join(test_output_path, 'test_video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for i in range(len(all_rendered_imgs)):
        video_out.write(cv2.cvtColor(all_rendered_imgs[i], cv2.COLOR_RGB2BGR))
    video_out.release()

if render_test_series:
    # test: new viewpoint and dynamic head & expression
    pose_range = (300,500)
    # load the render range
    test_output_path =  os.path.join(root_dir,f"./test_output_range_"+str(pose_range[0])+"_"+str(pose_range[1]))
    if not os.path.exists(test_output_path):
        os.makedirs(test_output_path)
        
    # Load the mesh and parameter of the mesh
    basedir = args.datadir
    # Load the image size
    H,W = 512,512
    mesh_canonical_path = os.path.join(basedir, 'mesh_canonical/canonical_mesh_colmap_coordinate.npy')
    # mesh_deformed_path = os.path.join(basedir, f'mesh_deformed/{mesh_id}_mesh_colmap_coordinate.npy')
    param_3DMM_path = os.path.join(basedir, 'exp_pose_dict.json')

    # Based on the dataset.json file, determine which images are train and test
    camera_path = os.path.join(basedir, 'test_camera/orbit-mild')
    scene_path = os.path.join(basedir, 'scene.json')
    # load 3DMM parameters
    param_3DMM = json.load(open(param_3DMM_path))
    test_cameras = os.listdir(camera_path)
    test_cameras = sorted(test_cameras, key=lambda x: int(x.split('.')[0]))
    warp_code, appearance_code = 0,0
    all_poses = []
    all_warp = []
    all_appearance = []
    all_mesh_canonical = []
    all_mesh_deformed = []
    all_param_3DMM = []

    for ind, test_camera in enumerate(test_cameras):
        # load the mesh 
        mesh_id_int = pose_range[0] + math.ceil((pose_range[1]-pose_range[0])*ind/len(test_cameras))
        mesh_id = 'p3_000'+str(mesh_id_int)
        print(ind, mesh_id)
        mesh_deformed_path = os.path.join(basedir, f'mesh_deformed/{mesh_id}_mesh_colmap_coordinate.npy')

        # load camera pose (4x4 matric)
        test_camera_path = os.path.join(camera_path, test_camera)
        camera = json.load(open(test_camera_path))
        camera_orientation = np.asarray(camera['orientation']).T
        camera_position = np.asarray(camera['position'])
        camera_pose = np.vstack([np.hstack([camera_orientation, camera_position.reshape(3,1)]), np.asarray([0,0,0,1])])
        camera_pose = camera_pose.astype(np.float32)
        all_poses.append(camera_pose)

        # Load the warp and appearance code 
        # Use 0 for validation
        all_warp.append(warp_code)
        all_appearance.append(appearance_code)

        # load mesh canonical
        mesh_canonical = np.load(mesh_canonical_path)
        mesh_canonical = mesh_canonical.astype(np.float32)
        all_mesh_canonical.append(mesh_canonical)

        # load mesh deformed
        mesh_deformed = np.load(mesh_deformed_path)
        mesh_deformed = mesh_deformed.astype(np.float32)
        all_mesh_deformed.append(mesh_deformed)
        
        # load param_3DMM
        param_3DMM_exp = param_3DMM[mesh_id]['exp'][0]
        param_3DMM_pose = param_3DMM[mesh_id]['pose'][0]
        param_3DMM_i = np.concatenate([param_3DMM_exp, param_3DMM_pose])
        param_3DMM_i = param_3DMM_i.astype(np.float32)
        all_param_3DMM.append(param_3DMM_i)


    assert len(all_poses) == len(all_warp) == len(all_appearance) == len(all_mesh_canonical) == len(all_mesh_deformed) == len(all_param_3DMM)

    # Load the camera focal length
    camera_one = json.load(open(os.path.join(camera_path, '000000.json')))
    camera_focal = camera_one['focal_length'] 
        
    # convert to numpy array
    all_poses = np.asarray(all_poses)
    all_warp = np.asarray(all_warp)
    all_appearance = np.asarray(all_appearance)
    all_mesh_canonical = np.asarray(all_mesh_canonical)
    all_mesh_deformed = np.asarray(all_mesh_deformed)
    all_param_3DMM = np.asarray(all_param_3DMM)

    # Move data to device
    all_poses = torch.from_numpy(all_poses).to(device)
    all_warp = torch.from_numpy(all_warp).to(device)
    all_appearance = torch.from_numpy(all_appearance).to(device)
    all_mesh_canonical = torch.from_numpy(all_mesh_canonical).to(device)
    all_mesh_deformed = torch.from_numpy(all_mesh_deformed).to(device)
    all_param_3DMM = torch.from_numpy(all_param_3DMM).to(device)

    all_rendered_imgs = []
    rendered_images_file = os.listdir(test_output_path)
    rendered_images = sorted([int(i.strip().split('.')[0].split('_')[1]) for i in rendered_images_file])
    # load rendered images
    for rendered_id in rendered_images:
        rendered_image = cv2.imread(os.path.join(test_output_path,f'test_{rendered_id}.png'))
        rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
        all_rendered_imgs.append(rendered_image)
        print(f'Loaded rendered image test_{rendered_id}.png')

    for i_img in tqdm(range(0,len(all_poses))):
        if i_img in rendered_images:
            continue
        pose = all_poses[i_img,:3, :4]
        warp = all_warp[i_img]
        appearance = all_appearance[i_img]
        mesh_canonical_i = all_mesh_canonical[i_img]
        mesh_deformed_i = all_mesh_deformed[i_img]
        param_3DMM_i = all_param_3DMM[i_img]

        with torch.no_grad():
            rays_o, rays_d = get_rays(H, W, camera_focal, pose, device)
            rgb_all = []

            # iteratively render the image by heights
            for i in range(0, H,2):
                rays_o_chunk = rays_o[i:i+2]
                rays_d_chunk = rays_d[i:i+2]
                batch_rays = torch.stack([rays_o_chunk, rays_d_chunk], dim=0)
                rgb,_,_,_,_ = render(H, W, camera_focal, chunk=args.chunk, rays=batch_rays, 
                                        frame_warp=warp, frame_appearance = appearance, 
                                        param_3DMM = param_3DMM_i, mesh_canonical = mesh_canonical_i, 
                                        mesh_deformed = mesh_deformed_i, **render_kwargs_test)
                rgb_all.append(rgb)


            rgb = torch.cat(rgb_all, dim=0)

            # Display the image
            # Convert to numpy first
            rgb = rgb.detach().cpu().numpy()
            rgb = to8b(rgb)
            # save image
            cv2.imwrite(os.path.join(test_output_path, f'test_{i_img}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            # plt.figure(2, figsize=(20,6))
            # plt.imshow(rgb)
            # plt.show()
            # add to all images
            all_rendered_imgs.append(rgb)

    fps = 30
    video_out = cv2.VideoWriter(os.path.join(test_output_path, 'test_video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for i in range(len(all_rendered_imgs)):
        video_out.write(cv2.cvtColor(all_rendered_imgs[i], cv2.COLOR_RGB2BGR))
    video_out.release()