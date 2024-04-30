import os
import imageio
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from scipy import spatial # for nearest neighbor search

from run_dnerf_helpers import *

from load_blender import load_blender_data
from load_nerfies import load_nerfies_data
try:
    from apex import amp
except ImportError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Detected device:", device)
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(network_input):
        num_batches = network_input.shape[0]
        out_list = []
        dx_list = []
        for i in range(0, num_batches, chunk):
            out, dx = fn(network_input[i:i+chunk].float())
            out_list += [out]
            dx_list += [dx]
        return torch.cat(out_list, 0), torch.cat(dx_list, 0)
    return ret
    


def run_network(inputs, viewdirs, warp_meta, appearance_meta,deform_mesh,canonical_mesh, param_3DMM, fn,iteration, embed_fn, embeddirs_fn, embed3DMM_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    inputs: [N_rays, N_points_per_ray,3] (x,y,z) points on the rays
    viewdirs: [N_rays,3]
    warp_meta: [N_rays,1]
    appearance_meta: [N_rays,1]
    deform_mesh: [5023, 3] (x,y,z) points of the mesh
    canonical_mesh: [5023, 3] (x,y,z) points of the canonical mesh
    param_3DMM: [N_rays,56(3DMM_params)]
    """

    assert len(torch.unique(warp_meta)) == 1 and len(torch.unique(appearance_meta)) == 1, "Only accepts all points from same time"
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) # flatten into (N_rays*N_samples, 3)
    # print("run_network fn inputs shape", inputs_flat.shape) 
    # Calculate the 3DMM deformation
    # Build the KDTree for deform_mesh
    deform_mesh_depatch = deform_mesh.detach().cpu().numpy()
    deform_mesh_tree = spatial.KDTree(deform_mesh_depatch)
    # For each point in the input, find the nearest point in the deform_mesh
    distance_to_mesh, deform_mesh_idx = deform_mesh_tree.query(inputs_flat.detach().cpu().numpy())
    # Get the corresponding point in the deform_mesh
    deform_mesh_points = deform_mesh[deform_mesh_idx].detach().cpu().numpy()
    # Get the corresponding point in the canonical mesh
    canonical_mesh_points = canonical_mesh[deform_mesh_idx].detach().cpu().numpy()
    # Calculate the deformation

    deform = (canonical_mesh_points - deform_mesh_points)/(np.exp(distance_to_mesh).reshape(-1,1)) # shape: (N_rays*N_samples, 3)
    deform = torch.from_numpy(deform).float().to(device)

    # Embed the 3DMM deformation
    deform_embed = embed3DMM_fn(deform,iteration)
    # embed position (XYZ)
    embedded = embed_fn(inputs_flat)
    # expand meta data (warp and appearance)
    if warp_meta is not None:
        B, N, _ = inputs.shape
        input_frame_warp = warp_meta[:, None].expand([B, N, 1])
        input_frame_warp_flat = torch.reshape(input_frame_warp, [-1, 1]) # flatten into (N_rays*N_samples, 1)
    
    if appearance_meta is not None:
        B, N, _ = inputs.shape
        input_frame_appearance = appearance_meta[:, None].expand([B, N, 1]) 
        input_frame_appearance_flat = torch.reshape(input_frame_appearance, [-1, 1]) # flatten into (N_rays*N_samples, 1)

    # expand the param_3DMM
    if param_3DMM is not None:
        B, N, _ = inputs.shape
        param_3DMM = param_3DMM[:, None].expand([B, N, param_3DMM.shape[-1]])
        param_3DMM_flat = torch.reshape(param_3DMM, [-1, param_3DMM.shape[-1]])
    
    embedded_dirs = None
    # embed views
    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape) 
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]]) # flatten into (N_rays*N_samples, 3)
        embedded_dirs = embeddirs_fn(input_dirs_flat) 

    network_input = torch.cat([embedded, deform_embed, embedded_dirs, param_3DMM_flat, input_frame_warp_flat, input_frame_appearance_flat], -1)
    outputs_flat, position_delta_flat = batchify(fn, netchunk)(network_input)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    position_delta = torch.reshape(position_delta_flat, list(inputs.shape[:-1]) + [position_delta_flat.shape[-1]])
    return outputs, position_delta

def batchify_rays(rays_o, rays_d, near,far,frame_warp,frame_appearance,param_3DMM, viewdirs, mesh_deformed ,mesh_canonical,iteration=200000, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_o.shape[0], chunk):
        ret = render_rays(rays_o[i:i+chunk], rays_d[i:i+chunk], near[i:i+chunk], far[i:i+chunk], frame_warp[i:i+chunk], frame_appearance[i:i+chunk], 
                            param_3DMM[i:i+chunk],viewdirs[i:i+chunk], mesh_deformed, mesh_canonical, iteration, **kwargs)
        for k in ret: 
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret



def render(H, W, focal, chunk=1024*32, rays=None, frame_warp=None, frame_appearance = None, 
                  param_3DMM = None, mesh_canonical = None, 
                  mesh_deformed = None, iteration = 200000, c2w=None, ndc=False,
                  near=0., far=1., use_viewdirs=False, c2w_staticcam=None, 
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      frame_warp: One value. The warp code of the image
      frame_appearance:  One value.. The appearance code of the image
      param_3DMM: array of shape [#6]. The 3DMM parameters of the image
      mesh_canonical: array of shape [5023, 3]. The canonical mesh of the image
      mesh_deformed: array of shape [5023, 3]. The deformed mesh of the image
      iteration: the current training iteration
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC (Normalized device coordinate) coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
      **kwargs: render_kwargs_train
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w, device)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs: # True
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam,device)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True) # normalize
        viewdirs = torch.reshape(viewdirs, [-1,3]).float() # flatten into (N_rays*N_samples, 3)

    sh = rays_d.shape # [..., 3] 
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float() # flatten into (N_rays*N_samples, 3)
    rays_d = torch.reshape(rays_d, [-1,3]).float() # flatten into (N_rays*N_samples, 3)
    # print("render fn: rays_o.shape", rays_o.shape) 
    # print("render fn: rays_d.shape", rays_d.shape)
    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    # print(f"render fn: near shape: {near.shape}, one value: {near[0]}")
    # print(f"render fn: far shape: {far.shape}, one value: {far[0]}")
    # Expand frame_warp, frame_appearance, param_3DMM
    frame_warp = frame_warp * torch.ones_like(rays_d[...,:1]) # [N_rays*N_samples, 1]
    frame_appearance = frame_appearance * torch.ones_like(rays_d[...,:1]) # [N_rays*N_samples, 1]
    # print(f"render fn: frame_warp shape: {frame_warp.shape}, one value: {frame_warp[0]}")
    # print(f"render fn: frame_appearance shape: {frame_appearance.shape}, one value: {frame_appearance[0]}")
    param_3DMM = param_3DMM * torch.ones_like(rays_d[...,:1]) # [N_rays*N_samples, 56]
    # print(f"render fn: param_3DMM shape: {param_3DMM.shape}")
    # NEW: Not concatenate them, just pass them separately
    # rays = torch.cat([rays_o, rays_d, near, far,frame_warp,frame_appearance,param_3DMM], -1)
    # if use_viewdirs:
    #     rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays_o, rays_d, near, far,frame_warp,frame_appearance,param_3DMM,viewdirs, mesh_deformed ,mesh_canonical,iteration,chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map','depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def create_nerf(args,frame_num):
    """Instantiate NeRF's MLP model.
    """

    # Embed function takes the 2D/3D as input and output the embedding (with dim = input_ch). 
    # We use input_ch_xxx because they are the input dimension for MLPs
    embed_fn, input_ch = get_embedder(args.multires, 3, args.i_embed) # positional embedding for x,y,z
    # positional embedding for 3DMM deformation (with regularization)
    embed3DMM_fn, input_ch_3DMM = get_deform_embedder(args.multires_3DMM, 3, args.deform_N, args.i_embed) 
    

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs: # True
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, 3, args.i_embed)

    output_ch = 4 # RGBA
    skips = [4] # resnet skip connections
    # Initialize coarse network
    model = NeRF.get_by_name(args.nerf_type, D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, input_ch_3DMM=input_ch_3DMM, 
                 input_ch_meta = args.input_ch_meta, use_viewdirs=args.use_viewdirs, embed_fn=embed_fn,
                 zero_canonical=not args.not_zero_canonical, frame_num = frame_num).to(device)
    grad_vars = list(model.parameters())

    model_fine = None # Initialize fine network
    if args.use_two_models_for_fine: # True
        print("Using two models for fine")
        model_fine = NeRF.get_by_name(args.nerf_type, D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views,input_ch_3DMM=input_ch_3DMM,
                          input_ch_meta = args.input_ch_meta, use_viewdirs=args.use_viewdirs, embed_fn=embed_fn,
                          zero_canonical=not args.not_zero_canonical, frame_num = frame_num).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, warp_meta, appearance_meta, deform_mesh,canonical_mesh, param_3DMM, network_fn,iteration : run_network(inputs, 
                                                                viewdirs, warp_meta, appearance_meta, 
                                                                deform_mesh,canonical_mesh, param_3DMM, network_fn,iteration,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                embed3DMM_fn=embed3DMM_fn,
                                                                netchunk=args.netchunk)
    print("fine and coarse network created, frame_num is",frame_num)
    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path) 

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict']) # load model weights
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine': model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model, # coarse network
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
        'use_two_models_for_fine' : args.use_two_models_for_fine,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(device)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1).to(device)

    rgb = torch.sigmoid(raw[...,:3]).to(device)  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        # print('Adding noise to raw')
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise).to(device)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
        # rgb_map = rgb_map + torch.cat([acc_map[..., None] * 0, acc_map[..., None] * 0, (1. - acc_map[..., None])], -1)

    return rgb_map, disp_map, acc_map, weights, depth_map

                            
def render_rays(rays_o,
                rays_d,
                near,
                far,
                frame_warp,
                frame_appearance,
                param_3DMM,
                viewdirs,
                mesh_deformed,
                mesh_canonical,
                iteration,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                z_vals=None,
                use_two_models_for_fine=False):
    """Volumetric rendering.
    Args:
      rays_o: array of shape [N_rays, 3], ray origin used to sample along a ray
      rays_d: array of shape [N_rays, 3] ray direction used to sample along a ray
      near: array of shape [N_rays, 1], near plane of the camera
      far: array of shape [N_rays, 1], far plane of the camera
      frame_warp: array of shape [N_rays, 1], warp code for the frame
      frame_appearance: array of shape [N_rays, 1], appearance code for the frame
      param_3DMM: array of shape [N_rays, 56], 3DMM parameters for the frame
      viewdirs: array of shape [N_rays, 3], viewing direction
      mesh_deformed: array of shape [5023,3], deformed mesh coordinates for the frame
      mesh_canonical: array of shape [5023,3], canonical mesh coordinates for the frame
      network_fn: function. Model for predicting RGB and density at each point in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    N_rays = rays_o.shape[0]
    z_samples = None
    rgb_map_0, disp_map_0, acc_map_0, position_delta_0 = None, None, None, None

    if z_vals is None: # True
        t_vals = torch.linspace(0., 1., steps=N_samples)
        near = near.to(device)
        far = far.to(device)
        t_vals = t_vals.to(device)
        if not lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.Tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

        # network_query_fn input: inputs, viewdirs, warp_meta, appearance_meta, deform_mesh,canonical_mesh, param_3DMM, network_fn 
        if N_importance <= 0: # false
            raw, position_delta = network_query_fn(pts, viewdirs, frame_warp,frame_appearance,mesh_deformed, mesh_canonical,param_3DMM,network_fn,iteration) # Query the model 
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

        else:
            if use_two_models_for_fine: # true
                # coarse model
                raw, position_delta_0 = network_query_fn(pts, viewdirs,frame_warp,frame_appearance,mesh_deformed, mesh_canonical,param_3DMM, network_fn,iteration)
                rgb_map_0, disp_map_0, acc_map_0, weights, depth_map_0 = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

            else:
                with torch.no_grad():
                    raw, _ = network_query_fn(pts, viewdirs, frame_warp,frame_appearance,mesh_deformed, mesh_canonical,param_3DMM, network_fn,iteration)
                    _, _, _, weights, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
            z_samples = z_samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

    # fine model
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
    run_fn = network_fn if network_fine is None else network_fine
    raw, position_delta = network_query_fn(pts, viewdirs, frame_warp,frame_appearance,mesh_deformed, mesh_canonical,param_3DMM, run_fn,iteration)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'z_vals' : z_vals, 'depth_map' : depth_map,
           'position_delta' : position_delta}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        if rgb_map_0 is not None:
            ret['rgb0'] = rgb_map_0
        if disp_map_0 is not None:
            ret['disp0'] = disp_map_0
        if acc_map_0 is not None:
            ret['acc0'] = acc_map_0
        if depth_map_0 is not None:
            ret['depth0'] = depth_map_0
        if position_delta_0 is not None:
            ret['position_delta_0'] = position_delta_0
        if z_samples is not None:
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--nerf_type", type=str, default="rignerf",
                        help='nerf network type')
    parser.add_argument("--N_iter", type=int, default=200000,
                        help='num training iterations')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=1500, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=4e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=120, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=128, 
                        help='number of coarse samples per ray')
    parser.add_argument("--not_zero_canonical", action='store_true',
                        help='if set zero time is not the canonic space')
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, # default embedding for location is 10
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--multires_3DMM", type=int, default=2, 
                        help='log2 of max freq for positional encoding (3D 3DMM deformation)')
    parser.add_argument("--input_ch_meta", type=int, default=8, 
                        help='output channels of metadata (deformation and appearance code) embedding network')
    parser.add_argument("--raw_noise_std", type=float, default=1e0, 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--use_two_models_for_fine", action='store_true',
                        help='use two models for fine results')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_iters_time", type=int, default=0,
                        help='number of steps to train on central time')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    parser.add_argument("--add_tv_loss", action='store_true',
                        help='evaluate tv loss')
    parser.add_argument("--tv_loss_weight", type=float,
                        default=1.e-4, help='weight of tv loss')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='nerfies', 
                        help='options: llff / blender / deepvoxels / nerfies')
    parser.add_argument("--testskip", type=int, default=2,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default= 2000,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default= 10000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default= 30000,
                        help='frequency of weight ckpt saving')

    parser.add_argument("--section",   type=str, default='p2,p3',
                        help='which section the data is trained')
    parser.add_argument("--downscale",  action='store_true',
                        help='downscale or not')
    parser.add_argument("--deform_N", type=int, default=40000,
                        help='when should reach the maximum number of frequencies m for deformation MLP')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data

    if args.dataset_type == 'blender':
        raise Exception('Only support nerfies dataset')
    elif args.dataset_type == 'nerfies':
       # NEW: load nerfies data
        images, poses, warps, appearances, mesh_canonicals, mesh_deformeds,param_3DMM, near, far, hwf, i_split = load_nerfies_data(args.datadir, args.section, downscale = args.downscale)
        print(f"Loaded nerfies: image shape: {images.shape}, poses shape: {poses.shape}, warps shape: {warps.shape}, appearances shape: {appearances.shape}, mesh_canonicals shape: {mesh_canonicals.shape}, mesh_deformeds shape: {mesh_deformeds.shape}, param_3DMM shape: {param_3DMM.shape}, near: {near}, far: {far}, hwf: {hwf}, datadir: {args.datadir}")
        i_train, i_val = i_split # indices to split data into train/val/test

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    
    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args,frame_num=(max(appearances)+1)) # init nerf model
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    # Move training data to GPU
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    warps = torch.Tensor(warps).to(device)
    appearances = torch.Tensor(appearances).to(device)
    param_3DMM = torch.Tensor(param_3DMM).to(device)
    mesh_canonicals = torch.Tensor(mesh_canonicals).to(device)
    mesh_deformeds = torch.Tensor(mesh_deformeds).to(device)



    N_iters = args.N_iter + 1
    print('Begin')

    # Summary writers
    writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))
    
    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()
        # Random from one image
        if i >= args.precrop_iters_time: # always true
            img_i = np.random.choice(i_train)
        else:
            skip_factor = i / float(args.precrop_iters_time) * len(i_train)
            max_sample = max(int(skip_factor), 3)
            img_i = np.random.choice(i_train[:max_sample])

        target = images[img_i]
        pose = poses[img_i, :3, :4]
        warp = warps[img_i]
        appearance = appearances[img_i]
        param_3DMM_i = param_3DMM[img_i]
        mesh_canonical_i = mesh_canonicals[img_i]
        mesh_deformed_i = mesh_deformeds[img_i]


        if N_rand is not None:
            rays_o, rays_d = get_rays(H, W, focal, pose, device)  # rays origin: (H, W, 3), rays direction: (H, W, 3)
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2), each row is (x, y) coordinate  
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,), random indices
            select_coords = coords[select_inds].long()  # (N_rand, 2), coordinates of the random pixels
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3), random origins
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3), random directions
            batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3), stack random rays origin and direction
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3), ground truth colors of the random pixels

        #####  Core optimization loop  #####
        rgb, disp, acc, depth, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, 
                                        frame_warp=warp, frame_appearance = appearance, 
                                        param_3DMM = param_3DMM_i, mesh_canonical = mesh_canonical_i, 
                                        mesh_deformed = mesh_deformed_i, iteration = i, verbose=i < 10, retraw=True, 
                                        **render_kwargs_train)


        optimizer.zero_grad()
        # TODO: use perceptual loss instead of l2 loss
        img_loss = img2mse(rgb, target_s)

        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras: # coarse model loss
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
        
        
        loss.backward()

        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            save_dict = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(), # coarse network weights
                'optimizer_state_dict': optimizer.state_dict(), # optimizer weights
            }
            if render_kwargs_train['network_fine'] is not None:
                save_dict['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict() # fine network weights

            torch.save(save_dict, path)
            print('Saved checkpoints at', path)

        if i % args.i_print == 0:
            tqdm_txt = f"[TRAIN] Iter: {i} Loss_fine: {img_loss.item()} PSNR: {psnr.item()}"
            if args.add_tv_loss:
                tqdm_txt += f" TV: {tv_loss.item()}"
            tqdm.write(tqdm_txt)

            writer.add_scalar('loss', img_loss.item(), i)
            writer.add_scalar('psnr', psnr.item(), i)
            if 'rgb0' in extras:
                writer.add_scalar('loss0', img_loss0.item(), i)
                writer.add_scalar('psnr0', psnr0.item(), i)
            if args.add_tv_loss:
                writer.add_scalar('tv', tv_loss.item(), i)

        del loss, img_loss, psnr, target_s
        if 'rgb0' in extras:
            del img_loss0, psnr0
        if args.add_tv_loss:
            del tv_loss
        del rgb, disp, acc, depth, extras

        if i%args.i_img==0: 
            torch.cuda.empty_cache()
            # Log a rendered validation view to Tensorboard
            img_i=np.random.choice(i_val) # validation images
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
                depth_all = []
                extras_all = []

                # iteratively render the image by heights
                for i in range(0, H, N_rand//W + 1):
                    rays_o_chunk = rays_o[i:i+N_rand//W+1]
                    rays_d_chunk = rays_d[i:i+N_rand//W+1]
                    batch_rays = torch.stack([rays_o_chunk, rays_d_chunk], dim=0)
                    rgb, disp, acc, depth, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays, 
                                            frame_warp=warp, frame_appearance = appearance, 
                                            param_3DMM = param_3DMM_i, mesh_canonical = mesh_canonical_i, 
                                            mesh_deformed = mesh_deformed_i, **render_kwargs_test)
                    rgb_all.append(rgb)
                    disp_all.append(disp)
                    acc_all.append(acc)
                    depth_all.append(depth)
                    extras_all.append(extras)

                rgb = torch.cat(rgb_all, dim=0)
                disp = torch.cat(disp_all, dim=0)
                acc = torch.cat(acc_all, dim=0)
                depth = torch.cat(depth_all, dim=0)
                extras = {k: torch.cat([v[k] for v in extras_all], dim=0) for k in extras_all[0]}

            psnr = mse2psnr(img2mse(rgb, target))
            writer.add_image('gt', to8b(target.cpu().numpy()), i, dataformats='HWC')
            writer.add_image('rgb', to8b(rgb.cpu().numpy()), i, dataformats='HWC')
            writer.add_image('disp', disp.cpu().numpy(), i, dataformats='HW')
            writer.add_image('acc', acc.cpu().numpy(), i, dataformats='HW')
            writer.add_image('depth', depth.cpu().numpy(), i, dataformats='HW')

            if 'rgb0' in extras:
                writer.add_image('rgb_rough', to8b(extras['rgb0'].cpu().numpy()), i, dataformats='HWC')
            if 'disp0' in extras:
                writer.add_image('disp_rough', extras['disp0'].cpu().numpy(), i, dataformats='HW')
            if 'depth0' in extras:
                writer.add_image('depth_rough', extras['depth0'].cpu().numpy(), i, dataformats='HW')
            if 'z_std' in extras:
                writer.add_image('acc_rough', extras['z_std'].cpu().numpy(), i, dataformats='HW')

            print("finish summary")
            writer.flush()
        
        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor') 
    train()
