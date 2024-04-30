from venv import create
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import vgg19

# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

# Regularized embedding for Deformation MLP
class DeformationEmbedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs,iteration):
        # Regularized embedding
        alpha = self.kwargs['num_freqs']*iteration/self.kwargs['N']
        bands = np.linspace(0, self.kwargs['max_freq_log2'], self.kwargs['num_freqs'])
        x = np.clip(alpha - bands, 0.0, 1.0)
        window = 0.5 * (1 + np.cos(np.pi * x + np.pi)) # shape: (num_freqs,)
        features=torch.cat([fn(inputs) for fn in self.embed_fns], -1) # shape: (N, num_freqs * d)
        # split the features into two original point and embedded features
        if self.kwargs['include_input']:
            identity, features = torch.split(features, [self.kwargs['input_dims'], features.shape[-1] - self.kwargs['input_dims']], dim=-1)
        # Apply the window by broadcasting to save on memory.
        features = torch.reshape(features, (-1, self.kwargs['num_freqs'], 2, self.kwargs['input_dims']))
        window = np.reshape(window, (-1, self.kwargs['num_freqs'],1, 1))
        features = features * torch.from_numpy(window).to(features.device).float()
        # concatenate the original point and the embedded features
        if self.kwargs['include_input']:
            features = torch.cat([identity, features.reshape(inputs.shape[0],-1)], dim=-1)
        return features
        
def get_deform_embedder(multires, input_dims, N=40000, i=0):
    if i == -1:
        return nn.Identity(), input_dims
    embed_kwargs = {
                'include_input' : True, # add the input to the embedding
                'input_dims' : input_dims, 
                'max_freq_log2' : multires-1, # maximum frequency of the positional encoding
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos], # periodic functions to use
                'N': N # when should reach the maximum number of frequencies m
    }
    
    embedder_obj = DeformationEmbedder(**embed_kwargs) # Regularized Positional encoding
    embed = lambda x,iteration, eo=embedder_obj : eo.embed(x,iteration)
    return embed, embedder_obj.out_dim


def get_embedder(multires, input_dims, i=0):
    if i == -1:
        return nn.Identity(), input_dims
    
    embed_kwargs = {
                'include_input' : True, # add the input to the embedding
                'input_dims' : input_dims, 
                'max_freq_log2' : multires-1, # maximum frequency of the positional encoding
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos], # periodic functions to use
    }
    
    embedder_obj = Embedder(**embed_kwargs) # Positional encoding
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class NeRF:
    @staticmethod
    def get_by_name(type,  *args, **kwargs):
        print ("NeRF type selected: %s" % type)

        if type == "original":
            model = NeRFOriginal(*args, **kwargs)
            model = model.float()
        elif type == "direct_temporal":
            model = DirectTemporalNeRF(*args, **kwargs)
            model = model.float()
        elif type == "rignerf":
            model = RigNeRF(*args, **kwargs)
            model = model.float()
        else:
            raise ValueError("Type %s not recognized." % type)
        return model

class RigNeRF(nn.Module):
    # default not consider embedding, but we embed the input
    def __init__(self, D=8, W=256, input_ch=3, output_ch=4, skips=[4], input_ch_views=3, input_ch_3DMM=3, 
                input_ch_meta = 8, use_viewdirs=True, embed_fn=None, zero_canonical=False, frame_num = 1400, 
                memory=[],param_3DMM_dim = 56):
        super(RigNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_3DMM = input_ch_3DMM
        self.input_ch_meta = input_ch_meta
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.memory = memory
        self.embed_fn = embed_fn
        self.zero_canonical = zero_canonical
        self.param_3DMM_dim = param_3DMM_dim # pose and expression parameters derived from 3DMM
        self.deformation_code_embed = self.create_embedding_net(frame_num = frame_num,output_size = input_ch_meta) # use 8 as the output size following paper
        self.appearance_code_embed = self.create_embedding_net(frame_num = frame_num,output_size = input_ch_meta) # use 8 as the output size following paper

        self._warp, self._warp_out = self.create_warp_net() # warp is the deformation MLP except the last layer, warp_out is the last layer

        # the NeRF MLP: take the x (embedded XYZ + viewdirs) as input, output RBGA
        #TODO: change this to the rignerf MLP
        self._nerfmlp = NeRFRigNeRF(D=D, W=W, input_ch=input_ch, input_ch_views=input_ch_views,
                                output_ch=output_ch, skips=skips,
                                 use_viewdirs=use_viewdirs, memory=memory,param_3DMM_dim= param_3DMM_dim,input_ch_meta = input_ch_meta,output_color_ch=3)

    def create_embedding_net(self,frame_num,output_size):
        return nn.Embedding(frame_num,output_size)

    def create_warp_net(self):
        # input layer: input: embedded XYZ + 3DMM deformation + embedded deformation code
        # the width of deformation MLP is half of the NeRF MLP
        layers = [nn.Linear(self.input_ch + self.input_ch_3DMM + self.input_ch_meta, self.W//2)] 
        for i in range(self.D - 1):
            if i in self.memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = self.W//2
            if i in self.skips: # residual connection: add the original embedded XYZ and embedded 3DMM deformation
                in_channels += self.input_ch
                in_channels += self.input_ch_3DMM
                in_channels += self.input_ch_meta

            layers += [layer(in_channels, self.W//2)] 
        return nn.ModuleList(layers), nn.Linear(self.W//2, 3)

    def deformation(self, new_pts, deformation_3DMM, deformation_embed, net, net_final):
        h = torch.cat([new_pts,deformation_3DMM,deformation_embed], dim=-1)
        # print("deformation h shape: ", h.shape)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([h,new_pts,deformation_3DMM,deformation_embed], -1)

        return h,net_final(h)

    def forward(self, x): # x: embedded XYZ + embedded 3DMM deformation + embedded_view + param + deformation code + appearance code
        input_pts, input_3DMM_deformation, input_views, param_3DMM, deformation_code, appearance_code= torch.split(x, [self.input_ch, self.input_ch_3DMM, self.input_ch_views, self.param_3DMM_dim,1,1], dim=-1)

        # deformation code and appearance code are long tensor
        deformation_code = deformation_code.long()
        appearance_code = appearance_code.long()
        deformation_embedding = self.deformation_code_embed(deformation_code).reshape(deformation_code.shape[0],-1) # embedding the deformation code
        # print("deformation_embedding shape",deformation_embedding.shape)
        feature, dx = self.deformation(input_pts,input_3DMM_deformation, deformation_embedding, self._warp, self._warp_out) # deformation MLP
        input_pts_orig = input_pts[:, :3] # the first 3 channels are the original XYZ
        deformation_3DMM_orig = input_3DMM_deformation[:, :3] # the first 3 channels are the original 3DMM deformation
        input_pts = self.embed_fn(input_pts_orig + dx + deformation_3DMM_orig) # positional embedding
        appearance_embedding = self.appearance_code_embed(appearance_code).reshape(appearance_code.shape[0],-1) # embedding the appearance code
        out, _ = self._nerfmlp(torch.cat([input_pts, input_views,feature,appearance_embedding,param_3DMM], dim=-1)) # NeRF MLP
        return out, dx + deformation_3DMM_orig


class NeRFRigNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4],
                 use_viewdirs=True, memory=[], param_3DMM_dim = 56, input_ch_meta = 8, output_color_ch=3):
        super(NeRFRigNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_meta = input_ch_meta
        self.param_3DMM_dim = param_3DMM_dim
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # The 3DMM parameters processing MLP
        self.param_linears = nn.ModuleList(
            [nn.Linear(param_3DMM_dim, W//4)] +
            [nn.Linear(W//4, W//4) for i in range(D//4-1)])
        layers = [nn.Linear(input_ch+W//2+W//4, W)] # input: embedded XYZ + deformation MLP feature vector + 3DMM parameters MLP feature vector 
        for i in range(D - 1):
            if i in memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = W
            if i in self.skips: # residual connection
                in_channels += input_ch
                in_channels += W//2
                in_channels += W//4

            layers += [layer(in_channels, W)]

        self.pts_linears = nn.ModuleList(layers)

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        # Input: pts_linear output(W), embedded_view(input_ch_views), encoded appearance (input_ch_meta), 3DMM parameter embedding (W//2)
        # self.views_linears = nn.ModuleList([nn.Linear(W+input_ch_views, W//2)])

        ### Implementation according to the paper
        self.views_linears = nn.ModuleList(
            [nn.Linear(W+input_ch_views+input_ch_meta, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//4)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, output_color_ch)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views,feature,appearance_embedding,param_3DMM= torch.split(x, [self.input_ch, self.input_ch_views,self.W//2,self.input_ch_meta,self.param_3DMM_dim], dim=-1)
        
        # 3DMM parameters processing MLP
        embedded_param_3DMM = param_3DMM
        for i,l in enumerate(self.param_linears):
            embedded_param_3DMM = self.param_linears[i](embedded_param_3DMM)
            embedded_param_3DMM = F.relu(embedded_param_3DMM)
        # embedded_param_3DMM: W//4
        h = torch.cat([input_pts,feature,embedded_param_3DMM],dim=-1) 

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([h,input_pts,feature,embedded_param_3DMM], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature_out = self.feature_linear(h)
            h = torch.cat([feature_out, input_views, appearance_embedding], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs, torch.zeros_like(input_pts[:, :3])

def hsv_to_rgb(h, s, v):
    '''
    h,s,v in range [0,1]
    '''
    hi = torch.floor(h * 6)
    f = h * 6. - hi
    p = v * (1. - s)
    q = v * (1. - f * s)
    t = v * (1. - (1. - f) * s)

    rgb = torch.cat([hi, hi, hi], -1) % 6
    rgb[rgb == 0] = torch.cat((v, t, p), -1)[rgb == 0]
    rgb[rgb == 1] = torch.cat((q, v, p), -1)[rgb == 1]
    rgb[rgb == 2] = torch.cat((p, v, t), -1)[rgb == 2]
    rgb[rgb == 3] = torch.cat((p, q, v), -1)[rgb == 3]
    rgb[rgb == 4] = torch.cat((t, p, v), -1)[rgb == 4]
    rgb[rgb == 5] = torch.cat((v, p, q), -1)[rgb == 5]
    return rgb

# Ray helpers
def get_rays(H, W, focal, c2w, device):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, (j-H*.5)/focal, torch.ones_like(i)], -1)
    dirs = dirs.to(device)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(c2w[:3,:3] * dirs[..., np.newaxis, :], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w): # get rays based on camera pose, focal and image resolution
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # Here they do the same thing as NeRF
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples





