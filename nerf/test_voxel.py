import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import NamedTuple, Tuple, Union, List, Callable
from collections import defaultdict
from mlp_models import MLP3D, MLPNeRF
from nerf.semantic_nerf import get_embedder
import os 

def make_3D_coordinates(box_coord, volume_size):
    xmin, xmax, ymin, ymax, zmin, zmax = box_coord
    H,W,Z = volume_size
    # print("x min max", xmin, xmax)
    # print("y min max", ymin, ymax)
    # print("z min max", zmin, zmax)
    # print("HWZ", H,W,Z)
    x_coords = torch.linspace(start=xmin, end=xmax, steps=W)
    y_coords = torch.linspace(start=ymin, end=ymax, steps=H)
    z_coords = torch.linspace(start=zmin, end=zmax, steps=Z)
    X, Y, Z = torch.meshgrid(x_coords, y_coords, z_coords)
    coordinates = torch.stack([X, Y, Z], dim=-1)
    return coordinates   


class NeRFDecoder(torch.nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        # Define the model.
        self.model = MLPNeRF()
        # self.embed_fn, self.input_ch = get_embedder(self.params.multires, self.params.i_embed, scalar_factor=10)
        # self.embeddirs_fn, self.input_ch_views = get_embedder(self.params.multires_views,
        #                                                 self.params.i_embed, scalar_factor=1)
        self.embed_fn, self.input_ch = get_embedder(10, 0, scalar_factor=10)
        self.embeddirs_fn, self.input_ch_views  = get_embedder(4,0,1)

        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path)['network_coarse_state_dict'])
        self.model.cuda()
        self.box_coord = make_3D_coordinates([-2.5, 2.5, -2.5, 2.5, -2.5, 2.5], [128, 128, 128])

        self.ray_dir="up"

    def forward(self, volume_coord=None):
        if volume_coord is None:
            volume_coord = self.box_coord
            # doing embedding

        H, W, Z = volume_coord.shape[:3]
        volume_coord = volume_coord.reshape(-1, 3)
        volume_coord_embed =self.embed_fn(volume_coord)

        if self.ray_dir == "up":
            volume_view_dir = torch.zeros_like(volume_coord).to(volume_coord.device)
            # print("volume_view_dir", volume_view_dir.size(), "volume_coord_embed", volume_coord_embed.size())
            volume_view_dir[:,-1] = -1
            volume_view_dir_embed = self.embeddirs_fn(volume_view_dir)

            # encode_rgb, encode_density = self.NeRF_forward(volume_coord_embed, volume_view_dir_embed)
            # encode_density = 1. - torch.exp(-F.relu(encode_density) *self.dists)

            # encode_rgb = torch.sigmoid(encode_rgb.reshape(self.batch_size, H,W,Z,-1))
            # encode_density = encode_density.reshape(self.batch_size,H,W,Z,-1)
        x = torch.cat([volume_coord_embed, volume_view_dir_embed], dim=-1)
        # x = x.unsqueeze(0)
        # torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        output = self.model(x)
        # outputs = torch.cat([rgb, alpha], -1)

        return output
  

    def run_network(self, pts, viewdirs):
        # # [N_rays, N_samples, 3]
        pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])
        # get to know the range of pts sampling
        embedded = self.embed_fn(pts_flat)

        input_dirs = viewdirs[:, None].expand(pts.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        embedded_dirs = self.embeddirs_fn(input_dirs_flat)
        
        embedded = torch.cat([embedded, embedded_dirs], -1)
        outputs_flat = self.batchify(self.model.forward)(embedded)
        outputs = torch.reshape(outputs_flat, list(pts.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs
    
    def volumetric_rendering(self, ray_batch):
        """
        Volumetric Rendering
        """
        N_rays = ray_batch.shape[0]

        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [N_rays, 1], [N_rays, 1]

        self.N_samples = 128
        t_vals = torch.linspace(0., 1., steps=self.N_samples).cuda()

        z_vals = near * (1. - t_vals) + far * (t_vals) # use linear sampling in depth space


        z_vals = z_vals.expand([N_rays, self.N_samples])

        pts_coarse_sampled = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        #print("pts_coarse_sampled x range",pts_coarse_sampled[...,0].min(), pts_coarse_sampled[...,0].max())
        #print("pts_coarse_sampled y range",pts_coarse_sampled[...,1].min(), pts_coarse_sampled[...,1].max())
        #print("pts_coarse_sampled z range",pts_coarse_sampled[...,2].min(), pts_coarse_sampled[...,2].max())

        raw_coarse = self.run_network(pts_coarse_sampled, viewdirs)
        

        rgb_coarse, disp_coarse, acc_coarse, weights_coarse, depth_coarse = \
            self.raw2outputs(raw_coarse, z_vals, rays_d)

        ret = {}
        ret['raw_coarse'] = raw_coarse
        ret['rgb_coarse'] = rgb_coarse
        ret['disp_coarse'] = disp_coarse
        ret['acc_coarse'] = acc_coarse
        ret['depth_coarse'] = depth_coarse

        # speed up
        for k in ret:
            # if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and self.config["experiment"]["debug"]:
            if (torch.isnan(ret[k]).any()):
                print(f"! [Numerical Error] {k} contains nan")
            if (torch.isinf(ret[k]).any()):
                print(f"! [Numerical Error] {k} contains inf.")
        return ret

    def batchify(self, fn):
        """Constructs a version of 'fn' that applies to smaller batches.
        """
        def ret(inputs):
            return torch.cat([fn(inputs[i:i + self.netchunk]) for i in range(0, inputs.shape[0], self.netchunk )], 0)
        return ret
     
    def batchify_rays(self, render_fn, rays_flat):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        self.chunk = 65536
        self.netchunk = 65536
        chunk = self.chunk 
        for i in range(0, rays_flat.shape[0], chunk):
            ret = render_fn(rays_flat[i:i + chunk])
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret
    
    def render_rays(self, flat_rays):
        """
        Render rays, run in optimisation loop
        Returns:
            List of:
            rgb_map: [batch_size, 3]. Predicted RGB values for rays.
            disp_map: [batch_size]. Disparity map. Inverse of depth.
            acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
            Dict of extras: dict with everything returned by render_rays().
        """

        # Render and reshape
        ray_shape = flat_rays.shape  # num_rays, 11

        # assert ray_shape[0] == self.n_rays  # this is not satisfied in test model
        fn = self.volumetric_rendering
        all_ret = self.batchify_rays(fn, flat_rays)

        for k in all_ret:
            k_sh = list(ray_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        return all_ret


    def raw2outputs(self, raw, z_vals, rays_d):
        """Transforms model's predictions to meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
            raw_noise_std: random perturbations added to ray samples
            
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists) # density

        dists = z_vals[..., 1:] - z_vals[..., :-1]  # # (N_rays, N_samples_-1)
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).cuda()], -1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        alpha = raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples]

        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        # [1, 1-a1, 1-a2, ...]
        # [N_rays, N_samples+1] sliced by [:, :-1] to [N_rays, N_samples]

        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        # [N_rays, 3], the accumulated opacity along the rays, equals "1 - (1-a1)(1-a2)...(1-an)" mathematically


        depth_map = torch.sum(weights * z_vals, -1)  # (N_rays,)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1) + 1e-10 ) ) # Check NaN
        acc_map = torch.sum(weights, -1)

        if True:
            # print("white background")
            rgb_map = rgb_map + (1.-acc_map[..., None])
        
        return rgb_map, disp_map, acc_map, weights, depth_map

 

def create_renders(model):
    # get rays
    rays = np.load('/home/qi_ma/neuips_2024/HyperDiffusion_NeRF/nerf/apple_first_pose_rays.npz')
    # print("rays", rays['sampled_rays'].shape)
    output = model.render_rays(torch.tensor(rays['sampled_rays']).float().cuda())
    return output



def create_voxel(model, save_name, N=128):
    coordinate = make_3D_coordinates([-2.5, 2.5, -2.5, 2.5, -2.5, 2.5], [N, N, N])
    coordinate = coordinate.to(next(model.model.parameters()).device)
    model = model.eval()
    outputs = model(coordinate)
    rgb, density = torch.split(outputs, [3, 1], dim=-1)
    rgb = torch.sigmoid(rgb)
    density = 1. - torch.exp(-F.relu(density))
    # save as npy
    npy_filename = save_name
    if npy_filename is not None:
        os.makedirs(os.path.dirname(npy_filename), exist_ok=True)
        np.save(npy_filename, rgb.cpu().detach().numpy())

    return rgb, density

