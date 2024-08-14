import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import NamedTuple, Tuple, Union, List, Callable
from collections import defaultdict

class NeRF_freeze_batch(nn.Module):
    def __init__(self, 
        params: dict,
        ):
        super(NeRF_freeze_batch, self).__init__()
        self.params = params
        self.H, self.W = params.image_size[0], params.image_size[1]
        self.aspect_ratio = self.W / self.H 
        self.hfov = params.hfov  # camera_angle_x
        self.fx = self.W / 2.0 / np.tan(.5 * self.hfov)
        self.fy = self.fx
        self.cx = (self.W - 1.0) / 2.0
        self.cy = (self.H - 1.0) / 2.0
        self.near, self.far = params.depth_range[0], params.depth_range[1]
        self.convention = params.convention
        self.N_samples = params.N_samples
        self.netchunk = params.netchunk
        self.chunk = params.chunk
        self.white_bkgd = params.white_bkgd
        self.skips = params.skips
        self.ray_dir = params.ray_dir

        t_vals = torch.linspace(0., 1., steps=self.N_samples)
        z_vals = self.near * (1. - t_vals) + self.far * (t_vals) # use linear sampling in depth space
        dists = z_vals[..., 1:] - z_vals[..., :-1]  # # (N_rays, N_samples_-1)
        self.dists = dists[0]



        self.embed_fn, self.input_ch = get_embedder(self.params.multires, self.params.i_embed, scalar_factor=10)
        self.embeddirs_fn, self.input_ch_views = get_embedder(self.params.multires_views,
                                                        self.params.i_embed, scalar_factor=1)
        
    def load_state_dict(self, pts_weights, pts_biases, alpha_weights, alpha_biases, feature_weights, feature_biases, view_weights, view_biases, rgb_weights, rgb_biases):
        self.pts_weights = pts_weights
        self.pts_biases = pts_biases
        self.alpha_weights = alpha_weights
        self.alpha_biases = alpha_biases
        self.feature_weights = feature_weights
        self.feature_biases = feature_biases
        self.view_weights = view_weights
        self.view_biases = view_biases
        self.rgb_weights = rgb_weights
        self.rgb_biases = rgb_biases
        self.batch_size = pts_weights[0].shape[0]
  

    def NeRF_forward(self, input_pts, input_views): #
        """
        x N_samples, 3

        """
        input_pts = input_pts.transpose(-2, -1)
        input_views = input_views.transpose(-2, -1)
        feature = input_pts.unsqueeze(0)

            
        for layer_i, (weight_i, bias_i) in enumerate(zip(self.pts_weights,self.pts_biases)):
            # weight_i: Bx128x63  
            # pts_biases: 
            weight_i.requires_grad = False 
            bias_i.requires_grad = False 
    
            if layer_i - 1 in self.skips:
                # print("feature", feature.size())
                # print("input_pts", input_pts.size())
                feature = torch.cat([input_pts, feature], 0)
            # print("weight_i", weight_i.size(), "feature", feature.size(), "bias_i", bias_i.size())
            feature = torch.matmul(weight_i, feature) + bias_i.unsqueeze(-1)
            feature = F.relu(feature)

        # get alpha 
        for layer_i, (weight_i, bias_i) in enumerate(zip(self.alpha_weights,self.alpha_biases)):
            weight_i.requires_grad = False 
            bias_i.requires_grad = False 
            alpha = torch.matmul(weight_i, feature) + bias_i.unsqueeze(-1)

        # get feature
        for layer_i, (weight_i, bias_i) in enumerate(zip(self.feature_weights,self.feature_biases)):
            weight_i.requires_grad = False 
            bias_i.requires_grad = False 
            feature = torch.matmul(weight_i, feature) + bias_i.unsqueeze(-1)

        # cat feature and views
        # print("feature", feature.size(), "input_views", input_views.unsqueeze(0).repeat(feature.shape[0],1,1) .size() ) # ([4, 128, 262144]) input_views torch.Size([27, 262144])
        h = torch.cat([feature, input_views.unsqueeze(0).repeat(feature.shape[0],1,1) ], dim=1) #  Shape x Batch

        # merge view and feature
        for layer_i, (weight_i, bias_i) in enumerate(zip(self.view_weights,self.view_biases)):
            weight_i.requires_grad = False 
            bias_i.requires_grad = False 
            h = torch.matmul(weight_i, h) + bias_i.unsqueeze(-1)
            h = F.relu(h)
        
        # output rgb color
        for layer_i, (weight_i, bias_i) in enumerate(zip(self.rgb_weights,self.rgb_biases)):
            weight_i.requires_grad = False 
            bias_i.requires_grad = False 
            rgb = torch.matmul(weight_i, h) + bias_i.unsqueeze(-1)
        # ([4, 3, 262144]) alpha torch.Size([4, 1, 262144])
   
        return rgb.transpose(-1,-2), alpha.transpose(-1,-2)

    def ray2feature_forward(self, ray_batch):
        N_rays = ray_batch.shape[0]

        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [N_rays, 1], [N_rays, 1]

        t_vals = torch.linspace(0., 1., steps=self.N_samples).cuda()

        z_vals = near * (1. - t_vals) + far * (t_vals) # use linear sampling in depth space


        z_vals = z_vals.expand([N_rays, self.N_samples])

        pts_coarse_sampled = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
        N_ray, N_sample, _ = pts_coarse_sampled.shape

        pts_coarse_sampled = pts_coarse_sampled.reshape(-1, 3)
        pts_coarse_sampled_embed =self.embed_fn(pts_coarse_sampled)
        volume_view_dir_embed = self.embeddirs_fn(viewdirs)

        encode_rgb, encode_density = self.NeRF_forward(pts_coarse_sampled_embed, volume_view_dir_embed)
        encode_rgb = torch.sigmoid(encode_rgb)
        # raw_coarse = self.run_network(pts_coarse_sampled, viewdirs)
        # return raw_coarse


    def rays2feature(self, flat_rays):
        """
        flat_rays: [B, HW, 11]
        pooling: ["mean, "max", "volmtric rendering"]
        """
        # Render and reshape
        ray_shape = flat_rays.shape  # num_rays, 11
        chunk = self.chunk 
        all_features = []
        for i in range(0, flat_rays.shape[0], chunk):
            feature_i = self.ray2feature_forward(flat_rays[i:i + chunk])
        

    def render_volume_coord(self, volume_coord):
        """
        volume_coord: Nx3, same coord share by all scenes
        for view direction, we pooling the feature from box 6 direction

        """
        H, W, Z = volume_coord.shape[:3]
        volume_coord = volume_coord.reshape(-1, 3)
        volume_coord_embed =self.embed_fn(volume_coord)
        if self.ray_dir == "up":
            volume_view_dir = torch.zeros_like(volume_coord).to(volume_coord.device)
            # print("volume_view_dir", volume_view_dir.size(), "volume_coord_embed", volume_coord_embed.size())
            volume_view_dir[:,-1] = -1
            volume_view_dir_embed = self.embeddirs_fn(volume_view_dir)
            encode_rgb, encode_density = self.NeRF_forward(volume_coord_embed, volume_view_dir_embed)
            encode_density = 1. - torch.exp(-F.relu(encode_density) *self.dists)

            encode_rgb = torch.sigmoid(encode_rgb.reshape(self.batch_size, H,W,Z,-1))
            encode_density = encode_density.reshape(self.batch_size,H,W,Z,-1)
        elif self.ray_dir == "pool":
            volume_view_dirs = [
                    [0, 0, -1],
                    [0, 0, 1],
                    [0, -1, 0],
                    [0, 1, 0],
                    [-1, 0, 0],
                    [1, 0, 0]
                ]
            encode_rgb = []
            encode_density = []
            for volume_view_dir in volume_view_dirs:
                volume_view_dir = torch.tensor(volume_view_dir).to(volume_coord.device).repeat(volume_coord.shape[0],1)
                volume_view_dir_embed = self.embeddirs_fn(volume_view_dir)
                encode_rgb_i, encode_density_i = self.NeRF_forward(volume_coord_embed, volume_view_dir_embed)
                encode_density_i = 1. - torch.exp(-F.relu(encode_density_i) *self.dists)
                encode_rgb_i = torch.sigmoid(encode_rgb_i.reshape(self.batch_size, H,W,Z,-1))
                encode_density_i = encode_density_i.reshape(self.batch_size,H,W,Z,-1)
                encode_rgb.append(encode_rgb_i)
                encode_density.append(encode_density_i)
            encode_rgb = torch.stack(encode_rgb, dim=0)
            encode_density = torch.stack(encode_density, dim=0)
            # assert torch.equal(encode_density[0], encode_density[1]), 'unequal densitys'
            # print("rgb difference", (torch.abs(encode_rgb[0] - encode_rgb[1]).mean() / torch.abs(encode_rgb[0])).mean(), torch.min(encode_rgb[0]),  torch.max(encode_rgb[0]))
            encode_rgb = torch.mean(encode_rgb, dim=0)
            encode_density = torch.mean(encode_density, dim=0)
            
        

        # elif self.ray_dir == "pool":
        #     # [0,0,-1], [0,0,1], [0,-1,0], [0,1,0], [-1,0,0], [1,0,0]
        #     volume_view_dir = torch.zeors_like(volume_coord_embed).to(volume_coord_embed.device)
        
        return encode_rgb, encode_density


    # TODO define the ray for encoding
    # def 

class NeRF_freeze(nn.Module):
    def __init__(self, 
        params: dict,
        ):
        super(NeRF_freeze, self).__init__()
        self.params = params
        self.H, self.W = params.image_size[0], params.image_size[1]
        self.aspect_ratio = self.W / self.H 
        self.hfov = params.hfov  # camera_angle_x
        self.fx = self.W / 2.0 / np.tan(.5 * self.hfov)
        self.fy = self.fx
        self.cx = (self.W - 1.0) / 2.0
        self.cy = (self.H - 1.0) / 2.0
        self.near, self.far = params.depth_range[0], params.depth_range[1]
        self.convention = params.convention
        self.N_samples = params.N_samples
        self.netchunk = params.netchunk
        self.chunk = params.chunk
        self.white_bkgd = params.white_bkgd
        self.skips = params.skips

        self.embed_fn, self.input_ch = get_embedder(self.params.multires, self.params.i_embed, scalar_factor=10)
        self.embeddirs_fn, self.input_ch_views = get_embedder(self.params.multires_views,
                                                        self.params.i_embed, scalar_factor=1)
        
    def sampling_index_active(self, n_rays, img):

        sample_weight_map = (1.0 - img.clone().mean(dim=0).reshape(-1))  # / (max_diff + 1e-9)
        sample_weight_map = torch.ceil(sample_weight_map) + 0.10  # + 0.3
        sample_weight_map = sample_weight_map / torch.sum(sample_weight_map)
        index_hw = sample_weight_map.multinomial(num_samples=n_rays, replacement=True)

        return index_hw
        
    def sampling_index(self, n_rays):
        # index_b = np.random.choice(np.arange(batch_size)).reshape((1, 1))  # sample one image from the full trainiing set
        index_hw = torch.randint(0, self.H * self.W, (1, n_rays))
        return index_hw

    def load_state_dict(self, pts_weights, pts_biases, alpha_weights, alpha_biases, feature_weights, feature_biases, view_weights, view_biases, rgb_weights, rgb_biases):
        # check dim
        if len(pts_weights[0].shape)==3:
            pts_weights = tuple(tensor[0] for tensor in pts_weights)
            pts_biases = tuple(tensor[0] for tensor in pts_biases)
            alpha_weights = tuple(tensor[0] for tensor in alpha_weights)
            alpha_biases = tuple(tensor[0] for tensor in alpha_biases)
            feature_weights = tuple(tensor[0] for tensor in feature_weights)
            feature_biases = tuple(tensor[0] for tensor in feature_biases)
            view_weights = tuple(tensor[0] for tensor in view_weights)
            view_biases = tuple(tensor[0] for tensor in view_biases)
            rgb_weights = tuple(tensor[0] for tensor in rgb_weights)
            rgb_biases = tuple(tensor[0] for tensor in rgb_biases)
        # print("pts_weights", pts_weights[0].shape)

        self.pts_weights = pts_weights
        self.pts_biases = pts_biases
        self.alpha_weights = alpha_weights
        self.alpha_biases = alpha_biases
        self.feature_weights = feature_weights
        self.feature_biases = feature_biases
        self.view_weights = view_weights
        self.view_biases = view_biases
        self.rgb_weights = rgb_weights
        self.rgb_biases = rgb_biases
  

    def NeRF_forward(self, x):
        """
        x N_samples, 90
        """
        # use first batch
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        input_pts = input_pts.transpose(-2, -1)
        input_views = input_views.transpose(-2, -1)
        feature = input_pts
        # print("feature", feature.size(), "input_views", input_views.size())
            
        for layer_i, (weight_i, bias_i) in enumerate(zip(self.pts_weights,self.pts_biases)):
            weight_i.requires_grad = False 
            bias_i.requires_grad = False 
            weight_i = weight_i
            bias_i = bias_i
            # print("feature", feature.size())
            # print("weight_i", weight_i.size())
            if layer_i - 1 in self.skips:
                feature = torch.cat([input_pts, feature], 0)
            feature = torch.matmul(weight_i, feature) + bias_i.unsqueeze(-1)
            feature = F.relu(feature)
        
        # get alpha 
        for layer_i, (weight_i, bias_i) in enumerate(zip(self.alpha_weights,self.alpha_biases)):
            weight_i.requires_grad = False 
            bias_i.requires_grad = False 
            weight_i = weight_i
            bias_i = bias_i
            alpha = torch.matmul(weight_i, feature) + bias_i.unsqueeze(-1)

        # get feature
        for layer_i, (weight_i, bias_i) in enumerate(zip(self.feature_weights,self.feature_biases)):
            weight_i.requires_grad = False 
            bias_i.requires_grad = False 
            weight_i = weight_i
            bias_i = bias_i
            feature = torch.matmul(weight_i, feature) + bias_i.unsqueeze(-1)

        # cat feature and views
        # print("feature", feature.size(), "input_views", input_views.size())
        
        h = torch.cat([feature, input_views], dim=0) #  Shape x Batch

        # merge view and feature
        for layer_i, (weight_i, bias_i) in enumerate(zip(self.view_weights,self.view_biases)):
            weight_i.requires_grad = False 
            bias_i.requires_grad = False 
            weight_i = weight_i
            bias_i = bias_i
            h = torch.matmul(weight_i, h) + bias_i.unsqueeze(-1)
            h = F.relu(h)
        
        # output rgb color
        for layer_i, (weight_i, bias_i) in enumerate(zip(self.rgb_weights,self.rgb_biases)):
            weight_i.requires_grad = False 
            bias_i.requires_grad = False 
            weight_i = weight_i
            bias_i = bias_i
            rgb = torch.matmul(weight_i, h) + bias_i.unsqueeze(-1)
        outputs= torch.cat([rgb, alpha], 0)
        outputs = outputs.transpose(1, 0)
        return outputs

    def batchify_rays(self, render_fn, rays_flat):
        """Render rays in smaller minibatches to avoid OOM.
        """
        all_ret = {}
        chunk = self.chunk 
        for i in range(0, rays_flat.shape[0], chunk):
            ret = render_fn(rays_flat[i:i + chunk])
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret

    def batchify(self, fn):
        """Constructs a version of 'fn' that applies to smaller batches.
        """
        def ret(inputs):
            return torch.cat([fn(inputs[i:i + self.netchunk]) for i in range(0, inputs.shape[0], self.netchunk )], 0)
        return ret


    def run_network(self, pts, viewdirs):
        # # [N_rays, N_samples, 3]
        pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])
        # get to know the range of pts sampling
        embedded = self.embed_fn(pts_flat)

        input_dirs = viewdirs[:, None].expand(pts.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])

        embedded_dirs = self.embeddirs_fn(input_dirs_flat)
        
        embedded = torch.cat([embedded, embedded_dirs], -1)
        outputs_flat = self.batchify(self.NeRF_forward)(embedded)
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

    def render_rays_chunk(self, flat_rays):
        """Render rays while moving resulting chunks to cpu to avoid OOM when rendering large images."""
        """flat rays: [B, HW, 11] """
        chunk_size = self.chunk
        B = flat_rays.shape[0]  # num_rays
        results = defaultdict(list)
        for i in range(0, B, chunk_size):
            rendered_ray_chunks = \
                self.render_rays(flat_rays[i:i+chunk_size])

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def get_rays_camera(self, B, depth_type):

        assert depth_type is "z" or depth_type is "euclidean"
        i, j = torch.meshgrid(torch.arange(self.W), torch.arange(self.H))  # pytorch's meshgrid has indexing='ij', we transpose to "xy" moode

        i = i.t().float()
        j = j.t().float()

        size = [B, self.H, self.W]

        i_batch = torch.empty(size)
        j_batch = torch.empty(size)
        i_batch[:, :, :] = i[None, :, :]
        j_batch[:, :, :] = j[None, :, :]

        if self.convention == "opencv":
            x = (i_batch - self.cx) / self.fx
            y = (j_batch - self.cy) / self.fy
            z = torch.ones(size)
        elif self.convention == "opengl":
            x = (i_batch - self.cx) / self.fx
            y = -(j_batch - self.cy) / self.fy
            z = -torch.ones(size)
        else:
            assert False
        dirs = torch.stack((x, y, z), dim=3)  # shape of [B, H, W, 3]

        if depth_type == 'euclidean':
            norm = torch.norm(dirs, dim=3, keepdim=True)
            dirs = dirs * (1. / norm)
        return dirs

    def get_rays_world(self, T_WC, dirs_C):
        R_WC = T_WC[:, :3, :3]  # Bx3x3
        dirs_W = torch.matmul(R_WC[:, None, ...], dirs_C[..., None]).squeeze(-1)
        origins = T_WC[:, :3, -1]  # Bx3
        origins = torch.broadcast_tensors(origins[:, None, :], dirs_W)[0]
        return origins, dirs_W

    def create_rays(self, rays_batch_num, Ts_c2w, depth_type = 'z'):
        """
        convention: 
        "opencv" or "opengl". It defines the coordinates convention of rays from cameras.
        OpenCv defines x,y,z as right, down, forward while OpenGl defines x,y,z as right, up, backward (camera looking towards forward direction still, -z!)
        Note: Use either convention is fine, but the corresponding pose should follow the same convention.

        """

        rays_cam = self.get_rays_camera(rays_batch_num, depth_type=depth_type) # [N, H, W, 3]
        rays_cam = rays_cam.to(Ts_c2w.device)

        dirs_C = rays_cam.view(rays_batch_num, -1, 3)  # [N, HW, 3]
        rays_o, rays_d = self.get_rays_world(Ts_c2w, dirs_C)  # origins: [B, HW, 3], dirs_W: [B, HW, 3]
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True).float()

        near, far = self.near * torch.ones_like(rays_d[..., :1]), self.far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        rays = torch.cat([rays, viewdirs], -1)
        return rays


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

        if self.white_bkgd:
            # print("white background")
            rgb_map = rgb_map + (1.-acc_map[..., None])
        
        return rgb_map, disp_map, acc_map, weights, depth_map



















# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:  # original raw input "x" is also included in the output
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)




def get_embedder(multires, i=0, scalar_factor=1):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x/scalar_factor)
    return embed, embedder_obj.out_dim


def fc_block(in_f, out_f):
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f),
        torch.nn.ReLU(out_f)
    )

class Semantic_NeRF(nn.Module):
    """
    Compared to the NeRF class wich also predicts semantic logits from MLPs, here we make the semantic label only a function of 3D position 
    instead of both positon and viewing directions.
    """
    def __init__(self, enable_semantic, num_semantic_classes, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False,
                 ):
        super(Semantic_NeRF, self).__init__()
        """
                D: number of layers for density (sigma) encoder
                W: number of hidden units in each layer
                input_ch: number of input channels for xyz (3+3*10*2=63 by default)
                in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
                skips: layer index to add skip connection in the Dth layer
        """
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.enable_semantic = enable_semantic

        # build the encoder
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)

        # Another layer is used to 
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            if enable_semantic:
                self.semantic_linear = nn.Sequential(fc_block(W, W // 2), nn.Linear(W // 2, num_semantic_classes))
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, show_endpoint=False):
        """
        Encodes input (xyz+dir) to rgb+sigma+semantics raw output
        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of 3D xyz position and viewing direction
            show_endpoint: end point from geometry network
        """
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            # if using view-dirs, output occupancy alpha as well as features for concatenation
            alpha = self.alpha_linear(h)
            if self.enable_semantic:
                sem_logits = self.semantic_linear(h)
            feature = self.feature_linear(h)

            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
                
            if show_endpoint:
                endpoint_feat = h
            rgb = self.rgb_linear(h)

            if self.enable_semantic:
                outputs = torch.cat([rgb, alpha, sem_logits], -1)
            else:
                outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        if show_endpoint is False:
            return outputs
        else:
            return torch.cat([outputs, endpoint_feat], -1)
