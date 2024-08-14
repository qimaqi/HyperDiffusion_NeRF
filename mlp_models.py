import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as dist
from torch import nn

from embedder import Embedder

# odict_keys(['pts_linears.0.weight', 'pts_linears.0.bias', 'pts_linears.1.weight', 'pts_linears.1.bias', 'pts_linears.2.weight', 'pts_linears.2.bias', 'pts_linears.3.weight', 'pts_linears.3.bias', 'views_linears.0.weight', 'views_linears.0.bias', 'feature_linear.weight', 'feature_linear.bias', 'alpha_linear.weight', 'alpha_linear.bias', 'rgb_linear.weight', 'rgb_linear.bias'])

class Embedder_NeRF:
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

    embedder_obj = Embedder_NeRF(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x/scalar_factor)
    return embed, embedder_obj.out_dim


class MLPNeRF(nn.Module):
    def __init__(
        self,
        D=4,
        W=128,
        input_ch=63,
        input_ch_views=27,
        skips=[],
        use_viewdirs=True,
        **kwargs,
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # build the encoder
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W // 2, 3)
        # self.embed_fn, self.input_ch = get_embedder(10, 0, scalar_factor=10)
        # self.embeddirs_fn, self.input_ch_views  = get_embedder(4,0,1)
    
    def forward(self, x):
        """
        Encodes input (xyz+dir) to rgb+sigma+semantics raw output
        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of 3D xyz position and viewing direction
            show_endpoint: end point from geometry network
        """
        # TODO add embedding inside here
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        # print("input_pts", input_pts.size(), 'input_views', input_views.size()) input_pts torch.Size([2097152, 63]) input_views torch.Size([2097152, 27])
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
    
        h = torch.cat([feature, input_views], -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)
            
        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)
        return outputs

  





class MLP(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_neurons,
        use_tanh=True,
        over_param=False,
        use_bias=True,
    ):
        super().__init__()
        multires = 1
        self.over_param = over_param
        if not over_param:
            self.embedder = Embedder(
                include_input=True,
                input_dims=2,
                max_freq_log2=multires - 1,
                num_freqs=multires,
                log_sampling=True,
                periodic_fns=[torch.sin, torch.cos],
            )
        self.layers = nn.ModuleList([])

        self.layers.append(nn.Linear(in_size, hidden_neurons[0], bias=use_bias))
        for i, _ in enumerate(hidden_neurons[:-1]):
            self.layers.append(
                nn.Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias)
            )
        self.layers.append(nn.Linear(hidden_neurons[-1], out_size, bias=use_bias))
        self.use_tanh = use_tanh

    def forward(self, x):
        if not self.over_param:
            x = self.embedder.embed(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x, None


class MLP3D(nn.Module):
    def __init__(
        self,
        out_size,
        hidden_neurons,
        use_leaky_relu=False,
        use_bias=True,
        multires=10,
        output_type=None,
        move=False,
        **kwargs,
    ):
        super().__init__()
        self.embedder = Embedder(
            include_input=True,
            input_dims=3 if not move else 4,
            max_freq_log2=multires - 1,
            num_freqs=multires,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.layers = nn.ModuleList([])
        self.output_type = output_type
        self.use_leaky_relu = use_leaky_relu
        in_size = self.embedder.out_dim
        self.layers.append(nn.Linear(in_size, hidden_neurons[0], bias=use_bias))
        for i, _ in enumerate(hidden_neurons[:-1]):
            self.layers.append(
                nn.Linear(hidden_neurons[i], hidden_neurons[i + 1], bias=use_bias)
            )
        self.layers.append(nn.Linear(hidden_neurons[-1], out_size, bias=use_bias))

    def forward(self, model_input):
        coords_org = model_input["coords"].clone().detach().requires_grad_(True)
        x = coords_org
        x = self.embedder.embed(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.leaky_relu(x) if self.use_leaky_relu else F.relu(x)
        x = self.layers[-1](x)

        if self.output_type == "occ":
            # x = torch.sigmoid(x)
            pass
        elif self.output_type == "sdf":
            x = torch.tanh(x)
        elif self.output_type == "logits":
            x = x
        else:
            raise f"This self.output_type ({self.output_type}) not implemented"
        x = dist.Bernoulli(logits=x).logits

        return {"model_in": coords_org, "model_out": x}



if __name__ == "__main__":
    
    mlp = MLPNeRF()
    state_dict_path = '/data/work-gcp-europe-west4-a/qimaqi/datasets/omni_weight_dataset_filter/output/apple_001/checkpoints/020000.ckpt'
    state_dict = torch.load(state_dict_path, map_location='cpu')
    state_dict = state_dict['network_coarse_state_dict']
    mlp.load_state_dict(state_dict)
    print(mlp)
