import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import NamedTuple, Tuple, Union, List, Callable
from collections import defaultdict

from evaluation_metrics_3d import compute_all_metrics, compute_all_metrics_4d
from hd_utils import (Config, calculate_fid_3d, generate_mlp_from_weights,
                      render_mesh, render_meshes)
from siren import sdf_meshing
from siren.dataio import anime_read
from siren.experiment_scripts.test_sdf import SDFDecoder
from nerf.test_voxel import NeRFDecoder , create_voxel, create_renders
from omegaconf import DictConfig

vox_path = '/home/qi_ma/neuips_2024/HyperDiffusion_NeRF/orig_voxels/run_dummy-np73mf6d/prev_sample_x_0s.pth'
vox = torch.load(vox_path)
print(vox.shape)
# take one voxels
vox_example = vox[-1]
weights = vox_example
mlp_kwargs =  {'model_type': 'mlp_nerf', 'use_leaky_relu': False, 'move': False}
#To Dictconfig
mlp_kwargs = DictConfig(mlp_kwargs)

# mlp = generate_mlp_from_weights(weights, mlp_kwargs)
# checkpoint_path = '/data/work-gcp-europe-west4-a/qimaqi/datasets/omni_weight_dataset_filter/output/apple_001/checkpoints/020000.ckpt'
# nerf_decoder = NeRFDecoder(checkpoint_path=checkpoint_path)

mlp = generate_mlp_from_weights(weights, mlp_kwargs)
nerf_decoder = NeRFDecoder(checkpoint_path=None)
nerf_decoder.model = mlp.cuda().eval()

# with torch.no_grad():
#     effective_file_name = None
#     rgb, density =  create_voxel(nerf_decoder, effective_file_name, N=128) 
# rgb = rgb.reshape(128, 128, 128, 3)
# density = density.reshape(128, 128, 128)
# print(rgb.shape)
# print(density.shape)

# # save to .nii.gz file
# rgb = rgb.cpu().numpy()
# density = density.cpu().numpy()

# import nibabel as nib
# rgb_nii = nib.Nifti1Image(rgb, np.eye(4))
# density_nii = nib.Nifti1Image(density, np.eye(4))
# nib.save(rgb_nii, 'rgb.nii.gz')
# nib.save(density_nii, 'density.nii.gz')


with torch.no_grad():
    effective_file_name = None
    results =  create_renders(nerf_decoder) 

rgb_results = results['rgb_coarse']
print(rgb_results.shape)
rgb_results = rgb_results.reshape(400,400,3)
rgb_results = rgb_results.cpu().numpy()
print(rgb_results.shape)
# save to .png file
from PIL import Image
image = Image.fromarray((rgb_results * 255).astype(np.uint8))
image.save('rgb.png')


# rgb = rgb.reshape(128, 128, 128, 3)
# density = density.reshape(128, 128, 128)
# print(rgb.shape)
# print(density.shape)

# # save to .nii.gz file
# rgb = rgb.cpu().numpy()
# density = density.cpu().numpy()

# import nibabel as nib
# rgb_nii = nib.Nifti1Image(rgb, np.eye(4))
# density_nii = nib.Nifti1Image(density, np.eye(4))
# nib.save(rgb_nii, 'rgb.nii.gz')
# nib.save(density_nii, 'density.nii.gz')
