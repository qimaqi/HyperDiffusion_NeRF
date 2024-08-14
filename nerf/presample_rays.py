import os 
import numpy as np
import json 
import glob 
import argparse
import hydra 
from semantic_nerf import NeRF_freeze
import torch 

@hydra.main(config_path="/home/qi_ma/neuips_2024/Weights_Space_Learning/experiments/omni_pose_registration/data", config_name="omniobject_ray", version_base="1.1")
def main(cfg):
    print(cfg)
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument("--from_cam_pose", type=bool, default=True)
    # arg_parser.add_argument("--samples_num", type=int, default=128*128*128)

    # args = arg_parser.parse_args()
    poses_all = []
    scenes_use = 1
  
    print("from_cam_pose")
    # load transforms from camera 
    data_root_path = '/data/work-gcp-europe-west4-a/qimaqi/datasets/omni_weight_dataset_filter/output/'
    transformers_file = os.path.join(data_root_path, 'apple_001', 'transforms.json')
    # glob.glob(os.path.join(data_root_path, '*' ,'transforms.json'))[0]

    with open(transformers_file, 'r') as fp:
        meta = json.load(fp)      
    for frame in meta['frames'][:1]:
        # print("transform_matrix")
        print(frame['transform_matrix'])
        poses_all.append(torch.tensor(frame['transform_matrix']).float())

    poses_all = torch.stack(poses_all)
        # total_rays = 400*400*len(poses_all)
        # print("total_rays", total_rays)

        # params: 
 
    nerf_example = NeRF_freeze(cfg.params)
    rays = nerf_example.create_rays(len(poses_all), poses_all, depth_type = 'z')
    sampled_rays = rays.reshape(-1,11)

    print("sampled_rays", sampled_rays.shape)
        # # save the rays
        # sampled_rays = sampled_rays.numpy()


    np.save('/home/qi_ma/neuips_2024/HyperDiffusion_NeRF/nerf/apple_first_pose_rays.npy', sampled_rays)

        # once we have the ray we can precalculate each scene. the ray feature
        # (rays_num, samples_num, [3Dpoints, view, rgb, density])



if __name__ == "__main__":
    main()