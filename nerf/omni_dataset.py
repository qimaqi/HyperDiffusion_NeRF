import torch
from torch.utils.data import Dataset
from typing import NamedTuple, Tuple, Union, List

import random
import copy
import torch.nn.functional as F
from src.model_zoo.neuralimage import NeuralImageFunction
import numpy as np

import torchvision.transforms as transforms
import torchvision
from PIL import Image
import io
import zipfile 
import hydra
from omegaconf import DictConfig
import copy 
from scipy.optimize import linear_sum_assignment
import time 
import os 
import cv2
import json
from PIL import Image
from time import time 
import h5py 

class Batch(NamedTuple):
    pts_weights: Tuple
    pts_biases: Tuple
    alpha_weights: Tuple
    alpha_biases: Tuple
    feature_weights: Tuple
    feature_biases: Tuple
    view_weights: Tuple
    view_biases: Tuple
    rgb_weights: Tuple
    rgb_biases: Tuple
    label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            pts_weights=tuple(w.to(device) for w in self.pts_weights),
            pts_biases=tuple(w.to(device) for w in self.pts_biases),
            alpha_weights=tuple(w.to(device) for w in self.alpha_weights),
            alpha_biases=tuple(w.to(device) for w in self.alpha_biases),
            feature_weights=tuple(w.to(device) for w in self.feature_weights),
            feature_biases=tuple(w.to(device) for w in self.feature_biases),
            view_weights=tuple(w.to(device) for w in self.view_weights),
            view_biases=tuple(w.to(device) for w in self.view_biases),
            rgb_weights=tuple(w.to(device) for w in self.rgb_weights),
            rgb_biases=tuple(w.to(device) for w in self.rgb_biases),
            label=self.label.to(device),
        )

    def __len__(self):
        return len(self.pts_weights[0])

class PoseBatch(NamedTuple):
    pts_weights: Tuple
    pts_biases: Tuple
    alpha_weights: Tuple
    alpha_biases: Tuple
    feature_weights: Tuple
    feature_biases: Tuple
    view_weights: Tuple
    view_biases: Tuple
    rgb_weights: Tuple
    rgb_biases: Tuple
    images: torch.Tensor
    poses: torch.Tensor
    scene: str


    def to(self, device):
        """move batch to device"""
        return self.__class__(
            pts_weights=tuple(w.to(device) for w in self.pts_weights),
            pts_biases=tuple(w.to(device) for w in self.pts_biases),
            alpha_weights=tuple(w.to(device) for w in self.alpha_weights),
            alpha_biases=tuple(w.to(device) for w in self.alpha_biases),
            feature_weights=tuple(w.to(device) for w in self.feature_weights),
            feature_biases=tuple(w.to(device) for w in self.feature_biases),
            view_weights=tuple(w.to(device) for w in self.view_weights),
            view_biases=tuple(w.to(device) for w in self.view_biases),
            rgb_weights=tuple(w.to(device) for w in self.rgb_weights),
            rgb_biases=tuple(w.to(device) for w in self.rgb_biases),
            images = self.images.to(device),
            poses = self.poses.to(device),
            scene = self.scene,
        )
    

    def __len__(self):
        return len(self.images)


def vector_to_checkpoint(state_dict, weights_key, biases_key, weights, biase):
    checkpoint = copy.deepcopy(state_dict)
    for idx, weight_name in enumerate(weights_key):
        assert weight_name in state_dict.keys()
        weights_shape = checkpoint[weight_name].shape
        checkpoint[weight_name] = weights[idx].reshape(*weights_shape)

    for idx, bias_name in enumerate(biases_key):
        assert bias_name in state_dict.keys()
        bias_shape =  checkpoint[bias_name].shape
        checkpoint[bias_name] = biase[idx].reshape(*bias_shape)
    return checkpoint

def vector_to_checkpoint_batch(state_dict, weights_key, biases_key, weights_batch, biase_batch, batch_idx):
    # weights_batch: 
    # biase_batch:
    # batch_idx:
    weights = []
    biase = []
    weights = [weights_batch[i][batch_idx] for i in range(len(weights_batch))]
    biase = [biase_batch[i][batch_idx] for i in range(len(biase_batch))]


    checkpoint = copy.deepcopy(state_dict)
    for idx, weight_name in enumerate(weights_key):
        assert weight_name in state_dict.keys()
        weights_shape = checkpoint[weight_name].shape
        checkpoint[weight_name] = weights[idx].reshape(*weights_shape)

    for idx, bias_name in enumerate(biases_key):
        assert bias_name in state_dict.keys()
        bias_shape =  checkpoint[bias_name].shape
        checkpoint[bias_name] = biase[idx].reshape(*bias_shape)
    return checkpoint

class OmniDataset(Dataset):
    def __init__(
        self, 
        data_path: str, 
        augmentation: DictConfig,
        params: DictConfig,
        task: str ='train',
        debug: bool = False,
        filter_bad: bool = True,
        device: str = 'cuda:0'
    ):
        self.data_path = data_path
        self.task = task
        self.labels = []
        self.image_files = []
        self.augmentation = augmentation
        self.debug = debug
        self.device = device 

        # this device is used for debugging permutation and augmentation
        print("using data_path", data_path)

        if filter_bad:
            # 1) use the good json 
            good_data_json = os.path.join(data_path, 'good.json')
            with open(good_data_json, 'r') as fp:
                good_data = json.load(fp)
        
        # 1) Load the data path checkpoitns
        self.checkpoints_path = []
        self.labels = []
        self.raw_root_path = []
        sorted_sub_dir = sorted(os.listdir(os.path.join(data_path, 'output')))
        print("sorted_sub_dir", sorted_sub_dir)
        for checkpoints_root_path in sorted_sub_dir:
            if filter_bad:
                if checkpoints_root_path not in good_data.keys():
                    print("skip data ", checkpoints_root_path)
                    continue
            scene_name = checkpoints_root_path[:-13]
            sub_index = scene_name.split('_')[-1]
            label = "_".join(scene_name.split('_')[:-1])

            self.checkpoints_path.append(os.path.join(data_path,'output',checkpoints_root_path, 'checkpoints', '025000.ckpt'))
            self.labels.append(label)
            self.raw_root_path.append(os.path.join('/data/omniobject3d/raw/blender_renders/', label, scene_name, 'render'))


    def __getitem__(self, index):
        state_dict = torch.load(self.checkpoints_path[index], map_location='cpu')
        label = self.labels[index]
        weights = tuple(
            [v for w, v in state_dict.items() if "weight" in w]
        )
        biases = tuple([v for w, v in state_dict.items() if "bias" in w])
        return Batch(weights=weights, biases=biases, label=label)

    def __len__(self):
        return len(self.checkpoints_path)
   


class OmniObjectPoseDataset(OmniDataset):
    def __init__(
        self, 
        data_path: str, 
        blender_path: str,
        h5_path: List[str],
        params: DictConfig,
        task: str ='train',
        debug: bool = False,
        device: str = 'cuda:0',
        samples_num: int = 4,
        inside_eval_num = 4 ,
        outside_eval_ratio = 0.1 ,
        mean: Tuple = (0.485, 0.456, 0.406),
        std: Tuple = (0.229, 0.224, 0.225),
        image_resize: Tuple = (224, 224),
        normalize_pose:Tuple = (0., 0., 2.4),
        normalize_image: bool = True
    ):
        self.data_path = data_path
        self.train_h5_path = os.path.join(data_path, h5_path[0])
        self.eval_h5_path = os.path.join(data_path, h5_path[1])
        self.image_resize = image_resize
        self.task = task
        self.mlp_files = []
        self.labels = []
        self.image_files = []
        self.debug = debug
        self.device = device 
        self.samples_num = samples_num
        self.params = params
        self.H = self.image_resize[0]
        self.W = self.image_resize[1]
        self.inside_eval_num = inside_eval_num
        self.outside_eval_ratio = outside_eval_ratio
        self.normalize_pose = normalize_pose
        self.normalize_image = normalize_image

        assert task in ['train', 'eval_inside', 'eval_outside','train_eval_inside']
        print("using data_path", data_path)

        # 1) Load the data path checkpoitns
        self.checkpoints_path = []
        self.blender_render_path = []  # raw root path to load images 
        self.scenes_name = []
        self.feature_root_path = []
        sorted_sub_dir = sorted(os.listdir(os.path.join(data_path, 'output')))
        train_test_split_path = os.path.join(data_path, 'omniobject_pose_regression_split.json')
        with open(train_test_split_path, 'r') as fp:
            valid_data = json.load(fp)
        # get data from json 
        considered_scenes = []
        for _, scenes_i in valid_data.items():
            eval_count = int(np.ceil(len(scenes_i) * self.outside_eval_ratio)) # use for eval
            # print("eval_count", eval_count)
            if task == 'train' or task == 'eval_inside' or task == 'train_eval_inside':
                to_add_scenes = scenes_i[:-eval_count]
            elif task == 'eval_outside':
                to_add_scenes = scenes_i[-eval_count:]
            considered_scenes = considered_scenes + to_add_scenes
     
        print(f"considered scenes {task}", len(considered_scenes))
        
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)

        for checkpoints_root_path in considered_scenes:
            scene_name = checkpoints_root_path
            label = "_".join(scene_name.split('_')[:-1])
            scene = scene_name
            self.checkpoints_path.append(os.path.join(data_path,'output' ,checkpoints_root_path, 'checkpoints', '020000.ckpt'))
            self.labels.append(label)
            self.scenes_name.append(scene)
            # to save the pose
            self.blender_render_path.append(os.path.join(blender_path, label, scene_name, 'render'))

        # using h5 files to accelerate the data loading 
        self.use_h5 = False
        if len(h5_path) > 0 and os.path.exists(self.train_h5_path) and os.path.exists(self.eval_h5_path):
            if task == 'train' or task == 'eval_inside' or task == 'train_eval_inside':
                self.h5_file = h5py.File(self.train_h5_path, 'r')
                assert self.h5_file['imgs'].shape[0] == len(self.checkpoints_path), f"loaded {self.h5_file['imgs']} scenes from h5 file but {len(self.checkpoints_path)} scenes from checkpoints"
                self.use_h5 = True
                self.n_imgs = self.h5_file['imgs'].shape[1]
                print("########## self.n_imgs ##########", self.n_imgs)
            elif task == 'eval_outside':
                self.h5_file = h5py.File(self.eval_h5_path, 'r')
                assert self.h5_file['imgs'].shape[0] == len(self.checkpoints_path), f"loaded {self.h5_file['imgs']} scenes from h5 file but {len(self.checkpoints_path)} scenes from checkpoints"
                self.use_h5 = True
            strings_data = self.h5_file['scene_name'][:]
            # Convert numpy array back to list of strings
            self.h5_strings_list  = strings_data.astype(str).tolist()
            print("##### using h5 files #######", h5_path)

        if not self.use_h5:
            print("##### using direcly load files #######")
            self.use_h5 = False
           
    def __getitem__(self, index, print_t=False):
        t0 = time()
        state_dict = torch.load(self.checkpoints_path[index], map_location='cpu')['network_coarse_state_dict']
        t1 = time()
        label = self.labels[index]
        scene = self.scenes_name[index]
        blender_path = self.blender_render_path[index]

        # current_time = int(time()+index)
        # np.random.seed(current_time)

        # sample image_samples images and correspond poses: 
        if self.use_h5:
            # sample directly here 
            if self.task == 'train':
                sampled_indices = np.random.choice(self.n_imgs - self.inside_eval_num , size=self.samples_num, replace=False)
                sampled_indices = np.sort(sampled_indices) # h5 files can only load in increasing order
                image_tensors = torch.tensor(self.h5_file['imgs'][index,sampled_indices])
                pose_tensors = torch.tensor(self.h5_file['c2ws'][index,sampled_indices])
            elif self.task == 'eval_inside':
                image_tensors = torch.tensor(self.h5_file['imgs'][index, -self.inside_eval_num:])
                pose_tensors = torch.tensor(self.h5_file['c2ws'][index,-self.inside_eval_num:])
            elif self.task == 'eval_outside':
                image_tensors = torch.tensor(self.h5_file['imgs'][index])
                pose_tensors = torch.tensor(self.h5_file['c2ws'][index])
            elif self.task == 'train_eval_inside':
                sampled_indices = np.random.choice(self.n_imgs - self.inside_eval_num , size=self.samples_num, replace=False)
                sampled_indices = np.sort(sampled_indices) # h5 files can only load in increasing order
                eval_sampled_indices = self.n_imgs - np.random.choice(self.inside_eval_num , size= 1) - 1
                total_indices = np.concatenate([sampled_indices, eval_sampled_indices])

                image_tensors = torch.tensor(self.h5_file['imgs'][index,total_indices])
                pose_tensors = torch.tensor(self.h5_file['c2ws'][index,total_indices]) 
                        
        else: # loading 
            with open(os.path.join(blender_path, 'transforms.json'), 'r') as fp:
                meta = json.load(fp)
            if self.task == 'train':
                image_tensors = torch.zeros((self.samples_num, 3, self.H, self.W))
                pose_tensors = torch.zeros((self.samples_num, 4, 4))
                sampled_indices = np.random.choice(len(meta['frames']) - self.inside_eval_num, size=self.samples_num, replace=False)
            elif self.task == 'eval_inside':
                image_tensors = torch.zeros((self.inside_eval_num, 3, self.H, self.W))
                pose_tensors = torch.zeros((self.inside_eval_num, 4, 4))
                sampled_indices = np.arange(len(meta['frames']) - self.inside_eval_num, len(meta['frames']))
            else:
                image_tensors = torch.zeros((len(meta['frames']), 3, self.H, self.W))
                pose_tensors = torch.zeros((len(meta['frames']), 4, 4))
                sampled_indices = np.arange(len(meta['frames']))
            
            j = 0
            for i, frame in enumerate(meta['frames']):
                if i in sampled_indices:
                    fname = os.path.join(blender_path, 'images')
                    image = Image.open(os.path.join(fname, frame['file_path'] + '.png')) # PIL open is faster than cv2
                    image = np.array(image) / 255.
                    image = image[...,:3]*image[...,-1:] + (1.-image[...,-1:])  # to RGB channel
                    image = torch.tensor(image).permute(2,0,1).float()
                    image_tensors[j] = F.interpolate(image.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=False).squeeze(0)
                    pose_tensors[j] = torch.tensor((np.array(frame['transform_matrix']))).float()
                    j+=1

        # h5 data do not yet noramlize or recenter the pose 
        if self.normalize_image:
            image_tensors = self.normalize(image_tensors)
        if self.normalize_pose is not None:
            pose_tensors[:,:3,3] = pose_tensors[:,:3,3] - torch.tensor(self.normalize_pose).reshape(-1,3)  # to origin
            pose_tensors[:,:3,3] = pose_tensors[:,:3,3] / 5.  # map to -1 to 1 for translation
        t2 = time()
        pts_weights = tuple(
            [v for w, v in state_dict.items() if "pts_linears" in w and "weight" in w]
        )
        pts_biases = tuple([v for w, v in state_dict.items() if "pts_linears" in w and "bias" in w])

        alpha_weights = tuple(
            [v for w, v in state_dict.items() if "alpha_linear" in w and "weight" in w]
        )
        alpha_biases = tuple([v for w, v in state_dict.items() if "alpha_linear" in w and "bias" in w])

        feature_weights = tuple(
            [v for w, v in state_dict.items() if "feature_linear" in w and "weight" in w]
        )
        feature_biases = tuple([v for w, v in state_dict.items() if "feature_linear" in w and "bias" in w])
        view_weights = tuple(
            [v for w, v in state_dict.items() if "views_linears" in w and "weight" in w]
        )
        view_biases = tuple([v for w, v in state_dict.items() if "views_linears" in w and "bias" in w])
        rgb_weights = tuple(
            [v for w, v in state_dict.items() if "rgb_linear" in w and "weight" in w]
        )
        rgb_biases = tuple([v for w, v in state_dict.items() if "rgb_linear" in w and "bias" in w])

        t3 = time()
        if print_t:
            print("load ckpt time", t1-t0)
            print("image pose time", t2-t1)
            print("weight to tuple", t3-t2)
            print("total time", t3-t0)

        return PoseBatch(
            pts_weights=pts_weights, 
            pts_biases=pts_biases, 
            alpha_weights=alpha_weights,
            alpha_biases=alpha_biases,
            feature_weights=feature_weights,
            feature_biases=feature_biases,
            view_weights=view_weights,
            view_biases=view_biases,
            rgb_weights=rgb_weights,
            rgb_biases=rgb_biases,
            images=image_tensors,
            poses=pose_tensors,
            scene = scene, 
        )


# class OmniImagePoseDataset(OmniDataset):
#     def __init__(
#         self, 
#         data_path: str, 
#         augmentation: DictConfig,
#         params: DictConfig,
#         task: str ='train',
#         debug: bool = False,
#         device: str = 'cuda:0',
#         samples_num: int = 4,
#         eval_num: int = 4,
#         mean: Tuple = (0.485, 0.456, 0.406),
#         std: Tuple = (0.229, 0.224, 0.225)
#     ):
#         self.data_path = data_path
#         self.task = task
#         self.mlp_files = []
#         self.labels = []
#         self.image_files = []
#         self.augmentation = augmentation
#         self.debug = debug
#         self.device = device 
#         self.samples_num = samples_num
#         self.params = params
#         self.H = self.params.image_size[0]
#         self.W = self.params.image_size[1]
#         self.eval_num = eval_num

#         # this device is used for debugging permutation and augmentation
#         print("using data_path", data_path)

#         # 1) Load the data path checkpoitns
#         self.checkpoints_path = []
#         self.labels = []
#         self.raw_root_path = []
#         self.scenes = []
#         self.feature_root_path = []
#         sorted_sub_dir = sorted(os.listdir(os.path.join(data_path, 'output')))
#         train_test_split_path = os.path.join(data_path, 'omniobject_pose_regression_split.json')
#         with open(train_test_split_path, 'r') as fp:
#             valid_data = json.load(fp)

#         # get data from json 
#         considered_scenes = []
#         for category_i, scenes_i in valid_data.items():
#             considered_scenes += scenes_i
            
#         self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)
#         transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((self.H, self.W)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=mean, std=std)
#         ])
        
#         # torchvision.transforms.Normalize(mean=mean, std=std)
#         for checkpoints_root_path in considered_scenes:
#             scene_name = checkpoints_root_path
#             sub_index = scene_name.split('_')[-1]
#             label = "_".join(scene_name.split('_')[:-1])
#             scene = scene_name

#             self.checkpoints_path.append(os.path.join(data_path,'output' ,checkpoints_root_path, 'checkpoints', '020000.ckpt'))
#             self.labels.append(label)
#             self.scenes.append(scene)
#             # to save the pose
#             self.raw_root_path.append(os.path.join('/data/omniobject3d/raw/blender_renders/', label, scene_name, 'render'))

#     def __getitem__(self, index, print_t=False):
#         # one item is one scene and all images possible
#         t0 = time()
#         state_dict = torch.load(self.checkpoints_path[index], map_location='cpu')['network_coarse_state_dict']
#         t1 = time()
#         label = self.labels[index]
#         scene = self.scenes[index]
#         root_path = self.raw_root_path[index]
#         feature_path = self.feature_root_path[index]

#         # sample image_samples images and correspond poses: 
#         with open(os.path.join(root_path, 'transforms.json'), 'r') as fp:
#             meta = json.load(fp)
        
#         if self.task == 'train':
#             image_tensors = torch.zeros((self.samples_num, 3, self.H, self.W))
#             pose_tensors = torch.zeros((self.samples_num, 4, 4))
#             sampled_indices = np.random.choice(len(meta['frames']) - self.eval_num, size=self.samples_num, replace=False)
#         else:
#             image_tensors = torch.zeros((self.eval_num, 3, self.H, self.W))
#             pose_tensors = torch.zeros((self.eval_num, 4, 4))
#             sampled_indices = np.arange(len(meta['frames']) - self.eval_num, len(meta['frames']))
#         j = 0
#         for i, frame in enumerate(meta['frames']):
#             if i in sampled_indices:
#                 fname = os.path.join(root_path, 'images')
#                 image = Image.open(os.path.join(fname, frame['file_path'] + '.png'))
#                 image = image.resize((self.H, self.W), Image.ANTIALIAS)
#                 image = np.array(image) / 255.
#                 image = image[...,:3]*image[...,-1:] + (1.-image[...,-1:])  # to RGB channel
#                 image_tensors[j] = (torch.tensor(image).permute(2,0,1).float())
#                 pose_tensors[j] = torch.tensor((np.array(frame['transform_matrix']))).float()
#                 j+=1
#         image_tensors = F.interpolate(image_tensors, size=(self.H, self.W), mode='bilinear', align_corners=False)
#         image_tensors = self.normalize(image_tensors)  # noramlize for later vit model
#         t2 = time()
#         # recenter the pose
#         pose_tensors[:,:3,3] = pose_tensors[:,:3,3] - pose_tensors[:,:3,3].mean(dim=0) # to origin
#         pose_tensors[:,:3,3] = pose_tensors[:,:3,3] / 5.  # map to -1 to 1 for translation
#         # after recenter
#         # camera translation x  range -3.9990973749011753 3.9366268835216762
#         # camera translation y  range -3.653199181780219 4.054219975247979
#         # camera translation z  range -2.018542287349701 1.5043730759620666

#         pts_weights = tuple(
#             [v for w, v in state_dict.items() if "pts_linears" in w and "weight" in w]
#         )
#         pts_biases = tuple([v for w, v in state_dict.items() if "pts_linears" in w and "bias" in w])

#         alpha_weights = tuple(
#             [v for w, v in state_dict.items() if "alpha_linear" in w and "weight" in w]
#         )
#         alpha_biases = tuple([v for w, v in state_dict.items() if "alpha_linear" in w and "bias" in w])

#         feature_weights = tuple(
#             [v for w, v in state_dict.items() if "feature_linear" in w and "weight" in w]
#         )
#         feature_biases = tuple([v for w, v in state_dict.items() if "feature_linear" in w and "bias" in w])
#         view_weights = tuple(
#             [v for w, v in state_dict.items() if "views_linears" in w and "weight" in w]
#         )
#         view_biases = tuple([v for w, v in state_dict.items() if "views_linears" in w and "bias" in w])
#         rgb_weights = tuple(
#             [v for w, v in state_dict.items() if "rgb_linear" in w and "weight" in w]
#         )
#         rgb_biases = tuple([v for w, v in state_dict.items() if "rgb_linear" in w and "bias" in w])

#         t3 = time()
#         if print_t:
#             print("load ckpt time", t1-t0)
#             print("image pose time", t2-t1)
#             print("weight to tuple", t3-t2)
#         return PoseBatch(
#             pts_weights=pts_weights, 
#             pts_biases=pts_biases, 
#             alpha_weights=alpha_weights,
#             alpha_biases=alpha_biases,
#             feature_weights=feature_weights,
#             feature_biases=feature_biases,
#             view_weights=view_weights,
#             view_biases=view_biases,
#             rgb_weights=rgb_weights,
#             rgb_biases=rgb_biases,
#             images=image_tensors,
#             poses=pose_tensors,
#             scene = scene, 
#         )



class OmniRawPoseDataset(OmniDataset):
    def __init__(
        self, 
        data_path: str, 
        augmentation: DictConfig,
        params: DictConfig,
        task: str ='train',
        debug: bool = False,
        device: str = 'cuda:0',
        samples_num: int = 4,
        eval_num: int = 4,
        mean: Tuple = (0.485, 0.456, 0.406),
        std: Tuple = (0.229, 0.224, 0.225)
    ):
        self.data_path = data_path
        self.task = task
        self.mlp_files = []
        self.labels = []
        self.image_files = []
        self.augmentation = augmentation
        self.debug = debug
        self.device = device 
        self.samples_num = samples_num
        self.params = params
        self.H = self.params.image_size[0]
        self.W = self.params.image_size[1]
        self.eval_num = eval_num

        # this device is used for debugging permutation and augmentation
        print("using data_path", data_path)

        # 1) Load the data path checkpoitns
        self.checkpoints_path = []
        self.labels = []
        self.raw_root_path = []
        self.scenes = []
        self.feature_root_path = []
            
        self.normalize = torchvision.transforms.Normalize(mean=mean, std=std)
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.H, self.W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        train_test_split_path = os.path.join(data_path, 'omniobject_pose_regression_split.json')
        with open(train_test_split_path, 'r') as fp:
            valid_data = json.load(fp)

        # get data from json 
        considered_scenes = []
        for category_i, scenes_i in valid_data.items():
            considered_scenes += scenes_i


        for checkpoints_root_path in considered_scenes:
            scene_name = checkpoints_root_path
            label = "_".join(scene_name.split('_')[:-1])
            scene = scene_name

            self.checkpoints_path.append(os.path.join(data_path,'output' ,checkpoints_root_path, 'checkpoints', '020000.ckpt'))
            self.labels.append(label)
            self.scenes.append(scene)
            # to save the pose
            self.raw_root_path.append(os.path.join('/data/omniobject3d/raw/blender_renders/', label, scene_name, 'render'))

           
    def __getitem__(self, index, print_t=False):
        # one item is one scene and all images possible
        t0 = time()
        state_dict = torch.load(self.checkpoints_path[index], map_location='cpu')['network_coarse_state_dict']
        t1 = time()
        scene = self.scenes[index]
        root_path = self.raw_root_path[index]

        # sample image_samples images and correspond poses: 
        with open(os.path.join(root_path, 'transforms.json'), 'r') as fp:
            meta = json.load(fp)
        
        # if len(meta['frames']) == 0:
        #     print("zero scene", scene)

        image_tensors = []
        pose_tensors = []

        j = 0
        for i, frame in enumerate(meta['frames']):
            fname = os.path.join(root_path, 'images')
            image = Image.open(os.path.join(fname, frame['file_path'] + '.png'))
            image = np.array(image) / 255.
            image = image[...,:3]*image[...,-1:] + (1.-image[...,-1:])  # to RGB channel
            image_tensors.append(torch.tensor(image).permute(2,0,1).float())
            pose_tensors.append(torch.tensor(np.array(frame['transform_matrix'])).float() )
            j+=1
        # image_tensors = self.normalize(image_tensors)
        image_tensors = torch.stack(image_tensors)
        image_tensors = F.interpolate(image_tensors, size=(self.H, self.W), mode='bilinear', align_corners=False)
        pose_tensors = torch.stack(pose_tensors)

        t2 = time()


        pts_weights = tuple(
            [v for w, v in state_dict.items() if "pts_linears" in w and "weight" in w]
        )
        pts_biases = tuple([v for w, v in state_dict.items() if "pts_linears" in w and "bias" in w])

        alpha_weights = tuple(
            [v for w, v in state_dict.items() if "alpha_linear" in w and "weight" in w]
        )
        alpha_biases = tuple([v for w, v in state_dict.items() if "alpha_linear" in w and "bias" in w])

        feature_weights = tuple(
            [v for w, v in state_dict.items() if "feature_linear" in w and "weight" in w]
        )
        feature_biases = tuple([v for w, v in state_dict.items() if "feature_linear" in w and "bias" in w])
        view_weights = tuple(
            [v for w, v in state_dict.items() if "views_linears" in w and "weight" in w]
        )
        view_biases = tuple([v for w, v in state_dict.items() if "views_linears" in w and "bias" in w])
        rgb_weights = tuple(
            [v for w, v in state_dict.items() if "rgb_linear" in w and "weight" in w]
        )
        rgb_biases = tuple([v for w, v in state_dict.items() if "rgb_linear" in w and "bias" in w])

        t3 = time()
        if print_t:
            print("load ckpt time", t1-t0)
            print("image pose time", t2-t1)
            print("weight to tuple", t3-t2)
        return PoseBatch(
            pts_weights=pts_weights, 
            pts_biases=pts_biases, 
            alpha_weights=alpha_weights,
            alpha_biases=alpha_biases,
            feature_weights=feature_weights,
            feature_biases=feature_biases,
            view_weights=view_weights,
            view_biases=view_biases,
            rgb_weights=rgb_weights,
            rgb_biases=rgb_biases,
            images=image_tensors,
            poses=pose_tensors,
            scene = scene, 
        )

