import json
import math
import os
from argparse import ArgumentParser
import numpy as np
from decord import VideoReader
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from utils.video_augmentation import DeleteFlowKeypoints,ToFloatTensor,Compose
from dataset.videoLoader import get_selected_indexs,pad_index
import cv2
import torchvision
from utils.video_augmentation import *
from dataset.utils import crop_hand
import glob

class HRCA_Dataset(Dataset):
    def __init__(self, base_url, split, dataset_cfg, train_labels=None, **kwargs):
        if train_labels is None:
            if dataset_cfg['dataset_name'] == "VN_SIGN":
                print("Label: ",
                      os.path.join(base_url, f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                self.train_labels = pd.read_csv(
                    os.path.join(base_url, f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"),
                    sep=',')
                # if split == 'test':
                #      self.train_labels = pd.concat([self.train_labels] * 5, ignore_index=True)
            elif dataset_cfg['dataset_name'] == "AUTSL":
                print("Label: ",
                      os.path.join(base_url, f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                self.train_labels = pd.read_csv(
                    os.path.join(base_url, f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"),
                    sep=',')

        else:
            print("Use labels from K-Fold")
            self.train_labels = train_labels

        print(split, len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.transform = self.build_transform(split)

    def build_transform(self, split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE']),
                MultiScaleCrop(
                    (self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']),
                    scales),
                RandomVerticalFlip(),
                # RandomRotate(p=0.3),
                # RandomShear(0.3, 0.3, p=0.3),
                # Salt(p=0.3),
                # GaussianBlur(sigma=1, p=0.3),
                ColorJitter(0.5, 0.5, 0.5, p=0.3),
                ToFloatTensor(), PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],
                          self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        else:
            print("Build test/val transform")
            transform = Compose(
                Resize(self.data_cfg['vid_transform']['IMAGE_SIZE']),
                # CenterCrop(self.data_cfg['vid_transform']['IMAGE_SIZE']),
                ToFloatTensor(),
                PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],
                          self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        return transform

    def count_frames(self, video_path):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file: {video_path}")

            # Đọc kích thước của video
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            cap.release()
            return total_frames, width, height

        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            return None, None, None

    def read_videos(self, name):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive', 'pad', 'central', 'pad'])

        path = f'{self.base_url}/crop_videos/{name}'
        vlen, width, height = self.count_frames(path)

        selected_index, pad = get_selected_indexs(vlen - 5, self.data_cfg['num_output_frames'], self.is_train,
                                                  index_setting, temporal_stride=self.data_cfg['temporal_stride'])
        
        if pad is not None:
            selected_index = pad_index(selected_index, pad).tolist()
        vr = VideoReader(path, width=320, height=256)
        frames = vr.get_batch(selected_index).asnumpy()

        clip = []
        heatmap_clip = []
        missing_wrists_left = []
        missing_wrists_right = []

        for frame, frame_index in zip(frames, selected_index):
            if self.data_cfg['crop_two_hand']:
                n = 0
                found = False
                while frame_index - n >= 0:
                    current_index = frame_index - n
                    kp_path = os.path.join(
                        self.base_url, 'poses', name.replace(".mp4", ""),
                        f"{name.replace('.mp4', '')}_{current_index:06d}_keypoints.json"
                    )
                    if os.path.exists(kp_path):
                        # load keypoints
                        with open(kp_path, 'r') as keypoints_file:
                            value = json.load(keypoints_file)
                            crop_keypoints = np.array(value['pose_threshold_02'])  # 26,3
                            x = 320 * crop_keypoints[:, 0] / width
                            y = 256 * crop_keypoints[:, 1] / height
                            crop_keypoints = np.stack((x, y), axis=0)
                        found = True
                        break
                    else:
                        n += 1
                if not found:
                    raise FileNotFoundError(
                        f"Could not find a keypoints file for frame_index {frame_index} or any prior frames."
                    )

            crops = None
            if self.data_cfg['crop_two_hand']:
                crops, missing_wrists_left, missing_wrists_right = crop_hand(frame, crop_keypoints,
                                                                             self.data_cfg['WRIST_DELTA'],
                                                                             self.data_cfg['SHOULDER_DIST_EPSILON'],
                                                                             self.transform, len(clip),
                                                                             missing_wrists_left, missing_wrists_right)
            else:
                crops = self.transform(frame)
            clip.append(crops)

            frame_index_heatmap = frame_index  # Sử dụng frame_index hiện tại
            heatmap = None
            if frame_index_heatmap > 0:
                full_path = os.path.join(self.base_url, 'heatmap', name.replace(".mp4", ""),
                                         'heatmap_{:05d}.jpg'.format(frame_index_heatmap))
                while not os.path.isfile(full_path):
                    frame_index_heatmap -= 1
                    if frame_index_heatmap <= 0:
                        break
                    full_path = os.path.join(self.base_url, 'heatmap', name.replace(".mp4", ""),
                                             'heatmap_{:05d}.jpg'.format(frame_index_heatmap))

                if os.path.isfile(full_path):
                    heatmap = Image.open(full_path).convert('RGB')
                    heatmap = np.array(heatmap)
                    heatmap = self.transform(heatmap)
                else:
                    heatmap = torch.zeros((3, self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']))
            else:
                heatmap = torch.zeros((3, self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']))
            heatmap_clip.append(heatmap)

        heatmap = torch.stack(heatmap_clip, dim=0)
        clip = torch.stack(clip, dim=0)
        return heatmap, clip

    def __getitem__(self, idx):
        self.transform.randomize_parameters()

        data = self.train_labels.iloc[idx].values
        name, label = data[0], data[1]

        heatmap, clip= self.read_videos(name)

        return heatmap, clip, torch.tensor(label)

    def __len__(self):
        return len(self.train_labels)


class HRMSSCA_Dataset(Dataset):
    def __init__(self, base_url, split, dataset_cfg, **kwargs):

        if dataset_cfg is None:
            self.train_labels = pd.read_csv(os.path.join(base_url, f'{split}.csv'), sep=',')
        else:
            if dataset_cfg['dataset_name'] == "VN_SIGN":
                print("Label: ",
                      os.path.join(base_url, f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                self.train_labels = pd.read_csv(
                    os.path.join(base_url, f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"),
                    sep=',')
                # if split == 'test':
                #      self.train_labels = pd.concat([self.train_labels] * 5, ignore_index=True)
            elif dataset_cfg['dataset_name'] == "AUTSL":
                self.train_labels = pd.read_csv(os.path.join(base_url, f'{split}.csv'), sep=',')
                self.labels = pd.read_csv(os.path.join(base_url, f'SignList_ClassId_TR_EN.csv'), sep=',')

        print(split, len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.transform = self.build_transform(split)

    def build_transform(self, split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7),
                MultiScaleCrop(
                    (self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']),
                    scales),
                RandomHorizontalFlip(),
                RandomRotate(p=0.3),
                RandomShear(0.3, 0.3, p=0.3),
                Salt(p=0.3),
                GaussianBlur(sigma=1, p=0.3),
                ColorJitter(0.5, 0.5, 0.5, p=0.3),
                ToFloatTensor(), PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],
                          self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        else:
            print("Build test/val transform")
            transform = Compose(
                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7),
                CenterCrop(self.data_cfg['vid_transform']['IMAGE_SIZE']),
                ToFloatTensor(),
                PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],
                          self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        return transform

    def count_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        # Đọc kích thước của video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        n_poses = len(glob.glob(video_path.replace("videos", "poses").replace('.mp4', '/*')))
        total_frames = min(total_frames, n_poses)
        return total_frames, width, height

    def transform_handflow(self, handflow):
        # Convert to a PyTorch tensor and transpose to get [C, V]
        handflow_tensor = torch.tensor(handflow, dtype=torch.float32).transpose(0, 1)
        return handflow_tensor

    def read_one_view(self, name, selected_index, width, height):

        clip = []
        heatmap_clip = []
        missing_wrists_left = []
        missing_wrists_right = []

        if self.data_cfg['dataset_name'] == "VN_SIGN":
            path = f'{self.base_url}/crop_videos/{name}'
        vr = VideoReader(path, width=320, height=256)
        frames = vr.get_batch(selected_index).asnumpy()
        for frame, frame_index in zip(frames, selected_index):
            if self.data_cfg['crop_two_hand']:
                kp_path = os.path.join(self.base_url, 'poses', name.replace(".mp4", ""),
                                       name.replace(".mp4", "") + '_{:06d}_'.format(frame_index) + 'keypoints.json')
                # load keypoints
                with open(kp_path, 'r') as keypoints_file:
                    value = json.loads(keypoints_file.read())

                    keypoints = np.array(value['pose_threshold_02'])  # 26,3
                    x = 320 * keypoints[:, 0] / width
                    y = 256 * keypoints[:, 1] / height

                keypoints = np.stack((x, y), axis=0)

            crops = None
            if self.data_cfg['crop_two_hand']:
                crops, missing_wrists_left, missing_wrists_right = crop_hand(frame, keypoints,
                                                                             self.data_cfg['WRIST_DELTA'],
                                                                             self.data_cfg['SHOULDER_DIST_EPSILON'],
                                                                             self.transform, len(clip),
                                                                             missing_wrists_left, missing_wrists_right)
            else:
                crops = self.transform(frame)
            clip.append(crops)

            frame_index_heatmap = frame_index  # Sử dụng frame_index hiện tại
            heatmap = None
            if frame_index_heatmap > 0:
                full_path = os.path.join(self.base_url, 'heatmap', name.replace(".mp4", ""),
                                         'heatmap_{:05d}.jpg'.format(frame_index_heatmap))
                while not os.path.isfile(full_path):
                    frame_index_heatmap -= 1
                    if frame_index_heatmap <= 0:
                        break
                    full_path = os.path.join(self.base_url, 'heatmap', name.replace(".mp4", ""),
                                             'heatmap_{:05d}.jpg'.format(frame_index_heatmap))

                if os.path.isfile(full_path):
                    heatmap = Image.open(full_path).convert('RGB')
                    heatmap = np.array(heatmap)
                    heatmap = self.transform(heatmap)
                else:
                    heatmap = torch.zeros((3, self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']))
            else:
                heatmap = torch.zeros((3, self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']))
            heatmap_clip.append(heatmap)

        heatmap = torch.stack(heatmap_clip, dim=0)
        clip = torch.stack(clip, dim=0)
        return heatmap, clip

    def read_videos(self, center, left, right):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive', 'pad', 'central', 'pad'])
        #
        vlen1, c_width, c_height = self.count_frames(os.path.join(self.base_url, 'videos', center))
        vlen2, l_width, l_height = self.count_frames(os.path.join(self.base_url, 'videos', left))
        vlen3, r_width, r_height = self.count_frames(os.path.join(self.base_url, 'videos', right))

        min_vlen = min(vlen1, min(vlen2, vlen3))
        max_vlen = max(vlen1, max(vlen2, vlen3))
        if max_vlen - min_vlen < 10:
            selected_index, pad = get_selected_indexs(min_vlen - 3, self.data_cfg['num_output_frames'], self.is_train,
                                                      index_setting, temporal_stride=self.data_cfg['temporal_stride'])

            if pad is not None:
                selected_index = pad_index(selected_index, pad).tolist()

            hmap_center, rgb_center = self.read_one_view(center, selected_index, width=c_width,height=c_height)

            hmap_left, rgb_left = self.read_one_view(left, selected_index, width=l_width, height=l_height)

            hmap_right, rgb_right = self.read_one_view(right, selected_index, width=r_width, height=r_height)
        else:
            selected_index, pad = get_selected_indexs(vlen1 - 3, self.data_cfg['num_output_frames'], self.is_train,
                                                      index_setting, temporal_stride=self.data_cfg['temporal_stride'])

            if pad is not None:
                selected_index = pad_index(selected_index, pad).tolist()

            hmap_center, rgb_center = self.read_one_view(center, selected_index, width=c_width,height=c_height)

            selected_index, pad = get_selected_indexs(vlen2 - 3, self.data_cfg['num_output_frames'], self.is_train,
                                                      index_setting, temporal_stride=self.data_cfg['temporal_stride'])

            if pad is not None:
                selected_index = pad_index(selected_index, pad).tolist()

            hmap_left, rgb_left = self.read_one_view(left, selected_index, width=l_width, height=l_height)

            selected_index, pad = get_selected_indexs(vlen3 - 3, self.data_cfg['num_output_frames'], self.is_train,
                                                      index_setting, temporal_stride=self.data_cfg['temporal_stride'])

            if pad is not None:
                selected_index = pad_index(selected_index, pad).tolist()

            hmap_right, rgb_right = self.read_one_view(right, selected_index, width=r_width, height=r_height)

        return rgb_left, rgb_center, rgb_right, hmap_left, hmap_center, hmap_right

    # def __getitem__(self, idx):
    #     self.transform.randomize_parameters()

    #     center,left,right,label = self.train_labels.iloc[idx].values

    #     center_video,center_pf,center_kp,left_video,left_pf,left_kp,right_video,right_pf,right_kp = self.read_videos(center,left,right)

    #     return center_video,center_pf,center_kp,left_video,left_pf,left_kp,right_video,right_pf,right_kp,torch.tensor(label)

    def __getitem__(self, idx):
        self.transform.randomize_parameters()

        center, left, right, label = self.train_labels.iloc[idx].values
        rgb_left, rgb_center, rgb_right, hmap_left, hmap_center, hmap_right = self.read_videos(center, left, right)

        random_number = np.random.randint(1, 1001)

        if not (random_number / 1000 > self.data_cfg.get('center_missing_rates', 0)):
            rgb_center[:] = 0
            hmap_center[:] = 0

        if not (random_number / 1000 > self.data_cfg.get('left_missing_rates', 0)):
            rgb_left[:] = 0
            hmap_left[:] = 0

        if not (random_number / 1000 > self.data_cfg.get('right_missing_rates', 0)):
            rgb_right[:] = 0
            hmap_right[:] = 0

        return rgb_left, rgb_center, rgb_right, hmap_left, hmap_center, hmap_right, torch.tensor(label)

    def __len__(self):
        return len(self.train_labels)

class HRMSSCA_debug_Dataset(Dataset):
    def __init__(self, base_url, split, dataset_cfg, **kwargs):

        if dataset_cfg is None:
            self.train_labels = pd.read_csv(os.path.join(base_url, f'{split}.csv'), sep=',')
        else:
            if dataset_cfg['dataset_name'] == "VN_SIGN":
                print("Label: ",
                      os.path.join(base_url, f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                self.train_labels = pd.read_csv(
                    os.path.join(base_url, f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"),
                    sep=',')
                # if split == 'test':
                #      self.train_labels = pd.concat([self.train_labels] * 5, ignore_index=True)
            elif dataset_cfg['dataset_name'] == "AUTSL":
                self.train_labels = pd.read_csv(os.path.join(base_url, f'{split}.csv'), sep=',')
                self.labels = pd.read_csv(os.path.join(base_url, f'SignList_ClassId_TR_EN.csv'), sep=',')

        print(split, len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.transform = self.build_transform(split)

    def build_transform(self, split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7),
                MultiScaleCrop(
                    (self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']),
                    scales),
                RandomHorizontalFlip(),
                # RandomRotate(p=0.3),
                # RandomShear(0.3, 0.3, p=0.3),
                # Salt(p=0.3),
                # GaussianBlur(sigma=1, p=0.3),
                ColorJitter(0.5, 0.5, 0.5, p=0.3),
                ToFloatTensor(), PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],
                          self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        else:
            print("Build test/val transform")
            transform = Compose(
                Resize(self.data_cfg['vid_transform']['IMAGE_SIZE']),
                ToFloatTensor(),
                PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],
                          self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        return transform

    def count_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        # Đọc kích thước của video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames, width, height

    def transform_handflow(self, handflow):
        # Convert to a PyTorch tensor and transpose to get [C, V]
        handflow_tensor = torch.tensor(handflow, dtype=torch.float32).transpose(0, 1)
        return handflow_tensor

    def read_one_view(self, name, selected_index, width, height):

        clip = []
        heatmap_clip = []
        missing_wrists_left = []
        missing_wrists_right = []

        if self.data_cfg['dataset_name'] == "VN_SIGN":
            path = f'{self.base_url}/crop_videos/{name}'
        vr = VideoReader(path, width=320, height=256)
        frames = vr.get_batch(selected_index).asnumpy()
        for frame, frame_index in zip(frames, selected_index):
            if self.data_cfg['crop_two_hand']:
                kp_path = os.path.join(self.base_url, 'poses', name.replace(".mp4", ""),
                                       name.replace(".mp4", "") + '_{:06d}_'.format(frame_index) + 'keypoints.json')
                # load keypoints
                with open(kp_path, 'r') as keypoints_file:
                    value = json.loads(keypoints_file.read())

                    keypoints = np.array(value['pose_threshold_02'])  # 26,3
                    x = 320 * keypoints[:, 0] / width
                    y = 256 * keypoints[:, 1] / height

                keypoints = np.stack((x, y), axis=0)

            crops = None
            if self.data_cfg['crop_two_hand']:
                crops, missing_wrists_left, missing_wrists_right = crop_hand(frame, keypoints,
                                                                             self.data_cfg['WRIST_DELTA'],
                                                                             self.data_cfg['SHOULDER_DIST_EPSILON'],
                                                                             self.transform, len(clip),
                                                                             missing_wrists_left, missing_wrists_right)
            else:
                crops = self.transform(frame)
            clip.append(crops)

            
        clip = torch.stack(clip, dim=0)
        return clip

    def read_videos(self, center, left, right):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive', 'pad', 'central', 'pad'])
        #
        vlen1, c_width, c_height = self.count_frames(os.path.join(self.base_url, 'crop_videos', center))
        vlen2, l_width, l_height = self.count_frames(os.path.join(self.base_url, 'crop_videos', left))
        vlen3, r_width, r_height = self.count_frames(os.path.join(self.base_url, 'crop_videos', right))

        min_vlen = min(vlen1, min(vlen2, vlen3))
        max_vlen = max(vlen1, max(vlen2, vlen3))
        if max_vlen - min_vlen < 10:
            selected_index, pad = get_selected_indexs(min_vlen - 3, self.data_cfg['num_output_frames'], self.is_train,
                                                      index_setting, temporal_stride=self.data_cfg['temporal_stride'])

            if pad is not None:
                selected_index = pad_index(selected_index, pad).tolist()

            rgb_center = self.read_one_view(center, selected_index, width=c_width,height=c_height)

            rgb_left = self.read_one_view(left, selected_index, width=l_width, height=l_height)

            rgb_right = self.read_one_view(right, selected_index, width=r_width, height=r_height)
        else:
            selected_index, pad = get_selected_indexs(vlen1 - 3, self.data_cfg['num_output_frames'], self.is_train,
                                                      index_setting, temporal_stride=self.data_cfg['temporal_stride'])

            if pad is not None:
                selected_index = pad_index(selected_index, pad).tolist()

            rgb_center = self.read_one_view(center, selected_index, width=c_width,height=c_height)

            selected_index, pad = get_selected_indexs(vlen2 - 3, self.data_cfg['num_output_frames'], self.is_train,
                                                      index_setting, temporal_stride=self.data_cfg['temporal_stride'])

            if pad is not None:
                selected_index = pad_index(selected_index, pad).tolist()

            rgb_left = self.read_one_view(left, selected_index, width=l_width, height=l_height)

            selected_index, pad = get_selected_indexs(vlen3 - 3, self.data_cfg['num_output_frames'], self.is_train,
                                                      index_setting, temporal_stride=self.data_cfg['temporal_stride'])

            if pad is not None:
                selected_index = pad_index(selected_index, pad).tolist()

            rgb_right = self.read_one_view(right, selected_index, width=r_width, height=r_height)

        return rgb_left, rgb_center, rgb_right

    # def __getitem__(self, idx):
    #     self.transform.randomize_parameters()

    #     center,left,right,label = self.train_labels.iloc[idx].values

    #     center_video,center_pf,center_kp,left_video,left_pf,left_kp,right_video,right_pf,right_kp = self.read_videos(center,left,right)

    #     return center_video,center_pf,center_kp,left_video,left_pf,left_kp,right_video,right_pf,right_kp,torch.tensor(label)

    def __getitem__(self, idx):
        self.transform.randomize_parameters()

        center, left, right, label, gloss = self.train_labels.iloc[idx].values
        rgb_left, rgb_center, rgb_right= self.read_videos(center, left, right)

        return rgb_left, rgb_center, rgb_right, gloss, torch.tensor(label)

    def __len__(self):
        return len(self.train_labels)