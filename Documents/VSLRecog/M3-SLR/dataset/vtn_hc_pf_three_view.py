import os
import numpy as np
import torch
from torch.utils.data import  Dataset
import pandas as pd
from transformers import AutoTokenizer
from dataset.videoLoader import get_selected_indexs,pad_index
import cv2
import torchvision
from dataset.utils import crop_hand
import json
from PIL import Image
from utils.video_augmentation import DeleteFlowKeypoints,ToFloatTensor,Compose
import glob
import time
from decord import VideoReader
import threading
import math
from utils.video_augmentation import *



class VTNHCPF_ThreeViewsData(Dataset):
    def __init__(self, base_url,split,dataset_cfg,**kwargs):
        
        if dataset_cfg is None:
            self.train_labels = pd.read_csv(os.path.join(base_url,f'{split}.csv'),sep=',')
        else:
            if dataset_cfg['dataset_name'] == "VN_SIGN":
                print("Label: ",os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                self.train_labels = pd.read_csv(os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"),sep=',')
                # if split == 'test':
                #      self.train_labels = pd.concat([self.train_labels] * 5, ignore_index=True)
            elif dataset_cfg['dataset_name'] == "AUTSL":
                self.train_labels = pd.read_csv(os.path.join(base_url,f'{split}.csv'),sep=',')
                self.labels = pd.read_csv(os.path.join(base_url,f'SignList_ClassId_TR_EN.csv'),sep=',')
       
        print(split,len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.pose_transform  = Compose(DeleteFlowKeypoints(list(range(112, 113))),
                                        DeleteFlowKeypoints(list(range(11, 92))),
                                        DeleteFlowKeypoints(list(range(0, 5))),
                                        ToFloatTensor())
        self.transform = self.build_transform(split)
    def build_transform(self,split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7),
                                MultiScaleCrop((self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']), scales),
                                RandomHorizontalFlip(), 
                                RandomRotate(p=0.3),
                                RandomShear(0.3,0.3,p = 0.3),
                                Salt( p = 0.3),
                                GaussianBlur( sigma=1,p = 0.3),
                                ColorJitter(0.5, 0.5, 0.5,p = 0.3),
                                ToFloatTensor(), PermuteImage(),
                                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        else:
            print("Build test/val transform")
            transform = Compose(
                                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7), 
                                CenterCrop(self.data_cfg['vid_transform']['IMAGE_SIZE']), 
                                ToFloatTensor(),
                                PermuteImage(),
                                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        return transform
    
    def count_frames(self,video_path):
        cap = cv2.VideoCapture(video_path)
        # Đọc kích thước của video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        n_poses = len(glob.glob(video_path.replace("videos","poses").replace('.mp4','/*')))
        total_frames = min(total_frames,n_poses)
        return total_frames,width,height
    def read_one_view(self,name,selected_index,width,height):
       
        clip = []
        poseflow_clip = []
        missing_wrists_left = []
        missing_wrists_right = []
       
        if self.data_cfg['dataset_name'] == "VN_SIGN":
            path = f'{self.base_url}/videos/{name}'   
        vr = VideoReader(path,width=320, height=256)
        frames = vr.get_batch(selected_index).asnumpy()
        for frame,frame_index in zip(frames,selected_index):
            if self.data_cfg['crop_two_hand']:
                
                kp_path = os.path.join(self.base_url,'poses',name.replace(".mp4",""),
                                    name.replace(".mp4","") + '_{:06d}_'.format(frame_index) + 'keypoints.json')
                # load keypoints
                with open(kp_path, 'r') as keypoints_file:
                    value = json.loads(keypoints_file.read())
                    
                    keypoints = np.array(value['pose_threshold_02']) # 26,3
                    x = 320*keypoints[:,0]/width
                    y = 256*keypoints[:,1]/height
                   
                keypoints = np.stack((x, y), axis=0)
               
           
            crops = None
            if self.data_cfg['crop_two_hand']:
                crops,missing_wrists_left,missing_wrists_right = crop_hand(frame,keypoints,self.data_cfg['WRIST_DELTA'],self.data_cfg['SHOULDER_DIST_EPSILON'],
                                                                       self.transform,len(clip),missing_wrists_left,missing_wrists_right)
            else:
                crops = self.transform(frame)
            clip.append(crops)

            # Let's say the first frame has a pose flow of 0 
            poseflow = None
            frame_index_poseflow = frame_index
            if frame_index_poseflow > 0:
                full_path = os.path.join(self.base_url,'poseflow',name.replace(".mp4",""),
                                        'flow_{:05d}.npy'.format(frame_index_poseflow))
                while not os.path.isfile(full_path):  # WORKAROUND FOR MISSING FILES!!!
                    print("Missing File",full_path)
                    frame_index_poseflow -= 1
                    full_path = os.path.join(self.base_url,'poseflow',name.replace(".mp4",""),
                                        'flow_{:05d}.npy'.format(frame_index_poseflow))

                value = np.load(full_path)
                poseflow = value
                # Normalize the angle between -1 and 1 from -pi and pi
                poseflow[:, 0] /= math.pi
                # Magnitude is already normalized from the pre-processing done before calculating the flow
            else:
                poseflow = np.zeros((135, 2))
            
            pose_transform = Compose(DeleteFlowKeypoints(list(range(114, 115))),
                                    DeleteFlowKeypoints(list(range(19,94))),
                                    DeleteFlowKeypoints(list(range(11, 17))),
                                    ToFloatTensor())

            poseflow = pose_transform(poseflow).view(-1)
            poseflow_clip.append(poseflow)
            
        clip = torch.stack(clip,dim = 0)
        poseflow = torch.stack(poseflow_clip, dim=0)
        return clip,poseflow

    def read_videos(self,center,left,right):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive','pad','central','pad'])
        # 
        vlen1,c_width,c_height = self.count_frames(os.path.join(self.base_url,'videos',center))
        vlen2,l_width,l_height = self.count_frames(os.path.join(self.base_url,'videos',left))
        vlen3,r_width,r_height = self.count_frames(os.path.join(self.base_url,'videos',right))

       
        min_vlen = min(vlen1,min(vlen2,vlen3))
        max_vlen = max(vlen1,max(vlen2,vlen3))
        if max_vlen - min_vlen < 10:
            selected_index, pad = get_selected_indexs(min_vlen - 3,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                print("Vlen",center,left,right)
                selected_index  = pad_index(selected_index,pad).tolist()
        
            center_video,center_pf = self.read_one_view(center,selected_index,width=c_width,height=c_height)
            
            left_video,left_pf = self.read_one_view(left,selected_index,width=l_width,height=l_height)

            right_video,right_pf = self.read_one_view(right,selected_index,width=r_width,height=r_height)
        else:
            selected_index, pad = get_selected_indexs(vlen1 - 3,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                print("Vlen",center)
                selected_index  = pad_index(selected_index,pad).tolist()

            center_video,center_pf = self.read_one_view(center,selected_index,width=c_width,height=c_height)

            selected_index, pad = get_selected_indexs(vlen2 - 3,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                print("Vlen",left)
                selected_index  = pad_index(selected_index,pad).tolist()
            
            
            left_video,left_pf = self.read_one_view(left,selected_index,width=l_width,height=l_height)

            selected_index, pad = get_selected_indexs(vlen3-3,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                print("Vlen",right)
                selected_index  = pad_index(selected_index,pad).tolist()

            right_video,right_pf = self.read_one_view(right,selected_index,width=r_width,height=r_height)

       

        return center_video,center_pf,left_video,left_pf,right_video,right_pf
        # return 1


    def __getitem__(self, idx):
        self.transform.randomize_parameters()
       
        center,left,right,label = self.train_labels.iloc[idx].values
       
        center_video,center_pf,left_video,left_pf,right_video,right_pf = self.read_videos(center,left,right)
        
        return center_video,center_pf,left_video,left_pf,right_video,right_pf,torch.tensor(label)
     
    
    def __len__(self):
        return len(self.train_labels)
    
class VTN3GCNData(Dataset):
    def __init__(self, base_url,split,dataset_cfg,**kwargs):
        
        if dataset_cfg is None:
            self.train_labels = pd.read_csv(os.path.join(base_url,f'{split}.csv'),sep=',')
        else:
            if dataset_cfg['dataset_name'] == "VN_SIGN":
                print("Label: ",os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"))
                self.train_labels = pd.read_csv(os.path.join(base_url,f"{dataset_cfg['label_folder']}/{split}_{dataset_cfg['data_type']}.csv"),sep=',')
                # if split == 'test':
                #      self.train_labels = pd.concat([self.train_labels] * 5, ignore_index=True)
            elif dataset_cfg['dataset_name'] == "AUTSL":
                self.train_labels = pd.read_csv(os.path.join(base_url,f'{split}.csv'),sep=',')
                self.labels = pd.read_csv(os.path.join(base_url,f'SignList_ClassId_TR_EN.csv'),sep=',')
       
        print(split,len(self.train_labels))
        self.split = split
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False
        self.base_url = base_url
        self.data_cfg = dataset_cfg
        self.data_name = dataset_cfg['dataset_name']
        self.pose_transform  = Compose(DeleteFlowKeypoints(list(range(112, 113))),
                                        DeleteFlowKeypoints(list(range(11, 92))),
                                        DeleteFlowKeypoints(list(range(0, 5))),
                                        ToFloatTensor())
        self.transform = self.build_transform(split)
    def build_transform(self,split):
        if split == 'train':
            print("Build train transform")
            transform = Compose(
                                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7),
                                MultiScaleCrop((self.data_cfg['vid_transform']['IMAGE_SIZE'], self.data_cfg['vid_transform']['IMAGE_SIZE']), scales),
                                RandomHorizontalFlip(), 
                                RandomRotate(p=0.3),
                                RandomShear(0.3,0.3,p = 0.3),
                                Salt( p = 0.3),
                                GaussianBlur( sigma=1,p = 0.3),
                                ColorJitter(0.5, 0.5, 0.5,p = 0.3),
                                ToFloatTensor(), PermuteImage(),
                                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        else:
            print("Build test/val transform")
            transform = Compose(
                                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7), 
                                CenterCrop(self.data_cfg['vid_transform']['IMAGE_SIZE']), 
                                ToFloatTensor(),
                                PermuteImage(),
                                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],self.data_cfg['vid_transform']['NORM_STD_IMGNET']))
        return transform
    
    def count_frames(self,video_path):
        cap = cv2.VideoCapture(video_path)
        # Đọc kích thước của video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        n_poses = len(glob.glob(video_path.replace("videos","poses").replace('.mp4','/*')))
        total_frames = min(total_frames,n_poses)
        return total_frames,width,height
    def transform_handflow(self, handflow):
        # Convert to a PyTorch tensor and transpose to get [C, V]
        handflow_tensor = torch.tensor(handflow, dtype=torch.float32).transpose(0, 1)
        return handflow_tensor
    def read_one_view(self,name,selected_index,width,height):
       
        clip = []
        poseflow_clip = []
        handkp_clip = []
        missing_wrists_left = []
        missing_wrists_right = []
       
        if self.data_cfg['dataset_name'] == "VN_SIGN":
            path = f'{self.base_url}/videos/{name}'   
        vr = VideoReader(path,width=320, height=256)
        frames = vr.get_batch(selected_index).asnumpy()
        for frame,frame_index in zip(frames,selected_index):
            if self.data_cfg['crop_two_hand']:
                
                kp_path = os.path.join(self.base_url,'poses',name.replace(".mp4",""),
                                    name.replace(".mp4","") + '_{:06d}_'.format(frame_index) + 'keypoints.json')
                # load keypoints
                with open(kp_path, 'r') as keypoints_file:
                    value = json.loads(keypoints_file.read())
                    
                    keypoints = np.array(value['pose_threshold_02']) # 26,3
                    x = 320*keypoints[:,0]/width
                    y = 256*keypoints[:,1]/height
                   
                keypoints = np.stack((x, y), axis=0)
               
           
            crops = None
            if self.data_cfg['crop_two_hand']:
                crops,missing_wrists_left,missing_wrists_right = crop_hand(frame,keypoints,self.data_cfg['WRIST_DELTA'],self.data_cfg['SHOULDER_DIST_EPSILON'],
                                                                       self.transform,len(clip),missing_wrists_left,missing_wrists_right)
            else:
                crops = self.transform(frame)
            clip.append(crops)

            # Let's say the first frame has a pose flow of 0 
            poseflow = None
            frame_index_poseflow = frame_index
            if frame_index_poseflow > 0:
                full_path = os.path.join(self.base_url,'poseflow',name.replace(".mp4",""),
                                        'flow_{:05d}.npy'.format(frame_index_poseflow))
                while not os.path.isfile(full_path):  # WORKAROUND FOR MISSING FILES!!!
                    frame_index_poseflow -= 1
                    full_path = os.path.join(self.base_url,'poseflow',name.replace(".mp4",""),
                                        'flow_{:05d}.npy'.format(frame_index_poseflow))

                value = np.load(full_path)
                poseflow = value
                # Normalize the angle between -1 and 1 from -pi and pi
                poseflow[:, 0] /= math.pi
                # Magnitude is already normalized from the pre-processing done before calculating the flow
            else:
                poseflow = np.zeros((135, 2))
            
            pose_transform = Compose(DeleteFlowKeypoints(list(range(114, 115))),
                                    DeleteFlowKeypoints(list(range(19,94))),
                                    DeleteFlowKeypoints(list(range(11, 17))),
                                    ToFloatTensor())

            poseflow = pose_transform(poseflow).view(-1)
            poseflow_clip.append(poseflow)

            frame_index_handkp = frame_index
            full_path = os.path.join(self.base_url, 'hand_keypoints', name.replace(".mp4", ""),
                                     f'hand_kp_{frame_index_handkp:05d}.npy')

            # Handle missing files by backtracking to previous frames
            while not os.path.isfile(full_path) and frame_index_handkp > 0:
                frame_index_handkp -= 1
                full_path = os.path.join(self.base_url, 'hand_keypoints', name.replace(".mp4", ""),
                                         f'hand_kp_{frame_index_handkp:05d}.npy')

            if os.path.isfile(full_path):
                # Load the keypoints data
                value = np.load(full_path)
                handkp_frame = value
            else:
                # If no handflow data is found, initialize with zeros
                handkp_frame = np.zeros((46, 2))

            # Apply transformations to handflow data
            handkp_frame = self.transform_handflow(handkp_frame)
            handkp_clip.append(handkp_frame)

        clip = torch.stack(clip,dim = 0)
        poseflow = torch.stack(poseflow_clip, dim=0)
        # Stack handflow frames into a tensor along the time dimension
        handkp = torch.stack(handkp_clip, dim=1)  # shape: [C, T, V]
        # Add the M dimension (number of persons), which is 1 in this case
        handkp = handkp.unsqueeze(-1)  # shape: [C, T, V, M]
        return clip,poseflow,handkp

    def read_videos(self,center,left,right):
        index_setting = self.data_cfg['transform_cfg'].get('index_setting', ['consecutive','pad','central','pad'])
        # 
        vlen1,c_width,c_height = self.count_frames(os.path.join(self.base_url,'videos',center))
        vlen2,l_width,l_height = self.count_frames(os.path.join(self.base_url,'videos',left))
        vlen3,r_width,r_height = self.count_frames(os.path.join(self.base_url,'videos',right))

       
        min_vlen = min(vlen1,min(vlen2,vlen3))
        max_vlen = max(vlen1,max(vlen2,vlen3))
        if max_vlen - min_vlen < 10:
            selected_index, pad = get_selected_indexs(min_vlen - 3,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                selected_index  = pad_index(selected_index,pad).tolist()
        
            center_video,center_pf,center_kp = self.read_one_view(center,selected_index,width=c_width,height=c_height)
            
            left_video,left_pf,left_kp = self.read_one_view(left,selected_index,width=l_width,height=l_height)

            right_video,right_pf,right_kp = self.read_one_view(right,selected_index,width=r_width,height=r_height)
        else:
            selected_index, pad = get_selected_indexs(vlen1 - 3,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                selected_index  = pad_index(selected_index,pad).tolist()

            center_video,center_pf,center_kp = self.read_one_view(center,selected_index,width=c_width,height=c_height)

            selected_index, pad = get_selected_indexs(vlen2 - 3,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                selected_index  = pad_index(selected_index,pad).tolist()
            
            
            left_video,left_pf,left_kp = self.read_one_view(left,selected_index,width=l_width,height=l_height)

            selected_index, pad = get_selected_indexs(vlen3-3,self.data_cfg['num_output_frames'],self.is_train,index_setting,temporal_stride=self.data_cfg['temporal_stride'])
            
            if pad is not None:
                selected_index  = pad_index(selected_index,pad).tolist()

            right_video,right_pf,right_kp = self.read_one_view(right,selected_index,width=r_width,height=r_height)

       

        return center_video,center_pf,center_kp,left_video,left_pf,left_kp,right_video,right_pf,right_kp
        # return 1


    # def __getitem__(self, idx):
    #     self.transform.randomize_parameters()
       
    #     center,left,right,label = self.train_labels.iloc[idx].values
       
    #     center_video,center_pf,center_kp,left_video,left_pf,left_kp,right_video,right_pf,right_kp = self.read_videos(center,left,right)
        
    #     return center_video,center_pf,center_kp,left_video,left_pf,left_kp,right_video,right_pf,right_kp,torch.tensor(label)

    def __getitem__(self, idx):
        self.transform.randomize_parameters()

        center, left, right, label = self.train_labels.iloc[idx].values
        center_video, center_pf, center_kp, left_video, left_pf, left_kp, right_video, right_pf, right_kp = self.read_videos(center, left, right)

        random_number = np.random.randint(1, 1001)

        if not (random_number / 1000 > self.data_cfg.get('center_missing_rates', 0)):
            center_video[:] = 0
            center_pf[:] = 0
            center_kp[:, :, :] = 0

        if not (random_number / 1000 > self.data_cfg.get('left_missing_rates', 0)):
            left_video[:] = 0
            left_pf[:] = 0
            left_kp[:, :, :] = 0
            
        if not (random_number / 1000 > self.data_cfg.get('right_missing_rates', 0)):
            right_video[:] = 0
            right_pf[:] = 0
            right_kp[:, :, :] = 0

        return center_video, center_pf, center_kp, left_video, left_pf, left_kp, right_video, right_pf, right_kp, torch.tensor(label)

    
    def __len__(self):
        return len(self.train_labels)