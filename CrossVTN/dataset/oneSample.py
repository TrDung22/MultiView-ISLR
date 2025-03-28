import os
import json
import glob
import math
import numpy as np
import torch
from torch.utils.data import Dataset
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

# Import các hàm crop, transform, v.v... mà bạn đã định nghĩa
# Ví dụ:
# from dataset.my_augmentations import (
#     DeleteFlowKeypoints, Compose, Scale, CenterCrop, ToFloatTensor,
#     PermuteImage, Normalize, ...
# )
# from dataset.my_utils import get_selected_indexs, pad_index, crop_hand, VideoReader, ...

class VTN3GCNDataOneSample(Dataset):
    """
    Dataset chỉ dùng để load DUY NHẤT 1 sample (gồm 3 view: center, left, right).
    Thích hợp cho việc inference 1 sample.
    """

    def __init__(
        self, 
        base_url,
        data_cfg,
        video_center,
        video_left,
        video_right,
        split="test",  # hoặc "val"
    ):
        """
        Khởi tạo dataset cho 1 sample.
        - base_url: thư mục gốc, chứa 'videos/', 'poseflow/', ...
        - dataset_cfg: dict cấu hình (giống trong code gốc)
        - video_center, video_left, video_right: tên file .mp4
        - split: "test" hoặc "val" (nếu cần xử lý transform theo mode test/val)
        """
        self.base_url = base_url
        self.data_cfg = data_cfg
        self.split = split

        # Lưu tên file để lát nữa đọc
        self.video_center = video_center
        self.video_left   = video_left
        self.video_right  = video_right

        # Tùy theo code gốc, bạn có thể giữ logic set self.is_train = (split == 'train')
        # Ở đây mình giả sử test => is_train = False
        self.is_train = (split == "train")

        # Build transform (hoặc có thể copy logic build_transform() từ code cũ)
        self.transform = self.build_transform(split)

        # Build pose_transform, v.v... (bạn copy logic từ code cũ)
        self.pose_transform = Compose(
            DeleteFlowKeypoints(list(range(112, 113))),
            DeleteFlowKeypoints(list(range(11, 92))),
            DeleteFlowKeypoints(list(range(0, 5))),
            ToFloatTensor()
        )

    def build_transform(self, split):
        """
        Xây dựng transform cho image/video frames. 
        Copy logic từ phần build_transform cũ,
        tuỳ bạn có cần augment khi test hay không.
        """
        if split == 'train':
            print("Build train transform (tùy chỉnh)")
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
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],self.data_cfg['vid_transform']['NORM_STD_IMGNET'])
            )
        else:
            print("Build test/val transform (tùy chỉnh)")
            transform = Compose(
                Scale(self.data_cfg['vid_transform']['IMAGE_SIZE'] * 8 // 7), 
                CenterCrop(self.data_cfg['vid_transform']['IMAGE_SIZE']),
                ToFloatTensor(),
                PermuteImage(),
                Normalize(self.data_cfg['vid_transform']['NORM_MEAN_IMGNET'],
                          self.data_cfg['vid_transform']['NORM_STD_IMGNET'])
            )
        return transform

    def __len__(self):
        # Dataset chỉ có 1 sample
        return 1

    def __getitem__(self, idx):
        """
        Duy nhất 1 sample => idx lúc nào cũng là 0
        """
        # Gọi hàm read_videos() (copy logic từ code cũ) 
        (
            center_video, center_pf, center_kp,
            left_video,   left_pf,   left_kp,
            right_video,  right_pf,  right_kp
        ) = self.read_videos(
            self.video_center, 
            self.video_left, 
            self.video_right
        )

        # Không trả về label, vì inference => 
        # HOẶC nếu bạn cần 10th = dummy label => 
        # return center_video, center_pf, center_kp, left_video, left_pf, left_kp, right_video, right_pf, right_kp, torch.tensor(-1)

        return (
            center_video, center_pf, center_kp,
            left_video,   left_pf,   left_kp,
            right_video,  right_pf,  right_kp
        )

    ###############################
    # COPY các hàm read_videos(), read_one_view(), count_frames(), v.v...
    # từ code cũ vào dưới đây
    ###############################
    
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
            
            crops,missing_wrists_left,missing_wrists_right = crop_hand(frame,keypoints,self.data_cfg['WRIST_DELTA'],self.data_cfg['SHOULDER_DIST_EPSILON'],
                                                                       self.transform,len(clip),missing_wrists_left,missing_wrists_right)
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
