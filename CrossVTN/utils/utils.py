import torch.nn as nn
import torch.optim as optim
from modelling.HRCA_model import HRCA, HRMSSCA
from modelling.Uniformer import UFOneView, UFThreeView, UsimKD
import torch
from trainer.tools import MyCustomLoss,OLM_Loss,MultipleMSELoss
from trainer.gsam import LinearScheduler
from torchvision import models
from torch.nn import functional as F
from collections import OrderedDict
from pytorch_lightning.utilities.migration import pl_legacy_patch

def load_criterion(train_cfg):
    criterion = None
    if train_cfg['criterion'] == "MyCustomLoss":
        criterion = MyCustomLoss(label_smoothing=train_cfg['label_smoothing'],weight_local=train_cfg.get('weight_local',1))
    if train_cfg['criterion'] == "MultipleMSELoss":
        criterion = MultipleMSELoss()
    if train_cfg['criterion'] == "OLM_Loss": 
        criterion = OLM_Loss(label_smoothing=train_cfg['label_smoothing'])
    assert criterion is not None
    return criterion

def load_optimizer(train_cfg,model,criterion=None):
    optimzer = None
    params = list(model.parameters())
    if criterion is not None:
        params += list(criterion.parameters())
    if train_cfg['optimzer'] == "SGD":
        optimzer = optim.SGD(params, lr=train_cfg['learning_rate'],weight_decay=float(train_cfg['w_decay']),momentum=0.9,nesterov=True)
    if train_cfg['optimzer'] == "Adam":
        optimzer = optim.AdamW(params, lr=train_cfg['learning_rate'],weight_decay=float(train_cfg['w_decay']))
    assert optimzer is not None
    return optimzer

def load_lr_scheduler(train_cfg,optimizer):
    scheduler = None
    if train_cfg['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=train_cfg['scheduler_factor'], patience=train_cfg['scheduler_patience'])
    if train_cfg['lr_scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg['lr_step_size'], gamma=train_cfg['gamma'])
    assert scheduler is not None
    return scheduler

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    try:
        if m.weight is not None:
            m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    except:
        pass


def load_model(cfg):
    if cfg['training']['pretrained']:
        print(f"load pretrained model: {cfg['training']['pretrained_model']}")
        if cfg['data']['model_name'] == 'vtn_att_poseflow':
            if ('.ckpt' in cfg['training']['pretrained_model']):
                model = VTNHCPF(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
                new_state_dict = {}
                with pl_legacy_patch():
                    checkpoint = torch.load(cfg['training']['pretrained_model'], map_location='cpu')['state_dict']
                    for key, value in checkpoint.items():
                        new_key = key.replace('model.', '')
                        if not new_key.startswith('classifier') and not new_key.startswith('self_attention_decoder'):
                            new_state_dict[new_key] = value
                model.load_state_dict(new_state_dict, strict=False)
            else:
                model = VTNHCPF(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
                # Hot fix cause cannot find file .ckpt
                # Only need the line below in root repo:
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
                # Fix path:
                # new_state_dict = {}
                # for key, value in torch.load(cfg['training']['pretrained_model'],map_location='cpu').items():
                #         new_state_dict[key.replace('model.','')] = value
                # model.reset_head(226) # AUTSL
                # model.load_state_dict(new_state_dict)
                # model.reset_head(model.num_classes)
        elif cfg['data']['model_name'] == 'VTNGCN':
            model = VTNHCPF_GCN(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            if '.ckpt' in cfg['training']['pretrained_model']: #Temporary
                new_state_dict = {}
                with pl_legacy_patch():
                    checkpoint = torch.load(cfg['training']['pretrained_model'], map_location='cpu')['state_dict']
                    for key, value in checkpoint.items():
                        new_key = key.replace('model.', '')
                        # if not new_key.startswith('bottle_mm') and not new_key.startswith('self_attention_decoder') and not new_key.startswith('classifier'):
                        if not new_key.startswith('bottle_mm') and not new_key.startswith('self_attention_decoder'):
                        # if not new_key.startswith('bottle_mm') and not new_key.startswith('classifier'):
                        # if not new_key.startswith('classifier'):
                            new_state_dict[new_key] = value
                model.reset_head(226)
                model.load_state_dict(new_state_dict, strict=False)
                model.reset_head(model.num_classes)
            else:
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))

        elif cfg['data']['model_name'] == 'HRCA':
            model = HRCA(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            # model.reset_head(400)
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
            # model.reset_head(model.num_classes)
        elif cfg['data']['model_name'] == 'HRMSSCA' or cfg['data']['model_name'] == 'HRMSSCA_debug':
            model = HRMSSCA(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'),strict=False)
        elif cfg['data']['model_name'] == 'UFOneView' or cfg['data']['model_name'] == 'MaskUFOneView':
            model = UFOneView(**cfg['model'],device=cfg['training']['device'])
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'),strict=False)
        elif cfg['data']['model_name'] == 'UFThreeView' or cfg['data']['model_name'] == 'MaskUFThreeView':
            model = UFThreeView(**cfg['model'],device=cfg['training']['device'])
            missing, unexpected = model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'),strict=False)
            if len(missing) > 0:
                print("Các tham số chưa khớp:", missing)
            if len(unexpected) > 0:
                print("Các tham số thừa không dùng:", unexpected)
        elif cfg['data']['model_name'] == 'UsimKD':
            model = UsimKD(**cfg['model'],device=cfg['training']['device'])
            model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'),strict=False)
        elif cfg['data']['model_name'] == 'VTNHCPF_Three_view':
            model = VTNHCPF_Three_View(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            if '.ckpt' in cfg['training']['pretrained_model']:
                new_state_dict = {}
                with pl_legacy_patch():
                    for key, value in torch.load(cfg['training']['pretrained_model'],map_location='cpu')['state_dict'].items():
                        new_state_dict[key.replace('model.','')] = value
                model.center.reset_head(226) # AUTSL
                model.left.reset_head(226) # AUTSL
                model.right.reset_head(226) # AUTSL
                # load autsl ckpt
                model.center.load_state_dict(new_state_dict)
                model.right.load_state_dict(new_state_dict)
                model.left.load_state_dict(new_state_dict)
                # add backbone
                model.add_backbone()
                # remove center, left and right backbone
                model.remove_head_and_backbone()
                model.freeze(layers = 0)
                print("Load VTNHCPF Three View")
            elif "IMAGENET" == cfg['training']['pretrained_model']:
                model.add_backbone()
                model.remove_head_and_backbone()
                print("Load VTNHCPF Three View IMAGENET")
            else:
                model.add_backbone()
                model.remove_head_and_backbone()
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
        
            print("Load VTNHCPF Three View")

        elif cfg['data']['model_name'] == 'VTN3GCN':
            model = VTN3GCN(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            if '.ckpt' in cfg['training']['pretrained_model']:
                new_state_dict = {}
                with pl_legacy_patch():
                    for key, value in torch.load(cfg['training']['pretrained_model'],map_location='cpu')['state_dict'].items():
                        new_state_dict[key.replace('model.','')] = value
                model.center.reset_head(226) # AUTSL
                model.left.reset_head(226) # AUTSL
                model.right.reset_head(226) # AUTSL
                # load autsl ckpt
                model.center.load_state_dict(new_state_dict)
                model.right.load_state_dict(new_state_dict)
                model.left.load_state_dict(new_state_dict)
                # add backbone
                model.add_backbone()
                # remove center, left and right backbone
                model.remove_head_and_backbone()
                model.freeze(layers = 0)
                print("Load VTN3GCN")
            elif "IMAGENET" == cfg['training']['pretrained_model']:
                model.add_backbone()
                model.remove_head_and_backbone()
                print("Load VTN3GCN IMAGENET")
            else:
                model.add_backbone()
                model.remove_head_and_backbone()
                model.load_state_dict(torch.load(cfg['training']['pretrained_model'],map_location='cpu'))
                for param in model.parameters():
                    param.requires_grad = False
        
            print("Load VTN3GCN")
    else:
        if cfg['data']['model_name'] == 'vtn_att_poseflow':
            model = VTNHCPF(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
        if cfg['data']['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation':
            model = VTNHCPF_OneView_Sim_Knowledge_Distilation(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
        if cfg['data']['model_name'] == 'VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference':
            model = VTNHCPF_OneView_Sim_Knowledge_Distilation_Inference(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            state_dict = torch.load("checkpoints/VTNHCPF_OneView_Sim_Knowledge_Distilation/VTNHCPF_OneView_Sim_Knowledge_Distilation with testing labels/best_checkpoints.pth",map_location='cpu')
            new_state_dict = {}
            for key,value in state_dict.items():
                if key.startswith('teacher'): # omit teacher state dict
                    if key.split('.')[1].startswith('classifier'): # save the classifier of the teacher model
                        new_state_dict[key.replace('teacher.','')] = value
                    continue
                new_state_dict[key] = value
            model.load_state_dict(new_state_dict)
        elif cfg['data']['model_name'] == 'VTNHCPF_Three_view':
            model = VTNHCPF_Three_View(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
            state_dict = torch.load("checkpoints/vtn_att_poseflow/vtn_att_poseflow autsl to vsl for one view/best_checkpoints.pth",map_location='cpu')
            model.center.load_state_dict(state_dict,strict = True)
            model.right.load_state_dict(state_dict,strict = True)
            model.left.load_state_dict(state_dict,strict = True)
            model.add_backbone()
            model.remove_head_and_backbone()
            model.freeze(layers = 0)
            print("Load VTNHCPF Three View")
        elif cfg['data']['model_name'] == 'VTN3GCN':
            if cfg['data']['center_kp']:
                model = VTN3GCN(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
                ckpt_path = "checkpoints/VTNGCN/VTNGCN finetune autsl to vsl for one view/best_checkpoints.pth"
                state_dict = torch.load(ckpt_path,map_location='cpu')
                print("Load VTN3GCN initialized weights: ",ckpt_path)
                model.center.load_state_dict(state_dict,strict = True)
                model.right.load_state_dict(state_dict,strict = True)
                model.left.load_state_dict(state_dict,strict = True)
                model.add_backbone()
                model.remove_head_and_backbone()
                model.freeze(layers = 0)
                print("Load VTN3GCN")
            else:
                model = VTN3GCN(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
                vtngcn_ckpt_path = "checkpoints/VTNGCN/VTNGCN finetune autsl for one view w lora/best_checkpoints.pth"
                vtn_ckpt_path = "checkpoints/vtn_att_poseflow/vtn_att_poseflow autsl to vsl for one view w lora/best_checkpoints.pth"
                vtngcn_state_dict = torch.load(vtngcn_ckpt_path,map_location='cpu')
                vtn_state_dict = torch.load(vtn_ckpt_path,map_location='cpu')
                new_vtn_state_dict = {}
                for key, value in vtn_state_dict.items():
                    # if not key.startswith('bottle_mm') and not key.startswith('self_attention_decoder.position_encoding.enc.weight'):
                    new_vtn_state_dict[key] = value
                print("Load VTN3GCN initialized weights: ",vtngcn_ckpt_path)
                print("Load VTN3GCN initialized weights: ",vtn_ckpt_path)
                model.center.load_state_dict(new_vtn_state_dict,strict = False)
                model.right.load_state_dict(vtngcn_state_dict,strict = True)
                model.left.load_state_dict(vtngcn_state_dict,strict = True)
                model.add_backbone()
                model.remove_head_and_backbone()
                model.freeze(layers = 0)
                print("Load VTN3GCN")
        elif cfg['data']['model_name'] == 'HRMSSCA' or cfg['data']['model_name'] == 'HRMSSCA_debug':
            model = HRMSSCA(**cfg['model'],sequence_length=cfg['data']['num_output_frames'])
        elif cfg['data']['model_name'] == 'UFOneView' or cfg['data']['model_name'] == 'MaskUFOneView':
            model = UFOneView(**cfg['model'],device=cfg['training']['device'])
        elif cfg['data']['model_name'] == 'UFThreeView' or cfg['data']['model_name'] == 'MaskUFThreeView':
            model = UFThreeView(**cfg['model'],device=cfg['training']['device'])
        elif cfg['data']['model_name'] == 'UsimKD':
            model = UsimKD(**cfg['model'],device=cfg['training']['device'])

    assert model is not None
    print("loaded model")
    return model
        