import numpy as np
import torch
import math
from einops import rearrange
import torch.nn.functional as F
import torch
from multiprocessing.sharedctypes import Value
import torch.nn as nn


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve or validation accuracy doesn't increase after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, 
                 path_loss='checkpoint_loss.pt', path_acc='checkpoint_acc.pt', trace_func=print):
        """
        Args:
            patience (int): Số epoch tối đa không cải thiện sau khi dừng huấn luyện.
                            Mặc định: 7
            verbose (bool): Nếu True, in ra thông báo mỗi khi có sự cải thiện.
                            Mặc định: False
            delta (float): Sự thay đổi tối thiểu trong giá trị được giám sát để xem xét là cải thiện.
                            Mặc định: 0
            path_loss (str): Đường dẫn để lưu checkpoint mô hình dựa trên loss.
                            Mặc định: 'checkpoint_loss.pt'
            path_acc (str): Đường dẫn để lưu checkpoint mô hình dựa trên accuracy.
                            Mặc định: 'checkpoint_acc.pt'
            trace_func (function): Hàm để in thông báo.
                            Mặc định: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter_loss = 0
        self.counter_acc = 0
        self.best_loss = None
        self.best_acc = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = -np.Inf
        self.delta = delta
        self.path_loss = path_loss
        self.path_acc = path_acc
        self.trace_func = trace_func

    def __call__(self, val_loss, val_acc, model):
        # Kiểm tra cải thiện của validation loss
        loss_improved = False
        acc_improved = False

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint_loss(val_loss, model)
            loss_improved = True
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.save_checkpoint_loss(val_loss, model)
            self.counter_loss = 0
            loss_improved = True
        else:
            self.counter_loss += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter (loss): {self.counter_loss} out of {self.patience}')

        # Kiểm tra cải thiện của validation accuracy
        if self.best_acc is None:
            self.best_acc = val_acc
            self.save_checkpoint_acc(val_acc, model)
            acc_improved = True
        elif val_acc > self.best_acc + self.delta:
            self.best_acc = val_acc
            self.save_checkpoint_acc(val_acc, model)
            self.counter_acc = 0
            acc_improved = True
        else:
            self.counter_acc += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter (acc): {self.counter_acc} out of {self.patience}')

        # Kiểm tra xem có cần dừng huấn luyện không
        if self.counter_loss >= self.patience and self.counter_acc >= self.patience:
            self.early_stop = True

    def save_checkpoint_loss(self, val_loss, model):
        '''Lưu mô hình khi validation loss giảm.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model (loss) ...')
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), self.path_loss)
        else:
            torch.save(model.state_dict(), self.path_loss)
        self.val_loss_min = val_loss

    def save_checkpoint_acc(self, val_acc, model):
        '''Lưu mô hình khi validation accuracy tăng.'''
        if self.verbose:
            self.trace_func(f'Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model (acc) ...')
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), self.path_acc)
        else:
            torch.save(model.state_dict(), self.path_acc)
        self.val_acc_max = val_acc
class MyCustomLoss(nn.Module):
    def __init__(self, reduction=None, label_smoothing=0, weight_local=1, cls_x=1, cts_x=1, cosine_x=1):
        super(MyCustomLoss, self).__init__()
        print("Use Label Smoothing: ", label_smoothing)
        self.crossentropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.mse = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.weight_local = weight_local 
        self.weight_global = 1-weight_local
        # print("Weight for local stages (1 and 2): ", self.weight_local)
        # print("Weight for global stages (3 and 4): ", self.weight_global)

    def classification_loss_mixup(self, logits, labels_a, labels_b, lam, epoch):
        """
        Tính classification loss với mixup:
          loss = lam * CE(logits, labels_a) + (1-lam) * CE(logits, labels_b)
        """
        loss_a = self.crossentropy(logits, labels_a)
        loss_b = self.crossentropy(logits, labels_b)
        loss = lam * loss_a + (1 - lam) * loss_b
        return loss, {'classification_loss': loss.item()}

    def forward(self, logits=None, labels=None, trans_feat_s=None, trans_feat_t=None, 
                student_features=None, teacher_features=None, student_logits=None, teacher_logits=None, 
                visual_fusion_ft=None, gloss_visual_fusion_ft=None, **kwargs):
        loss = 0
        loss_dict = {}

        if trans_feat_t is not None and trans_feat_s is not None:
            mse_loss = self.mse(trans_feat_s, trans_feat_t)
            loss += mse_loss
            loss_dict['mse_loss'] = mse_loss.item()
        
        if student_features is not None and teacher_features is not None:
            for stage in ['stage1', 'stage2']:
                stage_loss = self.mse(student_features[stage], teacher_features[stage])
                loss += self.weight_local * stage_loss
                loss_dict[f'mse_loss_{stage}'] = stage_loss.item()

            # Tính loss cho stage 3 và 4
            for stage in ['stage3', 'stage4']:
                stage_loss = self.mse(student_features[stage], teacher_features[stage])
                loss += self.weight_global * stage_loss
                loss_dict[f'mse_loss_{stage}'] = stage_loss.item()

        if student_logits is not None:
            student_logits = F.log_softmax(student_logits, dim=1)
            teacher_logits = teacher_logits.softmax(dim=1)
            kl_loss = self.kl_loss(student_logits, teacher_logits)
            loss += kl_loss
            loss_dict['kl_loss'] = kl_loss.item()

        if logits is not None:
            classification_loss = self.crossentropy(logits, labels)
            loss += classification_loss
            loss_dict['classification_loss'] = classification_loss.item()

        if visual_fusion_ft is not None and gloss_visual_fusion_ft is not None:
            fusion_mse_loss = self.mse(visual_fusion_ft, gloss_visual_fusion_ft.detach())
            loss += 0.05 * fusion_mse_loss
            loss_dict['fusion_mse_loss'] = fusion_mse_loss.item()

        return loss, loss_dict



class LabelSmoothCE(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(
        self, 
        lb_smooth=0.1, 
        reduction='mean', 
        ignore_index=-100, 
        word_emb_tab=None, 
        norm_type='softmax', 
        temp=1.0, 
        variant=None,
        margin=0.0,            # <-- Thêm margin, mặc định = 0.0 (nghĩa là không dùng)
    ):
        super(LabelSmoothCE, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.word_emb_sim = None
        self.variant = variant
        self.margin = margin    # <-- Lưu margin vào class
        self.norm_type = norm_type
        self.temp = temp

        if word_emb_tab is not None:
            # Ma trận độ tương đồng ngôn ngữ
            self.word_emb_sim = torch.matmul(
                F.normalize(word_emb_tab, dim=-1),
                F.normalize(word_emb_tab, dim=-1).T
            )

    def forward(self, logits, label, topk_idx=None, mixup_lam=None, y_a=None, y_b=None, **kwargs):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothCE(variant='word_sim', margin=0.2)
            >>> logits = torch.randn(8, 19)  # B=8, num_classes=19
            >>> lbs = torch.randint(0, 19, (8,))  # shape [B]
            >>> loss = criteria(logits, lbs)
        '''
        logits = logits.float()  # tránh nan, đảm bảo tính float32
        with torch.no_grad():
            if self.variant is None:
                # --------- Label smoothing cổ điển ----------
                num_classes = logits.size(1)
                label = label.clone().detach()
                ignore = label.eq(self.lb_ignore)
                n_valid = ignore.eq(0).sum()
                label[ignore] = 0

                lb_pos = 1.0 - self.lb_smooth
                lb_neg = self.lb_smooth / (num_classes - 1)
                lb_one_hot = torch.empty_like(logits).fill_(lb_neg)
                lb_one_hot.scatter_(1, label.unsqueeze(1), lb_pos)
                lb_one_hot = lb_one_hot.detach()

            elif 'dual' in self.variant:
                # --------- (Không thay đổi) -----------
                B, K, N = logits.shape
                label = label.clone().detach()

                lb_one_hot = torch.zeros(B*K, N, device=logits.device)
                label_exp = label.unsqueeze(1).expand(-1, K).reshape(-1)
                topk_idx = topk_idx.view(-1)
                idx = torch.arange(lb_one_hot.shape[0], device=logits.device)

                if mixup_lam is None:
                    lb_one_hot[idx, label_exp] = 0.5
                    lb_one_hot[idx, topk_idx] += 0.5
                else:
                    lb_one_hot[idx, topk_idx] += 0.5
                    y_a_exp = y_a.unsqueeze(1).expand(-1, K).reshape(-1)
                    y_b_exp = y_b.unsqueeze(1).expand(-1, K).reshape(-1)
                    lb_one_hot[idx, y_a_exp] += mixup_lam * 0.5
                    lb_one_hot[idx, y_b_exp] += (1. - mixup_lam) * 0.5

                lb_one_hot = lb_one_hot.detach().reshape(B, K, N)
                n_valid = B*K

            elif 'word_sim' in self.variant:
                # --------- Language-aware label smoothing ----------
                # Sử dụng ma trận word_emb_sim để tạo nhãn mềm
                assert self.word_emb_sim is not None, "word_emb_sim must not be None for 'word_sim'"
                
                # Tạo one_hot dựa vào độ tương đồng
                lb_one_hot = self.word_emb_sim[label]   # shape [B, N]
                ignore = label.eq(self.lb_ignore)
                n_valid = ignore.eq(0).sum()

                idx = torch.arange(label.shape[0], device=logits.device)
                # Loại bỏ ảnh hưởng nhãn thật trước khi normalize, tuỳ kiểu norm
                if self.norm_type == 'l1':
                    lb_one_hot[idx, label] = 0.0
                    lb_one_hot = F.normalize(lb_one_hot, p=1.0, dim=-1)
                elif self.norm_type == 'softmax':
                    lb_one_hot[idx, label] = float('-inf')
                    lb_one_hot /= self.temp
                    lb_one_hot = F.softmax(lb_one_hot, dim=-1)

                # Scale với lb_smooth, và đặt nhãn thật = (1 - lb_smooth)
                lb_one_hot *= self.lb_smooth
                lb_one_hot[idx, label] = 1.0 - self.lb_smooth
                lb_one_hot = lb_one_hot.detach()

            else:
                raise ValueError("Unsupported variant.")

        # -------------------- Áp dụng margin --------------------
        # Ví dụ: trừ margin vào logit của lớp đúng (CosFace style).
        # Chỉ thực hiện khi margin > 0 (và shape 2D: [B, C]) để code demo đơn giản.
        # Bạn có thể mở rộng nếu logits 4D (B,C,H,W).
        if self.margin > 0.0 and logits.dim() == 2:
            B = logits.size(0)
            idx = torch.arange(B, device=logits.device)
            # Trừ margin vào logit của nhãn thật
            logits[idx, label] = logits[idx, label] - self.margin

        # ------------------- Tính Loss --------------------------
        logs = self.log_softmax(logits)   # [B, C, ...]
        loss = -torch.sum(logs * lb_one_hot, dim=-1)

        # Giảm theo mean hoặc sum
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class MultipleMSELoss(nn.Module):
    """
    Compute multiple mse losses and return their average.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(MultipleMSELoss, self).__init__()
        self.mse_func = nn.MSELoss(reduction=reduction)

    def forward(self, x, y):
        loss_sum = 0.0
        multi_loss = []
        for xt, yt in zip(x, y):
            if isinstance(yt, (tuple,)):
                if len(yt) == 2:
                    yt, wt = yt
                    lt = "mse"
                elif len(yt) == 3:
                    yt, wt, lt = yt
                else:
                    raise NotImplementedError
            else:
                wt, lt = 1.0, "mse"
            if lt == "mse":
                loss = self.mse_func(xt, yt)
            else:
                raise NotImplementedError
            loss_sum += loss * wt
            multi_loss.append(loss)
        return loss_sum, multi_loss


class OLM_Loss(nn.Module):

    def __init__(self, reduction=None,label_smoothing = 0):
        """
        Args:
            reduction: defined for compatibility with other losses.
        """
        super(OLM_Loss, self).__init__()
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.weights = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
        self.optimizer = torch.optim.Adam([self.weights],lr = 1e-3)
        self.initial_loss = None

    def forward(self, logits_v,logists_k,logist_mt,labels,iteration,
                    **kwargs):
        loss = 0
        loss_dict = {
           
        }
        logits = torch.stack([logits_v, logists_k, logist_mt])
        loss_v =  self.criterion( logits_v, labels)
        loss_k =  self.criterion( logists_k, labels)
        loss_mt =  self.criterion(  logist_mt, labels)
        loss_dict['vision_cls'] = loss_v.item()
        loss_dict['keypoints_cls'] = loss_k.item()
        loss_dict['classification_loss'] = loss_mt.item()
        if iteration  == 0:
            self.initial_loss = torch.stack([loss_v, loss_k, loss_mt]).detach()
        loss = self.weights[0] *loss_v + self.weights[1] *loss_k + self.weights[2] *loss_mt
       

        logits_norm = [self.weights[i] * torch.norm(logit, dim=-1).detach() for i, logit in enumerate(logits)]
        # Optimize logit coefficients
        logits_norm = torch.stack(logits_norm, dim=-1)
        loss_ratio = torch.stack([loss_v, loss_k, loss_mt]).detach() / self.initial_loss
        rt = loss_ratio / loss_ratio.mean()
        logits_norm_avg = logits_norm.mean(-1).detach()
        constant = (logits_norm_avg.unsqueeze(-1) @ rt.unsqueeze(0)).detach()
        logitsnorm_loss = torch.abs(logits_norm - constant).sum()

       
        loss_dict['logitsnorm_loss'] = logitsnorm_loss.item()
        loss_dict['vision_w'] = self.weights[0].item()
        loss_dict['keypoints_w'] = self.weights[1].item()
        loss_dict['fusion_w'] = self.weights[2].item()
        loss_dict['iteration'] = iteration

        return loss,loss_dict,logitsnorm_loss