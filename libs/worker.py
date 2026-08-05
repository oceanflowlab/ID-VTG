from collections import OrderedDict, defaultdict
from copy import deepcopy
import os
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from libs import load_opt
import json

from .data import make_dataset, make_dataloader
from .dist_utils import get_rank, get_world_size, barrier, all_gather, print0
from .modeling import (
    PtGenerator, PtTransformer,
    sigmoid_focal_loss, ctr_giou_loss, ctr_diou_loss,
    make_optimizer, make_scheduler
)
from .nms import batched_nms
from .train_utils import Logger, AverageMeter, fix_random_seed, iou, time_str
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
except ImportError:
    FlopCountAnalysis = None
    flop_count_table = None




class Trainer:

    def __init__(self, opt):

        self.opt = opt

        # set random seed
        rng = fix_random_seed(opt.get('seed', 2022))
        self.ranking = opt['loss']['ranking']
        self.rank_threshold = opt['loss']['threshold']
        self.model = PtTransformer(opt['model']).cuda()
        self.comp_token_num=opt['model']['distractor_generator']['comp_token_num']
        self.model_ema = deepcopy(self.model).eval().requires_grad_(False)
        self.pt_gen = PtGenerator(**opt['pt_gen']).cuda()
        self.ema_beta = opt['train'].get('ema_beta', 0.999)
        # prepare dataset
        self.num_epochs = opt['train']['epochs'] + opt['train']['warmup_epochs']
        self.batch_size = batch_size = opt['train']['batch_size']
        self.dataset = make_dataset(
            opt['train']['data'], num_epochs=self.num_epochs, is_training=True
        )
        self.rank_before = opt['loss']['before']
        print("rank_before ",self.rank_before)
        self.dataloader, self.sampler = make_dataloader(
            self.dataset, generator=rng, is_training=True,
            batch_size=batch_size, num_workers=opt['train']['num_workers'],
            world_size=get_world_size(), rank=get_rank()
        )
        self.microbatch_size = opt['train'].get('microbatch_size', batch_size)
        self.num_microbatches = batch_size // self.microbatch_size
        assert batch_size % self.microbatch_size == 0

        # build training utilities
        self.itrs_per_epoch = opt['train']['scheduler']['itrs_per_epoch'] = len(self.dataloader)
        self.num_itrs = self.num_epochs * self.itrs_per_epoch
        self.epoch = self.itr = 0
        self.optimizer = make_optimizer(self.model, opt['train']['optimizer'])
        self.scheduler = make_scheduler(self.optimizer, opt['train']['scheduler'])
        self.clip_grad_norm = opt['train'].get('clip_grad_norm')

        # build logging utilities
        self.log_interval = opt['log'].get('log_interval', 100)
        self.checkpoint_epochs = opt['log'].get('checkpoint_epochs', (-1, ))
        if get_rank() == 0:
            self.logger = Logger(os.path.join(opt['_root'], 'log.txt'))
            self.tb_writer = SummaryWriter(os.path.join(opt['_root'], 'tensorboard'))
            self.loss_meters = OrderedDict()
            self.timer = AverageMeter()
        else:
            self.logger = self.tb_writer = self.loss_meters = self.timer = None

        # load model weights and training states
        if opt['_resume']:
            self.load()
            barrier()

        # set up distributed training
        if opt['_distributed']:
            self.model = DistributedDataParallel(self.model, [get_rank()], find_unused_parameters=True)
            self._ema_init()

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.max_text_len = opt['model']['max_text_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        # register annotation hyperparameters
        self.center_sampling = opt['train'].get('center_sampling', 'radius')
        self.center_sampling_radius = opt['train']['center_sampling_radius']

        # register optimization hyperparameters
        self.loss_norm_momentum = opt['train'].get('loss_norm_momentum', 0.9)
        self.loss_norm = opt['train']['loss_norm']
        self.loss_weight = opt['train'].get('loss_weight', 1.0)
        self.sim_loss_weight = opt['train'].get('sim_loss_weight', 1.0)
        self.reg_loss = opt['train'].get('reg_loss', 'diou')
        
        self.best_eval_metric = None
        self.freeze = False
        # self.evaluator = Evaluator(self.opt)
        
    def run(self):
        
        metric1, metric5 = None, None

        print0("Training started.")
        while self.epoch < self.num_epochs:
            print0(self.epoch)
            self.dataset.set_epoch(self.epoch)
            if self.opt['_distributed']:
                self.sampler.set_epoch(self.epoch)
            for data_list in self.dataloader:
                start_time = time.time()
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict = self.forward_backward(data_list)
                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.itr += 1
                self._ema_update()
                if get_rank() == 0:
                    # only track loss from rank 0 to avoid sync overhead
                    for k, v in loss_dict.items():
                        if k not in self.loss_meters:
                            self.loss_meters[k] = AverageMeter()
                        self.loss_meters[k].update(v.detach())
                    self.timer.update(time.time() - start_time)
                    if self.itr == 1 or self.itr % self.log_interval == 0:
                        self.log('train')
            self.epoch += 1
            self.checkpoint()
            barrier()
            if self.epoch > 1:
                # self.evaluator.opt['_ckpt'] = f"epoch_{self.epoch}"
                print0(f"Evaluating at epoch {self.epoch} ...")
                # if self.evaluator.multi_level:
                #     shutil.copy(
                #                 os.path.join(self.opt['_root'], 'models', 'last.pth'),
                #                 os.path.join(self.opt['_root'], 'models', f"{self.epoch}.pth")
                #             )
                # else:
                shutil.copy(
                    os.path.join(self.opt['_root'], 'models', 'last.pth'),
                    os.path.join(self.opt['_root'], 'models', f"best{self.epoch}.pth")
                )
                shutil.copy(
                    os.path.join(self.opt['_root'], 'states', 'last.pth'),
                    os.path.join(self.opt['_root'], 'states', f"best{self.epoch}.pth")
                )
                    
        print0("Training completed.")
        
    # def _run_eval_and_get_metric(self, is_training=True, level=None):
    #     self.evaluator.run(is_training)
        
    #     # 获取R1@0.3和R5@0.3
    #     metrics = self.evaluator.counts / self.evaluator.text_cnt
    #     try:
    #         i1 = self.evaluator.ranks.index(1) if 1 in self.evaluator.ranks else 0
    #         i5 = self.evaluator.ranks.index(5) if 5 in self.evaluator.ranks else 0
    #         j03 = list(self.evaluator.iou_threshs).index(0.3) if 0.3 in self.evaluator.iou_threshs else 0
    #         metric1 = float(metrics[i1, j03])
    #         metric5 = float(metrics[i5, j03])
            
    #         # 获取MR-full-mIoU
    #         if self.evaluator.text_cnt > 0:
    #             mr_full_miou = self.evaluator.rank1_iou_sum / self.evaluator.text_cnt
    #         else:
    #             mr_full_miou = 0.0
            
    #         # 记录到日志
    #         if get_rank() == 0:
    #             with open(os.path.join(self.opt['_root'], 'log.txt'), 'a') as f:
    #                 f.write(f"\n[Epoch {self.epoch}] MR-full-mIoU: {mr_full_miou*100:.2f}%\n")
            
    #         self.evaluator.refresh()
    #         return metric1, metric5, mr_full_miou
    #     except Exception as e:
    #         print0(f"Eval metric parse failed: {e}")
    #         self.evaluator.refresh()
    #         return 0.0, 0.0, 0.0

    def forward_backward(self, data_list):
        sim_loss=0
        cls_loss = reg_loss = total_loss = norm = 0
        for i in range(0, self.batch_size, self.microbatch_size):
            loss_dict = self._microbatch_forward_backward(
                data_list[i:i + self.microbatch_size],
                is_last=(i + self.microbatch_size >= self.batch_size)
            )
            cls_loss += loss_dict['cls']
            reg_loss += loss_dict['reg']
            sim_loss += loss_dict['sim']
            total_loss += loss_dict['total']
            norm += loss_dict['norm']

        # update EMA loss norm
        all_norms = [torch.zeros_like(norm) for _ in range(get_world_size())]
        all_gather(all_norms, norm)
        self.loss_norm = (
            self.loss_norm_momentum * self.loss_norm
            + (1. - self.loss_norm_momentum) * max(sum(all_norms).item(), 1)
        )
        return {'cls': cls_loss, 'reg': reg_loss, 'sim': sim_loss, 'total': total_loss}
        
    def _microbatch_forward_backward(self, data_list, is_last=False):
        # batch data
        vid, vid_masks, text, text_masks, text_size, frame, frame_masks = self._batchify(
                vid_list=[d['vid'] for d in data_list], 
                text_list=[d['text'] for d in data_list],
                frame_list=[d['frame'] for d in data_list]
            )
        vid = vid.cuda(non_blocking=True)
        vid_masks = vid_masks.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        text_masks = text_masks.cuda(non_blocking=True)
        text_size = text_size.cuda(non_blocking=True)
        if text_size is not None:
            logits_mask = vid_masks.repeat_interleave(text_size, dim=0)
        else:
            logits_mask = vid_masks
        
        frame = frame.cuda(non_blocking=True)
        frame_masks = frame_masks.cuda(non_blocking=True)

        targets = torch.cat([d['target'] / self.vid_stride for d in data_list])
        targets = targets.cuda(non_blocking=True)
        
        # forward pass
        if is_last or not self.opt['_distributed']:
            fpn_logits, fpn_offsets, fpn_masks, sim_logits = \
                self.model(vid, vid_masks, text, text_masks, frame, frame_masks, text_size=text_size)
        else:
            with self.model.no_sync():
                fpn_logits, fpn_offsets, fpn_masks, sim_logits = \
                    self.model(vid, vid_masks, text, text_masks, frame, frame_masks, text_size=text_size)
        fpn_n_points = [m.size(-1) for m in fpn_masks]
        fpn_points = self.pt_gen(fpn_n_points)

        # stitch model outputs
        fpn_logits = torch.cat(fpn_logits, dim=1)   # (bs, p)
        fpn_offsets = torch.cat(fpn_offsets, dim=1) # (bs, p, 2)
        fpn_masks = torch.cat(fpn_masks, dim=1)     # (bs, p)
        points = torch.cat(fpn_points)              # (p, 4)

        # annotate points
        gt_labels, gt_offsets = self._annotate_points(points, targets)

        # calculate point loss
        ## (1) loss norm
        pos_masks = torch.logical_and(gt_labels, fpn_masks)
        norm = pos_masks.sum()

        ## (2) classification loss on valid points
        cls_loss = self._calc_focal_loss(
            logits=fpn_logits[fpn_masks], labels=gt_labels[fpn_masks]
        ) / self.loss_norm * get_world_size()
        
        ## (3) regression loss on positive points
        reg_loss = self._calc_iou_loss(
            pred_offsets=fpn_offsets[pos_masks], gt_offsets=gt_offsets[pos_masks]
        ) / self.loss_norm * get_world_size()

        # (4) visual similarity loss
        if self.rank_before:
            sim_loss = self._calc_ranking_loss_gym(
                logits=sim_logits, labels=gt_labels[:, :(sim_logits.size(1)-1)]
            ) / self.loss_norm * get_world_size()
        else:
            sim_loss = self._calc_ranking_loss(
                logits=sim_logits, labels=gt_labels[:, :(sim_logits.size(1)-self.comp_token_num)], logits_mask=logits_mask, margin_bvn=self.rank_threshold, margin_pvb=self.rank_threshold
            ) / self.loss_norm * get_world_size()


        total_loss = cls_loss + self.loss_weight * reg_loss 
        total_loss = total_loss + self.sim_loss_weight*sim_loss
        total_loss.backward()
        return {
            'cls': cls_loss.detach(),
            'reg': self.loss_weight * (reg_loss .detach()),
            'sim': self.sim_loss_weight * (sim_loss.detach()),
            'total': total_loss.detach(),
            'norm': norm.detach(),
        }



    def _batchify_videos(self, vid_list):
        """
        Put video features and their masks in a batch.

        Args:
            vid_list (List[float tensor, (c1, t1)]): video features.

        Returns:
            vid (float tensor, (bs, c1, t1)): video feature sequences.
            vid_masks (bool tensor, (bs, t1)): video masks.
        """
        bs = len(vid_list)
        vid_dim = vid_list[0].size(0)
        vid_lens = [v.size(-1) for v in vid_list]
        vid = vid_list[0].new_full((bs, vid_dim, self.input_vid_len), 0.)
        for idx in range(bs):
            vid[idx, :, :vid_lens[idx]].copy_(vid_list[idx])
        vid_lens = torch.as_tensor(vid_lens)[:, None]
        vid_masks = torch.arange(self.input_vid_len)[None] < vid_lens
        return vid, vid_masks

    def _batchify_frames(self, frame_list):
        """
        Put frame features and their masks in a batch.

        Args:
            frmame_list (List[float tensor,  (dim)]): frame features.

        Returns:
            frame (float tensor, (bs, dim, 1)): frame feature sequences.
            frame_masks (bool tensor, (bs, 1)): frame masks.
        """
        bs = len(frame_list)
        frame_dim = frame_list[0].size(0)
        frame_lens = [1 for frame in frame_list]
        # print("frame_list[0] ",frame_list[0].size())
        # print("frame_dim ",frame_dim)
        # print("frame_lens ",frame_lens)
        frame = frame_list[0].new_full((bs, frame_dim, 1), 0.)
        for idx in range(bs):
            frame[idx, :, :frame_lens[idx]].copy_(frame_list[idx][..., None])
        frame_lens = torch.as_tensor(frame_lens)[:, None]
        frame_masks = torch.arange(1)[None] < frame_lens
        return frame, frame_masks

    def _batchify_text(self, text_list):
        """
        Put text features and their masks in a batch.

        Args:
            text_list (List[float tensor, (c2, t2)]): token features.

        Returns:
            text (float tensor, (bs, c2, t2)): token feature sequences.
            text_masks (bool tensor, (bs, t2)): token masks.
        """
        bs = len(text_list)
        text_dim = text_list[0].size(0)
        text_lens = [t.size(-1) for t in text_list]
        text = text_list[0].new_full((bs, text_dim, self.max_text_len), 0.)
        for idx in range(bs):
            text[idx, :, :text_lens[idx]].copy_(text_list[idx])
        text_lens = torch.as_tensor(text_lens)[:, None]
        text_masks = torch.arange(self.max_text_len)[None] < text_lens # (bs,self.max_text_len)
        return text, text_masks
      
    def _batchify(self, vid_list, text_list, frame_list):
        # print("frame_list: ",len(frame_list))
        assert len(vid_list) == len(text_list)
        bs = len(vid_list)

        # batch videos
        vid, vid_masks = self._batchify_videos(vid_list)

        if isinstance(frame_list[0], tuple):
            # many image queries are associated with the same video
            b_frame, b_frame_masks = tuple(), tuple()
            n = tuple()
            for f in frame_list:
                b_f, b_fm = self._batchify_frames(f) # (bs, dim, 1), (bs, 1)
                b_frame += (b_f, ) 
                b_frame_masks += (b_fm, ) 
                n += (len(f), ) 
            n_max = max(n)

            # (bs, n, c, t)
            frame_dim = b_frame[0].size(1)
            frame = b_frame[0].new_full(
                (bs, n_max, frame_dim, 1), 0.
            )
            for idx in range(bs):
                frame[idx, :n[idx]].copy_(b_frame[idx])

            # (bs, n, t)
            frame_masks = b_frame_masks[0].new_full(
                (bs, n_max, 1), 0, dtype=torch.bool
            )
            for idx in range(bs):
                frame_masks[idx, :n[idx]].copy_(b_frame_masks[idx])

        # batch text
        if isinstance(text_list[0], tuple):
             # many text queries are associated with the same video
            b_text, b_text_masks = tuple(), tuple()
            n = tuple()
            for t in text_list:
                b_t, b_tm = self._batchify_text(t)
                b_text += (b_t, ) #([len(t), text_dim, self.max_text_len], ...)
                b_text_masks += (b_tm, ) # ([len(t),self.max_text_len], ...)
                n += (len(t), ) # (len(t), ...)
            n_max = max(n)      # max number of text queries

            # (bs, n, c, t)
            text_dim = b_text[0].size(1)
            text = b_text[0].new_full(
                (bs, n_max, text_dim, self.max_text_len), 0.
            )
            for idx in range(bs):
                text[idx, :n[idx]].copy_(b_text[idx])

            # (bs, n, t)
            text_masks = b_text_masks[0].new_full(
                (bs, n_max, self.max_text_len), 0, dtype=torch.bool
            )
            for idx in range(bs):
                text_masks[idx, :n[idx]].copy_(b_text_masks[idx])
        else:
            n = bs * (1, )
            text, text_masks = self._batchify_text(text_list)

        text_size = torch.as_tensor(n)
        # print(vid.size())
        # print(vid_masks.size())
        # print(text.size())
        # print(text_masks.size())
        # print(text_size)
        # print(frame.size())
        # print(frame_masks.size())
        # vid: (bs, c1, t1)
        # vid_masks: (bs, t1)
        # text: (bs, (n,) c2, t2)
        # text_masks (bs, (n,) t2)
        # text_size: (bs,)
        # frames: (bs, (n，) c, h, w)
        return vid, vid_masks, text, text_masks, text_size, frame, frame_masks

    def _annotate_points(self, points, targets):
        """
        Assign ground-truth labels and offsets to candidate points.

        Args:
            fpn_points (List[float tensor, (p, 4)]): candidate points.
                (coordinate (1), regression range (2), stride(1))
            targets (float tensor, (bs, 2)): ground-truth segments.

        Returns:
            labels (bool tensor, (bs, p)): ground-truth binary labels.
            offsets (float tensor, (bs, p, 2)): ground-truth offsets.
        """
        labels_list, offsets_list = tuple(), tuple()
        for target in targets:
            labels, offsets = self._annotate_points_per_video(points, target)
            labels_list += (labels, )
            offsets_list += (offsets, )
        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)
        return labels, offsets

    def _annotate_points_per_video(self, points, target):
        """
        Args:
            points (float tensor, (p, 4)): candidate points from all levels.
                (coordinate (1), regression range (2), stride (1))
            target (float tensor, (2,)): ground-truth segment.

        Returns:
            labels (bool tensor, (p,)): ground-truth binary labels.
            offsets (float tensor, (p, 2)): ground-truth offsets.
        """
        # point distance to segment boundaries
        pt2start = points[:, 0] - target[0]     # (p,)
        pt2end = target[1] - points[:, 0]       # (p,)

        # offsets rescaled by down-sampling stride
        offsets = torch.stack((pt2start, pt2end), dim=-1) / points[:, 3:]

        inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

        # (1) whether a point lies in given sampling window
        if self.center_sampling == 'radius':
            ctr = 0.5 * (target[0] + target[1])
            radius = points[:, 3] * self.center_sampling_radius
            t_min = (ctr - radius).clamp_(min=target[0])
            t_max = (ctr + radius).clamp_(max=target[1])
            # point distance to window boundaries
            pt2left = points[:, 0] - t_min  # (p,)
            pt2right = t_max - points[:, 0] # (p,)
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

        # (2) whether event is within regression range of a point
        max_reg_dist = torch.maximum(pt2start, pt2end)
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
        )

        # a point is positive only if it meets both criteria
        labels = torch.logical_and(inside_window, inside_range)

        return labels, offsets


    def _calc_ranking_loss(
        self,
        logits,
        labels,
        logits_mask,
        pos_labels=None,
        neg_labels=None,
        margin_pvb=1.0,
        margin_bvn=1.0,
        alpha=2.0
    ):
        """
        Ranking loss.

        Args:
            logits (float tensor, (bs, p+1)): predicted logits of frames and one background token.
            labels (bool tensor, (bs, p)): ground-truth binary labels.
            logits_mask (bool tensor, (bs, p)): valid frame mask.
        """
        p = labels.size(1)

        if logits_mask.dim() == 3:
            logits_mask = logits_mask.squeeze(1)

        logits_mask = logits_mask[:, :p].to(torch.bool)
        labels = labels.to(torch.bool)


        frame_logits = logits[:, :p]              # [B, p]
        comp_logits = logits[:, p:]             # [B, K]

        if (pos_labels is not None) and (neg_labels is not None):
            pos_valid_mask = pos_labels & logits_mask
            neg_valid_mask = (~neg_labels) & logits_mask
        else:
            pos_valid_mask = labels & logits_mask
            neg_valid_mask = (~labels) & logits_mask

        has_pos = pos_valid_mask.any(dim=1)
        has_neg = neg_valid_mask.any(dim=1)
        valid_rows_mask = has_pos & has_neg

        if not valid_rows_mask.any():
            return torch.tensor(0.0, device=logits.device, dtype=frame_logits.dtype)

        pos_counts = pos_valid_mask.sum(dim=1, keepdim=True)   # (bs, 1)
        neg_counts = neg_valid_mask.sum(dim=1, keepdim=True)   # (bs, 1)

            
        pos_masked_logits = torch.where(pos_valid_mask, frame_logits, 0.0)
        pos_mean = pos_masked_logits.sum(dim=1, keepdim=True) / pos_counts.clamp(min=1).float()

        neg_masked_logits = torch.where(neg_valid_mask, frame_logits, 0.0)
        neg_mean = neg_masked_logits.sum(dim=1, keepdim=True) / neg_counts.clamp(min=1).float()

        pos_comp_loss = F.relu(margin_pvb - pos_mean + comp_logits)
        comp_neg_loss = F.relu(margin_bvn - comp_logits + neg_mean)
        pos_neg_loss = F.relu((margin_pvb + margin_bvn) - pos_mean + neg_mean)

        total_loss = alpha * (pos_comp_loss + comp_neg_loss) + pos_neg_loss

        final_loss = total_loss[valid_rows_mask].mean()
        return final_loss


    def _calc_ranking_loss_gym(self, logits, labels, margin_pvb=1.0, margin_bvn=1.0, alpha=2.0):
        """
        Ranking loss.

        Args:
            logits (float tensor, (bs, p+1)): predicted logits of frames and one background token.
            labels (bool tensor, (bs, p)): ground-truth binary labels.
        """
        frame_logits = logits[:, :-1]                 # (bs, p)
        bg_logits = logits[:, -1].unsqueeze(1)        # (bs, 1)

        has_pos = labels.any(dim=1)
        has_neg = (~labels).any(dim=1)
        valid_rows_mask = has_pos & has_neg           # Shape: (bs,)

        # If no rows are valid, return a zero loss
        if not valid_rows_mask.any():
            return torch.tensor(0.0, device=logits.device)

        # Use MEAN of all positive frames instead of max
        pos_masked_logits = torch.where(labels, frame_logits, 0.0)
        pos_counts = labels.sum(dim=1, keepdim=True).float()  # (bs, 1)
        pos_mean = pos_masked_logits.sum(dim=1, keepdim=True) / pos_counts.clamp(min=1)  # (bs, 1)
        
        # Use MEAN of all negative frames instead of max
        neg_masked_logits = torch.where(~labels, frame_logits, 0.0)
        neg_counts = (~labels).sum(dim=1, keepdim=True).float()  # (bs, 1)
        neg_mean = neg_masked_logits.sum(dim=1, keepdim=True) / neg_counts.clamp(min=1)  # (bs, 1)

        # Calculate the three loss components for the ENTIRE batch
        pos_bg_loss = F.relu(margin_pvb - pos_mean + bg_logits)
        bg_neg_loss = F.relu(margin_bvn - bg_logits + neg_mean)
        pos_neg_loss = F.relu((margin_pvb + margin_bvn) - pos_mean + neg_mean)
        
        # Combine the losses
        total_loss = alpha * (pos_bg_loss + bg_neg_loss) + pos_neg_loss # Shape: (bs, 1)
    
        # We only want to average the loss from rows that had both pos and neg examples.
        final_loss = total_loss[valid_rows_mask].mean()
        
        return final_loss

    def _calc_focal_loss(self, logits, labels, smoothing=0.2, alpha=0.5):
        labels = labels.to(logits.dtype) * (1.0 - smoothing) + smoothing / 2
        return sigmoid_focal_loss(logits, labels, alpha=alpha, reduction='sum')

    def _calc_iou_loss(self, pred_offsets, gt_offsets):
        iou_loss = ctr_diou_loss if self.reg_loss == 'diou' else ctr_giou_loss
        return iou_loss(pred_offsets, gt_offsets, reduction='sum')
    

    def _ema_init(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach())
        for b, b_ema in zip(self.model.buffers(), self.model_ema.buffers()):
            b_ema.copy_(b.detach())

    @torch.no_grad()
    def _ema_update(self):
        for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
            p_ema.copy_(p.detach().lerp(p_ema, self.ema_beta))

    def load(self):
        model_path = os.path.join(self.opt['_root'], 'models', 'last.pth')
        state_path = os.path.join(self.opt['_root'], 'states', 'last.pth')
        model_ckpt = torch.load(model_path, map_location='cpu')
        state_ckpt = torch.load(state_path, map_location='cpu')

        # MODIFIED: Load the state dicts with strict=False
        self.model.load_state_dict(model_ckpt['model'], strict=False)
        self.model_ema.load_state_dict(model_ckpt['model_ema'], strict=False)
        
        # Optimizer and scheduler loading remains the same
        self.optimizer.load_state_dict(state_ckpt['optimizer'])
        self.scheduler.load_state_dict(state_ckpt['scheduler'])
        self.epoch, self.itr = state_ckpt['epoch'], state_ckpt['itr']

        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        print0(f"Loaded checkpoint [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")

    def _unwrap(self, model):
        return model.module if self.opt['_distributed'] else model

    def checkpoint(self):
        e, t = len(str(self.num_epochs)), len(str(self.num_itrs))
        # Assuming print0 is a function that prints only on rank 0 in DDP setups
        print(f"Checkpointing at [epoch {self.epoch:0{e}d} / itr {self.itr:0{t}d}]...")
        model_dir = os.path.join(self.opt['_root'], 'models')
        state_dir = os.path.join(self.opt['_root'], 'states')

        # Ensure directories exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(state_dir, exist_ok=True)

        # Get the full state dictionary for the model and its EMA
        full_model_state_dict = self._unwrap(self.model).state_dict()
        full_ema_state_dict = self.model_ema.state_dict()

        # --- Intelligent filtering based on requires_grad ---
        trainable_model_weights = {}
        trainable_ema_weights = {}
        for name, param in self._unwrap(self.model).named_parameters():
            if param.requires_grad:
                # Use the state_dict to get the actual tensor value
                trainable_model_weights[name] = full_model_state_dict[name]
                trainable_ema_weights[name] = full_ema_state_dict[name]

        # Create the checkpoints with the smaller, filtered weights
        model_ckpt = {
            'model': trainable_model_weights,
            'model_ema': trainable_ema_weights,
        }
        state_ckpt = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'itr': self.itr,
        }
        # Save the smaller checkpoints
        torch.save(model_ckpt, os.path.join(model_dir, 'last.pth'))
        torch.save(state_ckpt, os.path.join(state_dir, 'last.pth'))

    def log(self, stage='train'):
        # 训练阶段日志
        t = len(str(self.num_itrs))
        log_str = f"[{self.itr:0{t}d}/{self.num_itrs:0{t}d}] "
        itr = self.itr
        lr = self.scheduler.get_last_lr()[0]

        # 记录损失值
        for k, v in self.loss_meters.items():
            log_str += f"{k} {v.item():.3f} | "
            self.tb_writer.add_scalar(f"{stage}/{k}", v.item(), itr)
            v.reset()

        # 记录学习率
        self.tb_writer.add_scalar(f"{stage}/lr", lr, itr)
        log_str += time_str(self.timer.item() * self.log_interval)
        self.timer.reset()
        self.logger.write(log_str)
        self.tb_writer.flush()

class Evaluator:

    def __init__(self, opt, vis=False):
        self.vis=vis # 是否对结果进行可视化
        if vis:
            self.opt = opt
        else:
            self.opt = load_opt(os.path.join(opt['_root'], 'opt.yaml'), is_training=False)
        # self.opt = opt
        self.opt['_root'] = opt['_root']
        # set random seed
        rng = fix_random_seed(self.opt.get('seed', 2022))
        self.rank1_iou_sum = 0.0  # 累加所有查询的top-1 IoU
        self.all_rank1_ious = []  # 存储所有查询的top-1 IoU（用于调试）
        # prepare dataset
        self.multi_level = self.opt['eval']['multi_level']
        print("multi_level evaluate: ", self.multi_level)
        self.level = self.opt['eval']['level']
        self.ckpt = opt.get('_ckpt', "best")
        print("loading ckpt ",self.ckpt)
        if self.multi_level:
            self.datasets = {}
            self.dataloaders={}
            self.num_itrs={}
            self.text_cnt = {}
            for lvl in self.level:
                self.datasets[lvl]=make_dataset(self.opt['eval']['data'], is_training=False, cur_level=lvl)
                self.dataloaders[lvl],_ = make_dataloader(
                    self.datasets[lvl], is_training=False, generator=rng, batch_size=1, num_workers=0
                )
                self.num_itrs[lvl] = len(self.dataloaders[lvl])
                self.text_cnt[lvl] = 0
        else:
            dataset = make_dataset(self.opt['eval']['data'], is_training=False)
            self.dataloader, _ = make_dataloader(
                dataset, is_training=False, generator=rng, batch_size=1, num_workers=0
            )
            self.num_itrs = len(self.dataloader)
            self.text_cnt = 0
            
        self.itr = 0

        # load model
        self.model = PtTransformer(self.opt['model']).cuda()
        self.pt_gen = PtGenerator(**self.opt['pt_gen']).cuda()
        self.logger = Logger(os.path.join(self.opt['_root'], f"log.txt"))

        # inference profiling
        self.profile_inference = self.opt['eval'].get('profile_inference', False)
        if self.profile_inference:
            print(f"cal inference: {self.profile_inference}")
        self.profile_warmup = self.opt['eval'].get('profile_warmup', 5)
        self.profile_max_samples = self.opt['eval'].get('profile_max_samples', 50)
        self.profile_flops_max_samples = self.opt['eval'].get('profile_flops_max_samples', 1)

        self.profile_seen = 0
        self.profile_records = []
        self.profile_flops_records = []

        # register model hyperparameters
        self.max_vid_len = opt['model']['max_vid_len']
        self.vid_stride = opt['model'].get('vid_stride', 1)
        self.input_vid_len = self.max_vid_len * self.vid_stride

        num_fpn_levels = opt['model']['num_fpn_levels']
        mha_win_size = opt['model']['mha_win_size']
        ds_strides = [2 ** i for i in range(num_fpn_levels)]
        min_chunk_size = 1
        for idx in range(num_fpn_levels):
            stride = ds_strides[idx]
            if mha_win_size > 0:
                stride *= (mha_win_size // 2) * 2
            min_chunk_size = max(min_chunk_size, stride)
        assert self.max_vid_len % min_chunk_size == 0, (
            f"max video length must be a multiple of {min_chunk_size}"
        )
        self.min_chunk_size = min_chunk_size

        # register evaluation hyperparameters
        self.ranks = opt['eval'].get('ranks', (1, 5))
        self.topk = max(self.ranks)
        self.iou_threshs = np.array(opt['eval'].get('iou_threshs', (0.3, 0.5)))
        if self.multi_level:
            self.counts = {}
            for lvl in self.level:
                self.counts[lvl] = np.zeros((len(self.ranks), len(self.iou_threshs)))
        else:
            self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))

        self.window_size = opt['eval'].get('window_size')
        self.window_stride = opt['eval'].get('window_stride')

        self.batched_nms = lambda segs, scores: batched_nms(
            segs, scores, **opt['eval']['nms']
        )
        self.pre_nms_topk = opt['eval']['pre_nms_topk']
        self.pre_nms_thresh = opt['eval']['pre_nms_thresh']
        self.seg_len_thresh = opt['eval']['seg_len_thresh']

    def load_model(self, name):
        filename = os.path.join(
            self.opt['_root'], 'models', f"{name}.pth"
        )
        print(filename)
        ckpt = torch.load(filename, map_location='cpu')

        # MODIFIED: Load the state dict with strict=False
        self.model.load_state_dict(ckpt['model_ema'], strict=False)
        
        print0(f"Loaded checkpoint [epoch {name}]...")


    def _reset_eval_metrics(self):
        self.itr = 0

        if self.multi_level:
            self.text_cnt = {lvl: 0 for lvl in self.level}
            self.counts = {
                lvl: np.zeros((len(self.ranks), len(self.iou_threshs)), dtype=np.float64)
                for lvl in self.level
            }
            self.map_sum = {lvl: 0.0 for lvl in self.level}
        else:
            self.text_cnt = 0
            self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)), dtype=np.float64)
            self.map_sum = 0.0

        self.rank1_iou_sum = 0.0
        self.all_rank1_ious = []

    @torch.no_grad()
    def run(self, is_training=False):
        if is_training:
            self.load_model("last")
        else:
            self.load_model(self.ckpt)
        self._reset_eval_metrics()
        self.model.eval().requires_grad_(False)
        self.rank1_iou_sum = 0.0
        self.all_rank1_ious = []

        if self.multi_level:
            print("multi level dataset")
            for cur_level in self.level:
                self.rank1_iou_sum = 0.0
                self.all_rank1_ious = []
                self.itr = 0
                print0("Evaluate level: ", cur_level)
                print0("Evaluation started.")
                start_time = time.time()
                cur_dataloader = self.dataloaders[cur_level]
                for data_list in cur_dataloader:
                    results = self.predict(data_list[0])
                    targets = data_list[0]['segment']
                    if len(results) == 0:
                        continue
                    assert len(results) == len(targets)
                    for result, target in zip(results, targets):
                        segs, scores = result['segments'], result['scores']
                        idx = scores.argsort(descending=True)
                        segs, scores = segs[idx[:self.topk]], scores[idx[:self.topk]]
                        target = torch.as_tensor(target, dtype=torch.float)
                        target = target.expand(len(segs), -1)
                        iou_topk = iou(segs, target)

                        # top-1 IoU
                        if len(iou_topk) > 0:
                            top1_iou = iou_topk[0].item()
                        else:
                            top1_iou = 0.0

                        # accumulate for MR-full-mIoU
                        self.rank1_iou_sum += top1_iou
                        self.all_rank1_ious.append(top1_iou)

                        iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
                        hit_mask = (iou_n[:, None] >= self.iou_threshs[None])  # shape: [num_ranks, num_threshs]
                        self.counts[cur_level] += hit_mask
                       

                    self.text_cnt[cur_level] += len(targets)
                    self.itr += 1

                self.log(is_last=True, level=cur_level)
                print0(f"Evaluation completed in {time_str(time.time() - start_time)}.")

        else:
            print0("Evaluation started.")
            start_time = time.time()
            for data_list in self.dataloader:
                data = data_list[0]
                if self.profile_inference:
                    results = self.predict_with_profile(data)
                else:
                    results = self.predict(data)
                targets = data['segment']
                if len(results) == 0:
                    continue
                assert len(results) == len(targets)
                for result, target in zip(results, targets):
                    segs, scores = result['segments'], result['scores']
                    idx = scores.argsort(descending=True)
                    segs, scores = segs[idx[:self.topk]], scores[idx[:self.topk]]
                    target = torch.as_tensor(target, dtype=torch.float)
                    target = target.expand(len(segs), -1)
                    iou_topk = iou(segs, target)

                
                    # top-1 IoU
                    if len(iou_topk) > 0:
                        top1_iou = iou_topk[0].item()
                    else:
                        top1_iou = 0.0

                    # accumulate for MR-full-mIoU
                    self.rank1_iou_sum += top1_iou
                    self.all_rank1_ious.append(top1_iou)

                    iou_n = np.array([iou_topk[:i].max().item() for i in self.ranks])
                    hit_mask = (iou_n[:, None] >= self.iou_threshs[None])  # shape: [num_ranks, num_threshs]
                    self.counts += hit_mask


                self.text_cnt += len(targets)
                self.itr += 1

            self.log(is_last=True)
            if self.profile_inference:
                self.log_inference_profile()
            print0(f"Evaluation completed in {time_str(time.time() - start_time)}.")

    @torch.no_grad()
    def predict_with_profile(self, data):
        """
        Run normal prediction and record peak GPU memory.
        FLOPs are profiled separately on a representative video-query pair.
        """
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        results = self.predict(data)

        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        self.profile_seen += 1

        # skip warmup iterations
        if self.profile_seen > self.profile_warmup:
            if len(self.profile_records) < self.profile_max_samples:
                n_queries = len(data['text']) if isinstance(data['text'], tuple) else 1
                vid_len = data['vid'].size(-1)

                self.profile_records.append({
                    "video_name": data.get("video_name", ""),
                    "vid_len": int(vid_len),
                    "num_queries": int(n_queries),
                    "peak_memory_mb": float(peak_mem_mb),
                })

            # fvcore is slow, so usually profile only one or a few samples
            if len(self.profile_flops_records) < self.profile_flops_max_samples:
                flops_info = self.profile_flops_one_video_query(data)
                if flops_info is not None:
                    self.profile_flops_records.append(flops_info)

        return results


    def _num_windows_for_data(self, data):
        vid_len = data['vid'].size(-1)
        window_size = min(self.window_size or vid_len, vid_len)
        window_stride = self.window_stride or window_size

        n = vid_len - window_size
        num_windows = 0
        idx = 0
        while idx <= n:
            num_windows += 1
            idx += window_stride
        if n > 0 and n % window_stride > 0:
            num_windows += 1
        return num_windows, window_size


    def _make_one_video_query_forward_inputs(self, data, query_idx=0):
        """
        Build inputs for one model.forward() with batch_size=1 and one query.
        This is used for FLOPs, not for evaluation accuracy.

        It follows the same tensor shapes as the training forward path:
        model(vid, vid_masks, text, text_masks, frame, frame_masks, ..., text_size)
        """
        device = next(self.model.parameters()).device

        vid = data['vid']
        vid_len = vid.size(-1)
        num_windows, window_size = self._num_windows_for_data(data)

        input_vid_len = self.input_vid_len
        if window_size > input_vid_len:
            stride = self.min_chunk_size * self.vid_stride
            input_vid_len = (window_size + (stride - 1)) // stride * stride

        # use the first sliding window as the representative input
        window = F.pad(vid[..., :window_size], (0, input_vid_len - window_size))[None]
        window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size

        window = window.to(device, non_blocking=True)
        window_mask = window_mask.to(device, non_blocking=True)

        tokens = data['text']
        frames = data['frame']
        if not isinstance(tokens, tuple):
            tokens = (tokens,)
        if not isinstance(frames, tuple):
            frames = (frames,)

        token = tokens[query_idx]
        frame = frames[query_idx]
        text_size = torch.as_tensor([1], device=device)


        # text: (bs=1, n_query=1, c, max_text_len)
        text_len = token.size(-1)
        text_dim = token.size(0)
        max_text_len = self.opt['model']['max_text_len']

        text = token.new_full((1, 1, text_dim, max_text_len), 0.)
        text[0, 0, :, :text_len].copy_(token)
        text_masks = torch.zeros((1, 1, max_text_len), dtype=torch.bool)
        text_masks[0, 0, :text_len] = True

        # frame: (bs=1, n_query=1, c, 1)
        frame_dim = frame.size(0)
        frame_tensor = frame.new_full((1, 1, frame_dim, 1), 0.)
        frame_tensor[0, 0, :, 0].copy_(frame)
        frame_masks = torch.ones((1, 1, 1), dtype=torch.bool)

        text = text.to(device, non_blocking=True)
        text_masks = text_masks.to(device, non_blocking=True)
        frame_tensor = frame_tensor.to(device, non_blocking=True)
        frame_masks = frame_masks.to(device, non_blocking=True)

        return {
            "inputs": (window, window_mask, text, text_masks, frame_tensor, frame_masks, text_size),
            "num_windows": num_windows,
            "window_size": window_size,
            "input_vid_len": input_vid_len,
        }


    def profile_flops_one_video_query(self, data):
        """
        Profile FLOPs for one video-query pair and one model forward.
        If sliding-window evaluation is used, also report an estimated full-video value.
        """
        if FlopCountAnalysis is None:
            print0("fvcore is not installed. Please install it with: pip install fvcore")
            return None

        model = self.model
        model.eval()

        pack = self._make_one_video_query_forward_inputs(data, query_idx=0)
        inputs = pack["inputs"]
        num_windows = pack["num_windows"]

        class FeatureForwardWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, vid, vid_masks, text, text_masks, frame, frame_masks, text_size):
                return self.model(
                    vid, vid_masks,
                    text, text_masks,
                    frame, frame_masks,
                    sampled_frames=None,
                    text_size=text_size
                )

        wrapper = FeatureForwardWrapper(model).eval()

        try:
            flops = FlopCountAnalysis(wrapper, inputs)
            flops.unsupported_ops_warnings(False)

            total_flops = float(flops.total())
            total_gflops = total_flops / 1e9

            module_flops = {
                name: float(val) / 1e9
                for name, val in flops.by_module().items()
            }

            module_op_flops = {
                name: {op: float(v) / 1e9 for op, v in ops.items()}
                for name, ops in flops.by_module_and_operator().items()
            }

            # This is an estimate for full-video inference when sliding windows are used.
            # It may slightly over-count query encoding because actual predict() encodes query once.
            estimated_full_video_gflops = total_gflops * num_windows

            unsupported_ops = dict(flops.unsupported_ops())

            record = {
                "video_name": data.get("video_name", ""),
                "gflops_per_video_query_window": total_gflops,
                "estimated_gflops_per_full_video_query": estimated_full_video_gflops,
                "num_windows": int(num_windows),
                "window_size": int(pack["window_size"]),
                "input_vid_len": int(pack["input_vid_len"]),
                "unsupported_ops": unsupported_ops,
                # 新增
                "module_gflops_per_video_query_window": module_flops,
                "module_operator_gflops_per_video_query_window": module_op_flops,
            }

            # optional: save detailed fvcore table once
            table_path = os.path.join(self.opt['_root'], "flops_table.txt")
            with open(table_path, "w") as f:
                f.write(flop_count_table(flops, max_depth=4))

            module_path = os.path.join(self.opt['_root'], "flops_by_module.json")
            with open(module_path, "w") as f:
                json.dump({
                    "by_module_gflops": module_flops,
                    "by_module_and_operator_gflops": module_op_flops,
                }, f, indent=2)


            return record

        except Exception as e:
            import traceback
            print0(f"FLOPs profiling failed: {e}")
            traceback.print_exc()
            return None


    def log_inference_profile(self, level=None):
        if len(self.profile_records) == 0 and len(self.profile_flops_records) == 0:
            return

        tag = f"_{level}" if level else ""
        save_path = os.path.join(self.opt['_root'], f"inference_profile{tag}.json")

        profile = {
            "memory_records": self.profile_records,
            "flops_records": self.profile_flops_records,
        }

        with open(save_path, "w") as f:
            json.dump(profile, f, indent=2)

        log_str = "\nInference Profile:"

        if len(self.profile_records) > 0:
            mem = np.array([r["peak_memory_mb"] for r in self.profile_records], dtype=np.float32)
            log_str += (
                f"\nPeak GPU memory allocated: "
                f"mean {mem.mean():.2f} MB, "
                f"max {mem.max():.2f} MB, "
                f"min {mem.min():.2f} MB "
                f"over {len(mem)} profiled samples"
            )

        if len(self.profile_flops_records) > 0:
            flops = np.array(
                [r["gflops_per_video_query_window"] for r in self.profile_flops_records],
                dtype=np.float32
            )
            full_flops = np.array(
                [r["estimated_gflops_per_full_video_query"] for r in self.profile_flops_records],
                dtype=np.float32
            )

            log_str += (
                f"\nFLOPs: "
                f"{flops.mean():.4f} GFLOPs per video-query window; "
                f"estimated {full_flops.mean():.4f} GFLOPs per full video-query"
            )

            unsupported = self.profile_flops_records[0].get("unsupported_ops", {})
            if len(unsupported) > 0:
                log_str += f"\nUnsupported ops in fvcore: {unsupported}"

        log_str += f"\nSaved inference profile to: {save_path}"
        self.logger.write(log_str)


    @torch.no_grad()
    def predict(self, data):
        """ Predict event segments given a single video and an arbitrary
        number of text queries. This function assumes single-GPU evaluation.
        """
        # parse text
        tokens = data['text']
        frames = data['frame']
        if not isinstance(tokens, tuple):
            tokens = (tokens, )
        if not isinstance(frames, tuple):
            frames = (frames, )

        q1_list, mask1_list, q2_list, mask2_list = tuple(), tuple(), tuple(), tuple()

        for text, frame in zip(tokens, frames):
            text = text[None]
            text_mask = text.new_full(
                (1, 1, text.size(-1)), 1, dtype=torch.bool
            )
            text = text.cuda(non_blocking=True)
            text_mask = text_mask.cuda(non_blocking=True)
            text, text_mask = self.model.encode_text(text, text_mask)

            frame = frame[None][..., None]
            frame_mask = frame.new_full(
                (1, 1, frame.size(-1)), 1, dtype=torch.bool
            )
            frame = frame.cuda(non_blocking=True)
            frame_mask = frame_mask.cuda(non_blocking=True)
            if frame_mask.ndim == 2:
                frame_mask = frame_mask.unsqueeze(1)

            visual_query, visual_mask = self.model.proj(frame, frame_mask)
            q1_list += (visual_query,)
            mask1_list += (visual_mask,)
            q2_list += (text,)
            mask2_list += (text_mask,)

        # parse video
        vid = data['vid']
        vid_len = vid.size(-1)

        # external scores (n, t)
        ext_scores = data['ext_scores']
        if ext_scores is not None and ext_scores.ndim == 1:
            ext_scores = ext_scores[None]

        # sliding-window evaluation
        window_size = min(self.window_size or vid_len, vid_len)
        window_stride = self.window_stride or window_size

        n = vid_len - window_size
        windows, window_offsets, window_ext_scores = tuple(), tuple(), tuple()
        
        idx = 0
        while idx <= n:
            windows += (vid[..., idx:idx + window_size], )
            window_offsets += (idx, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., idx:idx + window_size], )
            else:
                window_ext_scores += (None, )
            idx += window_stride
        
        if n > 0 and n % window_stride > 0:
            # backpad last window
            windows += (vid[..., -window_size:], )
            window_offsets += (n, )
            if ext_scores is not None:
                window_ext_scores += (ext_scores[..., -window_size:], )
            else:
                window_ext_scores += (None, )

        input_vid_len = self.input_vid_len
        if window_size > input_vid_len:
            stride = self.min_chunk_size * self.vid_stride
            input_vid_len = (window_size + (stride - 1)) // stride * stride

        segs_list, scores_list = tuple(), tuple()
        for window, window_offset, window_ext in zip(windows, window_offsets, window_ext_scores):
            window = F.pad(window, (0, input_vid_len - window_size))[None]
            window_mask = torch.arange(input_vid_len).view(1, 1, -1) < window_size
            window = window.cuda(non_blocking=True)
            window_mask = window_mask.cuda(non_blocking=True)

            if window_ext is not None:
                window_ext = F.pad(window_ext, (0, input_vid_len - window_size))
                window_ext = window_ext.cuda(non_blocking=True)

            fpn, fpn_masks, vid, vid_masks = self.model.encode_video(window, window_mask)
            fpn_n_points = [m.size(-1) for m in fpn_masks]
            fpn_points = self.pt_gen(fpn_n_points)

            duration = data['duration']
            fps = data['fps']

            fpn_logits_list, fpn_offsets_list = tuple(), tuple()
            for visual_query, visual_masks, text_query, text_masks in zip(q1_list, mask1_list, q2_list, mask2_list):

                comp_token, depress_value = self.model.encode_distractor(
                    vid, vid_masks, visual_query, visual_masks, text_query, text_masks
                )
                fpn_logits, fpn_offsets, _, _ = self.model.fuse_and_predict_with_distractor(
                    fpn, fpn_masks, visual_query, visual_masks, text_query, text_masks,
                    vid, vid_masks, comp_token, depress_value,
                )

                fpn_logits_list += (fpn_logits, )
                fpn_offsets_list += (fpn_offsets, )
                
            fpn_masks = [m.squeeze(1) for m in fpn_masks]

            # collect segments and their scores
            window_segs_list, window_scores_list = tuple(), tuple()
            for idx, (fpn_logits, fpn_offsets) in enumerate(zip(fpn_logits_list, fpn_offsets_list)):
                window_segs, window_scores = self._collect_segments(
                    fpn_points, fpn_logits, fpn_offsets, fpn_masks, 
                    window_ext[idx] if window_ext is not None else None
                )
                window_segs += window_offset / self.vid_stride
                window_segs_list += (window_segs.cpu(),)
                window_scores_list += (window_scores.cpu(),)

            segs_list += (window_segs_list,)
            scores_list += (window_scores_list,)

        segs_list = [torch.cat(x) for x in zip(*segs_list)]
        scores_list = [torch.cat(x) for x in zip(*scores_list)]

        results = tuple()
        for i, (segs, scores) in enumerate(zip(segs_list, scores_list)):
            n_topk = min(len(segs), self.pre_nms_topk)
            idx = scores.argsort(descending=True)[:n_topk]
            segs, scores = self.batched_nms(segs[idx], scores[idx])

            if len(segs) > 0:
                clip_stride = data['clip_stride']
                clip_size = data['clip_size']
                fps = data['fps']
                duration = data['duration']

                segs *= self.vid_stride
                segs = (segs * clip_stride + 0.5 * clip_size) / fps
                segs = torch.clamp(segs, min=0, max=duration)

            results += ({
                'segments': segs,
                'scores': scores
            },)

        return results


    def log(self, is_last=False, level=None):
        if level:
            metrics = self.counts[level] / self.text_cnt[level]
        else:
            metrics = self.counts / self.text_cnt
        
        log_str = "\nFinal:" if is_last else f"\n[{self.itr}/{self.num_itrs}]"
        if level:
            log_str = "\nFinal:" if is_last else f"\n[{self.itr}/{self.num_itrs[level]}]"
            log_str +=f"\nEvaluate {level} "
        
        # 计算MR-full-mIoU
        if level:
            total_queries = self.text_cnt[level]
        else:
            total_queries = self.text_cnt
        
        if total_queries > 0:
            mr_full_miou = self.rank1_iou_sum / total_queries * 100  # 转换为百分比
        else:
            mr_full_miou = 0.0
        
        # 输出R1@不同threshold
        for i, rank in enumerate(self.ranks):
            log_str += "\n-----"
            for j, thresh in enumerate(self.iou_threshs):
                log_str += (
                    f"\nRank@{rank}, IoU@{thresh:.1f}: "
                    f"{(metrics[i, j] * 100):.2f}")
        
        # 输出MR-full-mIoU
        log_str += f"\nMR-full-mIoU: {mr_full_miou:.2f}"
        
        self.logger.write(log_str)


    def _collect_segments(
        self,
        fpn_points,     # List[(p, 4) * #levels]
        fpn_logits,     # List[(1, p) * #levels]
        fpn_offsets,    # List[(1, p, 2) * #levels]
        fpn_masks,      # List[(1, p) * #levels]
        ext_scores,     # (p, )
    ):
        points_list, scores_list, offsets_list = tuple(), tuple(), tuple()

        # loop over all FPN levels
        for points, logits, offsets, masks in zip(
            fpn_points, fpn_logits, fpn_offsets, fpn_masks
        ):
            logits, offsets, masks = logits[0], offsets[0], masks[0]

            # compute point scores
            scores = torch.sigmoid(logits)
            if ext_scores is not None:
                # external scores has the same length as the video features
                scores *= ext_scores
                ext_scores = F.max_pool1d(
                    ext_scores[None, None], kernel_size=3, stride=2, padding=1
                )[0, 0]
            scores *= masks.float()

            # clean up predictions before NMS for efficiency
            ## (1) filter points by confidence threshold
            idx = scores > self.pre_nms_thresh
            points_list += (points[idx], )
            scores_list += (scores[idx], )
            offsets_list += (offsets[idx], )

        points = torch.cat(points_list)
        scores = torch.cat(scores_list)
        offsets = torch.cat(offsets_list)

        ## (2) only keep top-k scoring boxes
        n_topk = min(len(points), self.pre_nms_topk)
        idx = scores.argsort(descending=True)[:n_topk]
        points, scores, offsets = points[idx], scores[idx], offsets[idx]

        ## (3) assemble predicted segments
        pt_ctr = points[:, 0]
        left = pt_ctr - offsets[:, 0] * points[:, 3]
        right = pt_ctr + offsets[:, 1] * points[:, 3]
        segs = torch.stack((left, right), dim=-1)

        ## (4) filter segments by length threshold
        seg_lens = right - left
        idx = seg_lens > self.seg_len_thresh
        segs, scores = segs[idx], scores[idx]

        return segs, scores




    def refresh(self):
        """
        Reset all evaluation data and metrics to ensure clean state for next evaluation.
        This prevents previous evaluation results from affecting subsequent evaluations.
        """
        # Reset iteration and text counters
        self.itr = 0
        self.text_cnt = 0
        
        # Reset evaluation metrics
        self.counts = np.zeros((len(self.ranks), len(self.iou_threshs)))
        
        # 重置MR-full-mIoU相关变量
        self.rank1_iou_sum = 0.0
        self.all_rank1_ious = []
        
        print0("Evaluator state refreshed for new evaluation.")
