import torch
import torch.nn as nn

from .fusion import make_fusion
from .head import make_head
from .text_net import make_text_net
from .video_net import make_video_net
from .compose import make_compose_net
from .generate import make_generator
from .blocks import MaskedConv1D
import time
import os

class PtTransformer(nn.Module):
    """
    Transformer based model for single-stage sentence grounding
    """
    def __init__(self, opt):
        super().__init__()
        self.vid_net = make_video_net(opt['vid_net'])
        self.text_net = make_text_net(opt['text_net'])
        self.generator = make_generator(opt['distractor_generator'])
        self.proj = MaskedConv1D(opt['image_indim'], opt['text_net']['embd_dim'], 1)
        self.visual_fusion = make_fusion(opt['visual_fusion'])
        self.text_fusion = make_fusion(opt['text_fusion'])
        self.cls_head = make_head(opt['cls_head'])
        self.reg_head = make_head(opt['reg_head'])
        
    def count_parameters(self, module, trainable_only=True):
        if module is None:
            return 0
        if trainable_only:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in module.parameters())

    def print_parameters(self):
        vid_net_params = self.count_parameters(self.vid_net)
        text_encode_params = self.count_parameters(self.text_net)
        print(f"text_net params: {text_encode_params:,}")
        generator_params = self.count_parameters(self.generator)
        visual_fusion_params = self.count_parameters(self.visual_fusion)
        proj_params = self.count_parameters(self.proj)
        print(f"proj params: {proj_params:,}")
        print(f"vid_net params: {vid_net_params:,}")
        print(f"generator params: {generator_params:,}")
        print(f"visual_fusion params: {visual_fusion_params:,}")
        cls_head_params = self.count_parameters(self.cls_head)
        reg_head_params = self.count_parameters(self.reg_head)
        self.text_fusion.print_parameters()
        print(f"cls_head_params: {cls_head_params:,}")
        print(f"reg_head_params: {reg_head_params:,}")
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"all params: {trainable_params:,}")


    def encode_text(self, tokens, token_masks):
        text, text_masks = self.text_net(tokens, token_masks)
        return text, text_masks

    def encode_video(self, vid, vid_masks):
        fpn, fpn_masks, vid, vid_masks = self.vid_net(vid, vid_masks)
        return fpn, fpn_masks, vid, vid_masks
    def encode_distractor(self, vid, vid_masks, visual_query, visual_masks, text_query, text_masks, text_size=None):
        comp_token, depress_value = self.generator(vid, vid_masks, visual_query, visual_masks, text_query, text_masks, text_size)
        return comp_token, depress_value

    def fuse_and_predict_with_distractor (self, fpn, fpn_masks, visual_query, visual_masks, text_query, text_masks, origin, origin_masks, comp_token, depress_value, text_size=None):
        vid, logits, vid_masks = self.visual_fusion(visual_query, visual_masks, origin, origin_masks, comp_token, text_size)
        fpn, fpn_masks = self.text_fusion(fpn, fpn_masks, text_query, text_masks, vid, logits, vid_masks, depress_value, text_size)
        fpn_logits, _ = self.cls_head(fpn, fpn_masks)
        fpn_offsets, fpn_masks = self.reg_head(fpn, fpn_masks)
        return fpn_logits, fpn_offsets, fpn_masks, logits

    def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
        fpn, fpn_masks = self.fusion(fpn, fpn_masks, text, text_masks, text_size)
        fpn_logits, _ = self.cls_head(fpn, fpn_masks)
        fpn_offsets, fpn_masks = self.reg_head(fpn, fpn_masks)
        return fpn_logits, fpn_offsets, fpn_masks
    
    def forward(self, vid, vid_masks, text, text_masks, frame=None, frame_masks=None,text_size=None):
        # pack text features
        
        if text.ndim == 4:
            text = torch.cat([t[:k] for t, k in zip(text, text_size)])
            frame = torch.cat([t[:k] for t, k in zip(frame, text_size)])
        if text_masks.ndim == 3:
            text_masks = torch.cat([t[:k] for t, k in zip(text_masks, text_size)])
            frame_masks = torch.cat([t[:k] for t, k in zip(frame_masks, text_size)])
        text, text_masks = self.encode_text(text, text_masks)
        visual_query, visual_masks = frame, frame_masks
        if visual_masks.ndim == 2:
            visual_masks = visual_masks.unsqueeze(1)    # (bs, l) -> (bs, 1, l)
        visual_query, visual_masks = self.proj(visual_query, visual_masks)
        fpn, fpn_masks, vid, vid_masks = self.encode_video(vid, vid_masks)
        comp_token, depress_value = self.encode_distractor(vid, vid_masks, visual_query, visual_masks, text, text_masks, text_size)
        fpn_logits, fpn_offsets, fpn_masks, logits = \
        self.fuse_and_predict_with_distractor(fpn, fpn_masks, visual_query, visual_masks, text, text_masks, vid, vid_masks, comp_token, depress_value, text_size)
        return fpn_logits, fpn_offsets, fpn_masks, logits
        

class BufferList(nn.Module):

    def __init__(self, buffers):
        super().__init__()

        for i, buf in enumerate(buffers):
            self.register_buffer(str(i), buf, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class PtGenerator(nn.Module):
    """
    A generator for candidate points from specified FPN levels.
    """
    def __init__(
        self,
        max_seq_len,        # max sequence length
        num_fpn_levels,     # number of feature pyramid levels
        regression_range=4, # normalized regression range
        sigma=1,            # controls overlap between adjacent levels
        use_offset=False,   # whether to align points at the middle of two tics
    ):
        super().__init__()

        self.num_fpn_levels = num_fpn_levels
        assert max_seq_len % 2 ** (self.num_fpn_levels - 1) == 0
        self.max_seq_len = max_seq_len

        # derive regression range for each pyramid level
        self.regression_range = ((0, regression_range), )
        assert sigma > 0 and sigma <= 1
        for l in range(1, self.num_fpn_levels):
            assert regression_range <= max_seq_len
            v_min = regression_range * sigma
            v_max = regression_range * 2
            if l == self.num_fpn_levels - 1:
                v_max = max(v_max, max_seq_len + 1)
            self.regression_range += ((v_min, v_max), )
            regression_range = v_max

        self.use_offset = use_offset

        # generate and buffer all candidate points
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        # tics on the input grid
        tics = torch.arange(0, self.max_seq_len, 1.0)

        points_list = tuple()
        for l in range(self.num_fpn_levels):
            stride = 2 ** l
            points = tics[::stride][:, None]                    # (t, 1)
            if self.use_offset:
                points += 0.5 * stride

            reg_range = torch.as_tensor(
                self.regression_range[l], dtype=torch.float32
            )[None].repeat(len(points), 1)                      # (t, 2)
            stride = torch.as_tensor(
                stride, dtype=torch.float32
            )[None].repeat(len(points), 1)                      # (t, 1)
            points = torch.cat((points, reg_range, stride), 1)  # (t, 4)
            points_list += (points, )

        return BufferList(points_list)

    def forward(self, fpn_n_points):
        """
        Args:
            fpn_n_points (int list [l]): number of points at specified levels.

        Returns:
            fpn_point (float tensor [l * (p, 4)]): candidate points from speficied levels.
        """
        assert len(fpn_n_points) == self.num_fpn_levels

        fpn_points = tuple()
        for n_pts, pts in zip(fpn_n_points, self.buffer_points):
            assert n_pts <= len(pts), (
                'number of requested points {:d} cannot exceed max number '
                'of buffered points {:d}'.format(n_pts, len(pts))
            )
            fpn_points += (pts[:n_pts], )

        return fpn_points