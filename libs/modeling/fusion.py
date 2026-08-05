from copy import deepcopy

import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F  
import time
import os

from .blocks import LayerNorm, TransformerDecoder, MaskedConv1D


modules = dict()
def register_fusion(name):
    def decorator(module):
        modules[name] = module
        return module
    return decorator


@register_fusion('visual_xattn')
class VisualXAttNFusion(nn.Module):
    """ Fuse video and text_query features using attention.
    """

    def __init__(
        self,
        vid_dim,            # video feature dimension
        text_dim,           # query feature dimension
        n_layers=2,         # number of fusion layers
        n_heads=4,          # number of attention heads for MHA
        attn_pdrop=0.0,     # dropout rate for attention maps
        proj_pdrop=0.0,     # dropout rate for projection
        path_pdrop=0.0,     # dropout rate for residual paths
        xattn_mode='adaln', # cross-attention mode (adaln | affine)
        prior_prob=0.0,     # prior probability of positive class
        temperature=1.0,    # temperature for scaling cosine similarity

    ):
        super(VisualXAttNFusion, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                TransformerDecoder(
                    vid_dim, text_dim, 
                    n_heads=n_heads, 
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    xattn_mode=xattn_mode,
                )
            )

        self.ln_out = LayerNorm(vid_dim)
        
        # Temperature parameter for scaling cosine similarity
        self.temperature = temperature

        self.conv = MaskedConv1D(
                    vid_dim, vid_dim,
                    kernel_size=3, stride=1, padding=1, bias=False
                )
        self.norm = LayerNorm(vid_dim)

        self.score_head = MaskedConv1D(
            vid_dim, 1, kernel_size=3, stride=1, padding=1
        )

        bias_init = 0
        assert prior_prob >= 0 and prior_prob < 1
        if prior_prob > 0:
            bias_init = -np.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.score_head.conv.bias, bias_init)

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, visual_query, visual_masks, origin, origin_masks, comp_token, text_size=None):
        if text_size is not None and origin.size(0) != visual_query.size(0):
            origin = origin.repeat_interleave(text_size, dim=0)
            origin_masks = origin_masks.repeat_interleave(text_size, dim=0)
        vid = torch.cat([origin, comp_token], dim=2)
        comp_masks = origin_masks.new_ones(comp_token.size(0), 1, comp_token.size(2))
        vid_masks = torch.cat([origin_masks, comp_masks], dim=2)
        for layer in self.layers:
            vid, vid_masks = layer(vid, vid_masks, visual_query, visual_masks, text_size)
        vid = self.ln_out(vid)
        x, mask = self.conv(vid, vid_masks)
        x = F.relu(self.norm(x), inplace=True)
        logits, _ = self.score_head(x, mask)                  # (bs, 1, p)
        logits = logits.squeeze(1)
        return vid, logits, vid_masks


@register_fusion('xattn_slow_branch')
class XAttN_slow_Fusion(nn.Module):
    """ Fuse video and text features using attention.
    """

    def __init__(
        self,
        vid_dim,            # video feature dimension
        text_dim,           # text feature dimension
        n_layers=2,         # number of fusion layers
        n_heads=4,          # number of attention heads for MHA
        attn_pdrop=0.0,     # dropout rate for attention maps
        proj_pdrop=0.0,     # dropout rate for projection
        path_pdrop=0.0,     # dropout rate for residual paths
        xattn_mode='adaln', # cross-attention mode (adaln | affine)
    ):
        super(XAttN_slow_Fusion, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                TransformerDecoder(
                    vid_dim, text_dim, 
                    n_heads=n_heads, 
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    xattn_mode=xattn_mode,
                )
            )

        self.ln_out = LayerNorm(vid_dim)
        self.proj = MaskedConv1D(vid_dim, vid_dim, 1)
        self.residual = MaskedConv1D(vid_dim, vid_dim, 1)
        self.apply(self.__init_weights__)

    def count_parameters(self, module, trainable_only=True):
        if module is None:
            return 0
        if trainable_only:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in module.parameters())

    def print_parameters(self):
        text_layers_params = self.count_parameters(self.layers)
        text_ln_out_params = self.count_parameters(self.ln_out)
        text_params = text_layers_params+text_ln_out_params
        print(f"text ground params: {text_params:,}")
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        agg_params=trainable_params-text_params
        print(f"agg params: {agg_params:,}")


    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward(self, q, q_mask, kv, kv_mask, kv_size=None):
        for layer in self.layers:
            q, q_mask = layer(q, q_mask, kv, kv_mask, kv_size)
        q = self.ln_out(q)

        # repeat query to match the size of key / value
        if kv_size is not None and q.size(0) != kv.size(0):
            q = q.repeat_interleave(kv_size, dim=0)
            q_mask = q_mask.repeat_interleave(kv_size, dim=0)

        return q, q_mask
    

    def _slow_fast_aggregate_vectorized(self, x, mask, vid, logits, vid_masks, depress_value, comp_logits):
        """
        Vectorized Slow-Fast aggregation.

        Args:
            x (float tensor, (bs, c, t_fast)): fast/video features.
            mask (bool tensor, (bs, 1, t_fast)): fast masks.
            vid (float tensor, (bs, c, t_slow)): slow features.
            logits (float tensor, (bs, t_slow)): slow logits.
            vid_masks (bool tensor, (bs, 1, t_slow)): slow masks.
            depress_value (float tensor, (bs, c, 1) or None): depress value.
            comp_logits (float tensor, (bs, 1) or None): compare token logits.

        Returns:
            x: aggregated video features.
            mask: aggregated video masks.
        """
        bs, c, fast_length = x.size()
        _, _, slow_length = vid.size()

        scale = slow_length // fast_length
        recep_field = 2 * scale - 1
        window_size = recep_field

        x_proj, mask = self.proj(x, mask)

        device = x.device
        vid_masks_2d = vid_masks.squeeze(1).to(torch.bool)   # (bs, t_slow)

        # ---------------------------------------------------------
        # 1) Build indices for all fast positions at once
        #    start[i] = i * scale
        #    idx[i, j] = start[i] + j
        # ---------------------------------------------------------
        start_idx = torch.arange(fast_length, device=device) * scale              # (t_fast,)
        offset = torch.arange(window_size, device=device)                         # (window_size,)
        gather_idx = start_idx[:, None] + offset[None, :]                        # (t_fast, window_size)

        # # valid positions inside slow_length
        # valid_idx_mask = gather_idx < slow_length                                 # (t_fast, window_size)

        # # clamp invalid indices to 0 to make gather safe
        # # gather_idx_clamped = gather_idx.clamp(max=max(slow_length - 1, 0))        # (t_fast, window_size)
        # max_idx = (slow_length.to(gather_idx.device) - 1).clamp(min=0)
        # gather_idx_clamped = gather_idx.clamp(max=max_idx)


        # valid positions inside slow_length
        slow_length_idx = torch.as_tensor(
            slow_length,
            device=gather_idx.device,
            dtype=gather_idx.dtype
        )

        valid_idx_mask = gather_idx < slow_length_idx                              # (t_fast, window_size)

        # clamp invalid indices to slow_length - 1 to make gather safe
        max_idx = (slow_length_idx - 1).clamp_min(0)
        gather_idx_clamped = torch.minimum(gather_idx, max_idx)                    # (t_fast, window_size)


        # ---------------------------------------------------------
        # 2) Gather slow features/logits/masks
        #    slow_window:   (bs, c, t_fast, window_size)
        #    logits_window: (bs, t_fast, window_size)
        #    mask_window:   (bs, t_fast, window_size)
        # ---------------------------------------------------------
        # slow features
        vid_expand = vid.unsqueeze(2).expand(bs, c, fast_length, slow_length)     # (bs, c, t_fast, t_slow)
        gather_idx_feat = gather_idx_clamped.unsqueeze(0).unsqueeze(0).expand(bs, c, fast_length, window_size)
        slow_window = torch.gather(vid_expand, dim=3, index=gather_idx_feat)      # (bs, c, t_fast, window_size)

        # logits
        logits_expand = logits.unsqueeze(1).expand(bs, fast_length, slow_length)  # (bs, t_fast, t_slow)
        gather_idx_logit = gather_idx_clamped.unsqueeze(0).expand(bs, fast_length, window_size)
        logits_window = torch.gather(logits_expand, dim=2, index=gather_idx_logit)  # (bs, t_fast, window_size)

        # masks
        vid_masks_expand = vid_masks_2d.unsqueeze(1).expand(bs, fast_length, slow_length)  # (bs, t_fast, t_slow)
        mask_window = torch.gather(vid_masks_expand, dim=2, index=gather_idx_logit)         # (bs, t_fast, window_size)

        # mask out those indices beyond slow_length
        valid_idx_mask = valid_idx_mask.unsqueeze(0).expand(bs, fast_length, window_size)   # (bs, t_fast, window_size)
        mask_window = mask_window & valid_idx_mask


        if depress_value.dim() == 2:
            depress_value = depress_value.unsqueeze(2)  # [B, C] -> [B, C, 1]

        comp_token_num = depress_value.size(2)       # K

        # depress_value: [B, C, K] -> [B, C, T_fast, K]
        depress_value_expand = depress_value.unsqueeze(2).expand(
            bs, c, fast_length, comp_token_num
        )

        slow_window = torch.cat([slow_window, depress_value_expand], dim=3)
        # slow_window: [B, C, T_fast, window_size + K]

        # comp_logits: [B, K] -> [B, T_fast, K]
        if comp_logits.dim() == 1:
            comp_logits = comp_logits.unsqueeze(1)

        comp_logits_expand = comp_logits.unsqueeze(1).expand(
            bs, fast_length, comp_token_num
        )

        logits_window = torch.cat([logits_window, comp_logits_expand], dim=2)
        # logits_window: [B, T_fast, window_size + K]

        # comp masks: [B, T_fast, K]
        comp_mask = torch.ones(
            (bs, fast_length, comp_token_num),
            dtype=torch.bool,
            device=device
        )

        mask_window = torch.cat([mask_window, comp_mask], dim=2)


        # ---------------------------------------------------------
        # 4) Masked softmax
        # ---------------------------------------------------------
        masked_logits = logits_window.masked_fill(~mask_window, float('-inf'))
        weights = F.softmax(masked_logits, dim=-1)                                # (bs, t_fast, window_size[+1])

        # if all masked, set weights to 0
        has_valid = mask_window.any(dim=-1, keepdim=True)                         # (bs, t_fast, 1)
        weights = torch.where(has_valid, weights, torch.zeros_like(weights))

        # ---------------------------------------------------------
        # 5) Weighted sum
        #    slow_window: (bs, c, t_fast, w)
        #    weights:     (bs, t_fast, w) -> (bs, 1, t_fast, w)
        # ---------------------------------------------------------
        aggregated_slow = (slow_window * weights.unsqueeze(1)).sum(dim=-1)        # (bs, c, t_fast)

        # ---------------------------------------------------------
        # 6) Residual / output
        # ---------------------------------------------------------
        x, mask = self.residual(x_proj + aggregated_slow, mask)

        return x, mask



    def _slow_fast_aggregate(self, x, mask, vid, logits, vid_masks, depress_value, comp_logits):
        """
        Slow-Fast aggregation.
        Args:
            x (float tensor, (bs, c, t1)): video features.
            mask (bool tensor, (bs, 1, t1)): video masks.
            vid (float tensor, (bs, c, t2)): slow features.
            logits (float tensor, (bs, t2)): slow logits.
            vid_masks (bool tensor, (bs, 1, t2)): slow masks.
            depress_value (float tensor, (bs, c, 1)): depress value.
            comp_logits (float tensor, (bs, 1)): comp token logits.
        Returns:
            x: aggregated video features.
            mask: aggregated video masks.
        """
        # Create weighted aggregation of slow features
        bs, c, fast_length = x.size()
        bs, _, slow_length = vid.size()
        
        scale = slow_length // fast_length
        recep_field = 2 * scale - 1
        
        x_proj, mask = self.proj(x, mask)
        
        # Initialize aggregated slow features
        aggregated_slow = torch.zeros_like(x_proj)

        vid_masks = vid_masks.squeeze(1)
        
        # For each fast feature position
        for i in range(fast_length):
            # Calculate receptive field boundaries for this fast feature
            start = i * scale
            end = min(slow_length, start + recep_field)
            
            # Extract corresponding slow features and logits
            slow_window = vid[:, :, start:end]  # (bs, c, window_size)
            slow_window = torch.cat([slow_window, depress_value], dim=2)
            logits_window = logits[:, start:end]  # (bs, window_size)
            logits_window = torch.cat([logits_window, comp_logits], dim=1)  # Add background logits
            
            mask_window = vid_masks[:, start:end].to(torch.bool)  # (bs, window_size)
            mask_window = torch.cat([mask_window, vid_masks.new_ones(bs, 1)], dim=1).to(torch.bool)  # Add background mask
            
            # Apply mask to logits (set masked positions to very negative value)
            masked_logits = logits_window.clone()
            masked_logits[~mask_window] = -float('inf')
            
            # Compute softmax weights
            weights = F.softmax(masked_logits, dim=-1)  # (bs, window_size)
            
            # Handle case where all positions are masked
            weights = torch.where(mask_window.any(dim=-1, keepdim=True), 
                                weights, 
                                torch.zeros_like(weights))
            
            # Weighted average of slow features
            # weights: (bs, window_size) -> (bs, 1, window_size)
            # slow_window: (bs, c, window_size)
            weights_expanded = weights.unsqueeze(1)  # (bs, 1, window_size)
            weighted_slow = (slow_window * weights_expanded).sum(dim=-1)  # (bs, c)
            
            # Assign to aggregated slow features
            aggregated_slow[:, :, i] = weighted_slow
        
        # Combine projected fast features with aggregated slow features
        x, mask = self.residual(x_proj+aggregated_slow, mask)
        
        return x, mask

    def forward(self, fpn, fpn_masks, text_query, text_masks, vid, logits, vid_masks, depress_value, text_size=None):

        out, out_masks = tuple(), tuple()

        if depress_value.dim() == 2:
            depress_value = depress_value.unsqueeze(2)   # [B, C] -> [B, C, 1]

        comp_token_num = depress_value.size(2)

        vid = vid[:, :, :-comp_token_num]
        vid_masks = vid_masks[:, :, :-comp_token_num]

        logits, comp_logits = logits[:, :-comp_token_num], logits[:, -comp_token_num:]

        for x, mask in zip(fpn, fpn_masks):
            if text_size is not None and x.size(0) != vid.size(0):
                x = x.repeat_interleave(text_size, dim=0)
                mask = mask.repeat_interleave(text_size, dim=0)
            
            x, mask = self._slow_fast_aggregate_vectorized(x, mask, vid, logits, vid_masks, depress_value, comp_logits)

            x, mask = self._forward(x, mask, text_query, text_masks, text_size)

            out += (x, )
            out_masks += (mask, )
        return out, out_masks


@register_fusion('xattn')
class XAttNFusion(nn.Module):
    """ Fuse video and text features using attention.
    """

    def __init__(
        self,
        vid_dim,            # video feature dimension
        text_dim,           # text feature dimension
        n_layers=2,         # number of fusion layers
        n_heads=4,          # number of attention heads for MHA
        attn_pdrop=0.0,     # dropout rate for attention maps
        proj_pdrop=0.0,     # dropout rate for projection
        path_pdrop=0.0,     # dropout rate for residual paths
        xattn_mode='adaln', # cross-attention mode (adaln | affine)
    ):
        super(XAttNFusion, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                TransformerDecoder(
                    vid_dim, text_dim, 
                    n_heads=n_heads, 
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    xattn_mode=xattn_mode,
                )
            )

        self.ln_out = LayerNorm(vid_dim)

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _forward(self, q, q_mask, kv, kv_mask, kv_size=None):
        for layer in self.layers:
            q, q_mask = layer(q, q_mask, kv, kv_mask, kv_size)
        q = self.ln_out(q)

        # repeat query to match the size of key / value
        if kv_size is not None and q.size(0) != kv.size(0):
            q = q.repeat_interleave(kv_size, dim=0)
            q_mask = q_mask.repeat_interleave(kv_size, dim=0)

        return q, q_mask

    def forward(self, vid, vid_masks, text, text_mask, text_size=None):
        if not isinstance(vid, tuple):
            return self._forward(vid, vid_masks, text, text_mask, text_size)
            
        out, out_masks = tuple(), tuple()
        for x, mask in zip(vid, vid_masks):
            x, mask = self._forward(x, mask, text, text_mask, text_size)
            out += (x, )
            out_masks += (mask, )

        return out, out_masks


def make_fusion(opt):
    opt = deepcopy(opt)
    return modules[opt.pop('name')](**opt)