import numpy as np
from copy import deepcopy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import LayerNorm, TransformerDecoder, MaskedConv1D, TransformerEncoder
import json
import os
import numpy as np


modules = dict()
def register_generate(name):
    def decorator(module):
        modules[name] = module
        return module
    return decorator

@register_generate('generator')
class DistractorGenerator(nn.Module):
    """
    A generator for compare token and depress value.
    """
    def __init__(
        self,
        vid_dim,            # video feature dimension
        text_dim,           # query feature dimension
        n_layers=2,         # number of fusion layers
        n_heads=4,          # number of attention heads for MHA
        mha_win_size=5,    # local window size for MHA (0 for global attention)
        attn_pdrop=0.0,     # dropout rate for attention maps
        proj_pdrop=0.0,     # dropout rate for projection
        path_pdrop=0.0,     # dropout rate for residual paths
        comp_token_num=1,
        ):
        super().__init__()
        self.encoder = TransformerEncoder(
            vid_dim,
            stride=1,
            n_heads=n_heads, 
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
        )
        self.ln_encoder = LayerNorm(vid_dim)

        self.fusion_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.fusion_layers.append(
                TransformerDecoder(
                    vid_dim, text_dim, 
                    n_heads=n_heads, 
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
        self.ln_out = LayerNorm(vid_dim)
        self.comp_token_num=comp_token_num
        self.attention_net = nn.Linear(vid_dim, self.comp_token_num)

        # depress value
        value_token_num = self.comp_token_num
        self.attention_net_v = nn.Linear(vid_dim, value_token_num)
        self.fusion_layers_v = nn.ModuleList()
        for _ in range(n_layers):
            self.fusion_layers_v.append(
                TransformerDecoder(
                    vid_dim, text_dim, 
                    n_heads=n_heads, 
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
        self.ln_out_v = LayerNorm(vid_dim)

    def forward(self, vid, vid_masks, visual_query, visual_masks, text_query, text_masks, text_size=None):

        # compare token
        query=visual_query
        query_masks=visual_masks
        vid, vid_masks = self.encoder(vid, vid_masks)
        vid = self.ln_encoder(vid)
        vid_encoded=vid
        vid_masks_encoded = vid_masks
        for layer in self.fusion_layers:
            vid, vid_masks= layer(vid, vid_masks, query, query_masks, text_size)
        vid = self.ln_out(vid)
        logits = self.attention_net(vid.transpose(1, 2))         # [B, T, k] or [B, T]
        weights = F.softmax(logits, dim=1).transpose(1, 2)      # [B, k, T]
        comp_token = torch.einsum("bkt,bct->bck", weights, vid)

        # depress value
        vid_v=vid_encoded
        vid_masks_v = vid_masks_encoded
        query_v=text_query
        query_masks_v=text_masks
        for layer in self.fusion_layers_v:
            vid_v, vid_masks_v= layer(vid_v, vid_masks_v, query_v, query_masks_v, text_size)
        vid_v = self.ln_out_v(vid_v)
        logits_v = self.attention_net_v(vid_v.transpose(1, 2))      # [B, T, K]
        weights_v = F.softmax(logits_v, dim=1).transpose(1, 2)      # [B, K, T]
        depress_value = torch.einsum("bkt,bct->bck", weights_v, vid_v)   # [B, C, K]

        if comp_token.dim() == 2:
            comp_token = comp_token.unsqueeze(2)

        if depress_value.dim() == 2:
            depress_value = depress_value.unsqueeze(2)

        return comp_token, depress_value

def make_generator(opt):
    opt = deepcopy(opt)
    return modules[opt.pop('name')](**opt)