from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from .blip2 import Blip2Base, disabled_train
from .blocks import (
    sinusoid_encoding, MaskedConv1D, TransformerEncoder
)
from .weight_init import trunc_normal_

modules = dict()
def register_compose(name):
    def decorator(module):
        modules[name] = module
        return module
    return decorator

@register_compose("compose")
class Blip2Fusion(Blip2Base):
    """
    BLIP2图文特征融合模块
    输入: reference image和文本描述
    输出: 融合后的图文特征
    """
    def __init__(
        self,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        image_size=224,
        drop_path_rate=0,
        vit_model="eva_clip_g",
        use_grad_checkpoint=False,
        vit_precision="fp32",
        freeze_vision=True,
        pretrained_blip2_path="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth",  # Add this parameter for BLIP2 checkpoint
        
    ):
        super().__init__()
        
        # 视觉编码器
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, image_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        
        # Q-Former图文交互模块
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(30523) 

        # Load pretrained BLIP2 weights if provided
        if pretrained_blip2_path is not None:
            self.load_from_pretrained(pretrained_blip2_path)
            print(f"Loaded pretrained BLIP2 weights from {pretrained_blip2_path}")
        
        self.freeze_vision = freeze_vision
        if self.freeze_vision:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train

        for p in self.ln_vision.parameters():
            p.requires_grad = False
            
        for p in self.Qformer.parameters():
            p.requires_grad = False

        self.img_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.txt_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        
        self.max_txt_len = max_txt_len

    def forward(self, ref_img, input_ids, attention_mask):
        device = ref_img.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        if self.freeze_vision:
            with torch.no_grad():
                with autocast(enabled=(ref_img.dtype == torch.float16)):
                    image_embeds = self.ln_vision(self.visual_encoder(ref_img))
        else:
            with autocast(enabled=(ref_img.dtype == torch.float16)):
                image_embeds = self.ln_vision(self.visual_encoder(ref_img))
        
        image_attn_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
        query_tokens = self.query_tokens.expand(image_embeds.size(0), -1, -1)
        query_masks = torch.ones(
            (query_tokens.size(0), query_tokens.size(1)), 
            dtype=torch.long, 
            device=device
        )
        atten_mask = torch.cat([query_masks, attention_mask], dim=1)
        
        cls_output = self.Qformer.bert(
            input_ids=input_ids,
            attention_mask=atten_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attn_mask,
            return_dict=True,
        )

        num_img_tokens = query_tokens.size(1)
        fused_image_feat = cls_output.last_hidden_state[:, :num_img_tokens]

        reg_output = self.Qformer.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        fused_text_feat = reg_output.last_hidden_state

        num_txt_tokens = fused_text_feat.size(1)
        fused_image_feat = self.img_proj(fused_image_feat)
        fused_text_feat = self.txt_proj(fused_text_feat)   
        fused_image_feat = F.normalize(fused_image_feat, dim=-1)
        fused_text_feat = F.normalize(fused_text_feat, dim=-1)
        batch_size = ref_img.size(0)

        
        image_mask = torch.ones(
            (batch_size, 1, num_img_tokens),
            dtype=torch.long, 
            device=device     
        )
        text_mask = torch.ones(
            (batch_size, 1, num_txt_tokens),
            dtype=torch.long,
            device=device
        )

        return fused_image_feat.transpose(2, 1), image_mask, fused_text_feat.transpose(2, 1), text_mask

@register_compose('multimodal_transformer')
class MultimodalTransformer(nn.Module):
    """
    多模态Transformer，用于融合图像和文本特征
    
    图像向量 + 文本向量
    -> [分别投影到统一维度]
    -> [在token维度拼接]
    -> [自注意力Transformer x L]
    -> 拆分图像和文本表示
    """
    def __init__(
        self,
        img_in_dim,              # 图像特征维度
        text_in_dim,             # 文本特征维度
        embd_dim,                # 统一嵌入维度
        n_heads=4,                 # 注意力头数
        max_seq_len=128,             # 最大序列长度（图像+文本）
        n_layers=5,              # Transformer层数
        attn_pdrop=0.0,          # 注意力dropout
        proj_pdrop=0.0,          # 投影dropout
        path_pdrop=0.0,          # 残差路径dropout
        use_abs_pe=True,         # 使用绝对位置编码
        use_bkgd_token=True,     # 使用背景标记
        pe_type='sinusoid',      # 位置编码类型：sinusoid, learned, 2d
        separate_modality_pe=False, # 是否对模态使用独立的位置编码
        return_separate_masks=True, # 是否分别返回图像和文本的mask
    ):
        super().__init__()
        
        self.max_seq_len = max_seq_len
        self.embd_dim = embd_dim
        self.separate_modality_pe = separate_modality_pe
        self.return_separate_masks = return_separate_masks
        self.use_bkgd_token = use_bkgd_token
        
        # 图像特征投影层
        self.img_proj = MaskedConv1D(img_in_dim, embd_dim, 1)
        
        # 文本特征投影层
        self.text_proj = MaskedConv1D(text_in_dim, embd_dim, 1)
        
        # 位置编码
        if use_abs_pe:
            if pe_type == 'sinusoid':
                pe = sinusoid_encoding(max_seq_len, embd_dim // 2)
                pe /= embd_dim ** 0.5
                self.register_buffer('pe', pe, persistent=False)
            elif pe_type == 'learned':
                self.pe = nn.Parameter(torch.empty(1, embd_dim, max_seq_len))
                trunc_normal_(self.pe, mean=0.0, std=0.02)
            else:
                raise ValueError(f"Unknown PE type: {pe_type}")
        else:
            self.pe = None
        
        # 模态类型嵌入（可选）
        self.modality_embedding = nn.Parameter(torch.zeros(2, embd_dim, 1))
        trunc_normal_(self.modality_embedding, mean=0.0, std=0.02)
        
        # 背景标记（可选）
        if use_bkgd_token:
            self.bkgd_token = nn.Parameter(torch.empty(embd_dim, 1))
            trunc_normal_(self.bkgd_token, mean=0.0, std=0.02)
        else:
            self.bkgd_token = None
        
        # Transformer编码器堆叠
        self.transformer = nn.ModuleList()
        for _ in range(n_layers):
            self.transformer.append(
                TransformerEncoder(
                    embd_dim,
                    stride=0,
                    n_heads=n_heads,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop
                )
            )
        
        self.apply(self.__init_weights__)
    
    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, img_feats, text_feats, img_mask, text_mask):
        """
        前向传播
        
        Args:
            img_feats: 图像特征 [batch_size, img_in_dim, img_len]
            text_feats: 文本特征 [batch_size, text_in_dim, text_len]
            img_mask: 图像掩码 [batch_size, img_len] 或 [batch_size, 1, img_len]
            text_mask: 文本掩码 [batch_size, text_len] 或 [batch_size, 1, text_len]
        
        Returns:
            fused_img_feats: 融合后的图像特征 [batch_size, embd_dim, img_len]
            fused_text_feats: 融合后的文本特征 [batch_size, embd_dim, text_len]
            img_mask_out: 图像掩码 [batch_size, 1, img_len]
            text_mask_out: 文本掩码 [batch_size, 1, text_len]
            (可选) bkgd_feat: 背景标记特征 [batch_size, embd_dim, 1]
        """
        bs = img_feats.size(0)
        img_len = img_feats.size(2)
        text_len = text_feats.size(2)
        
        # 确保掩码维度正确
        if img_mask.ndim == 2:
            img_mask = img_mask.unsqueeze(1)  # (bs, l) -> (bs, 1, l)
        if text_mask.ndim == 2:
            text_mask = text_mask.unsqueeze(1)  # (bs, l) -> (bs, 1, l)
        
        # 保存原始的mask用于后续返回
        img_mask_orig = img_mask.clone()
        text_mask_orig = text_mask.clone()
        
        # 1. 分别投影图像和文本特征

        img_proj, img_mask_orig = self.img_proj(img_feats, img_mask)
        text_proj, text_mask_orig = self.text_proj(text_feats, text_mask)
        
        # 2. 添加模态类型嵌入（可选）
        img_proj = img_proj + self.modality_embedding[0].unsqueeze(0)
        text_proj = text_proj + self.modality_embedding[1].unsqueeze(0)
        
        # 3. 在token维度拼接图像和文本
        combined_feats = torch.cat([img_proj, text_proj], dim=2)
        combined_mask = torch.cat([img_mask, text_mask], dim=2)
        
        # 4. 添加位置编码
        if self.pe is not None:
            total_len = img_len + text_len
            pe = self.pe.to(combined_feats.dtype)
            
            if self.training:
                assert total_len <= self.max_seq_len
            else:
                if total_len > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None], size=total_len, mode='linear', align_corners=True
                    )[0]
            
            # 如果使用独立的位置编码，可以分别处理图像和文本
            if self.separate_modality_pe:
                # 为图像和文本使用不同的位置编码起始点
                combined_feats[:, :, :img_len] = combined_feats[:, :, :img_len] + pe[..., :img_len]
                combined_feats[:, :, img_len:] = combined_feats[:, :, img_len:] + pe[..., img_len:total_len]
            else:
                # 统一的位置编码
                combined_feats = combined_feats + pe[..., :total_len]
        
        # 5. 添加背景标记（可选）
        if self.bkgd_token is not None:
            bkgd_token = self.bkgd_token.repeat(bs, 1, 1)
            combined_feats = torch.cat([bkgd_token, combined_feats], dim=2)
            # 背景标记的掩码为1
            bkgd_mask = torch.ones(bs, 1, 1, device=combined_mask.device)
            combined_mask = torch.cat([bkgd_mask, combined_mask], dim=2)
        
        # 6. 通过Transformer层进行融合
        for transformer in self.transformer:
            combined_feats, _ = transformer(combined_feats, combined_mask)
        
        # 7. 拆分回图像和文本特征
        if self.bkgd_token is not None:
            # 如果有背景标记，跳过第一个token
            bkgd_feat = combined_feats[:, :, 0:1]
            img_start = 1
            text_start = 1 + img_len
        else:
            bkgd_feat = None
            img_start = 0
            text_start = img_len
        
        fused_img_feats = combined_feats[:, :, img_start:img_start+img_len]
        fused_text_feats = combined_feats[:, :, text_start:text_start+text_len]
        
        # 根据return_separate_masks参数决定返回什么
        if self.return_separate_masks:
            # 分别返回图像和文本的mask
            return fused_img_feats, fused_text_feats, img_mask_orig, text_mask_orig
        else:
            # 返回组合的mask（如果某些下游任务需要）
            return fused_img_feats, fused_text_feats, combined_mask
    
    def forward_with_bkgd(self, img_feats, text_feats, img_mask, text_mask):
        """
        前向传播并返回背景标记
        
        Args:
            img_feats: 图像特征 [batch_size, img_in_dim, img_len]
            text_feats: 文本特征 [batch_size, text_in_dim, text_len]
            img_mask: 图像掩码 [batch_size, img_len] 或 [batch_size, 1, img_len]
            text_mask: 文本掩码 [batch_size, text_len] 或 [batch_size, 1, text_len]
        
        Returns:
            fused_img_feats: 融合后的图像特征 [batch_size, embd_dim, img_len]
            fused_text_feats: 融合后的文本特征 [batch_size, embd_dim, text_len]
            img_mask_out: 图像掩码 [batch_size, 1, img_len]
            text_mask_out: 文本掩码 [batch_size, 1, text_len]
            bkgd_feat: 背景标记特征 [batch_size, embd_dim, 1]
        """
        # 调用forward但确保use_bkgd_token为True
        if not self.use_bkgd_token:
            raise ValueError("模型未启用背景标记，请设置use_bkgd_token=True")
        
        # 这里为了简化，我们重新计算一遍，实际中可以优化
        bs = img_feats.size(0)
        img_len = img_feats.size(2)
        text_len = text_feats.size(2)
        
        # 确保掩码维度正确
        if img_mask.ndim == 2:
            img_mask = img_mask.unsqueeze(1)
        if text_mask.ndim == 2:
            text_mask = text_mask.unsqueeze(1)
        
        # 保存原始的mask
        img_mask_orig = img_mask.clone()
        text_mask_orig = text_mask.clone()
        
        # 投影
        img_proj, _ = self.img_proj(img_feats, img_mask)
        text_proj, _ = self.text_proj(text_feats, text_mask)
        
        # 添加模态类型嵌入
        img_proj = img_proj + self.modality_embedding[0].unsqueeze(0)
        text_proj = text_proj + self.modality_embedding[1].unsqueeze(0)
        
        # 拼接
        combined_feats = torch.cat([img_proj, text_proj], dim=2)
        combined_mask = torch.cat([img_mask, text_mask], dim=2)
        
        # 添加背景标记
        bkgd_token = self.bkgd_token.repeat(bs, 1, 1)
        combined_feats = torch.cat([bkgd_token, combined_feats], dim=2)
        bkgd_mask = torch.ones(bs, 1, 1, device=combined_mask.device)
        combined_mask = torch.cat([bkgd_mask, combined_mask], dim=2)
        
        # 通过Transformer
        for transformer in self.transformer:
            combined_feats, _ = transformer(combined_feats, combined_mask)
        
        # 拆分
        bkgd_feat = combined_feats[:, :, 0:1]
        fused_img_feats = combined_feats[:, :, 1:1+img_len]
        fused_text_feats = combined_feats[:, :, 1+img_len:1+img_len+text_len]
        
        return fused_img_feats, fused_text_feats, img_mask_orig, text_mask_orig, bkgd_feat
    
    def get_cross_attention_maps(self, img_feats, text_feats, img_mask, text_mask, layer_idx=-1):
        """
        获取跨模态注意力图，用于可视化分析
        
        Args:
            layer_idx: 指定哪一层的注意力图，-1表示最后一层
        
        Returns:
            cross_attn_map: 跨模态注意力图 [batch_size, n_heads, total_len, total_len]
        """
        # 前向传播到指定层
        bs = img_feats.size(0)
        img_len = img_feats.size(2)
        text_len = text_feats.size(2)
        
        if img_mask.ndim == 2:
            img_mask = img_mask.unsqueeze(1)
        if text_mask.ndim == 2:
            text_mask = text_mask.unsqueeze(1)
        
        # 投影
        img_proj, _ = self.img_proj(img_feats, img_mask)
        text_proj, _ = self.text_proj(text_feats, text_mask)
        
        # 拼接
        combined_feats = torch.cat([img_proj, text_proj], dim=2)
        combined_mask = torch.cat([img_mask, text_mask], dim=2)
        
        # 添加背景标记
        if self.use_bkgd_token:
            bkgd_token = self.bkgd_token.repeat(bs, 1, 1)
            combined_feats = torch.cat([bkgd_token, combined_feats], dim=2)
            bkgd_mask = torch.ones(bs, 1, 1, device=combined_mask.device)
            combined_mask = torch.cat([bkgd_mask, combined_mask], dim=2)
        
        # 逐层前向传播，收集注意力图
        for i, transformer in enumerate(self.transformer):
            if i == layer_idx or (layer_idx == -1 and i == len(self.transformer) - 1):
                # 获取注意力图
                combined_feats, attn_info = transformer(combined_feats, combined_mask, return_attn=True)
                attn_maps = attn_info['attn_maps']  # [batch_size, n_heads, seq_len, seq_len]
                
                # 分析跨模态注意力
                if self.use_bkgd_token:
                    # 图像到文本的注意力
                    img_to_text_attn = attn_maps[:, :, 1:1+img_len, 1+img_len:]
                    # 文本到图像的注意力
                    text_to_img_attn = attn_maps[:, :, 1+img_len:, 1:1+img_len]
                    # 背景标记到其他token的注意力
                    bkgd_to_all = attn_maps[:, :, 0:1, :]
                else:
                    img_to_text_attn = attn_maps[:, :, :img_len, img_len:]
                    text_to_img_attn = attn_maps[:, :, img_len:, :img_len]
                    bkgd_to_all = None
                
                return {
                    'all_attn': attn_maps,
                    'img_to_text': img_to_text_attn,
                    'text_to_img': text_to_img_attn,
                    'bkgd_to_all': bkgd_to_all
                }
            else:
                combined_feats, _ = transformer(combined_feats, combined_mask)
        
        return None
@register_compose("img_proj")
class Img_Proj(nn.Module):
    """
    轻量卷积图像编码器：
    输入: ref_img (B,3,H,W)
    输出: image_feat (B, D, Q), image_mask (B, 1, Q)
    参数可在构造时设置：out_dim=D, num_query_token=Q, channels 控制卷积宽度
    """
    def __init__(self, out_dim: int = 256, num_query_token: int = 32, channels: int = 128):
        super().__init__()
        self.num_query_token = num_query_token
        self.out_dim = out_dim
        # 简单 conv stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # 投影到目标维度
        self.proj = nn.Conv2d(channels, out_dim, kernel_size=1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, ref_img: torch.Tensor):
        """
        ref_img: (B,3,H,W)
        返回:
          image_feat: (B, D, Q)
          image_mask: (B, 1, Q)
        """
        device = ref_img.device
        x = self.stem(ref_img)           # (B, C, Hf, Wf)
        x = self.proj(x)                 # (B, D, Hf, Wf)

        # 在 height 维度做平均，得到 (B, D, 1, Wf) —— 这个操作的 backward 是确定性的
        x = x.mean(dim=2, keepdim=True)  # (B, D, 1, Wf)

        # 如果宽度与目标 query 数不匹配，用插值调整到 (1, Q)
        if x.size(3) != self.num_query_token:
            # 使用 bilinear/area 插值通常是确定性的；align_corners=False 更稳健
            x = F.interpolate(x, size=(1, self.num_query_token), mode='bilinear', align_corners=False)

        # 变形为 (B, Q, D) -> 归一化 -> (B, D, Q)
        pooled = x.squeeze(2).permute(0, 2, 1).contiguous()  # (B, Q, D)
        pooled = self.norm(pooled)
        image_feat = pooled.transpose(1, 2).contiguous()     # (B, D, Q)


        batch_size = ref_img.size(0)
        image_mask = torch.ones((batch_size, 1, self.num_query_token), dtype=torch.long, device=device)
        return image_feat, image_mask


def make_compose_net(opt):
    opt = deepcopy(opt)
    return modules[opt.pop('name')](**opt)