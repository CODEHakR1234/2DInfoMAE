"""
InfoMAE: Information-Driven Masked Autoencoding
Main model implementation with Surprisal-Weighted Attention and Adaptive Masking
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import Block, PatchEmbed
import math
from typing import Optional, Tuple


class SurprisalWeightedAttention(nn.Module):
    """
    Self-attention with surprisal-based bias
    A_ij = softmax(QK^T / sqrt(d) + λ * S_j)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, surprisal_bias: Optional[torch.Tensor] = None, lambda_weight: float = 0.0):
        """
        Args:
            x: [B, N, C] input tokens
            surprisal_bias: [B, N] surprisal values for each token
            lambda_weight: weighting factor for surprisal bias
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Standard attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Add surprisal bias if provided
        if surprisal_bias is not None and lambda_weight > 0:
            # Expand surprisal to match attention shape
            surprisal_bias = surprisal_bias.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            attn = attn + lambda_weight * surprisal_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SurprisalBlock(nn.Module):
    """Transformer block with surprisal-weighted attention"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SurprisalWeightedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                                attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, surprisal_bias=None, lambda_weight=0.0):
        x = x + self.attn(self.norm1(x), surprisal_bias, lambda_weight)
        x = x + self.mlp(self.norm2(x))
        return x


class InfoMAE(nn.Module):
    """
    InfoMAE: Masked Autoencoder with Information-Driven Attention
    
    Key features:
    1. Surprisal-Weighted Attention (SWA)
    2. Adaptive Masking based on reconstruction error
    3. Information Bottleneck regularization
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        norm_layer=nn.LayerNorm,
        mask_ratio=0.75,
        use_surprisal_attention=True,
        adaptive_masking=True,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        self.use_surprisal_attention = use_surprisal_attention
        self.adaptive_masking = adaptive_masking
        
        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        
        # Encoder blocks
        if use_surprisal_attention:
            self.blocks = nn.ModuleList([
                SurprisalBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(depth)
            ])
        self.norm = norm_layer(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        
        # Surprisal tracking (exponential moving average)
        self.register_buffer('surprisal_ema', torch.ones(self.num_patches))
        self.surprisal_momentum = 0.9
        
        self.initialize_weights()
        
    def initialize_weights(self):
        """Initialize weights"""
        # Position embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        
        # Tokens
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Linear layers
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def patchify(self, imgs):
        """Convert images to patches"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * 3)
        return x
    
    def unpatchify(self, x):
        """Convert patches back to images"""
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * p, w * p)
        return imgs
    
    def random_masking(self, x, mask_ratio):
        """Random masking: uniform random"""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def adaptive_masking_strategy(self, x, mask_ratio, alpha=3.0, gamma=1.0):
        """
        Adaptive masking based on surprisal
        p_mask(i) = sigmoid(alpha - gamma * S_i)
        High surprisal → low mask probability → learn more often
        """
        N, L, D = x.shape
        
        # Use EMA of surprisal
        surprisal = self.surprisal_ema.unsqueeze(0).expand(N, -1)  # [B, L]
        
        # Compute masking probabilities
        mask_probs = torch.sigmoid(alpha - gamma * surprisal)
        
        # Sample masks
        mask = torch.bernoulli(mask_probs).to(x.device)
        
        # Ensure we mask approximately mask_ratio tokens
        target_masked = int(L * mask_ratio)
        current_masked = mask.sum(dim=1, keepdim=True)
        adjustment = target_masked - current_masked
        
        # Adjust mask to match target ratio
        for i in range(N):
            if adjustment[i] > 0:
                # Need to mask more
                unmasked_idx = (mask[i] == 0).nonzero(as_tuple=True)[0]
                if len(unmasked_idx) > 0:
                    perm = torch.randperm(len(unmasked_idx), device=x.device)
                    to_mask = unmasked_idx[perm[:int(adjustment[i])]]
                    mask[i, to_mask] = 1
            elif adjustment[i] < 0:
                # Need to unmask some
                masked_idx = (mask[i] == 1).nonzero(as_tuple=True)[0]
                if len(masked_idx) > 0:
                    perm = torch.randperm(len(masked_idx), device=x.device)
                    to_unmask = masked_idx[perm[:int(-adjustment[i])]]
                    mask[i, to_unmask] = 0
        
        # Get kept indices
        ids_keep = (mask == 0).nonzero(as_tuple=False)
        ids_keep = ids_keep[:, 1].reshape(N, -1)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Create restore indices
        ids_restore = torch.argsort(torch.argsort(mask, dim=1), dim=1)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio, lambda_weight=0.0, alpha=3.0, gamma=1.0):
        """Encoder with optional surprisal-weighted attention"""
        # Patch embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking
        if self.adaptive_masking and self.training:
            x, mask, ids_restore = self.adaptive_masking_strategy(x, mask_ratio, alpha, gamma)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Add cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Prepare surprisal bias (only for non-masked tokens)
        surprisal_bias = None
        if self.use_surprisal_attention and lambda_weight > 0:
            # Get surprisal for all patches
            surprisal_all = self.surprisal_ema.unsqueeze(0).expand(x.shape[0], -1)
            # Mask out the masked patches
            surprisal_masked = surprisal_all * (1 - mask)
            # Add cls token surprisal (set to mean)
            cls_surprisal = surprisal_masked.mean(dim=1, keepdim=True)
            surprisal_bias = torch.cat([cls_surprisal, surprisal_masked], dim=1)
        
        # Apply blocks
        for blk in self.blocks:
            if self.use_surprisal_attention:
                x = blk(x, surprisal_bias, lambda_weight)
            else:
                x = blk(x)
        
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """Decoder"""
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add pos embed
        x = x + self.decoder_pos_embed
        
        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predictor
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        Compute reconstruction loss
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        # Compute surprisal (reconstruction error) per patch
        with torch.no_grad():
            surprisal = loss.detach()
            # Update EMA
            if self.training:
                batch_surprisal = surprisal.mean(dim=0)  # Average over batch
                self.surprisal_ema = (self.surprisal_momentum * self.surprisal_ema + 
                                      (1 - self.surprisal_momentum) * batch_surprisal)
        
        # Loss only on masked patches
        loss = (loss * mask).sum() / mask.sum()
        return loss, surprisal
    
    def forward(self, imgs, mask_ratio=0.75, lambda_weight=0.0, alpha=3.0, gamma=1.0):
        """
        Forward pass
        Returns: loss, pred, mask, surprisal, latent
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, lambda_weight, alpha, gamma)
        pred = self.forward_decoder(latent, ids_restore)
        loss, surprisal = self.forward_loss(imgs, pred, mask)
        
        return loss, pred, mask, surprisal, latent
    
    def get_attention_maps(self, imgs):
        """Extract attention maps for visualization"""
        self.eval()
        with torch.no_grad():
            x = self.patch_embed(imgs)
            x = x + self.pos_embed[:, 1:, :]
            
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            attention_maps = []
            for blk in self.blocks:
                # Extract attention weights
                if hasattr(blk, 'attn'):
                    attn = blk.attn
                    B, N, C = x.shape
                    qkv = attn.qkv(blk.norm1(x)).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    attn_weights = (q @ k.transpose(-2, -1)) * attn.scale
                    attn_weights = attn_weights.softmax(dim=-1)
                    attention_maps.append(attn_weights.mean(dim=1))  # Average over heads
                
                x = blk(x) if not self.use_surprisal_attention else blk(x, None, 0.0)
            
        return attention_maps

