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
        
        # ✅ EPOCH-LEVEL SURPRISAL CACHE for image-specific adaptive masking
        # Note: EMA (position-based) has been removed in favor of epoch cache (content-based)
        # Instead of using position-based EMA (which becomes flat across dataset),
        # we cache the actual surprisal from previous epoch for each image.
        # This provides true image-specific, content-based surprisal.
        #
        # Key insight: Batch-level cache fails because different images appear in each batch.
        # Epoch-level cache works because same images appear across epochs!
        #
        # Memory: ~100 MB for ImageNet-100 (130k images × 196 patches × 4 bytes)
        # Can be stored on CPU to save GPU memory.
        self.use_epoch_cache = True  # Enable/disable epoch cache
        self.surprisal_memory = None  # Will be initialized based on dataset size
        self.surprisal_initialized = None  # Track which images have cached surprisal
        
        self.initialize_weights()
        
    def initialize_epoch_cache(self, dataset_size: int, device: str = 'cpu', use_half_precision: bool = False):
        """
        Initialize epoch-level surprisal cache
        
        Args:
            dataset_size: Number of images in the dataset
            device: 'cpu' or 'cuda' - CPU recommended to save GPU memory
            use_half_precision: Use float16 instead of float32 (2x memory saving)
                                Default False to match standard research practices (float32)
        
        Memory usage:
            - Float32: dataset_size × 196 × 4 bytes
            - Float16: dataset_size × 196 × 2 bytes
            Examples:
                - CIFAR-100 (50k): 39 MB (float32) / 20 MB (float16)
                - ImageNet-100 (130k): 102 MB (float32) / 51 MB (float16)
                - Full ImageNet-1K (1.2M): 1 GB (float32) / 500 MB (float16)
        
        Note: Float32 is the default to match standard research practices.
              Use float16 only if memory is constrained (e.g., Full ImageNet-1K).
        """
        dtype = torch.float16 if use_half_precision else torch.float32
        bytes_per_element = 2 if use_half_precision else 4
        memory_mb = dataset_size * self.num_patches * bytes_per_element / 1e6
        
        print(f"Initializing epoch cache for {dataset_size} images...")
        print(f"  Memory: {memory_mb:.2f} MB on {device} ({dtype})")
        
        # Inform about memory usage (note: CPU memory is usually plentiful)
        memory_gb = memory_mb / 1000
        
        # Only warn for very large caches (> 5GB)
        # Typical systems have 16-32GB+ RAM, so even 1-2GB cache is fine
        if memory_gb > 5.0:
            print(f"  ⚠️  WARNING: Very large cache size ({memory_gb:.2f} GB)")
            print(f"     If you encounter memory issues, consider:")
            print(f"     - Use use_half_precision=True (saves 50% memory, {memory_gb/2:.2f} GB)")
            print(f"     - Or disable cache (set use_epoch_cache=False)")
        elif memory_gb > 2.0:
            print(f"  ℹ️  Info: Large cache size ({memory_gb:.2f} GB)")
            print(f"     This is typically fine on modern systems (>8GB RAM)")
            if not use_half_precision and memory_gb > 1.0:
                print(f"     Optional: use_half_precision=True to reduce to ~{memory_gb/2:.2f} GB")
        
        self.surprisal_memory = torch.zeros(
            dataset_size, self.num_patches,
            dtype=dtype,
            device=device
        )
        self.surprisal_initialized = torch.zeros(
            dataset_size,
            dtype=torch.bool,
            device=device
        )
        print("✅ Epoch cache initialized!")
    
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
        """
        Random masking: uniform random
        Returns: x_masked, mask, ids_restore, ids_keep
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)  # [N, L] - shuffled indices
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # [N, L] - inverse permutation
        
        ids_keep = ids_shuffle[:, :len_keep]  # [N, len_keep] - kept patch indices
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore, ids_keep
    
    def adaptive_masking_strategy(self, x, mask_ratio, alpha=3.0, gamma=1.0, surprisal_override=None):
        """
        Adaptive masking based on surprisal
        p_mask(i) = sigmoid(alpha - gamma * S_i)
        High surprisal → low mask probability → learn more often
        
        Args:
            x: [N, L, D] input tokens
            mask_ratio: target masking ratio
            alpha: sigmoid offset parameter
            gamma: surprisal weight
            surprisal_override: [N, L] optional cached surprisal from previous iteration
        """
        N, L, D = x.shape
        
        # ✅ Use epoch-cached surprisal if available (image-specific, content-based)
        # If not available (first epoch or cache disabled), return None to use random masking
        if surprisal_override is not None and self.use_epoch_cache:
            surprisal = surprisal_override  # [B, L] - from previous epoch (masked patches only, unmasked = 0)
        else:
            # No cache available - return None to fallback to random masking
            # This happens in first epoch when cache is not yet built
            return None
        
        # Compute masking probabilities
        # p_mask = sigmoid(alpha - gamma * surprisal)
        # - surprisal > 0 (high): patch was difficult to reconstruct → low mask prob → learn more often
        # - surprisal = 0: patch was NOT masked (not yet learned) → high mask prob → should mask to learn
        mask_probs = torch.sigmoid(alpha - gamma * surprisal)
        
        # Sample masks
        mask = torch.bernoulli(mask_probs).to(x.device)
        
        # Ensure we mask approximately mask_ratio tokens
        target_masked = int(L * mask_ratio)
        current_masked = mask.sum(dim=1, keepdim=True)  # [N, 1] - long tensor
        adjustment = (target_masked - current_masked).squeeze(1)  # [N] - long tensor
        
        # ✅ FIXED: Adjust mask to match target ratio (type-safe: ensure int casting)
        for i in range(N):
            adj = int(round(adjustment[i].item()))  # Convert to Python int with rounding
            if adj > 0:
                # Need to mask more
                unmasked_idx = (mask[i] == 0).nonzero(as_tuple=True)[0]
                if len(unmasked_idx) > 0:
                    n_to_mask = min(adj, len(unmasked_idx))
                    n_to_mask = int(n_to_mask)  # Ensure int for indexing
                    perm = torch.randperm(len(unmasked_idx), device=x.device)
                    to_mask = unmasked_idx[perm[:n_to_mask]]
                    mask[i, to_mask] = 1
            elif adj < 0:
                # Need to unmask some
                masked_idx = (mask[i] == 1).nonzero(as_tuple=True)[0]
                if len(masked_idx) > 0:
                    n_to_unmask = min(-adj, len(masked_idx))
                    n_to_unmask = int(n_to_unmask)  # Ensure int for indexing
                    perm = torch.randperm(len(masked_idx), device=x.device)
                    to_unmask = masked_idx[perm[:n_to_unmask]]
                    mask[i, to_unmask] = 0
        
        # ✅ FIXED: Create ids_keep and ids_restore per sample with proper inverse permutation
        # Sample-by-sample: ids_shuffle = [keep_idx..., mask_idx...]
        # ids_restore[ids_shuffle[j]] = j (inverse permutation)
        ids_keep_list = []
        ids_restore_list = []
        
        for i in range(N):
            # Get kept and masked indices for this sample
            keep_idx = (mask[i] == 0).nonzero(as_tuple=True)[0]  # [N_keep]
            mask_idx = (mask[i] == 1).nonzero(as_tuple=True)[0]  # [N_masked]
            
            # ✅ FIXED: Create ids_shuffle and ids_restore matching random_masking logic
            # ids_shuffle: [keep_idx..., mask_idx...] - concatenated order
            # This represents the shuffled sequence: [x[keep_idx[0]], ..., x[keep_idx[N_keep-1]], mask, ..., mask]
            ids_shuffle = torch.cat([keep_idx, mask_idx], dim=0)  # [L]
            
            # ids_restore: inverse permutation of ids_shuffle
            # ids_restore[j] = "position in ids_shuffle where original patch j appears"
            # This is equivalent to: ids_restore = torch.argsort(torch.argsort(ids_shuffle))
            # But we compute it directly for clarity:
            ids_restore = torch.zeros(L, dtype=torch.long, device=x.device)
            for pos, orig_idx in enumerate(ids_shuffle):
                ids_restore[orig_idx] = pos
            
            ids_keep_list.append(keep_idx)
            ids_restore_list.append(ids_restore)
        
        # Stack ids_restore: [N, L]
        ids_restore = torch.stack(ids_restore_list, dim=0)
        
        # ✅ FIXED: Handle variable-length ids_keep - use actual lengths per sample
        # Note: All samples should have similar N_keep due to mask_ratio adjustment,
        # but we handle it properly to avoid padding issues
        max_keep = max(len(k) for k in ids_keep_list) if ids_keep_list else 0
        
        # Extract kept patches per sample (avoid padding issues)
        x_masked_list = []
        ids_keep_padded = torch.zeros(N, max_keep, dtype=torch.long, device=x.device) if max_keep > 0 else torch.zeros(N, 0, dtype=torch.long, device=x.device)
        
        for i, k in enumerate(ids_keep_list):
            # Extract kept patches for this sample using actual indices
            x_masked_list.append(torch.index_select(x[i], dim=0, index=k))  # [len(k), D]
            # Store ids_keep for later use (padded to max_keep)
            if len(k) > 0:
                ids_keep_padded[i, :len(k)] = k
        
        # ✅ FIXED: Stack x_masked - ensure consistent length across samples
        # After mask adjustment, all samples should have same N_keep
        if len(x_masked_list) > 0:
            lengths = [t.shape[0] for t in x_masked_list]
            # Ensure all samples have the same N_keep (should be after adjustment)
            if len(set(lengths)) == 1:
                # All same length - can stack directly
                x_masked = torch.stack(x_masked_list, dim=0)  # [N, N_keep, D]
                # ids_keep: remove padding, keep only actual N_keep
                actual_N_keep = lengths[0]
                ids_keep = ids_keep_padded[:, :actual_N_keep]  # [N, actual_N_keep]
            else:
                # Variable length detected (shouldn't happen after adjustment)
                # Pad to max length for consistency
                max_len = max(lengths)
                x_masked = torch.zeros(N, max_len, D, device=x.device, dtype=x.dtype)
                for i, t in enumerate(x_masked_list):
                    x_masked[i, :t.shape[0]] = t
                # Use padded ids_keep (already padded to max_keep which should equal max_len)
                ids_keep = ids_keep_padded[:, :max_len]  # [N, max_len]
        else:
            x_masked = torch.zeros(N, 0, D, device=x.device, dtype=x.dtype)
            ids_keep = ids_keep_padded
        
        return x_masked, mask, ids_restore, ids_keep
    
    def forward_encoder(self, x, mask_ratio, lambda_weight=0.0, alpha=3.0, gamma=1.0, surprisal_override=None):
        """
        Encoder with optional surprisal-weighted attention
        
        Args:
            x: [N, C, H, W] input images
            mask_ratio: masking ratio
            lambda_weight: surprisal attention weight
            alpha: adaptive masking offset
            gamma: adaptive masking surprisal weight
            surprisal_override: [N, L] optional cached surprisal from previous iteration
        """
        # Patch embedding
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking
        ids_keep = None  # Will be set by masking strategy
        if self.adaptive_masking and self.training:
            # ✅ Pass cached surprisal to adaptive masking
            mask_result = self.adaptive_masking_strategy(
                x, mask_ratio, alpha, gamma, surprisal_override=surprisal_override
            )
            if mask_result is not None:
                x, mask, ids_restore, ids_keep = mask_result
            else:
                # Fallback to random masking if no cache available (first epoch)
                x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        else:
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        
        # Prepare surprisal bias BEFORE adding cls token
        # ✅ Patch-wise surprisal bias: each patch gets its own surprisal value
        surprisal_bias = None
        if self.use_surprisal_attention and lambda_weight > 0:
            B = x.shape[0]
            N_keep = x.shape[1]  # Number of kept patches (before cls token)
            
            # ✅ Use cached surprisal per patch (image-specific, patch-specific, content-based)
            # Each patch gets its own surprisal value independently
            if surprisal_override is not None:
                # surprisal_override: [B, L] - each image has its own surprisal per patch
                # ✅ FIXED: Use ids_keep directly from masking strategy to ensure token order consistency
                # ids_keep: [B, N_keep] - indices of kept patches in original order
                # This ensures the surprisal mapping matches the actual kept patch order used in encoder
                # ✅ FIXED: Ensure ids_keep length matches actual N_keep
                # ids_keep should be [B, N_keep] after mask adjustment, but verify consistency
                actual_ids_keep_len = ids_keep.shape[1] if ids_keep.dim() > 1 else N_keep
                if actual_ids_keep_len != N_keep:
                    # Length mismatch - trim or handle appropriately
                    # This should not happen after mask adjustment, but handle gracefully
                    actual_N_keep = min(actual_ids_keep_len, N_keep)
                    ids_keep_trimmed = ids_keep[:, :actual_N_keep]
                    N_keep_use = actual_N_keep
                else:
                    ids_keep_trimmed = ids_keep
                    N_keep_use = N_keep
                
                kept_patch_surprisal = []
                for b in range(B):
                    # Use ids_keep[b] directly (ensures correct ordering with encoder tokens)
                    kept_idx = ids_keep_trimmed[b, :N_keep_use]  # [N_keep_use] - indices of kept patches
                    # Extract surprisal for kept patches using ids_keep
                    patch_surprisal = surprisal_override[b, kept_idx]  # [N_keep_use]
                    kept_patch_surprisal.append(patch_surprisal)
                
                # Stack: [B, N_keep_use] - each patch has its own surprisal value
                kept_patch_surprisal = torch.stack(kept_patch_surprisal, dim=0)  # [B, N_keep_use]
                
                # Ensure dimension matches x.shape[1] (actual N_keep)
                if kept_patch_surprisal.shape[1] != N_keep:
                    # Trim or pad to match actual encoder input length
                    if kept_patch_surprisal.shape[1] < N_keep:
                        # Pad with zeros (shouldn't happen, but handle it)
                        padding = torch.zeros(B, N_keep - kept_patch_surprisal.shape[1], 
                                            device=kept_patch_surprisal.device)
                        kept_patch_surprisal = torch.cat([kept_patch_surprisal, padding], dim=1)
                    else:
                        # Trim to match (shouldn't happen either)
                        kept_patch_surprisal = kept_patch_surprisal[:, :N_keep]
                
                # ✅ FIXED: Normalize surprisal to prevent scale issues and softmax saturation
                # Z-score normalization per batch + tanh for bounded range [-1, 1]
                # This prevents unbounded values that could saturate softmax
                batch_mean = kept_patch_surprisal.mean(dim=1, keepdim=True)  # [B, 1]
                batch_std = kept_patch_surprisal.std(dim=1, keepdim=True) + 1e-8  # [B, 1]
                kept_patch_surprisal = (kept_patch_surprisal - batch_mean) / batch_std  # Z-score
                kept_patch_surprisal = torch.tanh(kept_patch_surprisal)  # Bounded to [-1, 1]
                
                # For cls token: cls token is not a real patch, so it doesn't have its own surprisal.
                # We use zero bias for cls token (no surprisal-based attention modification).
                # Only real patches get patch-specific surprisal bias.
                cls_bias = torch.zeros(B, 1, device=x.device)  # [B, 1] - no bias for cls token
                
                # Concatenate: cls token (zero bias) + kept patches surprisal (patch-wise!)
                surprisal_bias = torch.cat([cls_bias, kept_patch_surprisal], dim=1)  # [B, 1 + N_keep]
                # Attention formula: A_ij = softmax(Q_i K_j^T / sqrt(d) + λ * S_j)
                # - When attending to cls token (j=0): uses zero bias (no surprisal modification)
                # - When attending to patch k (j=k): uses patch_k_bias (patch-specific surprisal)
                # This makes sense: only real patches have reconstruction difficulty (surprisal)
            else:
                # First epoch: no cache available, use zero (no bias)
                surprisal_bias = torch.zeros(B, 1 + N_keep, device=x.device)
        
        # Add cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # Now x shape: [B, 1 + N_keep, D]
        # surprisal_bias shape: [B, 1 + N_keep] - matches!
        
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
        # ✅ IMPORTANT: MAE only learns to reconstruct MASKED patches!
        # Unmasked patches have original information in encoder, so their
        # reconstruction error is NOT learned and cannot be trusted.
        # Therefore, we only store surprisal for masked patches (where learning happened).
        #
        # For adaptive masking in next epoch:
        # - Surprisal > 0: This patch was masked and learned → difficult to reconstruct
        #                  → low mask prob (learn more often)
        # - Surprisal = 0: This patch was NOT masked → not yet learned
        #                  → high mask prob (should be masked to learn)
        with torch.no_grad():
            # Only store surprisal for masked patches (where learning occurred)
            surprisal = loss.detach() * mask  # [N, L] - masked patches only!
            # Note: Surprisal is saved to epoch cache in forward() method
        
        # Loss only on masked patches
        loss = (loss * mask).sum() / mask.sum()
        return loss, surprisal  # surprisal: [N, L] - masked patches only (unmasked = 0)
    
    def forward(self, imgs, mask_ratio=0.75, lambda_weight=0.0, alpha=3.0, gamma=1.0, 
                image_ids=None):
        """
        Forward pass with epoch-level surprisal caching
        
        Args:
            imgs: [N, C, H, W] input images
            mask_ratio: masking ratio
            lambda_weight: surprisal attention weight
            alpha: adaptive masking offset
            gamma: adaptive masking surprisal weight
            image_ids: [N] image indices for epoch caching (optional)
        
        Returns:
            loss: reconstruction loss
            pred: [N, L, p*p*3] predictions
            mask: [N, L] mask (0=keep, 1=remove)
            surprisal: [N, L] surprisal (reconstruction error for masked patches only, unmasked = 0)
            latent: [N, N_keep+1, D] latent representations
        """
        # ✅ Get cached surprisal from epoch memory if available
        # Handle partial initialization: use cache for initialized images, zeros for others
        surprisal_override = None
        if image_ids is not None and self.use_epoch_cache and self.surprisal_memory is not None:
            # Move image_ids to same device as memory (usually CPU) for indexing
            image_ids_cpu = image_ids.cpu() if image_ids.device != self.surprisal_memory.device else image_ids
            
            # Check which images are initialized
            initialized = self.surprisal_initialized[image_ids_cpu]  # [B] bool
            
            if initialized.any():  # At least one image is initialized
                # Get from memory (initialized images have data, uninitialized have zeros)
                surprisal_override = self.surprisal_memory[image_ids_cpu].to(imgs.device)
                # Convert to float32 if needed (for computation)
                if surprisal_override.dtype == torch.float16:
                    surprisal_override = surprisal_override.float()
                # Note: Uninitialized images will have zeros (all patches surprisal = 0)
                # This is correct: surprisal = 0 means "not yet learned" → high mask prob (random-like)
        
        # Forward pass
        latent, mask, ids_restore = self.forward_encoder(
            imgs, mask_ratio, lambda_weight, alpha, gamma, 
            surprisal_override=surprisal_override
        )
        pred = self.forward_decoder(latent, ids_restore)
        loss, surprisal = self.forward_loss(imgs, pred, mask)
        
        # ✅ Save surprisal to epoch memory (during training)
        if self.training and image_ids is not None and self.use_epoch_cache and self.surprisal_memory is not None:
            # Move to same device as memory (usually CPU) for indexing
            image_ids_cpu = image_ids.cpu() if image_ids.device != self.surprisal_memory.device else image_ids
            surprisal_cpu = surprisal.detach().to(self.surprisal_memory.device)
            
            # Convert to cache dtype (float16 if cache uses half precision)
            if self.surprisal_memory.dtype == torch.float16:
                surprisal_cpu = surprisal_cpu.half()
            else:
                surprisal_cpu = surprisal_cpu.float()
            
            self.surprisal_memory[image_ids_cpu] = surprisal_cpu
            self.surprisal_initialized[image_ids_cpu] = True
        
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

