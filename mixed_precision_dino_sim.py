import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import Optional, Tuple, List
from vggt.layers.attention import Attention
from vggt.models.aggregator import Aggregator

# ============================================================================
# 1. Simulated Mixed Precision Attention
# ============================================================================

def quantize_fp4_blockwise(x: torch.Tensor, block_size: int = 64):
    """
    block wise fp4 quantization 
    x: [..., D]
    block_size: number of elements per block along the last dimension
    """
    qmax = 7
    eps = 1e-8

    orig_shape = x.shape
    D = orig_shape[-1]
    assert D % block_size == 0, "last dim must be divisible by block_size"

    # Flatten everything except last dim
    x_flat = x.reshape(-1, D)          # [N, D]
    x_blocks = x_flat.view(-1, block_size)  # [N * (D / B), B]

    max_vals = x_blocks.abs().amax(dim=1, keepdim=True)  # [num_blocks, 1]
    scales = (max_vals / qmax).clamp(min=eps)

    q = torch.round(x_blocks / scales).clamp(-8, 7).to(torch.int8)
    x_deq_blocks = (q.float() * scales).view_as(x_blocks)

    x_deq = x_deq_blocks.view_as(x_flat).reshape(orig_shape)
    return x_deq, q, scales


class SimulatedMixedPrecisionAttention(nn.Module):
    def __init__(self, original_layer: Attention):
        super().__init__()
        self.original_layer = original_layer
        # Copy attributes needed for forward pass
        self.num_heads = original_layer.num_heads
        self.head_dim = original_layer.head_dim
        self.scale = original_layer.scale
        self.attn_impl = original_layer.attn_impl
        self.rope = original_layer.rope
        
        # Mask to be provided by the Aggregator
        # Should be set before forward pass
        self.mask = None
        self._warned_fallback = False

    def _simulate_quantization(self, k: torch.Tensor) -> torch.Tensor:
        """
        Simulates mixed precision by casting 'unimportant' tokens to FP8 and back.
        """
        B, H, N, D = k.shape
        
        mask = self.mask
        
        if mask is None:
            if not self._warned_fallback:
                print("Warning: Mask not provided to SimulatedMixedPrecisionAttention. Falling back to 0, 1 alternative pattern.")
                self._warned_fallback = True
            
            # Fallback: Alternating 0, 1, 0, 1...
            mask = torch.arange(N, device=k.device) % 2
            # Reshape for broadcasting: [1, N, 1, 1] -> [B, H, N, D]
            mask = mask.view(1, 1, N, 1).expand(B, H, N, D)
        else:
            # Mask provided. Expected shape: [B, N] (0 or 1)
            # We need to expand it to [B, H, N, D]
            # Ensure mask is on the correct device
            if mask.device != k.device:
                mask = mask.to(k.device)
                
            # Check shape compatibility
            if mask.shape[0] != B or mask.shape[1] != N:
                # This might happen if batch size changes or logic is off
                # Fallback or error? Let's try to broadcast if possible, or error.
                # For now, assume shape is correct or broadcastable.
                pass

            # Expand: [B, N] -> [B, 1, N, 1] -> [B, H, N, D]
            mask = mask.view(B, 1, N, 1).expand(B, H, N, D)

        # k_noisy = k.clone()
        k_noisy = k
        
        # Identify low saliency tokens (mask == 0)
        low_saliency = (mask == 0)
        
        # Simulate FP8 quantization
        # We use float8_e4m3fn as it's a common format for weights/activations
        # if low_saliency.any():
        #     k_low = k[low_saliency]
        #     k_low_fp8 = k_low.to(torch.float8_e4m3fn)
        #     k_low_back = k_low_fp8.to(k.dtype)
        #     k_noisy[low_saliency] = k_low_back

        # Simulate fp4 quantization
        if low_saliency.any():
            k_low = k[low_saliency]
            k_low_back, q_low, s_low = quantize_fp4_blockwise(k_low, block_size=64)
            k_noisy[low_saliency] = k_low_back
            
        return k_noisy

    def forward(self, x: torch.Tensor, pos=None, attn_bias=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.original_layer.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.original_layer.q_norm(q), self.original_layer.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
            
        # --- INJECT NOISE HERE ---
        k = self._simulate_quantization(k)
        q = self._simulate_quantization(q)
        # -------------------------

        if self.attn_impl == 'sdpa':
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.original_layer.attn_drop.p if self.training else 0.0)
        elif self.attn_impl == 'triton':
            from vggt.layers.triton_attention import run_triton_attention
            x = run_triton_attention(q, k, v, self.scale)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.original_layer.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.original_layer.proj(x)
        x = self.original_layer.proj_drop(x)
        return x

# ============================================================================
# 2. Custom Aggregator Logic
# ============================================================================
def slice_expand_and_flatten(token_tensor, B, S):
    """
    Helper function from vggt.models.aggregator
    """
    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)
    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined

def custom_aggregator_forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
    """
    Custom forward pass for Aggregator that computes saliency mask and passes it to attention layers.
    """
    B, S, C_in, H, W = images.shape

    if C_in != 3:
        raise ValueError(f"Expected 3 input channels, got {C_in}")

    # Normalize images and reshape for patch embed 
    images = (images - self._resnet_mean) / self._resnet_std

    # Reshape to [B*S, C, H, W] for patch embedding
    images = images.view(B * S, C_in, H, W)
    
    # --- Custom Logic: Get Patch Tokens and Compute Mask ---
    patch_tokens = self.patch_embed(images)
    if isinstance(patch_tokens, dict):
        cls_tokens = patch_tokens["x_norm_clstoken"]
        patch_tokens = patch_tokens["x_norm_patchtokens"]

    # patch_tokens shape: [B*S, P, C]
    # Compute Saliency Mask
    # We want to compute saliency per image, but respect the batch dimension B.
    # So we treat it as [B, S*P, C] where M=S (images per video), N=P (patches per image).
    bs_dim, p_dim, c_dim = patch_tokens.shape # bs_dim = B*S
    
    # Compute patch saliency (L2 norm) per patch: [B*S, P] - Tried, too similar values across patches, not very useful
    # patch_saliency = patch_tokens.norm(dim=-1)

    # compute cosine similarity between cls token and patch tokens
    cls_tokens = cls_tokens.unsqueeze(1)
    # patch_tokens = patch_tokens.unsqueeze(0)
    patch_saliency = torch.nn.functional.cosine_similarity(cls_tokens, patch_tokens, dim=-1) # BS, P

    # Select top 50% patches per image to be High Precision
    high_precision_ratio = 0.5
    k = int(p_dim * (1 - high_precision_ratio))
    
    # save the saliency graph for visualization
    # threshold = torch.kthvalue(patch_saliency, k, dim=2, keepdim=True).values
    # threshold = threshold.squeeze(-1)
    # patch_saliency = patch_saliency.view(-1, p_dim).cpu().numpy()
    # import seaborn as sns
    # sns.heatmap(patch_saliency)
    # plt.savefig("saliency.png")
    # plt.close()
    # for b in range(patch_saliency.shape[0]):
    #     sns.kdeplot(patch_saliency[b], alpha=0.3)
    # plt.axvline(threshold.mean().cpu().numpy(), color="black", linestyle="--", linewidth=2)
    # plt.savefig("saliency_kde.png")
    # plt.close()
    
    if k > 0:
        # Get the k-th smallest value as threshold for each image
        threshold = torch.kthvalue(patch_saliency, k, dim=-1, keepdim=True).values # [B*S, 1]
        # Keep patches with saliency > threshold (Top 50%)
        mask_patches = (patch_saliency > threshold).long()
    else:
        # If k=0, it means we want 100% high precision (ratio=1.0)
        # or p_dim is small.
        mask_patches = torch.ones_like(patch_saliency, dtype=torch.long)
    # We also have Special Tokens (Camera + Register), they should be High Precision (1)
    num_special = self.patch_start_idx # 1 + num_register_tokens
    special_mask = torch.ones((bs_dim, num_special), device=mask_patches.device, dtype=mask_patches.dtype)
    
    # Full mask for Frame Attention: [B*S, P_total]
    frame_mask = torch.cat([special_mask, mask_patches], dim=1)
    
    # Full mask for Global Attention: [B, S*P_total]
    # We need to reshape frame_mask to [B, S, P_total] then flatten S and P_total
    # frame_mask is [B*S, P_total]
    global_mask = frame_mask.view(B, S, -1).view(B, -1)
    
    # Set masks on attention layers
    # Frame blocks
    for block in self.frame_blocks:
        if hasattr(block, 'attn') and isinstance(block.attn, SimulatedMixedPrecisionAttention):
            block.attn.mask = frame_mask
        elif hasattr(block, 'attention') and isinstance(block.attention, SimulatedMixedPrecisionAttention):
            block.attention.mask = frame_mask
            
    # Global blocks
    for block in self.global_blocks:
        if hasattr(block, 'attn') and isinstance(block.attn, SimulatedMixedPrecisionAttention):
            block.attn.mask = global_mask
        elif hasattr(block, 'attention') and isinstance(block.attention, SimulatedMixedPrecisionAttention):
            block.attention.mask = global_mask
            
    # -------------------------------------------------------
    _, P, C = patch_tokens.shape

    # Expand camera and register tokens to match batch size and sequence length
    camera_token = slice_expand_and_flatten(self.camera_token, B, S)
    register_token = slice_expand_and_flatten(self.register_token, B, S)

    # Concatenate special tokens with patch tokens
    tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

    pos = None
    if self.rope is not None:
        pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

    if self.patch_start_idx > 0:
        # do not use position embedding for special tokens (camera and register tokens)
        # so set pos to 0 for the special tokens
        pos = pos + 1
        pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
        pos = torch.cat([pos_special, pos], dim=1)

    # update P because we added special tokens
    _, P, C = tokens.shape

    frame_idx = 0
    global_idx = 0
    output_list = []

    for _ in range(self.aa_block_num):
        for attn_type in self.aa_order:
            if attn_type == "frame":
                # print('Processing frame attention')
                tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                    tokens, B, S, P, C, frame_idx, pos=pos
                )
            elif attn_type == "global":
                # print('Processing global attention')
                tokens, global_idx, global_intermediates = self._process_global_attention(
                    tokens, B, S, P, C, global_idx, pos=pos
                )
            else:
                raise ValueError(f"Unknown attention type: {attn_type}")

        for i in range(len(frame_intermediates)):
            # concat frame and global intermediates, [B x S x P x 2C]
            concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
            output_list.append(concat_inter)

    del concat_inter
    del frame_intermediates
    del global_intermediates
    return output_list, self.patch_start_idx


# ============================================================================
# 4. Monkey Patch Application
# ============================================================================

def apply_monkey_patch(model):
    """
    Applies the SimulatedMixedPrecisionAttention wrapper and Custom Aggregator logic to the model.
    """
    print("Applying Mixed-Precision Monkey Patch with DINO Saliency...")

    patches_applied = 0
    
    blocks = None
    try:
        blocks = model.aggregator.frame_blocks + model.aggregator.global_blocks
    except AttributeError:
        raise ValueError("Could not find 'frame_blocks' or 'global_blocks' attribute.")
    
    # Patch Attention Layers
    for i, block in enumerate(blocks):
        # Check for 'attn' or 'attention'
        target_attr = None
        if hasattr(block, 'attn') and isinstance(block.attn, Attention):
            target_attr = 'attn'
        elif hasattr(block, 'attention') and isinstance(block.attention, Attention):
            target_attr = 'attention'
            
        if target_attr:
            # print(f"Patching layer {i} attention ({target_attr})")
            original_attn = getattr(block, target_attr)
            setattr(block, target_attr, SimulatedMixedPrecisionAttention(original_attn))
            patches_applied += 1
        else:
            print(f"Skipping layer {i} (no attention found)")

    print(f"Total layers patched: {patches_applied}")
    
    # Patch Aggregator Forward Method
    print("Patching Aggregator forward method...")
    # Bind the custom forward method to the aggregator instance
    model.aggregator.forward = custom_aggregator_forward.__get__(model.aggregator, Aggregator)
    print("Aggregator forward method patched.")

if __name__ == "__main__":
    print("This script provides classes and functions for simulated mixed precision with DINO saliency.")
    print("Import 'apply_monkey_patch' and call it on your VGGT model.")
