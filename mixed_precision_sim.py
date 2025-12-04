import torch
import torch.nn as nn
import torch.nn.functional as F
from vggt.layers.attention import Attention

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
        
        # We don't need to copy weights, we'll use the original layer's weights
        # But we need to register them as submodules/parameters if we want them to be trainable/saveable
        # However, for a monkey patch simulation, we can just reference them.
        # To be safe and allow gradients to flow, we should probably just use the original layer's components.
        
    def _simulate_quantization(self, k: torch.Tensor) -> torch.Tensor:
        """
        Simulates mixed precision by casting 'unimportant' tokens to FP8 and back.
        Uses a simple alternating 0, 1, 0, 1 pattern for the mask.
        """
        B, H, N, D = k.shape
        # breakpoint()
        # Create alternating mask: 0, 1, 0, 1...
        # 1 means KEEP precision (High Saliency)
        # 0 means REDUCE precision (Low Saliency)
        mask = torch.arange(N, device=k.device) % 2
        
        # Reshape for broadcasting: [1, N, 1, 1]
        mask = mask.view(1, 1, N, 1).expand(B, H, N, D)
        
        # Clone k to avoid in-place modification issues if any
        # k_noisy = k.clone()
        k_noisy = k
        
        # Identify low saliency tokens
        low_saliency = (mask == 0)
        # breakpoint()
        # Simulate FP8 quantization
        # We use float8_e4m3fn as it's a common format for weights/activations
        if low_saliency.any():
            k_low = k[low_saliency]
            # breakpoint()
            k_low_fp8 = k_low.to(torch.float8_e4m3fn)
            k_low_back = k_low_fp8.to(k.dtype)
            k_noisy[low_saliency] = k_low_back
            
        return k_noisy

    def forward(self, x: torch.Tensor, pos=None, attn_bias=None) -> torch.Tensor:
        # Replicate the logic from vggt.layers.attention.Attention.forward
        # but inject noise into K
        
        B, N, C = x.shape
        # breakpoint()
        qkv = self.original_layer.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        # 3, B, H, N, D
        q, k, v = qkv.unbind(0)
        q, k = self.original_layer.q_norm(q), self.original_layer.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
            
        # --- INJECT NOISE HERE ---
        # shape of q and k: B, H, N, D
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

def apply_monkey_patch(model):
    """
    Applies the SimulatedMixedPrecisionAttention wrapper to the model.
    Sandwich Schedule:
    - Skip layers 0-2
    - Patch layers 3-18
    - Skip layers 19-24 (assuming 24 layers)
    """
    print("Applying Mixed-Precision Monkey Patch...")

    patches_applied = 0
    
    # blocks names: model.aggregator.frame_blocks
    # global_blocks names: model.aggregator.global_blocks
    blocks = None
    try:
        blocks = model.aggregator.frame_blocks + model.aggregator.global_blocks
    except AttributeError:
        raise ValueError("Could not find 'frame_blocks' or 'global_blocks' attribute.")
    # Iterate over blocks with index
    for i, block in enumerate(blocks):
        if 0 <= i <= 47:
            if hasattr(block, 'attn') and isinstance(block.attn, Attention):
                print(f"Patching layer {i} attention")
                original_attn = block.attn
                block.attn = SimulatedMixedPrecisionAttention(original_attn)
                patches_applied += 1
            elif hasattr(block, 'attention') and isinstance(block.attention, Attention): # Just in case
                print(f"Patching layer {i} attention")
                original_attn = block.attention
                block.attention = SimulatedMixedPrecisionAttention(original_attn)
                patches_applied += 1
        else:
            print(f"Skipping layer {i}")

    
            
    print(f"Total layers patched: {patches_applied}")
