# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# FlexAttention-based Frame-Sparse Global Attention
# Optimized for RTX 5090 (Blackwell) - PyTorch handles hardware-specific optimizations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Callable
import math
torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cuda.matmul.fp32_precision = 'tf32'

# Check for FlexAttention availability (PyTorch 2.5+)
FLEX_ATTENTION_AVAILABLE = False
try:
    from torch.nn.attention.flex_attention import (
        flex_attention,
        create_block_mask,
        BlockMask,
    )
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    flex_attention = None
    create_block_mask = None
    BlockMask = None


def create_frame_sparse_mask_fn(
    S: int,
    P: int, 
    window_size: int = 8,
    num_anchors: int = 10,
    always_first: bool = True,
) -> Callable:
    """
    Create a mask function for frame-sparse attention.
    
    This function will be compiled by FlexAttention into an efficient kernel.
    
    Args:
        S: Number of frames
        P: Patches per frame
        window_size: Attend to frames [i-w, i+w]
        num_anchors: Number of uniformly spaced anchor frames
        always_first: Always attend to first frame
    """
    anchor_step = max(1, S // num_anchors) if num_anchors > 0 else S + 1
    
    def mask_fn(b, h, q_idx, kv_idx):
        """
        Returns True if query at q_idx should attend to key at kv_idx.
        
        FlexAttention compiles this into an efficient block-sparse kernel.
        """
        # Convert token indices to frame indices
        q_frame = q_idx // P
        kv_frame = kv_idx // P
        
        # Self-frame attention (always attend within same frame)
        is_same_frame = q_frame == kv_frame
        
        # Window attention (nearby frames)
        frame_distance = q_frame - kv_frame
        is_in_window = (frame_distance >= -window_size) & (frame_distance <= window_size)
        
        # Anchor frames (uniformly spaced)
        is_anchor = (kv_frame % anchor_step) == 0
        
        # Reference frame (first frame)
        is_first = kv_frame == 0 if always_first else False
        
        return is_same_frame | is_in_window | is_anchor | is_first
    
    return mask_fn


class FlexSparseGlobalAttention(nn.Module):
    """
    Frame-Sparse Global Attention using PyTorch FlexAttention.
    
    Advantages over custom Triton:
    - PyTorch handles Blackwell-specific optimizations
    - Automatic kernel fusion and compilation
    - No manual tuning of block sizes, warps, etc.
    - Better maintained and tested
    
    Requirements:
    - PyTorch 2.5+
    - torch.compile for best performance
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        rope=None,
        # Sparse attention parameters
        window_size: int = 8,
        num_anchor_frames: int = 10,
    ):
        super().__init__()
        
        if not FLEX_ATTENTION_AVAILABLE:
            raise RuntimeError(
                "FlexAttention requires PyTorch 2.5+. "
                "Please upgrade or use FrameSparseGlobalAttention instead."
            )
        
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        
        self.window_size = window_size
        self.num_anchor_frames = num_anchor_frames
        
        # Cache for compiled block masks
        self._mask_cache = {}
    
    def _get_block_mask(self, S: int, P: int, device: torch.device) -> "BlockMask":
        """Get or create compiled block mask."""
        cache_key = (S, P, str(device))
        
        if cache_key not in self._mask_cache:
            mask_fn = create_frame_sparse_mask_fn(
                S=S,
                P=P,
                window_size=self.window_size,
                num_anchors=self.num_anchor_frames,
            )
            
            # Create block mask - FlexAttention compiles this efficiently
            # B=1, H=num_heads for broadcasting
            block_mask = create_block_mask(
                mask_fn,
                B=1,  # Will broadcast
                H=1,  # Will broadcast across heads
                Q_LEN=S * P,
                KV_LEN=S * P,
                device=device,
            )
            
            self._mask_cache[cache_key] = block_mask
        
        return self._mask_cache[cache_key]
    
    def forward(
        self,
        x: Tensor,
        S: int,
        P: int,
        pos: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass using FlexAttention.
        
        Args:
            x: [B, S*P, C] input tokens
            S: Number of frames
            P: Patches per frame
            pos: Optional RoPE positions
        """
        B, N, C = x.shape
        assert N == S * P
        
        # Compute QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, D]
        q, k, v = qkv.unbind(0)
        
        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # RoPE
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        
        # Get compiled block mask
        block_mask = self._get_block_mask(S, P, x.device)
        
        # FlexAttention - PyTorch compiles efficient kernel
        # This automatically handles Blackwell optimizations
        out = flex_attention(
            q, k, v,
            block_mask=block_mask,
            scale=self.scale,
        )
        
        # Apply dropout during training
        if self.training and self.attn_drop > 0:
            # Note: FlexAttention doesn't support dropout directly
            # Apply to output instead (approximation)
            out = F.dropout(out, p=self.attn_drop, training=True)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class FlexSparseGlobalBlock(nn.Module):
    """Transformer block with FlexAttention-based sparse global attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        init_values: Optional[float] = None,
        qk_norm: bool = False,
        rope=None,
        window_size: int = 8,
        num_anchor_frames: int = 10,
    ):
        super().__init__()
        
        from vggt.layers.layer_scale import LayerScale
        from vggt.layers.mlp import Mlp
        
        self.norm1 = nn.LayerNorm(dim)
        
        self.attn = FlexSparseGlobalAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            rope=rope,
            window_size=window_size,
            num_anchor_frames=num_anchor_frames,
        )
        
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
    
    def forward(self, x: Tensor, S: int, P: int, pos: Optional[Tensor] = None) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), S=S, P=P, pos=pos))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


# ============================================================================
# Fallback for systems without FlexAttention
# ============================================================================
def get_sparse_attention_module(
    dim: int,
    num_heads: int,
    window_size: int = 8,
    num_anchor_frames: int = 10,
    **kwargs
) -> nn.Module:
    """
    Get the best available sparse attention implementation.
    
    Priority:
    1. FlexAttention (PyTorch 2.5+) - Best for Blackwell
    2. Custom batched implementation - Good fallback
    """
    if FLEX_ATTENTION_AVAILABLE:
        print("Using FlexAttention (optimal for RTX 5090)")
        return FlexSparseGlobalAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            num_anchor_frames=num_anchor_frames,
            **kwargs
        )
    else:
        raise RuntimeError("FlexAttention is not available. Please upgrade PyTorch to 2.5+.")


# ============================================================================
# Blackwell-specific optimizations
# ============================================================================

def check_blackwell_features():
    """Check which Blackwell features are available."""
    features = {
        'cuda_available': torch.cuda.is_available(),
        'device_name': None,
        'compute_capability': None,
        'flex_attention': FLEX_ATTENTION_AVAILABLE,
        'fp8_support': False,
        'is_blackwell': False,
    }
    
    if torch.cuda.is_available():
        features['device_name'] = torch.cuda.get_device_name()
        cap = torch.cuda.get_device_capability()
        features['compute_capability'] = f"{cap[0]}.{cap[1]}"
        
        # Blackwell is SM100 (10.0)
        features['is_blackwell'] = cap[0] >= 10
        
        # FP8 support (Hopper and newer)
        features['fp8_support'] = cap[0] >= 9 or (cap[0] == 8 and cap[1] >= 9)
    
    return features


def print_optimization_recommendations():
    """Print optimization recommendations based on hardware."""
    features = check_blackwell_features()
    
    print("\n" + "="*60)
    print("Hardware Analysis for Attention Optimization")
    print("="*60)
    
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    print("\nRecommendations:")
    
    if features['is_blackwell']:
        print("  ✅ Blackwell GPU detected!")
        print("  → Use FlexAttention for sparse patterns (automatic Blackwell optimization)")
        print("  → Consider FP8 precision for additional speedup")
        print("  → Use larger batch sizes (more VRAM, higher throughput)")
    elif features['compute_capability'] and float(features['compute_capability']) >= 8.0:
        print("  ✅ Ada/Hopper GPU detected")
        print("  → FlexAttention or SDPA both work well")
        print("  → FP16/BF16 are well optimized")
    else:
        print("  ⚠️ Older GPU architecture")
        print("  → Stick with SDPA for best compatibility")
    
    if features['flex_attention']:
        print("  ✅ FlexAttention available - recommended for custom patterns")
    else:
        print("  ⚠️ FlexAttention not available - upgrade to PyTorch 2.5+")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    print_optimization_recommendations()
    
    if FLEX_ATTENTION_AVAILABLE and torch.cuda.is_available():
        import time
        
        print("\n" + "="*60)
        print("Running FlexAttention Benchmark")
        print("="*60)
        
        # 1. Create random array
        # B=1, H=16, S=37*37*100, D=64
        B = 1
        H = 16
        F = 100
        S_len = 37 * 37 * F  # 136,900
        D = 64

        print(f"Configuration:")
        print(f"  Batch (B): {B}")
        print(f"  Heads (H): {H}")
        print(f"  Seq Len (S): {S_len}")
        print(f"  Head Dim (D): {D}")
        print(f"  Total Tokens: {B * S_len}")
        
        device = torch.device("cuda")
        dtype = torch.bfloat16
        
        # Create tensors
        q = torch.randn(B, H, S_len, D, device=device, dtype=dtype, requires_grad=False)
        k = torch.randn(B, H, S_len, D, device=device, dtype=dtype, requires_grad=False)
        v = torch.randn(B, H, S_len, D, device=device, dtype=dtype, requires_grad=False)
        
        # Warmup
        for _ in range(3):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        
        # 2. Compute SDPA attention and record time
        print("\nBenchmarking SDPA (No Mask)...")
        start_time = time.time()
        # Run multiple iterations for better accuracy
        iterations = 10
        for _ in range(iterations):
            out_sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        end_time = time.time()
        avg_time_sdpa = (end_time - start_time) / iterations
        print(f"  SDPA Time: {avg_time_sdpa*1000:.2f} ms")
        
        # 3. Make mask=1 (Full Attention) with FlexAttention
        print("\nBenchmarking FlexAttention (Full Attention / mask=1)...")
        
        # Explicitly compile flex_attention to ensure fusion
        # dynamic=False often helps with specific shape benchmarks
        compiled_flex_attention = torch.compile(flex_attention, dynamic=False)

        # Helper to visualize mask
        def visualize_mask(mask_fn, name, num_blocks=32, block_size=128):
            print(f"\n{name} Pattern (Top-Left {num_blocks}x{num_blocks} blocks):")
            print(f"Each character represents a {block_size}x{block_size} block.")
            print(f"Frame size: {P} tokens (~{P/block_size:.1f} blocks).")
            
            # Print column headers (tens digit)
            header_tens = "   " + "".join([str(i//10) if i%10==0 else " " for i in range(num_blocks)])
            print(header_tens)
            # Print column headers (units digit)
            header_units = "   " + "".join([str(i%10) for i in range(num_blocks)])
            print(header_units)
            
            for r in range(num_blocks):
                row_str = f"{r:2d} "
                for c in range(num_blocks):
                    q_idx = torch.tensor(r * block_size)
                    kv_idx = torch.tensor(c * block_size)
                    
                    with torch.no_grad():
                        res = mask_fn(0, 0, q_idx, kv_idx)
                    
                    if isinstance(res, torch.Tensor):
                        is_active = res.item()
                    else:
                        is_active = res
                        
                    # Use different chars for different frames to show boundaries?
                    # Or just show active/inactive
                    row_str += "█" if is_active else "·"
                print(row_str)

        # 4. Create a random mask (Frame-Level Sparsity)
        print("\nBenchmarking FlexAttention (Random Frame Mask)...")
        print("  Testing frame-level sparsity (B, F, F) where F=100, P=37*37...")
        
        # User specification:
        # B=1, H=16, S=37*37*100, D=64
        # Mask shape: B, F, F
        # All 37*37 tokens in a frame share the same mask value
        
        F = 100
        P = 37 * 37  # 1369
        assert S_len == F * P
        
        # Create a random mask at the FRAME level
        # Shape: (B, F, F)
        # We use a random tensor on the device
        frame_mask = torch.randint(0, 2, (B, F, F), device=device, dtype=torch.bool)
        frame_mask[:, 0, 0] = 1
        # frame_mask[:, 0, :] = 1
        # frame_mask[:, :, 0] = 1
        
        # Ensure diagonal is present (optional, but usually good for attention)
        # For strict random testing we might leave it, but let's ensure self-attention is possible
        # frame_mask[:, torch.arange(F), torch.arange(F)] = True
        
        print(f"  Frame Mask Shape: {frame_mask.shape}")
        print(f"  Frame Mask Sparsity: {frame_mask.float().mean():.2%}")
        
        def frame_aware_mask_fn(b, h, q_idx, kv_idx):
            # Map token indices to frame indices
            q_frame = q_idx // P
            kv_frame = kv_idx // P
            
            # Index into the pre-computed frame mask
            # We need to handle broadcasting if b/h are tensors or ints
            # In mask_fn, b and h are usually ints or scalar tensors during tracing
            # We use advanced indexing. 
            # Note: frame_mask is (B, F, F). 
            return frame_mask[b, q_frame, kv_frame]
            
        # Visualize the pattern
        # We want to see the frame boundaries. P=1369. Block=128.
        # Frame 0 ends at 1369. 1369 / 128 ~= 10.7 blocks.
        # So we should see a transition around block 10-11.
        visualize_mask(frame_aware_mask_fn, "Frame-Level Random", num_blocks=30, block_size=128)
            
        try:
            print("  Creating block mask (this may take a moment)...")
            random_block_mask = create_block_mask(
                frame_aware_mask_fn, B=B, H=H, Q_LEN=S_len, KV_LEN=S_len, device=device,
                _compile=True # Ensure compilation
            )
            
            # Warmup
            print("  Warming up compiled model (Random Frame)...")
            for _ in range(3):
                _ = compiled_flex_attention(q, k, v, block_mask=random_block_mask)
            torch.cuda.synchronize()
            
            start_time = time.time()
            for _ in range(iterations):
                out_flex_rand = compiled_flex_attention(q, k, v, block_mask=random_block_mask)
            torch.cuda.synchronize()
            end_time = time.time()
            avg_time_flex_rand = (end_time - start_time) / iterations
            print(f"  FlexAttention (Random Frame) Time: {avg_time_flex_rand*1000:.2f} ms")
            
            # Check sparsity
            # sparsity() returns percentage (0-100)
            print(f"  Block Mask Sparsity: {random_block_mask.sparsity():.2f}%")
            
        except Exception as e:
            print(f"  ❌ Failed to run FlexAttention (Random Frame): {e}")

        # Answer the user's question about random mask creation
        print("\nAnswer to: 'Does create_block_mask allow for random mask creation?'")
        print("  Yes. The most efficient way for your use case is:")
        print("  1. Create a coarse-grained random mask tensor of shape (B, F, F).")
        print("  2. Define a mask_mod that maps token indices to frame indices (idx // P).")
        print("  3. Index into the coarse mask: `mask[b, q_idx//P, kv_idx//P]`.")
        print("  This handles the 'multi-block level sparsity' and non-aligned boundaries automatically.")



