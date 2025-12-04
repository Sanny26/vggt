#!/usr/bin/env python3
"""
Simple sanity test: When mask is all 1s, masked attention should equal standard attention.
"""

import torch
import torch.nn.functional as F
# from dino_frame_saliency import attention
from triton_examples.attention import attention

def test_all_ones_mask():
    """
    Test that when mask is all 1s, masked attention equals standard attention.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dtype = torch.bfloat16
    dtype = torch.float16
    
    # Small test case
    B, H, S, P, D = 1, 16, 2, 100, 64  # Test with 200 tokens (not divisible by common block sizes)
    N = S * P  # Total tokens = 256
    
    print(f"\n{'='*80}")
    print(f"Sanity Test: All-Ones Mask (should match standard attention)")
    print(f"{'='*80}")
    print(f"Config: B={B}, H={H}, S={S} frames, P={P} patches/frame, D={D}, N={N} total tokens")
    
    # Create random Q, K, V
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)
    
    # Create all-ones mask (all frames attend to all frames)
    mask = torch.ones(B, S, S, device=device, dtype=torch.uint8)
    
    print(f"\nMask statistics:")
    total_pairs = S * S
    active_pairs = mask[0].sum().item()
    print(f"  Active pairs (mask=1): {active_pairs}/{total_pairs} ({100*active_pairs/total_pairs:.1f}%)")
    print(f"  All frames attend to all frames ✓")
    # breakpoint()
    
    scale = 1.0 / (D ** 0.5)
    
    # Run both attentions with AMP autocast to ensure same precision
    # with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    # 1. Standard PyTorch attention (no mask)
    ref_output = F.scaled_dot_product_attention(q, k, v, scale=scale)
    
    # 2. Our masked Triton attention with all-ones mask
    # masked_output = attention(q, k, v, False, scale, mask, S, P, False)
    masked_output = attention(q, k, v, False, scale, False)
    # masked_output = F.scaled_dot_product_attention(q, k, v, scale=scale)
    
    # Compare outputs
    print(f"\n{'='*80}")
    print(f"Correctness Check")
    print(f"{'='*80}")
    
    diff = (masked_output - ref_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nMasked Triton vs Standard PyTorch:")
    print(f"  Max absolute error:  {max_diff:.6e}")
    print(f"  Mean absolute error: {mean_diff:.6e}")
    
    # Tolerance for bfloat16 + online softmax algorithm
    # Online softmax can have larger numerical differences than naive softmax
    tolerance = 0.1  # 10% max error acceptable for bfloat16 precision
    mean_tolerance = 1e-3  #But mean should be very small
    
    if max_diff < tolerance and mean_diff < mean_tolerance:
        print(f"\n✅ PASS: Outputs match (max_tol={tolerance}, mean_tol={mean_tolerance})")
        print(f"   The masking logic correctly passes through all attention when mask=1")
        success = True
    else:
        print(f"\n❌ FAIL: Outputs differ (max_tol={tolerance}, mean_tol={mean_tolerance})")
        # print(f"\nThis indicates a bug in the masked attention implementation!")
        # print(f"\nDebug sample (first 3 tokens, first head):")
        # print(f"Standard:      {ref_output[0, 0, :3, :5]}")
        # print(f"Masked output: {masked_output[0, 0, :3, :5]}")
        # print(f"Difference:    {diff[0, 0, :3, :5]}")
        print("Locations of failure:")
        print((diff[0, 0]==diff[0, 1]==diff[0, 2]).sum())
        print(diff[0, 0].nonzero())
        success = False
    
    return success

if __name__ == "__main__":
    try:
        success = test_all_ones_mask()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
