#!/usr/bin/env python3
"""
Validation test for masked attention correctness using real VGGT weights.
Compares the masked Triton attention kernel against PyTorch reference.
"""

import torch
import torch.nn.functional as F
import argparse
import os
from dino_frame_saliency import attention, compute_frame_saliency
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

def load_vggt_attention_layer(layer_idx: int = 0):
    """Load VGGT model and extract a specific global attention layer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading VGGT model...")
    model = VGGT()
    
    try:
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        state_dict = torch.hub.load_state_dict_from_url(_URL, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"✅ Loaded pretrained weights")
    except Exception as e:
        print(f"⚠️  Could not load weights (using random init): {e}")
    
    model.to(device=device)
    model.eval()
    
    # Get the specified global attention layer
    num_global_blocks = len(model.aggregator.global_blocks)
    if layer_idx >= num_global_blocks:
        raise ValueError(f"Layer index {layer_idx} out of range (0-{num_global_blocks-1})")
    
    attn_layer = model.aggregator.global_blocks[layer_idx].attn
    
    print(f"Extracted global attention layer {layer_idx}/{num_global_blocks-1}")
    print(f"  Config: num_heads={attn_layer.num_heads}, head_dim={attn_layer.head_dim}")
    
    return attn_layer, model

def test_masked_attention_with_real_weights(layer_idx: int = 0, num_frames: int = 20):
    """
    Test that masked attention produces correct results using real VGGT attention weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # Load attention layer
    attn_layer, model = load_vggt_attention_layer(layer_idx)
    attn_layer = attn_layer.to(dtype=dtype)
    
    # Load some real images to get realistic Q, K, V
    print(f"\nLoading test images...")
    images_dir = os.path.join("examples", "south", "person-hall", "images")
    image_names = [os.path.join(images_dir, name) for name in sorted(os.listdir(images_dir))]
    import random
    random.seed(42)
    random.shuffle(image_names)
    image_names = image_names[:num_frames]
    
    images = load_and_preprocess_images(image_names).to(device)
    images = images.unsqueeze(0)  # [1, S, 3, H, W]
    B, S, C_in, H, W = images.shape
    
    print(f"Loaded {S} frames: {images.shape}")
    
    # Get patch embeddings
    images_norm = (images - model.aggregator._resnet_mean.to(images.dtype)) / model.aggregator._resnet_std.to(images.dtype)
    images_flat = images_norm.view(B * S, C_in, H, W)
    
    with torch.no_grad():
        patch_tokens = model.aggregator.patch_embed(images_flat)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"].contiguous()
    
    _, P, C = patch_tokens.shape
    N = S * P
    
    print(f"Patch tokens: {patch_tokens.shape} -> total N={N} tokens")
    
    # Compute frame saliency mask
    frame_mask = compute_frame_saliency(patch_tokens.view(B, S, P, C))
    
    print(f"\n{'='*80}")
    print(f"Frame Saliency Mask (S={S} frames, P={P} patches/frame)")
    print(f"{'='*80}")
    mask_np = frame_mask[0].cpu().numpy()
    total_pairs = S * S
    active_pairs = mask_np.sum()
    print(f"Active pairs (mask=1): {active_pairs}/{total_pairs} ({100*active_pairs/total_pairs:.1f}%)")
    print(f"Skipped pairs (mask=0): {total_pairs - active_pairs}/{total_pairs} ({100*(total_pairs - active_pairs)/total_pairs:.1f}%)")
    print(f"\nMask pattern (showing first 10x10):")
    for i in range(min(S, 10)):
        row_str = " ".join(str(mask_np[i, j]) for j in range(min(S, 10)))
        if S > 10:
            row_str += " ..."
        print(f"  F{i:2d}: {row_str}")
    if S > 10:
        print("  ...")
    
    # Prepare input: add special tokens
    from vggt.models.aggregator import slice_expand_and_flatten
    camera_token = slice_expand_and_flatten(model.aggregator.camera_token, B, S)
    register_token = slice_expand_and_flatten(model.aggregator.register_token, B, S)
    tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1).to(dtype=dtype)
    
    _, P_total, C = tokens.shape  # P_total includes special tokens
    
    # Get position embeddings
    pos = None
    if model.aggregator.rope is not None:
        pos = model.aggregator.position_getter(B * S, H // model.aggregator.patch_size, 
                                               W // model.aggregator.patch_size, device=device)
        if model.aggregator.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * S, model.aggregator.patch_start_idx, 2).to(device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
    
    # Reshape to [B, S*P, C]
    tokens = tokens.view(B, S * P_total, C)
    if pos is not None:
        pos = pos.view(B, S, P_total, 2).view(B, S * P_total, 2)
    
    print(f"\n{'='*80}")
    print(f"Running Attention Comparison (Layer {layer_idx})")
    print(f"{'='*80}")
    
    # Compute Q, K, V using the real attention layer
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        norm_tokens = attn_layer.norm1(tokens) if hasattr(attn_layer, 'norm1') else tokens
        
        qkv = attn_layer.qkv(norm_tokens).reshape(B, S * P_total, 3, attn_layer.num_heads, 
                                                   attn_layer.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, H, N, D]
        q, k = attn_layer.q_norm(q), attn_layer.k_norm(k)
        
        # Apply RoPE if available
        if attn_layer.rope is not None and pos is not None:
            q = attn_layer.rope(q, pos)
            k = attn_layer.rope(k, pos)
        
        H, D = attn_layer.num_heads, attn_layer.head_dim
        scale = attn_layer.scale
        
        print(f"Q/K/V shapes: {q.shape}, dtypes: q={q.dtype}, k={k.dtype}, v={v.dtype}")
    
    # All attention computations with AMP autocast to ensure consistent precision
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        # 1. Reference: Standard PyTorch attention (no masking)
        ref_output = F.scaled_dot_product_attention(q, k, v, scale=scale)
        
        # 2. Our masked Triton attention
        masked_output = attention(q, k, v, False, scale, frame_mask, S, P_total, False)
        
        # 3. Expected: PyTorch with frame mask applied
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]
        
        # Create token-level mask from frame-level mask
        N = S * P_total
        token_mask = torch.zeros(B, 1, N, N, device=device, dtype=torch.bool)
        for i in range(S):
            for j in range(S):
                if frame_mask[0, i, j] == 1:
                    i_start, i_end = i * P_total, (i + 1) * P_total
                    j_start, j_end = j * P_total, (j + 1) * P_total
                    token_mask[:, :, i_start:i_end, j_start:j_end] = True
        
        masked_scores = scores.masked_fill(~token_mask, float('-inf'))
        attn_weights = F.softmax(masked_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        expected_output = torch.matmul(attn_weights, v)
    
    # Compare outputs
    print(f"\n{'='*80}")
    print(f"Correctness Analysis")
    print(f"{'='*80}")
    
    # Compare masked vs expected (both should apply the frame mask)
    diff_masked_expected = (masked_output - expected_output).abs()
    max_diff = diff_masked_expected.max().item()
    mean_diff = diff_masked_expected.mean().item()
    
    print(f"\nMasked Triton vs Expected PyTorch (both with frame mask):")
    print(f"  Max absolute error:  {max_diff:.6e}")
    print(f"  Mean absolute error: {mean_diff:.6e}")
   
    # Compare masked vs reference (to show the difference masking makes)
    diff_masked_ref = (masked_output - ref_output).abs()
    max_diff_ref = diff_masked_ref.max().item()
    mean_diff_ref = diff_masked_ref.mean().item()
    
    print(f"\nMasked Triton vs Reference PyTorch (no mask):")
    print(f"  Max absolute error:  {max_diff_ref:.6e}")
    print(f"  Mean absolute error: {mean_diff_ref:.6e}")
    print(f"  (This shows how much the mask changes the output)")
    
    # Tolerance check
    tolerance = 0.01  # BF16 tolerance
    if max_diff < tolerance:
        print(f"\n✅ PASS: Masked attention matches expected output (tolerance={tolerance})")
        success = True
    else:
        print(f"\n❌ FAIL: Masked attention differs from expected (tolerance={tolerance})")
        success = False
        
        # Debug output
        print(f"\nDebug sample (first 3 tokens, first head):")
        print(f"Expected:      {expected_output[0, 0, :3, :5]}")
        print(f"Masked output: {masked_output[0, 0, :3, :5]}")
        print(f"Difference:    {diff_masked_expected[0, 0, :3, :5]}")
    
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test masked attention with VGGT weights")
    parser.add_argument("--layer", type=int, default=0, 
                       help="Global attention layer index to test (0-23)")
    parser.add_argument("--frames", type=int, default=20,
                       help="Number of frames to test with")
    args = parser.parse_args()
    
    try:
        success = test_masked_attention_with_real_weights(args.layer, args.frames)
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
