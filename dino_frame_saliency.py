'''
This module implements the DINO frame saliency mask computation.
Given a B, F, P, D it computes the saliency mask between every frame pair
Symmetric matrix of size F x F. so solve only for F[i, j]=F[j, i]

F[i, j]] = int/bool mask as F X F would be too large.
I can simply store upper triangular part of the matrix and mirror it.
'''

import torch
import triton
import os
import numpy as np
import trimesh
import triton.language as tl
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from typing import List, Tuple
from vggt.models.aggregator import Aggregator, slice_expand_and_flatten
from vggt.layers.block import Block, drop_add_residual_stochastic_depth
from vggt.layers.attention import Attention
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
from triton.tools.tensor_descriptor import TensorDescriptor
from torch.cuda.amp import custom_fwd, custom_bwd

@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


# ==============================================================================
# Mixed Precision Attention: BF16 for important frames, FP8 for similar frames
# ==============================================================================
# 
# Frame saliency mask determines precision: Mask has shape B, S, S and attention input is of shape B, H, S*P, D
#   - mask[i,j] = 1 (similarity) → BF16 (important frame pairs, need precision)
#   - mask[i,j] = 0 (dissimilarity)  → FP8 (similar frames, less precision needed)
# The mask is constant and computed once before all the attention kernels. So using that to reorganize the attention input might be a good idea except the reference frame(zeroth frame)
# Uses online softmax with separate m/l accumulators per kernel, then combines.
# Use tensor cores for FP8. this is being run on rtx 5090. look for triton_examples/fused_attention for sdpa attention code in triton and other persistant and block scaled matmul desinged to work for latest architecture
# Simplify the problem as much as possible
# 
# Triton 
# ==============================================================================
def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 10

def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9

def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]

if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]

configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook) \
    for BM in [32, 64]\
    for BN in [32, 64]\
    for s in NUM_STAGES_OPTIONS \
    for w in [4, 8]\
]

def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (is_cuda() and torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128
                and conf.num_warps == 8)

def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    # Filter out configs where BLOCK_M > N_CTX
    return [conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX]

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    sparse_mask, frame_q, off_z, F, T, G, debug_counter, # F: num_frames, T: tokens_per_frame, G: num_global_tokens
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        #===========check sparsity mask and skip compute===========
        # frame_k = (start_n // P).to(tl.int16)
        # mask_idx = off_z * (F * F) + frame_q * F + frame_k
        # m_val = tl.load(sparse_mask + mask_idx)

        # token indices for this tile
        row_idx = start_m * BLOCK_M + offs_m          # [BLOCK_M]
        col_idx = start_n + offs_n                    # [BLOCK_N]

        # --- Hybrid Attention Logic ---
        # 1. Map token index -> frame id
        frame_q_row = (row_idx // T).to(tl.int32)
        frame_k_col = (col_idx // T).to(tl.int32)

        # 2. Identify Global Tokens (first G tokens of each frame)
        # Local index within the frame
        local_row = row_idx % T
        local_col = col_idx % T
        
        is_global_row = local_row < G
        is_global_col = local_col < G

        # 3. Build Mask
        # If either query or key is global, we ALWAYS attend (mask=1)
        # Otherwise, we check the saliency mask
        
        # Saliency mask lookup
        row_base = off_z * (F * F) + frame_q_row[:, None] * F   # [BM, 1]
        col_off  = frame_k_col[None, :]                         # [1, BN]
        saliency_val = tl.load(sparse_mask + row_base + col_off) # [BM, BN]

        # Combine: Global OR Saliency
        # We use tl.where to implement (is_global_row | is_global_col) | saliency_val
        # Note: is_global_row is [BM], is_global_col is [BN] -> broadcast to [BM, BN]
        is_global_any = (is_global_row[:, None] | is_global_col[None, :])
        block_mask = tl.where(is_global_any, 1, saliency_val)
        
        # Mask out-of-bounds columns (padding)
        # col_idx is [BN], broadcast to [BM, BN]
        block_mask = tl.where(col_idx[None, :] < N_CTX, block_mask, 0)

        # process all tile with 1 masks
        if tl.max(block_mask) == 1:
            tl.atomic_add(debug_counter, 1)
            # -- compute qk ----
            k = desc_k.load([offsetk_y, 0]).T
            qk = tl.dot(q, k)
            if STAGE == 2:
                mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                # Apply block_mask to causal mask
                mask = mask & (block_mask == 1)
                qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk -= m_ij[:, None]
            else:
                # Apply block_mask to non-causal attention
                qk = qk * qk_scale + tl.where(block_mask == 1, 0, -1.0e6)
                m_ij = tl.maximum(m_i, tl.max(qk, 1))
                qk -= m_ij[:, None]
            p = tl.math.exp2(qk)
            # -- compute correction factor
            alpha = tl.math.exp2(m_i - m_ij)
            l_ij = tl.sum(p, 1)
            # -- update output accumulator --
            if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
                BM: tl.constexpr = acc.shape[0]
                BN: tl.constexpr = acc.shape[1]
                acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
                acc0 = acc0 * alpha[:, None]
                acc1 = acc1 * alpha[:, None]
                acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
            else:
                acc = acc * alpha[:, None]
            # prepare p and v for the dot
            if dtype == tl.float8e5:
                v = desc_v.load([0, offsetv_y]).T
            else:
                v = desc_v.load([offsetv_y, 0])
            p = p.to(dtype)
            # note that this non transposed v for FP8 is only supported on Blackwell
            acc = tl.dot(p, v, acc)
            # update m_i and l_i
            # place this at the end of the loop to reduce register pressure
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i

@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)

@triton.autotune(configs=list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"],
                 prune_configs_by={'early_config_prune': prune_invalid_configs})
# @triton.autotune(configs=list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"])
@triton.jit
def _attn_fwd(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o, #    Z=B, HEAD_DIM=D
              sparsity_mask, F, T, G, debug_counter, # T=Total tokens per frame, G=Global tokens
              N_CTX, 
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              IS_HOPPER: tl.constexpr,  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.bfloat16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, y_dim], strides=[N_CTX, 1],
                                         block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE

    ## sparsity mask params 
    frame_q = (start_m * BLOCK_M // T).to(tl.int16)
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        sparsity_mask, frame_q, off_z, F, T, G, debug_counter,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        sparsity_mask, frame_q, off_z, F, T, G, debug_counter,
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


class _attention(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, q, k, v, causal, sm_scale, sparsity_mask, F, T, G, warp_specialize=True):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Use device_descriptor for Hopper + warpspec.
        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]

            dummy_block = [1, 1]
            desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1],
                                          block_shape=dummy_block)
            else:
                desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1],
                                          block_shape=dummy_block)
            desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        debug_counter = torch.zeros(1, dtype=torch.int32, device=q.device)

        # tma descriptor reuqire a gloobal memory allocator
        triton.set_allocator(alloc_fn)

        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        ctx.grid = grid
        if is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80
        _attn_fwd[grid](
            sm_scale, M,  #
            q.shape[0], q.shape[1],  #
            desc_q, desc_k, desc_v, desc_o,  #
            sparsity_mask, F, T, G, debug_counter, # sparse attention mask parameters
            N_CTX=q.shape[2],  # N_CTX
            HEAD_DIM=HEAD_DIM_K,  #
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
            STAGE=stage,  #
            warp_specialize=warp_specialize,  #
            IS_HOPPER=is_hopper(),  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        
        # Print debug count
        num_warps = _attn_fwd.best_config.num_warps
        num_threads = num_warps * 32
        total_blocks = debug_counter.item() // num_threads
        print(f"DEBUG: Processed {total_blocks} blocks (Threads: {num_threads}, Raw: {debug_counter.item()})")
        
        return o

attention = _attention.apply


class FastGlobalAttention(Attention):
    """
    Global attention that uses mixed-precision based on frame saliency.
    
    - Similar frames (mask=1) → FP8 attention (faster, lower precision)
    - Dissimilar frames (mask=0) → BF16 attention (higher precision)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(
        self, 
        x: torch.Tensor, 
        pos=None, 
        attn_bias=None, 
        frame_mask: torch.Tensor = None,
        num_frames: int = None,
        patch_start_idx: int = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] input tokens where N = S * P (frames * patches)
            pos: optional position embeddings for RoPE
            attn_bias: optional attention bias (unused with mixed precision)
            frame_mask: [B, S, S] or [S, S] from compute_frame_saliency
            num_frames: S - number of frames (required if frame_mask provided)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, H, N=S*P, D]
        q, k = self.q_norm(q), self.k_norm(k)
        
        # Apply RoPE if available
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        
        if frame_mask is not None and num_frames is not None:
            # Use mixed precision attention based on frame saliency
            # N = Total tokens (B * S * (G + P)) -> but here x is [B, N, C]
            # N = S * (G + P)
            total_tokens_per_frame = N // num_frames
            num_global_tokens = patch_start_idx
            
            # print(f"DEBUG: N={N}, F={num_frames}, T={total_tokens_per_frame}, G={num_global_tokens}")

            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            
            out = attention(q, k, v, False, self.scale, frame_mask, num_frames, total_tokens_per_frame, num_global_tokens, False)
            end_event.record()
            torch.cuda.synchronize()
            mixed_precision_time = start_event.elapsed_time(end_event)
            print(f"Mixed precision attention time: {mixed_precision_time} ms for {q.shape}, q.dtype: {q.dtype}, k.dtype: {k.dtype}, v.dtype: {v.dtype}")
        else:
            # Fall back to standard scaled dot-product attention
            print('Falling back to standard scaled dot-product attention')
            out = F.scaled_dot_product_attention(q, k, v)
        
        # Reshape back to [B, N, C]
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out


class FrameSaliencyBlock(Block):
    "Not using this for now"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_mask = None

    def forward(self, x: torch.Tensor, pos=None, frame_mask=None, ) -> torch.Tensor:
        def attn_residual_func(x: torch.Tensor, pos=None) -> torch.Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x, pos=pos, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
            x = drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos)
            x = x + ffn_residual_func(x)
        return x


@torch.compile(mode="reduce-overhead")
def compute_frame_saliency(
    frame_tokens: torch.Tensor, 
) -> torch.Tensor:
    """
    Compute frame saliency mask based on patch token similarity.
    Mask = 1 for similar frames, 0 for dissimilar frames
    """
    B, S, P, D = frame_tokens.shape
    # Flatten patches -> [B, S, P*D] (original approach)
    frame_repr = frame_tokens.reshape(B, S, -1)
    frame_repr = torch.nn.functional.normalize(frame_repr, dim=-1)
    mask = torch.bmm(frame_repr, frame_repr.transpose(-1, -2))
    mask = (mask >= 0.98).to(torch.uint8) # hardcoding threshold for better torch.compile
    
    # Ensuring reference frame is always processed 
    mask[:, :, 0] = 1
    mask[:, 0, :] = 1
    return mask

def custom_aggregator_forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
    """
    Args:
        images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

    Returns:
        (list[torch.Tensor], int):
            The list of outputs from the attention blocks,
            and the patch_start_idx indicating where patch tokens begin.
    """
    B, S, C_in, H, W = images.shape

    if C_in != 3:
        raise ValueError(f"Expected 3 input channels, got {C_in}")

    # Normalize images and reshape for patch embed 
    images = (images - self._resnet_mean.to(images.dtype)) / self._resnet_std.to(images.dtype)

    # Reshape to [B*S, C, H, W] for patch embedding
    images = images.view(B * S, C_in, H, W)
    # print(f"Before patch_embed: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    patch_tokens = self.patch_embed(images)
    # print(f"After patch_embed: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    if isinstance(patch_tokens, dict):
        patch_tokens = patch_tokens["x_norm_patchtokens"].contiguous()

    _, P, C = patch_tokens.shape

    frame_mask = compute_frame_saliency(patch_tokens.view(B, S, P, C))
    
    # fram
    # Debug: Print frame mask pattern
    print(f"\n{'='*60}")
    print(f"Frame Saliency Mask (B={B}, S={S}, P={P})")
    print(f"{'='*60}")
    for b in range(B):
        print(f"\nBatch {b}:")
        mask_np = frame_mask[b].cpu().numpy()
        total_pairs = S * S
        nonzero_pairs = mask_np.sum()
        print(f"  Active frame pairs (mask=1): {nonzero_pairs}/{total_pairs} ({100*nonzero_pairs/total_pairs:.1f}%)")
        print(f"  Skipped frame pairs (mask=0): {total_pairs - nonzero_pairs}/{total_pairs} ({100*(total_pairs - nonzero_pairs)/total_pairs:.1f}%)")
        print(f"  Mask pattern (1=process, 0=skip):")
        for i in range(min(S, 10)):  # Show first 10 frames
            row_str = "    " + " ".join(str(mask_np[i, j]) for j in range(min(S, 10)))
            if S > 10:
                row_str += " ..."
            print(f"    F{i:2d}: {row_str}")
        if S > 10:
            print("    ...")
    print(f"{'='*60}\n")

    # Expand camera and register tokens to match batch size and sequence length
    camera_token = slice_expand_and_flatten(self.camera_token, B, S)
    register_token = slice_expand_and_flatten(self.register_token, B, S)

    # Concatenate special tokens with patch tokens
    # breakpoint()
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

    for k in range(self.aa_block_num):
        for attn_type in self.aa_order:
            if attn_type == "frame":
                # print('Processing frame attention', k)
                tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                    tokens, B, S, P, C, frame_idx, pos=pos
                )
            elif attn_type == "global":
                # print('Processing global attention', k)
                tokens, global_idx, global_intermediates = self._process_global_attention(
                    tokens, B, S, P, C, global_idx, pos=pos, frame_mask=frame_mask
                )
            else:
                raise ValueError(f"Unknown attention type: {attn_type}")

        for i in range(len(frame_intermediates)):
            if k in self.intermediate_layer_idx: # until last two blocks we offload to CPU
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                if self.offload_intermediate and k != self.aa_block_num - 1: # keeping last block on GPU
                    # dequantize to lower precision
                    # concat_inter = concat_inter.to(torch.bfloat16)
                    concat_inter = concat_inter.to("cpu", non_blocking=True)
                output_list.append(concat_inter)
            
        del frame_intermediates
        del global_intermediates
    return output_list, self.patch_start_idx

def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, frame_mask=None):
        """
        Process global attention blocks with mixed-precision based on frame saliency.
        Tokens are in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            block = self.global_blocks[global_idx]
            
            # Manually call block components to pass frame_mask to attention
            # This mirrors Block.forward but allows us to pass extra args to attention
            x = tokens
            
            # Attention with residual
            if isinstance(block.attn, FastGlobalAttention):
                attn_out = block.attn(block.norm1(x), pos=pos, frame_mask=frame_mask, num_frames=S, patch_start_idx=self.patch_start_idx)
            else:
                attn_out = block.attn(block.norm1(x), pos=pos)
            x = x + block.ls1(attn_out)
            
            # FFN with residual  
            x = x + block.ls2(block.mlp(block.norm2(x)))
            
            tokens = x
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

def test_patch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using device: {device}, dtype: {dtype}")

    # Load Model
    print("Loading VGGT model...")
    model = VGGT()
    # We don't strictly need the weights for this structural test, but let's try to load them if possible
    # to get realistic outputs.
    try:
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        state_dict = torch.hub.load_state_dict_from_url(_URL, map_location='cpu')
        model.load_state_dict(state_dict)
        print("Loaded pretrained weights.")
    except Exception as e:
        print(f"Could not load weights (using random init): {e}")
    
    model.to(device=device) # using amp autocast while inferencing 
    model.aggregator.forward = custom_aggregator_forward.__get__(model.aggregator, Aggregator)
    model.aggregator._process_global_attention = _process_global_attention.__get__(model.aggregator, Aggregator)

    # swap global attention blocks with FastGlobalAttention and copy weights from original global attention blocks
    # for i in range(len(model.aggregator.global_blocks)):
    for i in range(10):
        block = model.aggregator.global_blocks[i]
        old_attn = block.attn
        
        # Create new FastGlobalAttention with same config
        new_attn = FastGlobalAttention(
            dim=old_attn.qkv.in_features,
            num_heads=old_attn.num_heads,
            qkv_bias=old_attn.qkv.bias is not None,
            proj_bias=old_attn.proj.bias is not None,
            attn_drop=old_attn.attn_drop.p,
            proj_drop=old_attn.proj_drop.p if hasattr(old_attn, 'proj_drop') else 0.0,
            qk_norm=not isinstance(old_attn.q_norm, torch.nn.Identity),
            rope=old_attn.rope,
            attn_impl=old_attn.attn_impl,
        )
        
        # Copy weights from old attention to new
        new_attn.qkv.load_state_dict(old_attn.qkv.state_dict())
        new_attn.proj.load_state_dict(old_attn.proj.state_dict())
        new_attn.q_norm.load_state_dict(old_attn.q_norm.state_dict())
        new_attn.k_norm.load_state_dict(old_attn.k_norm.state_dict())
        
        # Move to same device and dtype
        new_attn = new_attn.to(device=device, dtype=dtype)
        
        # Replace attention module in block
        block.attn = new_attn
        
    print(f"Swapped {len(model.aggregator.global_blocks)} global attention blocks with FastGlobalAttention")

    model.eval()

    # VGGT expects images [B, S, 3, H, W]
    # images_dir = os.path.join("examples", "south", "person-hall", "images")
    # images_dir = os.path.join("examples", "pyramid",)
    images_dir = os.path.join("examples", "south", "gerrard-hall", "images")
    pcd_name = images_dir.split("/")[-1]
    image_names = [os.path.join(images_dir, name) for name in sorted(os.listdir(images_dir))]
    # import random
    # random.shuffle(image_names)
    image_names = image_names[:50]
    images = load_and_preprocess_images(image_names).to(device)
    # images = images.repeat(1, 1, 1, 1) # 100 images at 27 GB memory, 200 images cuda out of memory
    images = images.unsqueeze(0) # introducing batch dimension
    print(f"Images shape: {images.shape}")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        aggregated_tokens_list, ps_idx = model.aggregator(images)
        point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)

        print(f"Before Confidence range: [{point_conf.min():.3f}, {point_conf.max():.3f}]")
        conf_threshold = 2
        # conf_threshold = 0
        conf_mask = point_conf >= conf_threshold
        points = point_map[conf_mask].view(-1, 3).cpu().numpy()
        print(points.shape)
        
        # Permute images from [B, F, C, H, W] to [B, F, H, W, C] so channels are last
        images_permuted = images.permute(0, 1, 3, 4, 2)  # [B, F, H, W, C]
        rgb = images_permuted[conf_mask].cpu().numpy()
        
        # Get confidence values and normalize them
        conf = point_conf[conf_mask].cpu().numpy()
        conf -= -0.4
        conf = conf / conf.max()
        
        # Concatenate RGB with confidence as RGBA (alpha channel)
        rgba = np.concatenate((rgb, conf[:, None]), axis=-1)
        
        print(f"Points shape: {points.shape}, RGBA shape: {rgba.shape}")
        print(f"Confidence range: [{conf.min():.3f}, {conf.max():.3f}]")
        
        # Export as PLY file
        cloud = trimesh.points.PointCloud(vertices=points, colors=rgba)
        cloud.export(f".outputs/{pcd_name}_085.ply")
        print(f"Saved point cloud to .outputs/{pcd_name}_085.ply")
    
if __name__ == "__main__":
    test_patch()