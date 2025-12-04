"""
Mixed precision attention kernels with selective precision allocation.

Implements various strategies for using different precisions within attention:
1. Random block selection (baseline for testing)
2. Head-level mixed precision
3. Token-level mixed precision (importance-based)
"""

import torch
import numpy as np

TRITON_AVAILABLE = False

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def attention_kernel_mixed_blocks(
        Q,
        K,
        V,
        Out,
        BlockMask,  # [B, H, num_blocks_m] - which blocks to compute in FP32 (1) vs FP16 (0)
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_ob,
        stride_oh,
        stride_om,
        stride_ok,
        stride_maskb,
        stride_maskh,
        B,
        H,
        M,
        N,
        D,
        scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Mixed precision attention with block-level precision selection.
        
        Uses BlockMask to determine which query blocks should use FP32 accumulation
        vs standard mixed precision (FP16 input, FP32 accumulation).
        
        This is a proof-of-concept for selective precision allocation.
        """
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_m = tl.program_id(2)

        # Check if this block should use higher precision
        mask_ptr = BlockMask + pid_b * stride_maskb + pid_h * stride_maskh + pid_m
        use_fp32 = tl.load(mask_ptr)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)

        q_ptrs = (
            Q
            + pid_b * stride_qb
            + pid_h * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_k[None, :] * stride_qk
        )
        q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

        # If use_fp32, convert Q to fp32 immediately
        # Otherwise keep in original precision
        if use_fp32 > 0:
            q = q.to(tl.float32)

        acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        for start_n in range(0, N, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = (
                K
                + pid_b * stride_kb
                + pid_h * stride_kh
                + offs_n[:, None] * stride_kn
                + offs_k[None, :] * stride_kk
            )
            k = tl.load(k_ptrs, mask=offs_n[:, None] < N, other=0.0)

            if use_fp32 > 0:
                k = k.to(tl.float32)

            scores = tl.dot(q, tl.trans(k))
            scores = scores * scale

            m_ij = tl.max(scores, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)

            alpha = tl.exp(m_i - m_i_new)
            acc = acc * alpha[:, None]
            l_i = l_i * alpha

            scores_shifted = scores - m_i_new[:, None]
            p = tl.exp(scores_shifted)
            l_ij = tl.sum(p, axis=1)
            l_i = l_i + l_ij

            v_ptrs = (
                V
                + pid_b * stride_vb
                + pid_h * stride_vh
                + offs_n[:, None] * stride_vn
                + offs_k[None, :] * stride_vk
            )
            v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)

            if use_fp32 > 0:
                v = v.to(tl.float32)
                acc = acc + tl.dot(p, v)
            else:
                acc = acc + tl.dot(p.to(v.dtype), v)

            m_i = m_i_new

        acc = acc / l_i[:, None]

        out_ptrs = (
            Out
            + pid_b * stride_ob
            + pid_h * stride_oh
            + offs_m[:, None] * stride_om
            + offs_k[None, :] * stride_ok
        )
        tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < M)

    @triton.jit
    def attention_kernel_token_mixed(
        Q,
        K,
        V,
        Out,
        TokenMask,  # [B, M] - per-token precision selection
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_ob,
        stride_oh,
        stride_om,
        stride_ok,
        stride_tokb,
        stride_tokm,
        B,
        H,
        M,
        N,
        D,
        scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Token-level mixed precision attention.
        
        Each token can have its own precision level based on importance/saliency.
        """
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_m = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)

        # Load per-token precision mask
        token_mask_ptrs = TokenMask + pid_b * stride_tokb + offs_m * stride_tokm
        token_precision = tl.load(token_mask_ptrs, mask=offs_m < M, other=0.0)

        q_ptrs = (
            Q
            + pid_b * stride_qb
            + pid_h * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_k[None, :] * stride_qk
        )
        q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

        acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        for start_n in range(0, N, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = (
                K
                + pid_b * stride_kb
                + pid_h * stride_kh
                + offs_n[:, None] * stride_kn
                + offs_k[None, :] * stride_kk
            )
            k = tl.load(k_ptrs, mask=offs_n[:, None] < N, other=0.0)

            scores = tl.dot(q, tl.trans(k))
            scores = scores * scale

            # Per-token precision scaling (higher precision for important tokens)
            # This is a simplified version - in practice you'd want more sophisticated logic
            scores = scores * (1.0 + 0.1 * token_precision[:, None])

            m_ij = tl.max(scores, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)

            alpha = tl.exp(m_i - m_i_new)
            acc = acc * alpha[:, None]
            l_i = l_i * alpha

            scores_shifted = scores - m_i_new[:, None]
            p = tl.exp(scores_shifted)
            l_ij = tl.sum(p, axis=1)
            l_i = l_i + l_ij

            v_ptrs = (
                V
                + pid_b * stride_vb
                + pid_h * stride_vh
                + offs_n[:, None] * stride_vn
                + offs_k[None, :] * stride_vk
            )
            v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)

            acc = acc + tl.dot(p.to(v.dtype), v)

            m_i = m_i_new

        acc = acc / l_i[:, None]

        out_ptrs = (
            Out
            + pid_b * stride_ob
            + pid_h * stride_oh
            + offs_m[:, None] * stride_om
            + offs_k[None, :] * stride_ok
        )
        tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < M)


# ============================================================================
# Precision Selection Strategies
# ============================================================================


def random_block_selection(B: int, H: int, M: int, block_m: int, fp32_ratio: float = 0.3) -> torch.Tensor:
    """
    Randomly select blocks to compute in FP32.
    
    Args:
        B: Batch size
        H: Number of heads
        M: Sequence length
        block_m: Block size for M dimension
        fp32_ratio: Fraction of blocks to compute in FP32
        
    Returns:
        Block mask [B, H, num_blocks_m] where 1 = FP32, 0 = FP16
    """
    num_blocks_m = triton.cdiv(M, block_m)
    mask = torch.rand(B, H, num_blocks_m) < fp32_ratio
    return mask.float()


def saliency_based_selection(
    attention_scores: torch.Tensor, 
    block_m: int, 
    fp32_ratio: float = 0.3
) -> torch.Tensor:
    """
    Select blocks based on attention score statistics from previous layer.
    
    Args:
        attention_scores: [B, H, M, N] attention scores from previous layer
        block_m: Block size
        fp32_ratio: Fraction of blocks to use FP32
        
    Returns:
        Block mask [B, H, num_blocks_m]
    """
    B, H, M, N = attention_scores.shape
    num_blocks_m = triton.cdiv(M, block_m)
    
    # Compute saliency: variance or max of attention scores per token
    saliency = attention_scores.var(dim=-1)  # [B, H, M]
    
    # Pool saliency into blocks
    block_saliency = []
    for i in range(num_blocks_m):
        start = i * block_m
        end = min(start + block_m, M)
        block_sal = saliency[:, :, start:end].mean(dim=-1)  # [B, H]
        block_saliency.append(block_sal)
    
    block_saliency = torch.stack(block_saliency, dim=-1)  # [B, H, num_blocks_m]
    
    # Select top-k blocks
    num_fp32_blocks = max(1, int(num_blocks_m * fp32_ratio))
    _, top_indices = torch.topk(block_saliency, num_fp32_blocks, dim=-1)
    
    mask = torch.zeros(B, H, num_blocks_m, device=attention_scores.device)
    mask.scatter_(-1, top_indices, 1.0)
    
    return mask


def head_based_selection(B: int, H: int, M: int, block_m: int, fp32_heads: list) -> torch.Tensor:
    """
    Select specific heads to compute in FP32.
    
    Args:
        B: Batch size
        H: Number of heads
        M: Sequence length
        block_m: Block size
        fp32_heads: List of head indices to compute in FP32
        
    Returns:
        Block mask [B, H, num_blocks_m]
    """
    num_blocks_m = triton.cdiv(M, block_m)
    mask = torch.zeros(B, H, num_blocks_m)
    
    for head_idx in fp32_heads:
        if 0 <= head_idx < H:
            mask[:, head_idx, :] = 1.0
    
    return mask


def token_importance_from_norm(q: torch.Tensor, top_k_ratio: float = 0.3) -> torch.Tensor:
    """
    Compute token importance based on query norm.
    
    Args:
        q: Query tensor [B, H, M, D]
        top_k_ratio: Fraction of tokens to mark as important
        
    Returns:
        Token importance [B, M] where 1.0 = important, 0.0 = not important
    """
    B, H, M, D = q.shape
    
    # Compute per-token norm (averaged across heads)
    token_norm = q.norm(dim=-1).mean(dim=1)  # [B, M]
    
    # Select top-k tokens
    num_important = max(1, int(M * top_k_ratio))
    _, top_indices = torch.topk(token_norm, num_important, dim=-1)
    
    importance = torch.zeros(B, M, device=q.device)
    importance.scatter_(-1, top_indices, 1.0)
    
    return importance


# ============================================================================
# Wrapper Functions
# ============================================================================


def run_mixed_precision_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    strategy: str = "random",
    fp32_ratio: float = 0.3,
    prev_attention_scores: torch.Tensor = None,
    fp32_heads: list = None,
) -> tuple[torch.Tensor, dict]:
    """
    Run mixed precision attention with various selection strategies.
    
    Args:
        q, k, v: Query, Key, Value tensors [B, H, M/N, D]
        scale: Attention scale factor
        strategy: Selection strategy - "random", "saliency", "heads", "token_norm"
        fp32_ratio: Fraction of computation to do in FP32
        prev_attention_scores: Previous layer attention for saliency-based selection
        fp32_heads: List of head indices for head-based selection
        
    Returns:
        output: Attention output [B, H, M, D]
        stats: Dictionary with statistics about precision usage
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")

    if q.device.type != "cuda":
        raise ValueError("Only CUDA tensors supported")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    B, H, M, D = q.shape
    N = k.shape[2]

    block_m = 64
    block_n = 64
    block_k = D

    num_blocks_m = triton.cdiv(M, block_m)
    
    # Select precision allocation strategy
    if strategy == "random":
        block_mask = random_block_selection(B, H, M, block_m, fp32_ratio).to(q.device)
    elif strategy == "saliency":
        if prev_attention_scores is None:
            raise ValueError("saliency strategy requires prev_attention_scores")
        block_mask = saliency_based_selection(prev_attention_scores, block_m, fp32_ratio).to(q.device)
    elif strategy == "heads":
        if fp32_heads is None:
            # Default: use first 30% of heads in FP32
            num_fp32 = max(1, int(H * fp32_ratio))
            fp32_heads = list(range(num_fp32))
        block_mask = head_based_selection(B, H, M, block_m, fp32_heads).to(q.device)
    elif strategy == "token_norm":
        token_mask = token_importance_from_norm(q, fp32_ratio).to(q.device)
        # For block kernel, convert token mask to block mask
        block_mask = torch.zeros(B, H, num_blocks_m, device=q.device)
        for i in range(num_blocks_m):
            start = i * block_m
            end = min(start + block_m, M)
            # If any token in block is important, mark block as FP32
            block_importance = token_mask[:, start:end].max(dim=-1)[0]
            block_mask[:, :, i] = block_importance.unsqueeze(1).expand(-1, H)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    block_mask = block_mask.contiguous()
    out = torch.empty_like(q)

    grid = (B, H, num_blocks_m)

    attention_kernel_mixed_blocks[grid](
        q,
        k,
        v,
        out,
        block_mask,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        block_mask.stride(0),
        block_mask.stride(1),
        B,
        H,
        M,
        N,
        D,
        scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )

    # Compute statistics
    stats = {
        "total_blocks": B * H * num_blocks_m,
        "fp32_blocks": block_mask.sum().item(),
        "fp32_percentage": (block_mask.sum() / (B * H * num_blocks_m)).item() * 100,
        "strategy": strategy,
        "block_mask": block_mask.cpu(),
    }

    return out, stats

