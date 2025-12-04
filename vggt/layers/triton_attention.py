import torch

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
    def attention_kernel(
        Q,
        K,
        V,
        Out,
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
        Fused attention kernel (similar to FlashAttention).
        Supports fp16, bf16, and fp32 dtypes.

        Q: [B, H, M, D]
        K: [B, H, N, D]
        V: [B, H, N, D]
        Out: [B, H, M, D]
        """
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_m = tl.program_id(2)

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


def run_triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Run Triton fused attention kernel.
    Supports fp16, bf16, and fp32 dtypes.
    
    Args:
        q: Query tensor [B, H, M, D]
        k: Key tensor [B, H, N, D]
        v: Value tensor [B, H, N, D]
        scale: Attention scale factor
        
    Returns:
        Output tensor [B, H, M, D]
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available on this system")

    if q.device.type != "cuda":
        raise ValueError("Triton attention kernel only supports CUDA tensors")

    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Triton attention expects fp16/bf16/fp32 tensors, got {q.dtype}")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    B, H, M, D = q.shape
    N = k.shape[2]

    block_m = 64
    block_n = 64
    block_k = D

    out = torch.empty_like(q)
    grid = (B, H, triton.cdiv(M, block_m))

    attention_kernel[grid](
        q,
        k,
        v,
        out,
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

    return out


# Backward compatibility alias
run_triton_attention_fp32 = run_triton_attention
