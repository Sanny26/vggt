#!/usr/bin/env python3
"""
VGGT Profiling Script
---------------------
Profiles CUDA operations including memory layout changes, attention, MLP, and LayerNorm.

Usage:
    python memvggt.py --num_frames 50 --profile          # Profile CUDA kernels
    python memvggt.py --num_frames 50 --memory           # Memory tracking
    python memvggt.py --num_frames 50 --compile          # With torch.compile
    python memvggt.py --num_frames 50 --profile --memory # Both
"""

import argparse
import torch
torch.backends.cudnn.conv.fp32_precision = 'tf32'
torch.backends.cuda.matmul.fp32_precision = 'tf32'

# ============================================================================
# SECTION 1: Configuration & Setup
# ============================================================================

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description="VGGT Profiling Script")
parser.add_argument("--attn_impl", type=str, default="sdpa", help="Attention implementation")
parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type (float16, bfloat16, float32)")
parser.add_argument("--num_frames", type=int, default=100, help="Number of frames to process")
parser.add_argument("--compile", action="store_true", help="Enable torch.compile on aggregator")
parser.add_argument("--profile", action="store_true", help="Profile CUDA operations with torch.profiler")
parser.add_argument("--memory", action="store_true", help="Enable memory tracking and snapshot")
parser.add_argument("--offload", action="store_true", help="Offload intermediate tokens to CPU")
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this profiling script")

device = torch.device("cuda", 0)
dtype = getattr(torch, args.dtype)

print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Config: attn={args.attn_impl}, dtype={args.dtype}, frames={args.num_frames}")
print(f"Flags: compile={args.compile}, profile={args.profile}, memory={args.memory}")

# ============================================================================
# SECTION 2: Model Loading & Preparation
# ============================================================================

from vggt.models.vggt import VGGT

print("\n=== Loading Model ===")
model = VGGT(attn_impl=args.attn_impl)

_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
model.eval()
model = model.to(device)

# whether to offload intermediate tokens to CPU
if args.offload:
    offload_intermediate = True
else:
    offload_intermediate = False
model.aggregator.offload_intermediate = offload_intermediate
model.depth_head.offload_intermediate = offload_intermediate
model.camera_head.offload_intermediate = offload_intermediate
model.point_head.offload_intermediate = offload_intermediate

if args.compile:
    print("Compiling aggregator with torch.compile (default mode)...")
    model.aggregator = torch.compile(model.aggregator)

# Create input
images = torch.randn((args.num_frames, 3, 518, 518), device=device, dtype=dtype)
print(f"Input shape: {images.shape}")

# Warmup
print("\n=== Warmup ===")
with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    _ = model(images[None])
    torch.cuda.synchronize()
print("Warmup complete")

# ============================================================================
# SECTION 3: Profiling & Benchmarking
# ============================================================================

def run_inference():
    """Single inference pass."""
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        return model(images[None])

# --- CUDA Profiling ---
if args.profile:
    print("\n=== CUDA Profiling ===")
    print("Capturing all ops: reshapes, attention, MLP, LayerNorm, etc.\n")
    
    torch.cuda.synchronize()
    
    with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True) as prof:
        run_inference()
        torch.cuda.synchronize()
    
    # Self CUDA time (time spent in kernel itself, not including children)
    print("Top operations by self CUDA time:")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
    
    # Total CUDA time (inclusive of children)
    print("\nTop operations by total CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Group by input shape to see reshape/layout operations
    print("\nOperations grouped by input shape:")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=25))
    
    # Export trace for visualization in chrome://tracing or Perfetto
    prof.export_chrome_trace("trace.json")
    print("\nTrace exported to trace.json (open in chrome://tracing or Perfetto)")

# --- Memory Tracking ---
if args.memory:
    print("\n=== Memory Profiling ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Try to record memory history for snapshot
    try:
        torch.cuda.memory._record_memory_history(True)
    except AttributeError:
        print("Note: Memory history recording not available in this PyTorch version")
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    run_inference()
    end.record()
    torch.cuda.synchronize()
    
    print(f"Inference time: {start.elapsed_time(end) / 1000:.3f}s")
    
    peak_mem = torch.cuda.max_memory_allocated(device)
    print(f"Peak memory: {peak_mem / 1024**3:.2f} GB")
    
    # Save memory snapshot
    try:
        torch.cuda.memory._record_memory_history(False)
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        print("Memory snapshot saved to memory_snapshot.pickle")
    except AttributeError:
        pass
    
    print("\n" + torch.cuda.memory_summary(device))

# --- Benchmark (always runs) ---
# print("\n=== Benchmark ===")
# num_iters = 3
# times = []

# for i in range(num_iters):
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
    
#     start.record()
#     run_inference()
#     end.record()
#     torch.cuda.synchronize()
    
#     elapsed = start.elapsed_time(end) / 1000
#     times.append(elapsed)
#     print(f"  Iter {i+1}: {elapsed:.3f}s")

# print(f"\nAvg: {sum(times)/len(times):.3f}s | Min: {min(times):.3f}s | Max: {max(times):.3f}s")
