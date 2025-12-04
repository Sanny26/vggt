import os
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.layers.attention import Attention
from utils import profiling
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--attn_impl", type=str, default="sdpa")
parser.add_argument("--dtype", type=str, default="float32")
parser.add_argument(
    "--checkpoint_eval",
    action="store_true",
    help="Enable gradient checkpointing inside the aggregator even when the model is in eval mode.",
)
parser.add_argument(
    "--force_flash_sdpa",
    action="store_true",
    help="Force PyTorch SDPA to prefer FlashAttention/memory-efficient kernels (only affects attn_impl=sdpa).",
)
args = parser.parse_args()

def _print_profiler_tables(prof: torch.profiler.profile, row_limit: int = 20) -> None:
    """Emit CUDA time and memory summaries from a torch.profiler run."""
    key_averages = prof.key_averages()
    print("\n=== Top ops by total CUDA time ===")
    print(key_averages.table(sort_by="cuda_time_total", row_limit=row_limit))
    print("\n=== Top ops by CUDA memory usage ===")
    print(key_averages.table(sort_by="self_cuda_memory_usage", row_limit=row_limit))


if not torch.cuda.is_available():
    raise RuntimeError("VGGT profiling script expects a CUDA-enabled PyTorch build")

# device = torch.device("cuda", 0)
# compute_cap = torch.cuda.get_device_capability(device.index)
# compute_cap_num = compute_cap[0] + compute_cap[1] / 10.0
# # Select dtype based on GPU capability
# if compute_cap[0] >= 9:  # H100 (Hopper)
#     dtype = torch.float8_e4m3fn
#     print(f"Using FP8 (E4M3) on {torch.cuda.get_device_name(device.index)}")
# elif compute_cap_num >= 8.9:  # Ada Lovelace (RTX 40-series)
#     dtype = torch.float8_e4m3fn
#     print(f"Using FP8 (E4M3) on {torch.cuda.get_device_name(device.index)}")
# elif compute_cap[0] >= 8:  # Ampere (A100, RTX 30-series)
#     dtype = torch.bfloat16
#     print(f"Using BF16 on {torch.cuda.get_device_name(device.index)}")
# else:
#     dtype = torch.float16
#     print(f"Using FP16 on {torch.cuda.get_device_name(device.index)}")


device = torch.device("cuda", 0)
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability(device.index)[0] >= 8 else torch.float16
# dtype=torch.float32
# dtype = torch.float8_e4m3fn

# attn_impl = 'sdpa'
# attn_impl = 'triton'
attn_impl = args.attn_impl


model = VGGT(attn_impl=attn_impl)
print(f'Using {attn_impl} attention implementation with {dtype} dtype')
if args.checkpoint_eval:
    print("Aggregator will use checkpointing even in eval mode to reduce activation memory.")

_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))


model.eval()
model = model.to(device)
print(f"Using device: cuda:{device.index} {torch.cuda.get_device_name(device.index)}")

# Load and preprocess example images (replace with your own image paths)
# images_dir = os.path.join(os.path.dirname(__file__), "examples", "kitchen", "images")
# image_names = [os.path.join(images_dir, name) for name in sorted(os.listdir(images_dir))]
# images = load_and_preprocess_images(image_names).to(device, dtype)
# images = images.repeat(100, 1, 1, 1) # 100 images at 27 GB memory, 200 images cuda out of memory
images = torch.randn((25, 3, 518, 518)).to(device, dtype)
print(f"Images shape: {images.shape}")

# warmup
with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    try:
        # _ = model(images[None])
        model.aggregator(images[None])
    except torch.OutOfMemoryError:
        pass # Ignore OOM during warmup
torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as prof:
    # Autocast context for layers such as Conv2d, Linear, Attention, etc.
    # - Patch embedding (Conv2d or Vision Transformer): autocast to dtype (bfloat16/float16)
    # - Transformer blocks (including Attention/MLP): autocast to dtype (bfloat16/float16), 
    #   though within VGGT some heads (CameraHead, DPTHead) may explicitly disable autocast.
    # - DPTHead/CameraHead/TrackHead: custom forward routines, see model code for their mixed-precision behavior.
    # for _ in range(10):
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        # _ = model(images)
        # aggregated_tokens, patch_idx = model.aggregator(images[None])
        model.aggregator(images[None])
    prof.step()

prof.export_chrome_trace(f"traces/vggt_{attn_impl}_{dtype}.json")
_print_profiler_tables(prof)