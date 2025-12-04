import torch
import os
import numpy as np
from vggt.models.vggt import VGGT
# from mixed_precision_sim import apply_monkey_patch
from mixed_precision_dino_sim import apply_monkey_patch
from vggt.utils.load_fn import load_and_preprocess_images
import trimesh

def _print_profiler_tables(prof: torch.profiler.profile, row_limit: int = 20) -> None:
    """Emit CUDA time and memory summaries from a torch.profiler run."""
    key_averages = prof.key_averages()
    print("\n=== Top ops by total CUDA time ===")
    print(key_averages.table(sort_by="cuda_time_total", row_limit=row_limit))
    print("\n=== Top ops by CUDA memory usage ===")
    print(key_averages.table(sort_by="self_cuda_memory_usage", row_limit=row_limit))

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
    
    # model.to(device=device, dtype=dtype)
    model.to(device=device) # using amp autocast while inferencing 
    model.eval()

    # Create dummy input
    # VGGT expects images [B, S, 3, H, W]
    # Let's use small size for speed
    # Load and preprocess example images (replace with your own image paths)
    images_dir = os.path.join("examples", "kitchen", "images")
    image_names = [os.path.join(images_dir, name) for name in sorted(os.listdir(images_dir))]
    images = load_and_preprocess_images(image_names).to(device)
    images = images.repeat(1, 1, 1, 1) # 100 images at 27 GB memory, 200 images cuda out of memory
    images = images.unsqueeze(0) # introducing batch dimension
    print(f"Images shape: {images.shape}")
    
    # print("Running baseline forward pass...")
    # with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    #     output_baseline, _ = model.aggregator(images)
    #     final_baseline = output_baseline[-1]

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=False) as prof:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            output_baseline = model(images)
            final_baseline = output_baseline['world_points']
            final_baseline_conf = output_baseline['world_points_conf']
    _print_profiler_tables(prof)

    apply_monkey_patch(model)

    # print("Running patched forward pass...")
    # with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    #     output_patched, _ = model.aggregator(images)
    #     final_patched = output_patched[-1]

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True, with_stack=False) as prof:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            output_patched = model(images)
            final_patched = output_patched['world_points']
            final_patched_conf = output_patched['world_points_conf']
    _print_profiler_tables(prof)

    diff = (final_baseline - final_patched).abs()
    diff_conf = (final_baseline_conf - final_patched_conf).abs()
    print("Pointmap Diff mean: ", diff.mean().item())
    print("Conf diff mean: ", diff_conf.mean().item())
    
    if diff.max().item() > 0:
        print("SUCCESS: Patch is active and affecting outputs.")
    else:
        print("WARNING: No difference detected. Patch might not be active or mask is all-1s.")

    # write a function to save the world along with conf per point as a ply file to load in vscode ply extension
    # reshape B, S, C, H, W to B, S, H, W, C
    images = images.permute(0, 1, 3, 4, 2) 
    conf_threshold = 3
    final_baseline = final_baseline[final_baseline_conf > conf_threshold].cpu().numpy()
    final_baseline_rgb = images[final_baseline_conf > conf_threshold].cpu().numpy()
    
    final_patched = final_patched[final_patched_conf > conf_threshold].cpu().numpy()
    final_patched_rgb = images[final_patched_conf > conf_threshold].cpu().numpy()

    # normalize to 0 - 1 
    # final_baseline_conf = torch.nn.functional.normalize(final_baseline_conf, p=2, dim=(2, 3)).unsqueeze(-1)
    # final_patched_conf = torch.nn.functional.normalize(final_patched_conf, p=2, dim=(2, 3)).unsqueeze(-1)
    final_baseline_conf = final_baseline_conf[final_baseline_conf > conf_threshold].cpu().numpy()
    final_patched_conf = final_patched_conf[final_patched_conf > conf_threshold].cpu().numpy()
    final_baseline_conf -= -0.4
    final_baseline_conf = final_baseline_conf / final_baseline_conf.max()
    final_patched_conf -= -0.4
    final_patched_conf = final_patched_conf / final_patched_conf.max()
    print(final_baseline_conf.shape, final_baseline_rgb.shape, final_patched_conf.min(), final_patched_conf.max())
    # add conf as rgba
    final_baseline_rgb = np.concatenate((final_baseline_rgb, final_baseline_conf[:, None]), axis=-1)
    final_patched_rgb = np.concatenate((final_patched_rgb, final_patched_conf[:, None]), axis=-1)
    cloud = trimesh.points.PointCloud(vertices=final_baseline, colors=final_baseline_rgb)
    cloud.export(".outputs/baseline.ply")
    
    cloud = trimesh.points.PointCloud(vertices=final_patched, colors=final_patched_rgb)
    cloud.export(".outputs/patched.ply")

if __name__ == "__main__":
    test_patch()
