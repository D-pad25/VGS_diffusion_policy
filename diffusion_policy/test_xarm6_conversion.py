#!/usr/bin/env python3
import argparse
import zarr
import numpy as np
import cv2
from pathlib import Path

# Import your conversion function
from diffusion_policy.real_world.real_xarm6_data_conversion import real_data_to_replay_buffer   # <- replace with actual filename/module

def print_group(name, group):
    for k, arr in group.items():
        if isinstance(arr, zarr.Array):
            print(f"{name}/{k:15s} shape={arr.shape} dtype={arr.dtype} chunks={arr.chunks}")
        elif isinstance(arr, zarr.Group):
            print_group(f"{name}/{k}", arr)
            
def main():
    ap = argparse.ArgumentParser(description="Test the real_data_to_replay_buffer conversion.")
    ap.add_argument("--input_root", type=Path, required=True,
                    help="Folder containing Episode*/step*.pkl")
    ap.add_argument("--out_path", type=Path, default=Path("agrivla_replay.zarr"),
                    help="Output Zarr directory (will be overwritten).")
    ap.add_argument("--res", nargs=2, type=int, default=[224, 224],
                    metavar=("W", "H"), help="Resize images to (W,H).")
    ap.add_argument("--preview", action="store_true",
                    help="Show first wrist/base frame with OpenCV (press key to close).")
    args = ap.parse_args()

    # --- Run conversion ---
    print("▶️ Converting dataset...")
    store = zarr.DirectoryStore(str(args.out_path))
    rb = real_data_to_replay_buffer(
        dataset_path=str(args.input_root),
        out_store=store,
        out_resolutions=(args.res[0], args.res[1]),
        lowdim_keys=['robot_state','action','timestamp','episode_ends','episode_lengths'],
        image_keys=['wrist_rgb','base_rgb']
    )
    print("✅ Conversion complete.")

    # --- Inspect contents ---
    g = zarr.open(store, mode='r')
    print("\n=== Stored arrays ===")
    for group_name in ["data", "meta"]:
        if group_name in g:
            group = g[group_name]
            print(f"\nGroup: {group_name}")
            print(f"\nGroup: {group_name}")
            print_group(group_name, group)


    # --- Integrity checks ---
    T = g["robot_state"].shape[0]
    E = g["episode_ends"].shape[0]
    print(f"\nTotal steps (T): {T}, Episodes (E): {E}")

    # episode_lengths vs episode_ends consistency
    ep_lengths = g["episode_lengths"][:]
    ep_ends = g["episode_ends"][:]
    assert np.all(ep_ends == np.cumsum(ep_lengths)), "episode_ends != cumsum(episode_lengths)"
    assert ep_ends[-1] == T, "Final episode_end != total steps"
    print("✅ Episode metadata consistent.")

    # low-dim checks
    assert g["robot_state"].shape == (T, 7)
    assert g["action"].shape == (T, 7)
    print("✅ robot_state and action have shape (T,7).")

    # sample values
    i = 0
    rs = g["robot_state"][i]
    act = g["action"][i]
    print(f"\nSample step {i}:")
    print(" robot_state:", rs)
    print(" action     :", act)
    print(" timestamp  :", g['timestamp'][i])

    # optional preview
    if args.preview:
        for cam in ["wrist_rgb", "base_rgb"]:
            if cam in g:
                img = g[cam][i]   # (H,W,3) uint8
                cv2.imshow(cam, img[..., ::-1])  # BGR for OpenCV
        print("Press any key in the OpenCV window(s) to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
