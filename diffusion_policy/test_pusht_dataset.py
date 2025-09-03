#!/usr/bin/env python3
import argparse
import zarr
import numpy as np

def print_group(name, group):
    for k, arr in group.items():
        if isinstance(arr, zarr.Array):
            print(f"{name}/{k:15s} shape={arr.shape} dtype={arr.dtype} chunks={arr.chunks}")
        elif isinstance(arr, zarr.Group):
            print_group(f"{name}/{k}", arr)

def main():
    ap = argparse.ArgumentParser(description="Inspect an existing .zarr dataset (like pusht_cchi_v7_replay.zarr).")
    ap.add_argument("--zarr_path", required=True,
                    help="Path to the .zarr dataset (folder).")
    args = ap.parse_args()

    # open the root group
    g = zarr.open(args.zarr_path, mode="r")

    print("\n=== Stored arrays ===")
    for group_name in g.keys():
        if group_name in g:
            print(f"\nGroup: {group_name}")
            print_group(group_name, g[group_name])

    # Integrity checks (if arrays exist)
    if "data" in g and "meta" in g:
        if "episode_ends" in g["meta"]:
            T = None
            if "state" in g["data"]:
                T = g["data"]["state"].shape[0]
            elif "robot_state" in g["data"]:
                T = g["data"]["robot_state"].shape[0]

            E = g["meta"]["episode_ends"].shape[0]
            print(f"\nTotal steps (T): {T}, Episodes (E): {E}")

            if "episode_lengths" in g["meta"]:
                ep_lengths = g["meta"]["episode_lengths"][:]
                ep_ends = g["meta"]["episode_ends"][:]
                assert np.all(ep_ends == np.cumsum(ep_lengths)), "episode_ends != cumsum(episode_lengths)"
                if T is not None:
                    assert ep_ends[-1] == T, "Final episode_end != total steps"
                print("✅ Episode metadata consistent.")

    # Sample values
    print("\n=== Sample values (first step) ===")
    try:
        if "action" in g["data"]:
            print(" action:", g["data"]["action"][0])
        if "state" in g["data"]:
            print(" state :", g["data"]["state"][0])
        if "robot_state" in g["data"]:
            print(" robot_state:", g["data"]["robot_state"][0])
        if "timestamp" in g["data"]:
            print(" timestamp:", g["data"]["timestamp"][0])
    except Exception as e:
        print("⚠️ Could not fetch sample:", e)

if __name__ == "__main__":
    main()
