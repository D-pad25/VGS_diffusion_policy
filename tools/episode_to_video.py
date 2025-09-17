#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
episode_to_video.py
Create a video for one trajectory (episode) with text overlays of key data.

Inputs supported:
  1) --cache_zarr_zip <path/to/hash.zarr.zip>   (fastest; from your RealXArm6ImageDataset cache)
  2) --dataset_path  <path/to/raw_episodes_root>  (will build a ReplayBuffer in memory)

Overlays per frame:
  - episode index, step index in episode, global t
  - timestamp (if present)
  - robot_state q[0..5] + gripper
  - action u[0..5] + gripper
  - Δu (this - previous) for quick “jerk/change” inspection

Layout:
  - If both 'wrist_rgb' and 'base_rgb' exist -> horizontal concat [wrist | base]
  - If only one exists -> single view
"""

import os
import argparse
import numpy as np
import cv2
import zarr
from typing import Optional, Tuple, List
from tqdm import tqdm

# Import your repo utilities (as you already use):
from diffusion_policy.common.replay_buffer import ReplayBuffer
try:
    # Your conversion util (provided in your message)
    from diffusion_policy.real_world.real_xarm6_data_conversion import real_data_to_replay_buffer
except Exception:
    real_data_to_replay_buffer = None


def load_replay_buffer(
    cache_zarr_zip: Optional[str],
    dataset_path: Optional[str],
    out_resolutions: Tuple[int, int]=(224, 224)
) -> ReplayBuffer:
    """
    Load a ReplayBuffer from either a zipped zarr cache or by building from raw episodes.
    """
    assert cache_zarr_zip or dataset_path, "Provide --cache_zarr_zip or --dataset_path"
    if cache_zarr_zip:
        assert os.path.isfile(cache_zarr_zip), f"Not found: {cache_zarr_zip}"
        with zarr.ZipStore(cache_zarr_zip, mode="r") as zip_store:
            # Copy into fast MemoryStore for quick random access
            rb = ReplayBuffer.copy_from_store(src_store=zip_store, store=zarr.MemoryStore())
        return rb

    # Fallback: build from raw files
    assert real_data_to_replay_buffer is not None, (
        "real_data_to_replay_buffer import failed and no cache provided."
    )
    rb = real_data_to_replay_buffer(
        dataset_path=dataset_path,
        out_store=zarr.MemoryStore(),
        out_resolutions=out_resolutions,      # resize frames
        lowdim_keys=['robot_state', 'action', 'timestamp', 'episode_ends', 'episode_lengths'],
        image_keys=['wrist_rgb', 'base_rgb']
    )
    return rb


def episode_range(episode_ends: np.ndarray, epi_idx: int) -> Tuple[int, int]:
    """
    Convert episode index to [start, end) global frame indices.
    """
    assert 0 <= epi_idx < len(episode_ends), f"episode_index out of range (0..{len(episode_ends)-1})"
    end = int(episode_ends[epi_idx])
    start = 0 if epi_idx == 0 else int(episode_ends[epi_idx - 1])
    return start, end


def ensure_same_size(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """
    Resize HxWx3 uint8 image to target (H, W).
    """
    th, tw = target_hw
    if img.shape[0] == th and img.shape[1] == tw:
        return img
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)


def draw_text_block(
    canvas: np.ndarray,
    lines: List[str],
    topleft: Tuple[int, int]=(8, 8),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float=0.45,
    line_height: int=18,
    text_color=(255, 255, 255),
    box_color=(0, 0, 0),
    alpha: float=0.55,
    padding: int=6,
    thickness: int=1
) -> None:
    """
    Draws a semi-transparent text box with multiple lines.
    """
    x0, y0 = topleft
    # Determine box size
    (w_max, h_total) = (0, 0)
    for i, text in enumerate(lines):
        (w, h) = cv2.getTextSize(text, font, font_scale, thickness)[0]
        w_max = max(w_max, w)
    h_total = len(lines) * line_height

    # Background
    x1, y1 = x0 + w_max + 2*padding, y0 + h_total + 2*padding
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), box_color, -1)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)

    # Text lines
    y = y0 + padding + int(line_height * 0.75)
    for text in lines:
        cv2.putText(canvas, text, (x0 + padding, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += line_height


def format_vec(v: np.ndarray, k: int) -> str:
    """
    Format first k elements with compact width; assumes v is 1D.
    """
    return "[" + ", ".join(f"{float(x): .3f}" for x in v[:k]) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_zarr_zip", type=str, default=None,
                    help="Path to zipped zarr cache (e.g., <hash>.zarr.zip). Preferred.")
    ap.add_argument("--dataset_path", type=str, default=None,
                    help="Raw episodes root. Uses real_data_to_replay_buffer if no cache given.")
    ap.add_argument("--episode_index", type=int, default=0, help="Episode to render")
    ap.add_argument("--out", type=str, required=True, help="Output video path (e.g., out.mp4)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="FPS override. If 0, will infer from timestamps or default to 30.")
    ap.add_argument("--max_joints", type=int, default=6,
                    help="How many joints from q/u to print (before gripper).")
    ap.add_argument("--resize_h", type=int, default=224, help="Resize height for each view")
    ap.add_argument("--resize_w", type=int, default=224, help="Resize width for each view")
    args = ap.parse_args()

    rb = load_replay_buffer(args.cache_zarr_zip, args.dataset_path, out_resolutions=(args.resize_w, args.resize_h))

    # Check available keys
    have_wrist = 'wrist_rgb' in rb
    have_base  = 'base_rgb' in rb
    assert have_wrist or have_base, "ReplayBuffer must contain at least one of 'wrist_rgb' or 'base_rgb'."

    # Episode segmentation
    ep_ends = rb.episode_ends[:]
    start, end = episode_range(ep_ends, args.episode_index)
    T = end - start
    assert T > 0, "Empty episode."

    # Pull arrays (slice once for speed)
    wrist = rb['wrist_rgb'][start:end] if have_wrist else None  # (T,H,W,3) uint8
    base  = rb['base_rgb'][start:end]  if have_base  else None
    q     = rb['robot_state'][start:end]               # (T,7) float
    u     = rb['action'][start:end]                    # (T,7) float
    tstamp = rb['timestamp'][start:end] if 'timestamp' in rb else None

    # Infer FPS
    if args.fps > 0:
        fps = float(args.fps)
    else:
        if tstamp is not None and len(tstamp) > 1:
            dt = np.diff(tstamp).astype(np.float64)
            med_dt = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
            fps = (1.0 / med_dt) if med_dt > 0 else 30.0
        else:
            fps = 30.0

    # Compute final frame size
    H, W = args.resize_h, args.resize_w
    # Compose a dummy frame to set writer size
    def make_frame(idx: int) -> np.ndarray:
        views = []
        if have_wrist:
            views.append(ensure_same_size(wrist[idx], (H, W)))
        if have_base:
            views.append(ensure_same_size(base[idx], (H, W)))
        frame = views[0] if len(views) == 1 else np.hstack(views)
        return frame

    # Pre-create writer
    sample_frame = make_frame(0)
    vid_h, vid_w = sample_frame.shape[0], sample_frame.shape[1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, fps, (vid_w, vid_h))
    assert vw.isOpened(), f"Failed to open VideoWriter for {args.out}"

    # Render loop
    last_u = None
    with tqdm(total=T, desc=f"Rendering episode {args.episode_index}", unit="f") as pbar:
        for i in range(T):
            frame = make_frame(i).copy()

            # Prepare overlay lines
            global_idx = start + i
            step_in_ep = i
            lines = []
            lines.append(f"epi={args.episode_index}  step={step_in_ep}/{T-1}  t={global_idx}")
            if tstamp is not None:
                lines.append(f"timestamp: {tstamp[i]:.3f}")

            # q and u
            q_i = q[i]
            u_i = u[i]
            maxj = min(args.max_joints, 6)

            lines.append(f"q[0:{maxj}]: {format_vec(q_i, maxj)}  g: {q_i[6]: .3f}")
            lines.append(f"u[0:{maxj}]: {format_vec(u_i, maxj)}  g: {u_i[6]: .3f}")

            # Δu vs previous
            if last_u is not None:
                du = u_i - last_u
                lines.append(f"d u:      {format_vec(du, maxj)}  g: {du[6]: .3f}")
            else:
                lines.append(f"d u:      {'[ 0.000, ...]':>18}  g:  0.000")

            draw_text_block(frame, lines, topleft=(8, 8), font_scale=0.48, line_height=19, alpha=0.55)

            vw.write(frame)
            last_u = u_i
            pbar.update(1)

    vw.release()
    print(f"Saved video to: {args.out}\nFPS used: {fps:.2f}, frames: {T}, size: {vid_w}x{vid_h}")


if __name__ == "__main__":
    main()
