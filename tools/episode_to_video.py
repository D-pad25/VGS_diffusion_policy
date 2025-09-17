#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
episode_to_video.py  (Script A ➜ styled like Script B)

- Loads from a zipped zarr ReplayBuffer (preferred) OR builds from raw episodes.
- Renders a single episode to a video with a clean header HUD similar to Script B:
    Prompt line
    Joints: [j0..j5] | Gripper
    Actions: [u0..u5] | Gripper cmd
- Wrist|Base views are stacked horizontally after resizing to a target height.

Examples (HPC):
  python -m tools.episode_to_video \
    --cache_zarr_zip /mnt/hpccs01/home/n10813934/data/diffusion/converted_padded/93aacbb843da37ac6da4922061c722db.zarr.zip \
    --episode_index 0 \
    --out /mnt/hpccs01/home/n10813934/data/diffusion/videos/episode0.mp4 \
    --fps 30 --target_h 480 \
    --prompt "Prompt: Pick a ripe, red tomato and drop it in the blue bucket."
"""

import os
import argparse
from typing import Optional, Tuple, List

import numpy as np
import cv2
import zarr
from tqdm import tqdm

from diffusion_policy.common.replay_buffer import ReplayBuffer
try:
    from diffusion_policy.real_world.real_xarm6_data_conversion import real_data_to_replay_buffer
except Exception:
    real_data_to_replay_buffer = None


# -------------------------- I/O helpers --------------------------
def load_replay_buffer(
    cache_zarr_zip: Optional[str],
    dataset_path: Optional[str],
    out_resolutions: Tuple[int, int]=(224, 224)
) -> ReplayBuffer:
    assert cache_zarr_zip or dataset_path, "Provide --cache_zarr_zip or --dataset_path"
    if cache_zarr_zip:
        assert os.path.isfile(cache_zarr_zip), f"Not found: {cache_zarr_zip}"
        with zarr.ZipStore(cache_zarr_zip, mode="r") as zip_store:
            return ReplayBuffer.copy_from_store(src_store=zip_store, store=zarr.MemoryStore())
    assert real_data_to_replay_buffer is not None, "real_data_to_replay_buffer not available."
    return real_data_to_replay_buffer(
        dataset_path=dataset_path,
        out_store=zarr.MemoryStore(),
        out_resolutions=out_resolutions,
        lowdim_keys=['robot_state', 'action', 'timestamp', 'episode_ends', 'episode_lengths'],
        image_keys=['wrist_rgb', 'base_rgb']
    )


def episode_range(episode_ends: np.ndarray, epi_idx: int):
    assert 0 <= epi_idx < len(episode_ends), f"episode_index out of range (0..{len(episode_ends)-1})"
    end = int(episode_ends[epi_idx])
    start = 0 if epi_idx == 0 else int(episode_ends[epi_idx-1])
    return start, end


# ----------------------- Frame compositor -----------------------
def safe_to_rgb(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    new_w = int(round(w * (target_h / float(h))))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def put_text_outlined(img, text, org, font, scale, color=(255,255,255),
                      thickness=1, outline=2):
    cv2.putText(img, text, (org[0]+2, org[1]+2), font, scale, (0,0,0), outline, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def render_frame(wrist_img: np.ndarray,
                 base_img: Optional[np.ndarray],
                 q7: np.ndarray,
                 u7: np.ndarray,
                 prompt_text: str,
                 header_h: int = 110,
                 font_scale: float = 0.6) -> np.ndarray:
    # Normalize inputs & equalize heights
    wrist_img = safe_to_rgb(wrist_img)
    if base_img is not None:
        base_img = safe_to_rgb(base_img)
        h = min(wrist_img.shape[0], base_img.shape[0])
        wrist_img = resize_to_height(wrist_img, h)
        base_img  = resize_to_height(base_img,  h)
        frame_rgb = np.hstack([wrist_img, base_img])
    else:
        frame_rgb = wrist_img

    # (Keep color order as-is; if your stored frames are RGB and you prefer cv2 colors, uncomment below)
    # frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    frame = frame_rgb.copy()

    # Header overlay bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], min(header_h, frame.shape[0])), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Compose header lines (Script B style)
    j_txt = "Joints: [" + ", ".join(f"{j:+.2f}" for j in q7[:6]) + f"]  |  Gripper: {q7[-1]:+.4f}"
    a_txt = "Actions: [" + ", ".join(f"{a:+.2f}" for a in u7[:6]) + f"]  |  Gripper cmd: {u7[-1]:+.4f}"
    lines = [prompt_text, j_txt, a_txt]

    # Draw outlined text
    font = cv2.FONT_HERSHEY_DUPLEX
    y = 28
    for t in lines:
        put_text_outlined(frame, t, (12, y), font, font_scale)
        y += 26

    return frame


# --------------------------- Main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_zarr_zip", type=str, default=None,
                    help="Path to zipped zarr cache (e.g., <hash>.zarr.zip).")
    ap.add_argument("--dataset_path", type=str, default=None,
                    help="Raw episodes root. Uses real_data_to_replay_buffer if no cache given.")
    ap.add_argument("--episode_index", type=int, default=0, help="Episode to render.")
    ap.add_argument("--out", type=str, required=True, help="Output video path (e.g., out.mp4)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="FPS override. If 0, infer from timestamps else 30.")
    ap.add_argument("--target_h", type=int, default=480,
                    help="Target height per view before horizontal stack (improves readability).")
    ap.add_argument("--prompt", type=str, default="Prompt: Pick a ripe, red tomato and drop it in the blue bucket.",
                    help="Header prompt text to show on the first line.")
    args = ap.parse_args()

    rb = load_replay_buffer(args.cache_zarr_zip, args.dataset_path)

    have_wrist = 'wrist_rgb' in rb
    have_base  = 'base_rgb' in rb
    assert have_wrist or have_base, "ReplayBuffer must contain at least one of 'wrist_rgb' or 'base_rgb'."

    ep_ends = rb.episode_ends[:]
    start, end = episode_range(ep_ends, args.episode_index)
    T = end - start
    assert T > 0, "Empty episode."

    wrist = rb['wrist_rgb'][start:end] if have_wrist else None
    base  = rb['base_rgb'][start:end]  if have_base else None
    q     = rb['robot_state'][start:end]     # (T,7)
    u     = rb['action'][start:end]          # (T,7)
    ts    = rb['timestamp'][start:end] if 'timestamp' in rb else None

    # Determine FPS
    if args.fps > 0:
        fps = float(args.fps)
    else:
        if ts is not None and len(ts) > 1:
            dt = np.diff(ts).astype(np.float64)
            dt = dt[dt > 0]
            fps = (1.0 / float(np.median(dt))) if dt.size > 0 else 30.0
        else:
            fps = 30.0

    # Compose one sample to size the writer (resize to target_h first)
    def compose_raw(i):
        w = wrist[i] if wrist is not None else None
        b = base[i]  if base  is not None else None
        if w is None and b is None:
            raise RuntimeError("No frames available.")
        # Match Script B: preserve aspect ratio per view, then stack
        if w is not None:
            w = resize_to_height(safe_to_rgb(w), args.target_h)
        if b is not None:
            b = resize_to_height(safe_to_rgb(b), args.target_h)
        return w, b

    w0, b0 = compose_raw(0)
    sample = render_frame(w0, b0, q[0], u[0], args.prompt)
    H, W = sample.shape[:2]
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    assert vw.isOpened(), f"Failed to open VideoWriter for {args.out}"

    with tqdm(total=T, desc=f"Rendering episode {args.episode_index}", unit="f") as pbar:
        for i in range(T):
            wi, bi = compose_raw(i)
            frame = render_frame(wi, bi, q[i], u[i], args.prompt)
            vw.write(frame)
            pbar.update(1)

    vw.release()
    print(f"Saved video to: {args.out}  ({W}x{H} @ {fps:.2f} fps, frames={T})")


if __name__ == "__main__":
    main()
