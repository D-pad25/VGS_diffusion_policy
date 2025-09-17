#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, numpy as np, cv2, zarr
from typing import Optional, Tuple, List
from tqdm import tqdm
from diffusion_policy.common.replay_buffer import ReplayBuffer
try:
    from diffusion_policy.real_world.real_xarm6_data_conversion import real_data_to_replay_buffer
except Exception:
    real_data_to_replay_buffer = None

def load_replay_buffer(cache_zarr_zip: Optional[str], dataset_path: Optional[str],
                       out_resolutions: Tuple[int,int]=(224,224)) -> ReplayBuffer:
    assert cache_zarr_zip or dataset_path
    if cache_zarr_zip:
        with zarr.ZipStore(cache_zarr_zip, mode="r") as zip_store:
            return ReplayBuffer.copy_from_store(src_store=zip_store, store=zarr.MemoryStore())
    assert real_data_to_replay_buffer is not None, "Need cache_zarr_zip or real_data_to_replay_buffer."
    return real_data_to_replay_buffer(dataset_path=dataset_path, out_store=zarr.MemoryStore(),
                                      out_resolutions=out_resolutions,
                                      lowdim_keys=['robot_state','action','timestamp','episode_ends','episode_lengths'],
                                      image_keys=['wrist_rgb','base_rgb'])

def episode_range(episode_ends: np.ndarray, epi_idx: int):
    end = int(episode_ends[epi_idx]); start = 0 if epi_idx == 0 else int(episode_ends[epi_idx-1]); return start, end

def ensure_size(img: np.ndarray, hw: Tuple[int,int]) -> np.ndarray:
    h,w = hw;  return img if img.shape[:2]==(h,w) else cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)

def put_text_outlined(img, text, org, font, scale, color=(255,255,255), thickness=1, outline=2):
    cv2.putText(img, text, org, font, scale, (0,0,0), outline, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)

def draw_card(img, lines: List[str], anchor: str, margin: int, pad: int, font, scale, lh, alpha=0.45):
    H,W,_ = img.shape
    # measure width
    wmax = 0; htot = len(lines)*lh
    for s in lines:
        w,_ = cv2.getTextSize(s, font, scale, 1)[0]
        wmax = max(wmax, w)
    # anchor position
    if anchor == "top":
        x0, y0 = margin, margin
    elif anchor == "bl":
        x0, y0 = margin, H - margin - (htot + 2*pad)
    elif anchor == "br":
        x0, y0 = W - margin - (wmax + 2*pad), H - margin - (htot + 2*pad)
    else:
        x0, y0 = margin, margin
    x1, y1 = x0 + wmax + 2*pad, y0 + htot + 2*pad
    overlay = img.copy()
    cv2.rectangle(overlay, (x0,y0), (x1,y1), (0,0,0), -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, dst=img)
    # text
    y = y0 + pad + int(0.8*lh)
    for s in lines:
        put_text_outlined(img, s, (x0+pad, y), font, scale)
        y += lh

def fmt_vec(v: np.ndarray, k: int, dec: int=3) -> str:
    return "[" + ", ".join(f"{float(x):.{dec}f}" for x in v[:k]) + "]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_zarr_zip", type=str, default=None)
    ap.add_argument("--dataset_path", type=str, default=None)
    ap.add_argument("--episode_index", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--resize_h", type=int, default=360)   # bigger default for readability
    ap.add_argument("--resize_w", type=int, default=360)
    ap.add_argument("--show_joints", type=int, default=3)  # compact
    ap.add_argument("--decimals", type=int, default=3)
    args = ap.parse_args()

    rb = load_replay_buffer(args.cache_zarr_zip, args.dataset_path,
                            out_resolutions=(args.resize_w, args.resize_h))

    have_wrist = 'wrist_rgb' in rb; have_base = 'base_rgb' in rb
    assert have_wrist or have_base, "Need at least one camera key."

    ep_ends = rb.episode_ends[:]
    s,e = episode_range(ep_ends, args.episode_index)
    T = e - s;  assert T > 0

    wrist = rb['wrist_rgb'][s:e] if have_wrist else None
    base  = rb['base_rgb'][s:e]  if have_base else None
    q     = rb['robot_state'][s:e]
    u     = rb['action'][s:e]
    ts    = rb['timestamp'][s:e] if 'timestamp' in rb else None

    # FPS
    if args.fps > 0: fps = float(args.fps)
    else:
        if ts is not None and len(ts) > 1:
            dt = np.diff(ts); dt = dt[dt>0]
            fps = (1.0/float(np.median(dt))) if dt.size>0 else 30.0
        else: fps = 30.0

    H,W = args.resize_h, args.resize_w
    def compose(i):
        views = []
        if have_wrist: views.append(ensure_size(wrist[i], (H,W)))
        if have_base:  views.append(ensure_size(base[i],  (H,W)))
        return views[0] if len(views)==1 else np.hstack(views)

    sample = compose(0)
    vh, vw = sample.shape[:2]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
    assert writer.isOpened(), f"Cannot open writer: {args.out}"

    # adaptive font scaling
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.38, min(vh/540.0, 0.9))   # ~0.5 for 360p height, smaller for 224p
    th, _ = cv2.getTextSize("Hg", font, scale, 1)[0]
    lh = int(th*1.35)

    last_u = None
    with tqdm(total=T, desc=f"Episode {args.episode_index}") as pbar:
        for i in range(T):
            frame = compose(i).copy()

            # --- top HUD (slim) ---
            meta = [
                f"epi={args.episode_index}  step={i}/{T-1}  t={s+i}"
                + (f"  time={ts[i]:.3f}" if ts is not None else "")
            ]
            draw_card(frame, meta, anchor="top", margin=6, pad=6, font=font, scale=scale, lh=lh, alpha=0.35)

            # --- bottom-left: q ---
            maxj = max(1, min(args.show_joints, 6))
            qi = q[i]; ui = u[i]
            q_lines = [
                f"q[0:{maxj}]: {fmt_vec(qi, maxj, args.decimals)}",
                f"g: {qi[6]:.{args.decimals}f}"
            ]
            draw_card(frame, q_lines, anchor="bl", margin=8, pad=6, font=font, scale=scale, lh=lh, alpha=0.35)

            # --- bottom-right: u & Δu ---
            if last_u is not None:
                du = ui - last_u
                du_str = fmt_vec(du, maxj, args.decimals)
                dg = f"{du[6]:.{args.decimals}f}"
            else:
                du_str = "[" + ", ".join(["0.000"]*maxj) + "]"
                dg = f"{0.0:.{args.decimals}f}"
            u_lines = [
                f"u[0:{maxj}]: {fmt_vec(ui, maxj, args.decimals)}   g: {ui[6]:.{args.decimals}f}",
                f"Δu:        {du_str}   g: {dg}"
            ]
            draw_card(frame, u_lines, anchor="br", margin=8, pad=6, font=font, scale=scale, lh=lh, alpha=0.35)

            writer.write(frame); last_u = ui; pbar.update(1)

    writer.release()
    print(f"Saved: {args.out}  ({vw}x{vh} @ {fps:.2f} fps, frames={T})")

if __name__ == "__main__":
    main()
