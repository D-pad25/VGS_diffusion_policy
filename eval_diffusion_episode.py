#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained Diffusion Policy on a recorded episode of step-wise .pkl files.
The goal is to mirror your live evaluation loop as closely as possible while
running *open-loop* on logged observations.

Inputs
------
- An episode directory containing N step .pkl files (each file = one step).
  Each .pkl must contain at least:
    'wrist_rgb' (H,W,3 uint8), 'base_rgb' (H,W,3 uint8),
    'joint_positions' (7,), 'gripper_position' (scalar), and ideally 'control' (7,)
  Optional: 'wrist_depth', 'base_depth', 'joint_velocities', 'ee_pos_quat'

- A diffusion-policy checkpoint (.ckpt) trained with shape_meta that expects:
    obs: { wrist_rgb: rgb, base_rgb: rgb, robot_state: low_dim[7] }
    action: [7]

Outputs
-------
- Metrics printed to console and saved to <out_dir>/metrics.csv
- Optional per-step JSON logs in <out_dir>/steps/*.json (enable with --save_steps)

Notes
-----
- We feed the model *exactly* the same observation tuple you use live:
    base_rgb (T,3,H,W), wrist_rgb (T,3,H,W), robot_state (T,7)
  using an ObsBuffer of size n_obs_steps from the checkpoint cfg (default 2).
- We execute the returned action chunk step-by-step to mimic your control loop.
- For quantitative evaluation, we compare predicted actions against the dataset's
  recorded 'control' vectors (if present). We also compute a proxy metric that
  compares the predicted action to the observed joint delta (q[t+1]-q[t]).
"""
from __future__ import annotations

import os
import re
import io
import cv2
import sys
import json
import time
import math
import dill
import glob
import tyro
import torch
import hydra
import pickle
import shutil
import random
import pathlib
import warnings
import numpy as np
import dataclasses as dc
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass

# ---------- Resize helper (prefer your existing implementation) ----------
from diffusion_policy.common.cv2_util import get_image_transform

# Image conversion
img_tf_cache = {}  # (k, in_w,in_h,out_w,out_h) -> callable
def get_tf(k, in_w, in_h, out_w, out_h):
    key = (k, in_w, in_h, out_w, out_h)
    if key not in img_tf_cache:
        img_tf_cache[key] = get_image_transform(
            input_res=(in_w, in_h), output_res=(out_w, out_h), bgr_to_rgb=False
        )
    return img_tf_cache[key]

try:
    # If you have this in your repo
    from tools.resize_pkl import resize_with_pad as _resize_with_pad
except Exception:
    from PIL import Image
    def _resize_with_pad(img: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
        """Minimal tf.image.resize_with_pad equivalent for a single HWC image."""
        ih, iw = img.shape[:2]
        oh, ow = height, width
        # compute resize keeping aspect
        if iw/ih >= ow/oh:
            rw = ow
            rh = math.ceil(rw / iw * ih)
        else:
            rh = oh
            rw = math.ceil(rh / ih * iw)
        pil = Image.fromarray(img)
        pil = pil.resize((rw, rh), resample=method)
        resized = np.asarray(pil)
        # center crop if larger than target
        y0 = max((rh - oh)//2, 0)
        x0 = max((rw - ow)//2, 0)
        crop = resized[y0:y0+oh, x0:x0+ow]
        # pad if smaller than target
        out = np.zeros((oh, ow, img.shape[2]), dtype=img.dtype)
        y_off = max((oh - crop.shape[0])//2, 0)
        x_off = max((ow - crop.shape[1])//2, 0)
        out[y_off:y_off+crop.shape[0], x_off:x_off+crop.shape[1]] = crop
        return out


def to_chw01(img_hwc_uint8: np.ndarray) -> np.ndarray:
    img = img_hwc_uint8.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))  # CHW


class ObsBuffer:
    def __init__(self, n_obs_steps: int = 2, H: int = 224, W: int = 224):
        self.n = n_obs_steps
        self.H, self.W = H, W
        self.base_rgb = deque(maxlen=self.n)
        self.wrist_rgb = deque(maxlen=self.n)
        self.robot_state = deque(maxlen=self.n)

    def push(self, obs: Dict):
        base = _resize_with_pad(obs["base_rgb"], self.H, self.W)
        wrist = _resize_with_pad(obs["wrist_rgb"], self.H, self.W)
        self.base_rgb.append(to_chw01(base))
        self.wrist_rgb.append(to_chw01(wrist))
        state7 = np.concatenate([obs["joint_position"], np.atleast_1d(obs["gripper_position"]).astype(np.float32)])
        self.robot_state.append(state7.astype(np.float32))
        while len(self.base_rgb) < self.n:
            self.base_rgb.appendleft(self.base_rgb[0].copy())
            self.wrist_rgb.appendleft(self.wrist_rgb[0].copy())
            self.robot_state.appendleft(self.robot_state[0].copy())

    def ready(self) -> bool:
        return len(self.base_rgb) == self.n

    def as_np_dict(self) -> Dict[str, np.ndarray]:
        base = np.stack(list(self.base_rgb), axis=0)
        wrist = np.stack(list(self.wrist_rgb), axis=0)
        state = np.stack(list(self.robot_state), axis=0)
        return dict(base_rgb=base, wrist_rgb=wrist, robot_state=state)


# ---------- Episode Loader ----------
@dc.dataclass
class EpisodeStep:
    idx: int
    obs: Dict[str, np.ndarray]
    control: Optional[np.ndarray]  # (7,) if available


def _coerce_obs_keys(d: Dict) -> Dict:
    # Map dataset fields -> live obs keys consumed by your loop
    out = {}
    # images
    for cam_key, alias in [("base_rgb", "base_rgb"), ("wrist_rgb", "wrist_rgb")]:
        if cam_key not in d:
            raise KeyError(f"Missing key '{cam_key}' in step")
        out[alias] = d[cam_key]
    # joints
    jp = None
    if "joint_position" in d:
        jp = d["joint_position"]
    elif "joint_positions" in d:
        jp = d["joint_positions"]
    else:
        raise KeyError("Missing 'joint_position(s)' in step")
    out["joint_position"] = np.asarray(jp, dtype=np.float32).reshape(-1)[:6]
    # gripper
    if "gripper_position" in d:
        gp = d["gripper_position"]
        gp = np.asarray(gp, dtype=np.float32).reshape(()).item()
    else:
        warnings.warn("Missing 'gripper_position' in step; defaulting to 0.0")
        gp = 0.0
    out["gripper_position"] = np.float32(gp)
    return out


def _read_pkl(path: str) -> Dict:
    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return dill.load(f)


def load_episode(episode_dir: str) -> List[EpisodeStep]:
    paths = sorted(glob.glob(os.path.join(episode_dir, "*.pkl")))
    if not paths:
        raise FileNotFoundError(f"No .pkl files found in {episode_dir}")

    # Try to natural sort by any integer in the filename
    def nat_key(p: str):
        nums = re.findall(r"\d+", os.path.basename(p))
        return [int(n) for n in nums] if nums else [0]
    paths.sort(key=nat_key)

    steps: List[EpisodeStep] = []
    for i, p in enumerate(paths):
        data = _read_pkl(p)
        obs = _coerce_obs_keys(data)
        control = None
        for k in ("control", "action", "command"):
            if k in data:
                control = np.asarray(data[k], dtype=np.float32).reshape(-1)[:7]
                break
        steps.append(EpisodeStep(idx=i, obs=obs, control=control))
    return steps


# ---------- Diffusion Policy Loader ----------
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


def load_diffusion_policy(ckpt_path: str) -> Tuple[BaseImagePolicy, dict, int, torch.device]:
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy: BaseImagePolicy = workspace.model
    try:
        if getattr(cfg, "training", None) and getattr(cfg.training, "use_ema", False):
            policy = workspace.ema_model
    except Exception:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval().to(device)

    if not hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = 16

    n_obs_steps = int(getattr(cfg, "n_obs_steps", 2))
    return policy, cfg, n_obs_steps, device


def make_obs_torch(obs_np: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in obs_np.items():
        t = torch.from_numpy(v).unsqueeze(0).to(device)  # (1, T, ...)
        out[k] = t
    return out


# ---------- Metrics ----------
@dataclass
class StepMetrics:
    step: int
    mae_all7: float
    rmse_all7: float
    mae_joints6: float
    rmse_joints6: float
    mae_grip: float
    rmse_grip: float
    mae_vs_qdelta_all7: Optional[float] = None
    rmse_vs_qdelta_all7: Optional[float] = None


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


# ---------- Main Evaluation Loop ----------
@dataclass
class Args:
    ckpt: str = "/home/d_pad25/Thesis/diffusion_checkpoints/latest.ckpt"
    episode_dir: str = "/home/d_pad25/Thesis/Data/Evalauation/0828_173511"
    out_dir: str = "/home/d_pad25/Thesis/Data/Evalauation/eval_out"
    control_hz: float = 30.0
    H: int = 224
    W: int = 224
    delta_threshold_deg: float = 0.25  # only for info; offline eval does not interpolate robot
    save_steps: bool = False


def main(args: Args):
    os.makedirs(args.out_dir, exist_ok=True)
    steps = load_episode(args.episode_dir)
    print(f"[EPISODE] Loaded {len(steps)} steps from {args.episode_dir}")

    policy, cfg, n_obs_steps, device = load_diffusion_policy(args.ckpt)
    print(f"[POLICY] Device={device}, n_obs_steps={n_obs_steps}")
    try:
        policy.reset()
    except Exception:
        pass

    buf = ObsBuffer(n_obs_steps=n_obs_steps, H=args.H, W=args.W)

    # Metrics accumulation
    metrics: List[StepMetrics] = []

    # Control rate emulation (no sleep by default; we match index->index)
    dt = 1.0 / float(args.control_hz)

    action_chunk = None
    chunk_len = 0
    chunk_used = 0

    # We'll iterate through the episode in lockstep with the dataset index.
    for i in range(len(steps)):
        obs = steps[i].obs
        buf.push(obs)

        if (action_chunk is None) or (chunk_used >= chunk_len):
            if not buf.ready():
                continue  # need T frames to start
            with torch.no_grad():
                obs_np = buf.as_np_dict()
                obs_t = make_obs_torch(obs_np, device)
                model_obs = {"base_rgb": obs_t["base_rgb"],
                             "wrist_rgb": obs_t["wrist_rgb"],
                             "robot_state": obs_t["robot_state"]}
                result = policy.predict_action(model_obs)
                action_chunk = result["action"][0].detach().to("cpu").numpy()  # (A,7)
                chunk_len = action_chunk.shape[0]
                chunk_used = 0
                if action_chunk.shape[-1] != 7:
                    raise RuntimeError(f"Unexpected action shape {action_chunk.shape}; expected (*,7)")
                print(f"[POLICY] New action chunk @ step {i}: {chunk_len} actions")

        # Current action from chunk (open-loop)
        a_pred = action_chunk[chunk_used]
        chunk_used += 1

        # Ground-truth action if available at this index
        a_true = steps[i].control

        # Compute metrics if we have labels
        if a_true is not None:
            mae_all7 = _mae(a_pred, a_true)
            rmse_all7 = _rmse(a_pred, a_true)
            mae_j = _mae(a_pred[:6], a_true[:6])
            rmse_j = _rmse(a_pred[:6], a_true[:6])
            mae_g = _mae(np.array([a_pred[-1]]), np.array([a_true[-1]]))
            rmse_g = _rmse(np.array([a_pred[-1]]), np.array([a_true[-1]]))
        else:
            mae_all7 = rmse_all7 = mae_j = rmse_j = mae_g = rmse_g = float('nan')

        # Proxy metric vs observed joint delta (requires i+1)
        mae_vs_qdelta = rmse_vs_qdelta = None
        if i + 1 < len(steps):
            q_now = np.concatenate([steps[i].obs["joint_position"], np.atleast_1d(steps[i].obs["gripper_position"])])
            q_nxt = np.concatenate([steps[i+1].obs["joint_position"], np.atleast_1d(steps[i+1].obs["gripper_position"])])
            q_delta = q_nxt - q_now
            mae_vs_qdelta = _mae(a_pred, q_delta)
            rmse_vs_qdelta = _rmse(a_pred, q_delta)

        metrics.append(StepMetrics(
            step=i,
            mae_all7=mae_all7,
            rmse_all7=rmse_all7,
            mae_joints6=mae_j,
            rmse_joints6=rmse_j,
            mae_grip=mae_g,
            rmse_grip=rmse_g,
            mae_vs_qdelta_all7=mae_vs_qdelta,
            rmse_vs_qdelta_all7=rmse_vs_qdelta,
        ))

        if args.save_steps:
            rec = dict(
                step=i,
                a_pred=a_pred.tolist(),
                a_true=(a_true.tolist() if a_true is not None else None),
                q=steps[i].obs["joint_position"].tolist(),
                g=float(steps[i].obs["gripper_position"]),
                mae_all7=mae_all7,
                rmse_all7=rmse_all7,
                mae_vs_qdelta_all7=mae_vs_qdelta,
                rmse_vs_qdelta_all7=rmse_vs_qdelta,
            )
            step_dir = os.path.join(args.out_dir, "steps")
            os.makedirs(step_dir, exist_ok=True)
            with open(os.path.join(step_dir, f"step_{i:06d}.json"), "w") as f:
                json.dump(rec, f, indent=2)

    # Write CSV summary
    import csv
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step","mae_all7","rmse_all7","mae_joints6","rmse_joints6","mae_grip","rmse_grip",
                    "mae_vs_qdelta_all7","rmse_vs_qdelta_all7"]) 
        for m in metrics:
            w.writerow([m.step,m.mae_all7,m.rmse_all7,m.mae_joints6,m.rmse_joints6,m.mae_grip,m.rmse_grip,
                        m.mae_vs_qdelta_all7,m.rmse_vs_qdelta_all7])

    # Print aggregate stats
    def _nanmean(arr):
        arr = np.asarray(arr, dtype=np.float64)
        return float(np.nanmean(arr)) if arr.size else float('nan')

    agg = {
        "mae_all7": _nanmean([m.mae_all7 for m in metrics]),
        "rmse_all7": _nanmean([m.rmse_all7 for m in metrics]),
        "mae_joints6": _nanmean([m.mae_joints6 for m in metrics]),
        "rmse_joints6": _nanmean([m.rmse_joints6 for m in metrics]),
        "mae_grip": _nanmean([m.mae_grip for m in metrics]),
        "rmse_grip": _nanmean([m.rmse_grip for m in metrics]),
        "mae_vs_qdelta_all7": _nanmean([m.mae_vs_qdelta_all7 for m in metrics]),
        "rmse_vs_qdelta_all7": _nanmean([m.rmse_vs_qdelta_all7 for m in metrics]),
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print("\n===== Episode Summary =====")
    for k, v in agg.items():
        print(f"{k:>22s}: {v:.6f}")
    print(f"\nWritten: {csv_path}\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
