#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Diffusion Policy on a recorded episode while sweeping:
  - num_inference_steps (denoise steps)
  - JPEG compression (simulate on-wire compression loss)

Outputs a single CSV (metrics_by_combo.csv) with rows per (nis, jpeg_q).

Notes
-----
- Uses the same obs tuple you use live:
    base_rgb (T,3,H,W), wrist_rgb (T,3,H,W), robot_state (T,7)
  via an ObsBuffer of size n_obs_steps from the checkpoint cfg (default 2).
- Executes returned action chunks step-by-step to mimic your control loop.
- Quantitative eval compares predicted actions vs dataset 'control' vectors
  (if present), and a proxy metric vs observed joint deltas (q[t+1]-q[t]).
"""

from __future__ import annotations
import os, re, glob, json, math, time, pickle, warnings, csv, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np
import torch, dill, hydra, cv2

# ---------------- Utilities ----------------
def apply_jpeg_rgb(img_rgb: np.ndarray, quality: int) -> np.ndarray:
    """Simulate on-wire JPEG: RGB -> encode/decode -> RGB."""
    if quality <= 0:
        return img_rgb
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img_rgb
    bgr2 = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)

def to_chw01(img_hwc_uint8: np.ndarray) -> np.ndarray:
    img = img_hwc_uint8.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))  # CHW

def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# ---------------- Resize helper (with fallback) ----------------
try:
    from tools.resize_pkl import resize_with_pad  # your existing util if available
except Exception:
    from PIL import Image
    def resize_with_pad(img: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
        """Minimal tf.image.resize_with_pad equivalent for a single HWC image."""
        ih, iw = img.shape[:2]; oh, ow = height, width
        if iw/ih >= ow/oh:
            rw = ow; rh = max(1, int(round(rw / iw * ih)))
        else:
            rh = oh; rw = max(1, int(round(rh / ih * iw)))
        pil = Image.fromarray(img)
        pil = pil.resize((rw, rh), resample=method)
        resized = np.asarray(pil)
        # center crop if larger than target
        y0 = max((rh - oh)//2, 0); x0 = max((rw - ow)//2, 0)
        crop = resized[y0:y0+oh, x0:x0+ow]
        # pad if smaller than target
        out = np.zeros((oh, ow, img.shape[2]), dtype=img.dtype)
        y_off = max((oh - crop.shape[0])//2, 0); x_off = max((ow - crop.shape[1])//2, 0)
        out[y_off:y_off+crop.shape[0], x_off:x_off+crop.shape[1]] = crop
        return out

# ---------------- Episode I/O ----------------
def _read_pkl(path: str) -> Dict:
    with open(path, 'rb') as f:
        try:
            return pickle.load(f)
        except Exception:
            f.seek(0)
            return dill.load(f)

def _coerce_obs_keys(d: Dict) -> Dict:
    out = {}
    # images
    for cam in ("base_rgb", "wrist_rgb"):
        if cam not in d:
            raise KeyError(f"Missing key '{cam}'")
        out[cam] = np.asarray(d[cam], dtype=np.uint8)
    # joints
    if "joint_position" in d:
        jp = d["joint_position"]
    elif "joint_positions" in d:
        jp = d["joint_positions"]
    else:
        raise KeyError("Missing joint_position(s)")
    out["joint_position"] = np.asarray(jp, dtype=np.float32).reshape(-1)[:6]
    # gripper
    gp = float(np.asarray(d.get("gripper_position", 0.0), dtype=np.float32).reshape(()))
    out["gripper_position"] = np.float32(gp)
    return out

@dataclass
class EpisodeStep:
    idx: int
    obs: Dict[str, np.ndarray]
    control: Optional[np.ndarray]  # (7,) if available

def load_episode(episode_dir: str) -> List[EpisodeStep]:
    paths = sorted(glob.glob(os.path.join(episode_dir, "*.pkl")))
    if not paths:
        raise FileNotFoundError(f"No .pkl in {episode_dir}")

    # natural sort by numbers in filename
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
        steps.append(EpisodeStep(i, obs, control))
    return steps

# ---------------- Buffer ----------------
class ObsBuffer:
    """
    Maintains rolling obs and (optionally) tracks PSNR between
    original resized frame and compressed-decoded frame.
    """
    def __init__(self, n_obs_steps: int, H: int, W: int, jpeg_quality: int, compress_before_resize: bool):
        self.n = n_obs_steps
        self.H, self.W = H, W
        self.jpeg_q = int(jpeg_quality)
        self.compress_before_resize = bool(compress_before_resize)
        self.base_rgb = deque(maxlen=self.n)
        self.wrist_rgb = deque(maxlen=self.n)
        self.robot_state = deque(maxlen=self.n)
        # PSNR accumulators
        self.psnr_sum_base = 0.0
        self.psnr_sum_wrist = 0.0
        self.psnr_count = 0

    def _prep_pair(self, img: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """Return (compressed_for_model_uint8, psnr_or_None)."""
        # reference: resized original (what the model would see without compression)
        ref = resize_with_pad(img, self.H, self.W)
        if self.jpeg_q <= 0:
            return ref, None  # no compression; PSNR undefined/infinite
        if self.compress_before_resize:
            comp = apply_jpeg_rgb(img, self.jpeg_q)
            comp = resize_with_pad(comp, self.H, self.W)
        else:
            comp = resize_with_pad(img, self.H, self.W)
            comp = apply_jpeg_rgb(comp, self.jpeg_q)
        psnr = float(cv2.PSNR(ref, comp))
        return comp, psnr

    def push(self, obs: Dict):
        base_img = obs["base_rgb"]
        wrist_img = obs["wrist_rgb"]

        base_u8, psnr_b = self._prep_pair(base_img)
        wrist_u8, psnr_w = self._prep_pair(wrist_img)

        if psnr_b is not None and psnr_w is not None:
            self.psnr_sum_base += psnr_b
            self.psnr_sum_wrist += psnr_w
            self.psnr_count += 1

        self.base_rgb.append(to_chw01(base_u8))
        self.wrist_rgb.append(to_chw01(wrist_u8))

        state7 = np.concatenate([obs["joint_position"], np.atleast_1d(obs["gripper_position"])], dtype=np.float32)
        self.robot_state.append(state7.astype(np.float32))

        # bootstrap until buffer full
        while len(self.base_rgb) < self.n:
            self.base_rgb.appendleft(self.base_rgb[0].copy())
            self.wrist_rgb.appendleft(self.wrist_rgb[0].copy())
            self.robot_state.appendleft(self.robot_state[0].copy())

    def ready(self) -> bool:
        return len(self.base_rgb) == self.n

    def as_np_dict(self) -> Dict[str, np.ndarray]:
        return dict(
            base_rgb=np.stack(list(self.base_rgb), axis=0),
            wrist_rgb=np.stack(list(self.wrist_rgb), axis=0),
            robot_state=np.stack(list(self.robot_state), axis=0),
        )

    def psnr_means(self) -> Tuple[float, float]:
        if self.psnr_count == 0:
            return float('nan'), float('nan')
        return self.psnr_sum_base / self.psnr_count, self.psnr_sum_wrist / self.psnr_count

# ---------------- Policy ----------------
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

def load_policy(ckpt_path: str) -> Tuple[BaseImagePolicy, dict, int, torch.device]:
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    ws: BaseWorkspace = cls(cfg)
    ws.load_payload(payload, exclude_keys=None, include_keys=None)

    policy: BaseImagePolicy = ws.model
    try:
        if getattr(cfg, "training", None) and getattr(cfg.training, "use_ema", False):
            policy = ws.ema_model
    except Exception:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval().to(device)

    if not hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = 16

    n_obs_steps = int(getattr(cfg, "n_obs_steps", 2))
    return policy, cfg, n_obs_steps, device

def to_torch_batch(obs_np: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in obs_np.items():
        t = torch.from_numpy(v).unsqueeze(0).to(device, non_blocking=True)  # (1, T, ...)
        out[k] = t
    return out

# ---------------- Metrics ----------------
def mae(a, b) -> float:
    return float(np.mean(np.abs(a - b)))

def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

# ---------------- CLI ----------------
@dataclass
class Args:
    ckpt: str
    episode_dir: str
    out_dir: str = "./eval_ablation_out"
    H: int = 224
    W: int = 224
    control_hz: float = 30.0
    save_steps: bool = False
    # Sweeps
    num_inference_steps: List[int] = None
    jpeg_qualities: List[int] = None  # 0 => no jpeg
    compress_before_resize: bool = True
    use_amp: bool = True
    seed: int = 0

# ---------------- Main ----------------
def main(args: Args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    cv2.setNumThreads(1)

    if args.num_inference_steps is None: args.num_inference_steps = [6, 8, 12, 16]
    if args.jpeg_qualities   is None: args.jpeg_qualities   = [0, 95, 85, 75]

    steps = load_episode(args.episode_dir)
    print(f"[EPISODE] Loaded {len(steps)} steps from {args.episode_dir}")

    policy, cfg, n_obs_steps, device = load_policy(args.ckpt)
    print(f"[POLICY] device={device}, n_obs_steps={n_obs_steps}")

    results_rows = []
    dt = 1.0 / float(args.control_hz)

    for nis in args.num_inference_steps:
        # set denoise steps for this run
        if hasattr(policy, "set_num_inference_steps"):
            policy.set_num_inference_steps(int(nis))
        else:
            policy.num_inference_steps = int(nis)

        for q in args.jpeg_qualities:
            print(f"\n=== Run: nis={nis}, jpeg_q={q} (compress_before_resize={args.compress_before_resize}) ===")
            if hasattr(policy, "reset"):
                try:
                    policy.reset()
                except Exception:
                    pass

            buf = ObsBuffer(
                n_obs_steps=n_obs_steps, H=args.H, W=args.W,
                jpeg_quality=int(q), compress_before_resize=args.compress_before_resize
            )

            infer_times: List[float] = []
            chunk_lens: List[int] = []

            # accumulate metrics
            m_mae_all, m_rmse_all = [], []
            m_mae_j, m_rmse_j, m_mae_g, m_rmse_g = [], [], [], []
            m_mae_qd, m_rmse_qd = [], []

            action_chunk = None; chunk_len = 0; chunk_used = 0
            did_warmup = False

            for i in range(len(steps)):
                obs = steps[i].obs
                buf.push(obs)

                # need a full buffer before first forward
                if (action_chunk is None) or (chunk_used >= chunk_len):
                    if not buf.ready():
                        continue

                    obs_np = buf.as_np_dict()
                    model_obs = to_torch_batch(obs_np, device)

                    # one warm-up forward (not timed) to avoid including CUDA kernel setup
                    if not did_warmup:
                        with torch.inference_mode():
                            _ = policy.predict_action(model_obs)
                            if device.type == 'cuda': torch.cuda.synchronize()
                        did_warmup = True

                    # timed forward
                    t0 = time.perf_counter()
                    with torch.inference_mode():
                        if device.type == 'cuda' and args.use_amp:
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                result = policy.predict_action(model_obs)
                        else:
                            result = policy.predict_action(model_obs)
                        if device.type == 'cuda': torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    infer_times.append((t1 - t0) * 1e3)

                    action_chunk = result["action"][0].detach().to("cpu").numpy()  # (A,7)
                    chunk_len = int(action_chunk.shape[0]); chunk_used = 0
                    chunk_lens.append(chunk_len)

                a_pred = action_chunk[chunk_used]; chunk_used += 1
                a_true = steps[i].control

                if a_true is not None:
                    m_mae_all.append(mae(a_pred, a_true))
                    m_rmse_all.append(rmse(a_pred, a_true))
                    m_mae_j.append(mae(a_pred[:6], a_true[:6]))
                    m_rmse_j.append(rmse(a_pred[:6], a_true[:6]))
                    m_mae_g.append(mae(np.array([a_pred[-1]]), np.array([a_true[-1]])))
                    m_rmse_g.append(rmse(np.array([a_pred[-1]]), np.array([a_true[-1]])))

                if i + 1 < len(steps):
                    q_now = np.concatenate([steps[i].obs["joint_position"], [steps[i].obs["gripper_position"]]])
                    q_nxt = np.concatenate([steps[i+1].obs["joint_position"], [steps[i+1].obs["gripper_position"]]])
                    q_delta = q_nxt - q_now
                    m_mae_qd.append(mae(a_pred, q_delta))
                    m_rmse_qd.append(rmse(a_pred, q_delta))

            # aggregate for this (nis, q)
            def nanmean(x): return float(np.nan) if len(x)==0 else float(np.mean(x))
            psnr_b, psnr_w = buf.psnr_means()
            row = {
                "num_inference_steps": int(nis),
                "jpeg_quality": int(q),
                "compress_before_resize": bool(args.compress_before_resize),
                "mae_all7": nanmean(m_mae_all),
                "rmse_all7": nanmean(m_rmse_all),
                "mae_joints6": nanmean(m_mae_j),
                "rmse_joints6": nanmean(m_rmse_j),
                "mae_grip": nanmean(m_mae_g),
                "rmse_grip": nanmean(m_rmse_g),
                "mae_vs_qdelta_all7": nanmean(m_mae_qd),
                "rmse_vs_qdelta_all7": nanmean(m_rmse_qd),
                "infer_ms_mean": nanmean(infer_times),
                "infer_ms_median": float(np.median(infer_times)) if infer_times else float('nan'),
                "action_chunk_len_mean": nanmean(chunk_lens),
                "psnr_base_mean": float(psnr_b),
                "psnr_wrist_mean": float(psnr_w),
            }
            print({k: (round(v,4) if isinstance(v, float) else v) for k,v in row.items()})
            results_rows.append(row)

    # write CSV summary
    out_csv = os.path.join(args.out_dir, "metrics_by_combo.csv")
    cols = [
        "num_inference_steps","jpeg_quality","compress_before_resize",
        "mae_all7","rmse_all7","mae_joints6","rmse_joints6","mae_grip","rmse_grip",
        "mae_vs_qdelta_all7","rmse_vs_qdelta_all7",
        "infer_ms_mean","infer_ms_median","action_chunk_len_mean",
        "psnr_base_mean","psnr_wrist_mean"
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in results_rows: w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nWrote: {out_csv}")

if __name__ == "__main__":
    import tyro
    args = tyro.cli(Args)
    main(args)
