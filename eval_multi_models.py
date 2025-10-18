#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluator for Diffusion Policy checkpoints.

What it adds:
- Scans multiple checkpoint roots (your six models below).
- Flexible selection: all / latest / best-by-val_loss / top-k-by-val_loss.
- Writes per-ckpt metrics (CSV + summary.json) in tidy subfolders.
- Builds per-model Excel workbooks + one global Excel with all sheets.

Usage examples
--------------
# Evaluate only latest.ckpt in each folder
python eval_batch.py --episode-dir "C:\\path\\to\\episode" --out-root "C:\\tmp\\eval" --ckpt-mode latest

# Evaluate best-by-val_loss in each folder
python eval_batch.py --episode-dir "C:\\path\\to\\episode" --out-root "C:\\tmp\\eval" --ckpt-mode best

# Evaluate top-3 lowest val_loss checkpoints in each folder
python eval_batch.py --episode-dir "C:\\path\\to\\episode" --out-root "C:\\tmp\\eval" --ckpt-mode topk --topk 3

# Evaluate every epoch=*.ckpt and latest.ckpt
python eval_batch.py --episode-dir "C:\\path\\to\\episode" --out-root "C:\\tmp\\eval" --ckpt-mode all
"""
from __future__ import annotations

import os, re, json, math, glob, pickle, warnings, csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np
import dill
import torch
import hydra
import tyro

# ---------- YOUR ORIGINAL IMPORTS ----------
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# ---------- Resize helper (unchanged) ----------
img_tf_cache = {}
def get_tf(k, in_w, in_h, out_w, out_h):
    key = (k, in_w, in_h, out_w, out_h)
    if key not in img_tf_cache:
        img_tf_cache[key] = get_image_transform(
            input_res=(in_w, in_h), output_res=(out_w, out_h), bgr_to_rgb=False
        )
    return img_tf_cache[key]

try:
    from tools.resize_pkl import resize_with_pad as _resize_with_pad
except Exception:
    from PIL import Image
    def _resize_with_pad(img: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
        ih, iw = img.shape[:2]; oh, ow = height, width
        if iw/ih >= ow/oh:
            rw = ow; rh = math.ceil(rw / iw * ih)
        else:
            rh = oh; rw = math.ceil(rh / ih * iw)
        pil = Image.fromarray(img).resize((rw, rh), resample=method)
        resized = np.asarray(pil)
        y0 = max((rh - oh)//2, 0); x0 = max((rw - ow)//2, 0)
        crop = resized[y0:y0+oh, x0:x0+ow]
        out = np.zeros((oh, ow, img.shape[2]), dtype=img.dtype)
        y_off = max((oh - crop.shape[0])//2, 0); x_off = max((ow - crop.shape[1])//2, 0)
        out[y_off:y_off+crop.shape[0], x_off:x_off+crop.shape[1]] = crop
        return out

def to_chw01(img_hwc_uint8: np.ndarray) -> np.ndarray:
    img = img_hwc_uint8.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))

# ---------- Episode loader ----------
@dataclass
class EpisodeStep:
    idx: int
    obs: Dict[str, np.ndarray]
    control: Optional[np.ndarray]

def _coerce_obs_keys(d: Dict) -> Dict:
    out = {}
    for cam_key in ("base_rgb", "wrist_rgb"):
        if cam_key not in d:
            raise KeyError(f"Missing key '{cam_key}' in step")
        out[cam_key] = d[cam_key]
    jp = d["joint_position"] if "joint_position" in d else d.get("joint_positions", None)
    if jp is None: raise KeyError("Missing 'joint_position(s)' in step")
    out["joint_position"] = np.asarray(jp, dtype=np.float32).reshape(-1)[:6]
    gp = d.get("gripper_position", 0.0)
    out["gripper_position"] = np.float32(np.asarray(gp, dtype=np.float32).reshape(()).item())
    return out

def _read_pkl(path: str) -> Dict:
    with open(path, 'rb') as f:
        try: return pickle.load(f)
        except Exception:
            f.seek(0); return dill.load(f)

def load_episode(episode_dir: str) -> List[EpisodeStep]:
    paths = sorted(glob.glob(os.path.join(episode_dir, "*.pkl")))
    if not paths: raise FileNotFoundError(f"No .pkl files found in {episode_dir}")
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

# ---------- Obs buffer ----------
class ObsBuffer:
    def __init__(self, n_obs_steps: int = 2, H: int = 224, W: int = 224):
        self.n = n_obs_steps; self.H, self.W = H, W
        self.base_rgb, self.wrist_rgb, self.robot_state = deque(maxlen=self.n), deque(maxlen=self.n), deque(maxlen=self.n)
    def push(self, obs: Dict):
        base = _resize_with_pad(obs["base_rgb"], self.H, self.W)
        wrist = _resize_with_pad(obs["wrist_rgb"], self.H, self.W)
        self.base_rgb.append(to_chw01(base)); self.wrist_rgb.append(to_chw01(wrist))
        state7 = np.concatenate([obs["joint_position"], np.atleast_1d(obs["gripper_position"]).astype(np.float32)])
        self.robot_state.append(state7.astype(np.float32))
        while len(self.base_rgb) < self.n:
            self.base_rgb.appendleft(self.base_rgb[0].copy()); self.wrist_rgb.appendleft(self.wrist_rgb[0].copy()); self.robot_state.appendleft(self.robot_state[0].copy())
    def ready(self) -> bool: return len(self.base_rgb) == self.n
    def as_np_dict(self) -> Dict[str, np.ndarray]:
        return dict(base_rgb=np.stack(self.base_rgb, axis=0),
                    wrist_rgb=np.stack(self.wrist_rgb, axis=0),
                    robot_state=np.stack(self.robot_state, axis=0))

# ---------- Policy loader ----------
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
        import torch as _T
        t = _T.from_numpy(v).unsqueeze(0).to(device)  # (1, T, ...)
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

def _mae(a: np.ndarray, b: np.ndarray) -> float: return float(np.mean(np.abs(a - b)))
def _rmse(a: np.ndarray, b: np.ndarray) -> float: return float(np.sqrt(np.mean((a - b) ** 2)))

# ---------- Single-ckpt evaluation (your loop, lightly wrapped) ----------
@dataclass
class EvalArgs:
    ckpt: str
    episode_dir: str
    out_dir: str
    control_hz: float = 30.0
    H: int = 224
    W: int = 224
    save_steps: bool = False

def evaluate_one(args: EvalArgs) -> Dict[str, float]:
    os.makedirs(args.out_dir, exist_ok=True)
    steps = load_episode(args.episode_dir)
    policy, cfg, n_obs_steps, device = load_diffusion_policy(args.ckpt)
    try: policy.reset()
    except Exception: pass
    buf = ObsBuffer(n_obs_steps=n_obs_steps, H=args.H, W=args.W)
    metrics: List[StepMetrics] = []
    action_chunk = None; chunk_len = 0; chunk_used = 0

    for i in range(len(steps)):
        obs = steps[i].obs; buf.push(obs)
        if (action_chunk is None) or (chunk_used >= chunk_len):
            if not buf.ready(): continue
            with torch.no_grad():
                obs_np = buf.as_np_dict(); obs_t = make_obs_torch(obs_np, device)
                model_obs = {"base_rgb": obs_t["base_rgb"], "wrist_rgb": obs_t["wrist_rgb"], "robot_state": obs_t["robot_state"]}
                result = policy.predict_action(model_obs)
                action_chunk = result["action"][0].detach().to("cpu").numpy()
                chunk_len = action_chunk.shape[0]; chunk_used = 0
                if action_chunk.shape[-1] != 7:
                    raise RuntimeError(f"Unexpected action shape {action_chunk.shape}; expected (*,7)")
        a_pred = action_chunk[chunk_used]; chunk_used += 1
        a_true = steps[i].control
        if a_true is not None:
            mae_all7 = _mae(a_pred, a_true); rmse_all7 = _rmse(a_pred, a_true)
            mae_j = _mae(a_pred[:6], a_true[:6]); rmse_j = _rmse(a_pred[:6], a_true[:6])
            mae_g = _mae(np.array([a_pred[-1]]), np.array([a_true[-1]])); rmse_g = _rmse(np.array([a_pred[-1]]), np.array([a_true[-1]]))
        else:
            mae_all7 = rmse_all7 = mae_j = rmse_j = mae_g = rmse_g = float('nan')
        mae_vs_qdelta = rmse_vs_qdelta = None
        if i + 1 < len(steps):
            q_now = np.concatenate([steps[i].obs["joint_position"], np.atleast_1d(steps[i].obs["gripper_position"])])
            q_nxt = np.concatenate([steps[i+1].obs["joint_position"], np.atleast_1d(steps[i+1].obs["gripper_position"])])
            q_delta = q_nxt - q_now
            mae_vs_qdelta = _mae(a_pred, q_delta); rmse_vs_qdelta = _rmse(a_pred, q_delta)
        metrics.append(StepMetrics(
            step=i, mae_all7=mae_all7, rmse_all7=rmse_all7,
            mae_joints6=mae_j, rmse_joints6=rmse_j,
            mae_grip=mae_g, rmse_grip=rmse_g,
            mae_vs_qdelta_all7=mae_vs_qdelta, rmse_vs_qdelta_all7=rmse_vs_qdelta
        ))
        if args.save_steps:
            rec = dict(
                step=i, a_pred=a_pred.tolist(),
                a_true=(a_true.tolist() if a_true is not None else None),
                q=steps[i].obs["joint_position"].tolist(),
                g=float(steps[i].obs["gripper_position"]),
                mae_all7=mae_all7, rmse_all7=rmse_all7,
                mae_vs_qdelta_all7=mae_vs_qdelta, rmse_vs_qdelta_all7=rmse_vs_qdelta
            )
            step_dir = os.path.join(args.out_dir, "steps"); os.makedirs(step_dir, exist_ok=True)
            with open(os.path.join(step_dir, f"step_{i:06d}.json"), "w") as f: json.dump(rec, f, indent=2)

    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step","mae_all7","rmse_all7","mae_joints6","rmse_joints6","mae_grip","rmse_grip","mae_vs_qdelta_all7","rmse_vs_qdelta_all7"])
        for m in metrics:
            w.writerow([m.step,m.mae_all7,m.rmse_all7,m.mae_joints6,m.rmse_joints6,m.mae_grip,m.rmse_grip,m.mae_vs_qdelta_all7,m.rmse_vs_qdelta_all7])

    def _nanmean(arr): arr = np.asarray(arr, dtype=np.float64); return float(np.nanmean(arr)) if arr.size else float('nan')
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
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f: json.dump(agg, f, indent=2)
    return agg

# ---------- Batch runner ----------
CKPT_MAP = {
    # Label -> checkpoints directory

    "ImageNet_FT":  r"/home/n10813934/gitRepos/VGS_diffusion_policy/data/outputs/2025.09.15/22.35.26_train_xarm6_diffusion_unet_real_finetuned_workspace_real_xarm_image/checkpoints",
    "ImageNet":     r"/home/n10813934/gitRepos/VGS_diffusion_policy/data/outputs/2025.09.08/22.12.28_train_xarm6_diffusion_unet_real_pretrained_workspace_real_xarm_image/checkpoints",
    "E2E":          r"/home/n10813934/gitRepos/VGS_diffusion_policy/data/outputs/2025.09.17/22.17.33_train_xarm6_diffusion_unet_real_image_workspace_NoImagenetNorm_real_xarm_image/checkpoints",
    "R3M":          r"/home/n10813934/gitRepos/VGS_diffusion_policy/data/outputs/2025.10.17/20.02.49_train_xarm6_diffusion_unet_real_image_workspace_r3m_real_xarm_image/checkpoints",
    "R3M_FT":       r"/home/n10813934/gitRepos/VGS_diffusion_policy/data/outputs/2025.10.17/20.12.32_train_xarm6_diffusion_unet_real_image_workspace_r3m_fine_tune_real_xarm_image/checkpoints",
    "Shared_RGB":   r"/home/n10813934/gitRepos/VGS_diffusion_policy/data/outputs/2025.10.17/20.13.45_train_xarm6_diffusion_unet_real_image_workspace_shared_rgb_real_xarm_image/checkpoints",
}

def parse_epoch_and_valloss(fname: str) -> Tuple[Optional[int], Optional[float]]:
    # epoch=0015-val_loss=0.263.ckpt  -> (15, 0.263)
    m = re.search(r"epoch=(\d+)-val_loss=([0-9.]+)\.ckpt$", fname)
    if m: return int(m.group(1)), float(m.group(2))
    if fname == "latest.ckpt": return None, None
    return None, None

def scan_checkpoints(root: str) -> List[Tuple[str, Optional[int], Optional[float]]]:
    if not os.path.isdir(root): return []
    files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".ckpt")]
    items = []
    for p in files:
        e, v = parse_epoch_and_valloss(os.path.basename(p))
        items.append((p, e, v))
    # Always put epoch=*.ckpt before latest for sorting; we'll control selection mode later
    items.sort(key=lambda t: (t[1] is None, t[1] if t[1] is not None else 10**9))
    return items

@dataclass
class BatchArgs:
    episode_dir: str = "/home/n10813934/data/0828_173511/"
    out_dir: str = "/home/n10813934/data/diffusion_eval_out"
    ckpt_mode: str = "best"     # one of {"all","latest","best","topk"}
    topk: int = 3               # used if ckpt_mode == "topk"
    save_steps: bool = False
    H: int = 224
    W: int = 224
    control_hz: float = 30.0

def _ensure_dir(d: str): os.makedirs(d, exist_ok=True)

def select_ckpts(items: List[Tuple[str, Optional[int], Optional[float]]], mode: str, topk: int) -> List[Tuple[str, Optional[int], Optional[float]]]:
    if not items: return []
    if mode == "all":
        return items
    if mode == "latest":
        lat = [t for t in items if os.path.basename(t[0]) == "latest.ckpt"]
        return lat if lat else [items[-1]]
    # filter epoch=*.ckpt with val_loss
    with_loss = [t for t in items if t[2] is not None]
    if not with_loss:
        return [items[-1]]
    with_loss.sort(key=lambda t: t[2])  # ascending val_loss
    if mode == "best":
        return [with_loss[0]]
    if mode == "topk":
        return with_loss[:max(1, topk)]
    return [with_loss[0]]

def write_excels(per_label_rows: Dict[str, List[Dict]], out_root: str):
    import pandas as pd
    # Per-model workbooks
    for label, rows in per_label_rows.items():
        if not rows: continue
        df = pd.DataFrame(rows)
        df.to_excel(os.path.join(out_root, f"{label}_metrics.xlsx"), index=False)
    # Global workbook with a sheet per model + combined
    with pd.ExcelWriter(os.path.join(out_root, "all_models_metrics.xlsx")) as wx:
        combined = []
        for label, rows in per_label_rows.items():
            if not rows: continue
            df = pd.DataFrame(rows)
            df.to_excel(wx, sheet_name=label[:31], index=False)
            combined.append(df)
        if combined:
            pd.concat(combined, ignore_index=True).to_excel(wx, sheet_name="ALL", index=False)

def run_batch(bargs: BatchArgs):
    _ensure_dir(bargs.out_root)
    per_label_rows: Dict[str, List[Dict]] = {k: [] for k in CKPT_MAP.keys()}
    for label, ckpt_root in CKPT_MAP.items():
        print(f"\n=== {label} ===\n{ckpt_root}")
        items = scan_checkpoints(ckpt_root)
        chosen = select_ckpts(items, bargs.ckpt_mode.lower(), bargs.topk)
        if not chosen:
            print(f"[WARN] No checkpoints found in {ckpt_root}")
            continue
        for ckpt_path, epoch, valloss in chosen:
            suffix = os.path.splitext(os.path.basename(ckpt_path))[0]
            out_dir = os.path.join(bargs.out_root, label, suffix)
            agg = evaluate_one(EvalArgs(
                ckpt=ckpt_path,
                episode_dir=bargs.episode_dir,
                out_dir=out_dir,
                control_hz=bargs.control_hz,
                H=bargs.H, W=bargs.W,
                save_steps=bargs.save_steps
            ))
            row = dict(
                label=label,
                ckpt=os.path.basename(ckpt_path),
                epoch=epoch,
                val_loss=valloss,
                **agg
            )
            per_label_rows[label].append(row)
    write_excels(per_label_rows, bargs.out_root)
    print("\nDone. Per-ckpt CSVs + summaries are under out_root/<label>/<ckpt>/")
    print("Per-model Excel: <label>_metrics.xlsx; Global: all_models_metrics.xlsx")

# ---------- CLI ----------
def main(bargs: BatchArgs):
    run_batch(bargs)

if __name__ == "__main__":
    tyro.cli(main)
