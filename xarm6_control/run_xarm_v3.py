#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a Diffusion Policy checkpoint on xArm6 using pi0-style loop.
- Local mode: load checkpoint and infer locally
- Remote policy mode: send single-step obs to HPC server (WebSocket), receive action chunk

Usage examples:

# MOCK + REMOTE (tunnel on 127.0.0.1:8765)
python -m xarm6_control.run_xarm --ckpt DUMMY --mock --use_remote_policy \
  --policy_server_host 127.0.0.1 --policy_server_port 8765 --max_steps 120 --control_hz 10

# REAL + REMOTE
python -m xarm6_control.run_xarm --ckpt DUMMY --use_remote_policy \
  --remote_host localhost --wrist_camera_port 5000 --base_camera_port 5001 \
  --control_hz 30 --max_steps 300 --policy_server_host 127.0.0.1 --policy_server_port 8765

# LOCAL (no server)
python -m xarm6_control.run_xarm --ckpt /path/to/ckpt --mock --max_steps 60 --control_hz 10
"""

import os
import time
import copy
import math
import numpy as np
import torch
import dill
import hydra
from collections import deque
from typing import Dict, Tuple

# env/cameras
from xarm6_control.xarm_env import XArmRealEnv, MockXArmEnv
from xarm6_control.zmq_core.camera_node import ZMQClientCamera

import msgpack_numpy
import asyncio
import websockets

from diffusion_policy.common.cv2_util import get_image_transform

# -----------------------------
# Image conversion + transforms
# -----------------------------
img_tf_cache = {}  # (k, in_w,in_h,out_w,out_h) -> callable
def get_tf(k, in_w, in_h, out_w, out_h):
    key = (k, in_w, in_h, out_w, out_h)
    if key not in img_tf_cache:
        img_tf_cache[key] = get_image_transform(
            input_res=(in_w, in_h), output_res=(out_w, out_h), bgr_to_rgb=False
        )
    return img_tf_cache[key]

try:
    import cv2  # optional, only used for resize fallback if tools not present
except Exception:
    cv2 = None

# pad/resize: use your tools if available, otherwise fallback
try:
    from tools.resize_pkl import resize_with_pad as _rp_resize_with_pad
    def resize_with_pad(img, H=224, W=224): return _rp_resize_with_pad(img, H, W)
except Exception:
    def resize_with_pad(img_hwc_uint8, H=224, W=224):
        h, w = img_hwc_uint8.shape[:2]
        scale = min(W / w, H / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        if cv2 is not None:
            resized = cv2.resize(
                img_hwc_uint8, (nw, nh),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            )
        else:
            # very simple nearest-neighbor fallback if cv2 not available
            y_idx = (np.linspace(0, h-1, nh)).astype(int)
            x_idx = (np.linspace(0, w-1, nw)).astype(int)
            resized = img_hwc_uint8[y_idx][:, x_idx]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        x0 = (W - nw) // 2; y0 = (H - nh) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        return canvas


def _to_chw01(img_hwc_uint8: np.ndarray) -> np.ndarray:
    img = img_hwc_uint8.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))  # CHW


# -----------------
# Websocket client
# -----------------
class WebsocketClientPolicy:
    """Minimal pi0-like client for our diffusion server."""
    def __init__(self, host="127.0.0.1", port=8765):
        self.uri = f"ws://{host}:{port}"
        self.packer = msgpack_numpy.Packer()
        # start event loop
        self.loop = asyncio.get_event_loop()
        self.connection = self.loop.run_until_complete(
            websockets.connect(self.uri, max_size=None)
        )
        # read metadata frame
        meta = self.loop.run_until_complete(self.connection.recv())
        print(f"[CLIENT] Connected. Metadata: {msgpack_numpy.unpackb(meta)}")

    def infer(self, obs: dict) -> dict:
        msg = self.packer.pack(obs)
        self.loop.run_until_complete(self.connection.send(msg))
        reply = self.loop.run_until_complete(self.connection.recv())
        return msgpack_numpy.unpackb(reply)


# -----------------
# Diffusion policy
# -----------------
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

def _load_diffusion_policy(ckpt_path: str) -> Tuple[BaseImagePolicy, dict, int]:
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg.__target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy: BaseImagePolicy = workspace.model
    try:
        if getattr(cfg, "training", None) and getattr(cfg.training, "use_ema", False):
            policy = workspace.ema_model
    except Exception:
        pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.to(device).eval()
    if not hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = 16
    policy.num_inference_steps = 16
    n_obs_steps = int(getattr(cfg, "n_obs_steps", 2))
    return policy, cfg, n_obs_steps

def _make_obs_torch(obs_np: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in obs_np.items():
        out[k] = torch.from_numpy(v).unsqueeze(0).to(device)  # (1,T,...)
    return out


# -----------
# Obs buffer
# -----------
class _ObsBuffer:
    def __init__(self, n_obs_steps: int = 2, H: int = 224, W: int = 224):
        self.n = n_obs_steps
        self.H, self.W = H, W
        self.base_rgb = deque(maxlen=self.n)    # each: (3,H,W), float32 [0,1]
        self.wrist_rgb = deque(maxlen=self.n)   # each: (3,H,W), float32 [0,1]
        self.robot_state = deque(maxlen=self.n) # each: (7,), float32 (rad * 6, grip)

    def push(self, obs: Dict):
        base = resize_with_pad(obs["base_rgb"], self.H, self.W)
        wrist = resize_with_pad(obs["wrist_rgb"], self.H, self.W)
        self.base_rgb.append(_to_chw01(base))
        self.wrist_rgb.append(_to_chw01(wrist))
        state7 = np.concatenate([obs["joint_position"], obs["gripper_position"]]).astype(np.float32)
        self.robot_state.append(state7)
        while len(self.base_rgb) < self.n:
            self.base_rgb.appendleft(self.base_rgb[0].copy())
            self.wrist_rgb.appendleft(self.wrist_rgb[0].copy())
            self.robot_state.appendleft(self.robot_state[0].copy())

    def ready(self) -> bool:
        return len(self.base_rgb) == self.n

    def as_np_dict(self) -> Dict[str, np.ndarray]:
        base = np.stack(list(self.base_rgb), axis=0)      # (T,3,H,W)
        wrist = np.stack(list(self.wrist_rgb), axis=0)    # (T,3,H,W)
        state = np.stack(list(self.robot_state), axis=0)  # (T,7)
        return dict(base_rgb=base, wrist_rgb=wrist, robot_state=state)


# -------------------
# Debug print helpers
# -------------------
_JLBL = ["J1","J2","J3","J4","J5","J6"]

def _fmt_vec_deg(vec6: np.ndarray, width: int = 7, prec: int = 1) -> str:
    return " ".join(f"{v:>{width}.{prec}f}" for v in vec6.tolist())

def _lap_var(gray01: np.ndarray) -> float:
    """Fast Laplacian-like variance without cv2. gray01 in [0,1], float32."""
    g = gray01
    # inner region to avoid border issues
    c = g[1:-1,1:-1]
    lap = (-4.0*c
           + g[1:-1,:-2] + g[1:-1,2:]
           + g[:-2,1:-1] + g[2:,1:-1])
    return float(lap.var())

def _vision_stats(chw01: np.ndarray) -> Dict[str, float]:
    """Compute simple vision stats from CHW float [0,1]."""
    c, h, w = chw01.shape
    r, g, b = chw01[0], chw01[1], chw01[2]
    # luma (BT.601-ish)
    y = 0.299*r + 0.587*g + 0.114*b
    mean = float(y.mean() * 255.0)
    p5   = float(np.percentile(y, 5) * 255.0)
    p95  = float(np.percentile(y, 95) * 255.0)
    dark = float((y < 0.06).mean() * 100.0)
    bright = float((y > 0.94).mean() * 100.0)
    sharp = _lap_var(y)
    return dict(h=h, w=w, mean=mean, p5=p5, p95=p95, dark=dark, bright=bright, sharp=sharp)

def _print_debug_block(
    step_idx: int,
    action_idx: int,
    action_len: int,
    buf: _ObsBuffer,
    current_obs: Dict,
    action: np.ndarray
):
    # Ensure we have two frames in buffer
    if len(buf.robot_state) < 2:
        print(f"[DEBUG] Warming up buffer (have {len(buf.robot_state)}/2 frames).")
        return

    # Robot state buffer (t-1, t0)
    state_tm1 = buf.robot_state[-2]
    state_t0  = buf.robot_state[-1]
    j_tm1_deg = np.degrees(state_tm1[:6])
    j_t0_deg  = np.degrees(state_t0[:6])
    grip_tm1  = float(state_tm1[6])
    grip_t0   = float(state_t0[6])

    # Vision stats for both timesteps (base & wrist)
    b_tm1 = _vision_stats(buf.base_rgb[-2])
    b_t0  = _vision_stats(buf.base_rgb[-1])
    w_tm1 = _vision_stats(buf.wrist_rgb[-2])
    w_t0  = _vision_stats(buf.wrist_rgb[-1])

    # Action + deltas in degrees
    act_deg   = np.degrees(action[:6])
    cur_deg   = np.degrees(current_obs["joint_position"][:6])
    delta_deg = act_deg - cur_deg
    grip_act  = float(action[-1])

    header = f"===== DEBUG :: STEP {step_idx+1} | ACTION {action_idx}/{action_len} ====="
    print("\n" + header)
    print("-" * len(header))

    # Buffer states
    lbl_row = " ".join(f"{lbl:>7}" for lbl in _JLBL)
    print("OBS BUFFER (robot_state) in degrees [J1..J6] | grip")
    print(f"      {'':>7}{lbl_row} | {'grip':>5}")
    print(f"   t-1: {_fmt_vec_deg(j_tm1_deg)} | {grip_tm1:>5.2f}")
    print(f"    t0: {_fmt_vec_deg(j_t0_deg)} | {grip_t0:>5.2f}")

    # Vision
    def _vline(tag: str, s: Dict[str, float]) -> str:
        return (f"{tag}: {s['w']}x{s['h']}  mean={s['mean']:.1f}  "
                f"p5={s['p5']:.1f}  p95={s['p95']:.1f}  "
                f"dark%={s['dark']:.1f}  bright%={s['bright']:.1f}  "
                f"sharp={s['sharp']:.2f}")

    print("\nVISION (per frame; luma 0-255, sharp=var(Lap))")
    print(_vline(" base t-1", b_tm1))
    print(_vline(" base  t0", b_t0))
    print(_vline("wrist t-1", w_tm1))
    print(_vline("wrist  t0", w_t0))

    # Action details
    print("\nACTION (deg) vs CURRENT (deg) + Δ (deg) | gripper (target)")
    print(f" current: {_fmt_vec_deg(cur_deg)} | {float(current_obs['gripper_position']):>5.2f}")
    print(f" target : {_fmt_vec_deg(act_deg)} | {grip_act:>5.2f}")
    print(f" delta  : {_fmt_vec_deg(delta_deg)} |")
    print(f" max|Δ| : {np.max(np.abs(delta_deg)):.2f}°   mean|Δ|: {np.mean(np.abs(delta_deg)):.2f}°")
    print("-" * len(header))


# -----------
# Mock camera
# -----------
class _MockCamera:
    def read(self, img_size=None):
        rgb = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        depth = np.random.randint(0, 65536, size=(480, 640), dtype=np.uint16)
        return rgb, depth


# ----
# Main
# ----
def main(
    ckpt: str,
    use_pad: bool = True,          # whether to use pad/resize (True) or stretch-resize (False)
    remote_host: str = "localhost",
    remote_port: int = 8000,              # your ZMQ camera server (unchanged)
    wrist_camera_port: int = 5000,
    base_camera_port: int = 5001,
    max_steps: int = 5000,
    prompt: str = "Pick a ripe, red tomato and drop it in the blue bucket.",
    mock: bool = False,
    control_hz: float = 30.0,
    step_through_instructions: bool = False,
    delta_threshold: float = 0.25,        # degrees per joint
    log_dir: str = "/media/acrv/DanielsSSD/Test_sem2/diffusion",
    save: bool = False,

    # Remote policy settings
    use_remote_policy: bool = False,
    policy_server_host: str = "127.0.0.1",
    policy_server_port: int = 8000,

    # NEW: debug printing
    debug: bool = False,
):
    # --- Environment & cameras ---
    if mock:
        camera_clients = {"wrist": _MockCamera(), "base": _MockCamera()}
        env = MockXArmEnv(camera_dict=camera_clients)
    else:
        camera_clients = {
            "wrist": ZMQClientCamera(port=wrist_camera_port, host=remote_host),
            "base":  ZMQClientCamera(port=base_camera_port,  host=remote_host),
        }
        env = XArmRealEnv(camera_dict=camera_clients)

    if save:
        import datetime
        log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)
        print(f"[LOG] Saving to: {log_dir}")

    # --- Policy setup ---
    policy = cfg = None
    n_obs_steps = 2
    device = torch.device('cpu')

    policy_client = None
    if use_remote_policy:
        policy_client = WebsocketClientPolicy(
            host=policy_server_host, port=policy_server_port
        )
        print(f"[REMOTE] Connected to policy server at ws://{policy_server_host}:{policy_server_port}")
    else:
        policy, cfg, n_obs_steps = _load_diffusion_policy(ckpt)
        device = next(policy.parameters()).device
        print(f"[LOCAL POLICY] Loaded on device: {device}, n_obs_steps={n_obs_steps}")

    H = 224; W = 224
    H_IN = 240; W_IN = 424
    buf = _ObsBuffer(n_obs_steps=n_obs_steps, H=H, W=W)

    dt = 1.0 / float(control_hz)
    actions_from_chunk_completed = 0
    action_chunk = None
    action_chunk_len = 0

    try:
        if policy is not None and hasattr(policy, "reset"):
            policy.reset()
    except Exception:
        pass

    for step_idx in range(max_steps):
        loop_t0 = time.time()
        obs = env.get_observation()
        buf.push(obs)

        # Need at least n_obs_steps before local inference; remote server handles its own buffer
        if (action_chunk is None) and (not use_remote_policy) and (not buf.ready()):
            time.sleep(max(0.0, dt - (time.time() - loop_t0)))
            continue

        need_new_chunk = (action_chunk is None) or (actions_from_chunk_completed >= action_chunk_len)
        if need_new_chunk:
            if use_remote_policy:
                # Single-step obs to server; server keeps its own 2-step buffer
                if use_pad:
                    base_rgb = resize_with_pad(obs["base_rgb"], H, W)
                    wrist_rgb = resize_with_pad(obs["wrist_rgb"], H, W)
                else:
                    tf = get_tf("rgb", W_IN, H_IN, W, H)
                    base_rgb = tf(obs["base_rgb"])   # uint8, HxWx3
                    wrist_rgb = tf(obs["wrist_rgb"]) # uint8, HxWx3
                observation = {
                    "base_rgb": base_rgb,
                    "wrist_rgb": wrist_rgb,
                    "joint_position": obs["joint_position"],
                    "gripper_position": obs["gripper_position"],
                    # optional: "prompt": prompt
                }
                reply = policy_client.infer(observation)  # {'actions': [[..7..], ...]}
                action_chunk = np.array(reply.get("actions", []), dtype=np.float32)
                if action_chunk.size == 0:
                    # warm-up on first tick or empty reply
                    time.sleep(max(0.0, dt - (time.time() - loop_t0)))
                    continue
                action_chunk_len = action_chunk.shape[0]
                actions_from_chunk_completed = 0
                print(f"[REMOTE POLICY] New action chunk: {action_chunk_len} steps")
            else:
                with torch.no_grad():
                    obs_np = buf.as_np_dict()
                    obs_torch = _make_obs_torch(obs_np, device=device)
                    model_obs = {
                        "base_rgb":    obs_torch["base_rgb"],      # (1,T,3,H,W)
                        "wrist_rgb":   obs_torch["wrist_rgb"],     # (1,T,3,H,W)
                        "robot_state": obs_torch["robot_state"],   # (1,T,7)
                    }
                    result = policy.predict_action(model_obs)
                    action_chunk = result["action"][0].detach().to("cpu").numpy()
                    action_chunk_len = action_chunk.shape[0]
                    actions_from_chunk_completed = 0
                    if action_chunk.shape[-1] != 7:
                        raise RuntimeError(f"Policy returned unexpected action shape {action_chunk.shape}; expected (*,7).")
                    print(f"[LOCAL POLICY] New action chunk: {action_chunk_len} steps")

        # Pull next action (REQUEST)
        action = action_chunk[actions_from_chunk_completed]
        actions_from_chunk_completed += 1

        # Optional DEBUG print on each requested action
        if debug:
            _print_debug_block(
                step_idx=step_idx,
                action_idx=actions_from_chunk_completed,
                action_len=action_chunk_len,
                buf=buf,
                current_obs=obs,
                action=action
            )

        # Safety check in degrees
        current_joints_deg = np.degrees(obs["joint_position"][:6])
        action_joints_deg  = np.degrees(action[:6])
        delta_deg = action_joints_deg - current_joints_deg

        if step_through_instructions:
            print(f"\n[STEP {step_idx+1}, ACTION {actions_from_chunk_completed}/{action_chunk_len}]")
            print("Current (deg):", np.round(current_joints_deg, 2))
            print("Action  (deg):", np.round(action_joints_deg, 2))
            print("Delta   (deg):", np.round(delta_deg, 2))
            print(f"Gripper pose: {obs['gripper_position']}, Gripper action: {action[-1]:.3f}")
            if np.any(np.abs(delta_deg) > delta_threshold):
                print("⚠️  Warning: large joint delta detected!")
            cmd = input("Press [Enter] to execute, 's' to skip, or 'q' to quit: ").strip().lower()
            if cmd == "q":
                print("Exiting policy execution.")
                break
            elif cmd == "s":
                print("Skipping this action.")
                actions_from_chunk_completed = action_chunk_len
                continue
            if np.any(np.abs(delta_deg) > delta_threshold):
                state = np.concatenate([obs["joint_position"], obs["gripper_position"]])
                interp = env.generate_joint_trajectory(state, action, delta_threshold * math.pi / 180.0)
                obs = env.step_through_interpolated_trajectory(interp, obs, step_idx, log_dir, control_hz, True, save)
            else:
                if save:
                    env.save_step_data(log_dir, step_idx, copy.deepcopy(obs), action)
                env.step(np.array(action))

            elapsed = time.time() - loop_t0
            time.sleep(max(0.0, dt - elapsed))
            continue

        # Non-step-through: interpolate if too large
        if np.any(np.abs(delta_deg) > delta_threshold):
            state = np.concatenate([obs["joint_position"], obs["gripper_position"]])
            interp = env.generate_joint_trajectory(state, action, delta_threshold * math.pi / 180.0)
            obs = env.step_through_interpolated_trajectory(interp, obs, step_idx, log_dir, control_hz, False, save)
        else:
            if save:
                env.save_step_data(log_dir, step_idx, copy.deepcopy(obs), action)
            env.step(np.array(action))

        elapsed = time.time() - loop_t0
        time.sleep(max(0.0, dt - elapsed))


if __name__ == "__main__":
    # tyro if available; fall back to argparse
    try:
        import tyro
        tyro.cli(main)
    except Exception:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--ckpt", required=True)
        p.add_argument("--remote_host", default="localhost")
        p.add_argument("--remote_port", type=int, default=8000)
        p.add_argument("--wrist_camera_port", type=int, default=5000)
        p.add_argument("--base_camera_port", type=int, default=5001)
        p.add_argument("--max_steps", type=int, default=5000)
        p.add_argument("--prompt", default="Pick a ripe, red tomato and drop it in the blue bucket.")
        p.add_argument("--mock", action="store_true")
        p.add_argument("--control_hz", type=float, default=30.0)
        p.add_argument("--step_through_instructions", action="store_true")
        p.add_argument("--delta_threshold", type=float, default=0.25)
        p.add_argument("--log_dir", default=os.path.expanduser("/media/acrv/DanielsSSD/Test_sem2/diffusion"))
        p.add_argument("--save", action="store_true")
        p.add_argument("--use_remote_policy", action="store_true")
        p.add_argument("--policy_server_host", default="127.0.0.1")
        p.add_argument("--policy_server_port", type=int, default=8765)
        p.add_argument("--debug", action="store_true", help="Print buffer (2 obs), vision stats, and action each step.")
        args = p.parse_args()
        main(**vars(args))
