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

# try to re-use your pi0 websocket client
try:
    from openpi_client import websocket_client_policy
    _HAS_PI0_CLIENT = True
except Exception:
    _HAS_PI0_CLIENT = False

# diffusion policy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# pad/resize: use your tools if available, otherwise fallback
try:
    from tools.resize_pkl import resize_with_pad as _rp_resize_with_pad
    def resize_with_pad(img, H=224, W=224): return _rp_resize_with_pad(img, H, W)
except Exception:
    import cv2
    def resize_with_pad(img_hwc_uint8, H=224, W=224):
        h, w = img_hwc_uint8.shape[:2]
        scale = min(W / w, H / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(img_hwc_uint8, (nw, nh),
                             interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        x0 = (W - nw) // 2; y0 = (H - nh) // 2
        canvas[y0:y0+nh, x0:x0+nw] = resized
        return canvas


def _to_chw01(img_hwc_uint8: np.ndarray) -> np.ndarray:
    img = img_hwc_uint8.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))  # CHW


class _ObsBuffer:
    def __init__(self, n_obs_steps: int = 2, H: int = 224, W: int = 224):
        self.n = n_obs_steps
        self.H, self.W = H, W
        self.base_rgb = deque(maxlen=self.n)
        self.wrist_rgb = deque(maxlen=self.n)
        self.robot_state = deque(maxlen=self.n)

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


def _load_diffusion_policy(ckpt_path: str) -> Tuple[BaseImagePolicy, dict, int]:
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
    policy.to(device).eval()
    if not hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = 16
    n_obs_steps = int(getattr(cfg, "n_obs_steps", 2))
    return policy, cfg, n_obs_steps


def _make_obs_torch(obs_np: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in obs_np.items():
        out[k] = torch.from_numpy(v).unsqueeze(0).to(device)  # (1,T,...)
    return out


class _MockCamera:
    def read(self, img_size=None):
        rgb = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        depth = np.random.randint(0, 65536, size=(480, 640), dtype=np.uint16)
        return rgb, depth


def main(
    ckpt: str,
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
    log_dir: str = os.path.expanduser("~/diffusion_logs"),
    save: bool = False,

    # Remote policy settings
    use_remote_policy: bool = False,
    policy_server_host: str = "127.0.0.1",
    policy_server_port: int = 8765,
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
        if not _HAS_PI0_CLIENT:
            raise RuntimeError("openpi_client.websocket_client_policy not found. Install pi0 client or set use_remote_policy=False.")
        policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=policy_server_host, port=policy_server_port
        )
        print(f"[REMOTE] Connected to policy server at ws://{policy_server_host}:{policy_server_port}")
    else:
        policy, cfg, n_obs_steps = _load_diffusion_policy(ckpt)
        device = next(policy.parameters()).device
        print(f"[LOCAL POLICY] Loaded on device: {device}, n_obs_steps={n_obs_steps}")

    H = 224; W = 224
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

        # Need at least 2 timesteps before local inference; remote server handles its own buffer
        if (action_chunk is None) and (not use_remote_policy) and (not buf.ready()):
            time.sleep(max(0.0, dt - (time.time() - loop_t0)))
            continue

        need_new_chunk = (action_chunk is None) or (actions_from_chunk_completed >= action_chunk_len)
        if need_new_chunk:
            if use_remote_policy:
                # Single-step obs to server; server keeps its own 2-step buffer
                base_rgb = resize_with_pad(obs["base_rgb"], H, W)
                wrist_rgb = resize_with_pad(obs["wrist_rgb"], H, W)
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

        # Pull next action
        action = action_chunk[actions_from_chunk_completed]
        actions_from_chunk_completed += 1

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
        p.add_argument("--log_dir", default=os.path.expanduser("~/diffusion_logs"))
        p.add_argument("--save", action="store_true")
        p.add_argument("--use_remote_policy", action="store_true")
        p.add_argument("--policy_server_host", default="127.0.0.1")
        p.add_argument("--policy_server_port", type=int, default=8765)
        args = p.parse_args()
        main(**vars(args))
