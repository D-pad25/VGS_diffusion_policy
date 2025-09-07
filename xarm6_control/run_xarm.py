#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a Diffusion Policy checkpoint on the xArm6 using your existing pi0-style loop.
- Two-step observation window (n_obs_steps=2)
- Action chunk execution (e.g., 16 x 7)
- Works with your XArmRealEnv / MockXArmEnv and ZMQClientCamera
- Keeps your step-through & delta-threshold interpolation logic

Usage:
  (robodiff)$ python run_diffusion_xarm6.py \
      --ckpt /path/to/checkpoint.ckpt \
      --remote_host localhost \
      --remote_port 8000 \
      --wrist_camera_port 5000 \
      --base_camera_port 5001 \
      --control_hz 30 \
      --prompt "Pick a ripe, red tomato and drop it in the blue bucket." \
      --save False \
      --mock False
"""

import os
import time
import copy
import math
import tyro
import torch
import dill
import hydra
import numpy as np
from collections import deque
from typing import Dict, Tuple

# --- Your existing modules ---
from xarm_env import XArmRealEnv, MockXArmEnv
from zmq_core.camera_node import ZMQClientCamera


# --- Diffusion Policy imports ---
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from tools.resize_pkl import resize_with_pad

# ---------- Helpers ----------

def to_chw01(img_hwc_uint8: np.ndarray) -> np.ndarray:
    """HWC uint8 -> CHW float32 in [0,1]."""
    img = img_hwc_uint8.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))  # CHW

class ObsBuffer:
    """Keeps last n_obs_steps frames for each key."""
    def __init__(self, n_obs_steps: int = 2, H: int = 224, W: int = 224):
        self.n = n_obs_steps
        self.H, self.W = H, W
        self.base_rgb = deque(maxlen=self.n)
        self.wrist_rgb = deque(maxlen=self.n)
        self.robot_state = deque(maxlen=self.n)

    def push(self, obs: Dict):
        # Expect these keys from env.get_observation()
        base = resize_with_pad(obs["base_rgb"], self.H, self.W)
        wrist = resize_with_pad(obs["wrist_rgb"], self.H, self.W)

        self.base_rgb.append(to_chw01(base))  # CHW
        self.wrist_rgb.append(to_chw01(wrist))

        # robot_state = [j1..j6, gripper] as float32
        state7 = np.concatenate([obs["joint_position"], obs["gripper_position"]]).astype(np.float32)
        self.robot_state.append(state7)

        # If not yet full, front-fill with first frame
        while len(self.base_rgb) < self.n:
            self.base_rgb.appendleft(self.base_rgb[0].copy())
            self.wrist_rgb.appendleft(self.wrist_rgb[0].copy())
            self.robot_state.appendleft(self.robot_state[0].copy())

    def ready(self) -> bool:
        return len(self.base_rgb) == self.n

    def as_np_dict(self) -> Dict[str, np.ndarray]:
        """Return numpy arrays with leading obs-step dimension (T, ...)."""
        base = np.stack(list(self.base_rgb), axis=0)        # (T, 3, H, W)
        wrist = np.stack(list(self.wrist_rgb), axis=0)      # (T, 3, H, W)
        state = np.stack(list(self.robot_state), axis=0)    # (T, 7)
        return dict(base_rgb=base, wrist_rgb=wrist, robot_state=state)

def load_diffusion_policy(ckpt_path: str) -> Tuple[BaseImagePolicy, dict, int]:
    """Load workspace & policy from a diffusion-policy checkpoint; return (policy, cfg, n_obs_steps)."""
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Prefer EMA if available and enabled
    policy: BaseImagePolicy = workspace.model
    try:
        if getattr(cfg, "training", None) and getattr(cfg.training, "use_ema", False):
            policy = workspace.ema_model
    except Exception:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval().to(device)

    # Set sensible inference defaults if not present
    if not hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = 16  # DDIM steps
    # Let the model decide n_action_steps; we’ll consume whatever it returns

    n_obs_steps = int(getattr(cfg, "n_obs_steps", 2))
    return policy, cfg, n_obs_steps

def make_obs_torch(obs_np: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert (T, ...) numpy to (B=1, T, ...) torch on device."""
    out = {}
    for k, v in obs_np.items():
        t = torch.from_numpy(v).unsqueeze(0).to(device)  # (1, T, ...)
        out[k] = t
    return out

# ---------- Main runner ----------

def main(
    ckpt: str,
    remote_host: str = "localhost",
    remote_port: int = 8000,              # (kept for parity; not used by diffusion)
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
):
    """
    Notes:
      - Diffusion policy expects: T=2 obs steps for each of base_rgb, wrist_rgb, robot_state.
      - Policy returns an action chunk (e.g., 16 x 7), which we play at control_hz.
    """

    # --- Environment & cameras (your existing setup) ---
    if mock:
        camera_clients = {
            "wrist": _MockCamera(),
            "base": _MockCamera(),
        }
        env = MockXArmEnv(camera_dict=camera_clients)
    else:
        camera_clients = {
            "wrist": ZMQClientCamera(port=wrist_camera_port, host=remote_host),
            "base":  ZMQClientCamera(port=base_camera_port,  host=remote_host),
        }
        env = XArmRealEnv(camera_dict=camera_clients)

    # Optional logging dir
    if save:
        import datetime
        log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)
        print(f"[LOG] Saving to: {log_dir}")

    # --- Load diffusion policy ---
    policy, cfg, n_obs_steps = load_diffusion_policy(ckpt)
    device = next(policy.parameters()).device
    print(f"[POLICY] Loaded on device: {device}, n_obs_steps={n_obs_steps}")

    # Validate shape_meta keys (we’ll just warn if different)
    try:
        shape_meta = cfg.task.shape_meta
        expected = {"base_rgb", "wrist_rgb", "robot_state"}
        found = set(shape_meta["obs"].keys())
        if not expected.issubset(found):
            print(f"[WARN] shape_meta obs keys {found} do not include all expected {expected}. "
                  f"Proceeding with keys we produce.")
    except Exception:
        print("[WARN] Could not read shape_meta from cfg; proceeding with default keys.")

    # --- Obs buffer (2 timesteps) ---
    H = 224; W = 224
    buf = ObsBuffer(n_obs_steps=n_obs_steps, H=H, W=W)

    # --- Control loop ---
    dt = 1.0 / float(control_hz)
    actions_from_chunk_completed = 0
    action_chunk = None
    action_chunk_len = 0

    # (Optional) policy reset
    try:
        policy.reset()
    except Exception:
        pass

    for step_idx in range(max_steps):
        loop_t0 = time.time()
        obs = env.get_observation()
        buf.push(obs)

        # Need at least 2 timesteps before first inference
        if action_chunk is None and not buf.ready():
            # Sleep to maintain control rate
            time.sleep(max(0.0, dt - (time.time() - loop_t0)))
            continue

        # Get a new action chunk if needed
        if (action_chunk is None) or (actions_from_chunk_completed >= action_chunk_len):
            with torch.no_grad():
                obs_np = buf.as_np_dict()
                obs_torch = make_obs_torch(obs_np, device=device)

                # Build full obs dict expected by policy
                # If your policy uses other names, add remapping here.
                model_obs = {
                    "base_rgb":    obs_torch["base_rgb"],      # (1, T, 3, H, W)
                    "wrist_rgb":   obs_torch["wrist_rgb"],     # (1, T, 3, H, W)
                    "robot_state": obs_torch["robot_state"],   # (1, T, 7)
                    # "prompt":    (not used by diffusion-policy)
                }

                result = policy.predict_action(model_obs)
                # result['action'] -> (B=1, A, 7) typically; pull the first batch
                action_chunk = result["action"][0].detach().to("cpu").numpy()
                action_chunk_len = action_chunk.shape[0]
                actions_from_chunk_completed = 0

                if action_chunk.shape[-1] != 7:
                    raise RuntimeError(f"Policy returned unexpected action shape {action_chunk.shape}; expected (*,7).")

                print(f"[POLICY] New action chunk: {action_chunk_len} steps")

        # Current state & next action
        action = action_chunk[actions_from_chunk_completed]
        actions_from_chunk_completed += 1

        # Convert to degrees for thresholding (your env uses radians internally)
        current_joints_deg = np.degrees(obs["joint_position"][:6])
        action_joints_deg  = np.degrees(action[:6])
        delta_deg = action_joints_deg - current_joints_deg

        # Step-through mode for safety/debug
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
                actions_from_chunk_completed = action_chunk_len  # force refresh next loop
                continue

            # Interpolate if needed
            if np.any(np.abs(delta_deg) > delta_threshold):
                state = np.concatenate([obs["joint_position"], obs["gripper_position"]])
                interp = env.generate_joint_trajectory(state, action, delta_threshold * math.pi / 180.0)
                obs = env.step_through_interpolated_trajectory(
                    interp, obs, step_idx, log_dir, control_hz, step_through_instructions, save
                )
            else:
                if save:
                    env.save_step_data(log_dir, step_idx, copy.deepcopy(obs), action)
                env.step(np.array(action))

            # Keep loop period
            elapsed = time.time() - loop_t0
            time.sleep(max(0.0, dt - elapsed))
            continue

        # Non step-through: safety interpolation if delta too large
        if np.any(np.abs(delta_deg) > delta_threshold):
            state = np.concatenate([obs["joint_position"], obs["gripper_position"]])
            interp = env.generate_joint_trajectory(state, action, delta_threshold * math.pi / 180.0)
            obs = env.step_through_interpolated_trajectory(
                interp, obs, step_idx, log_dir, control_hz, step_through_instructions, save
            )
        else:
            if save:
                env.save_step_data(log_dir, step_idx, copy.deepcopy(obs), action)
            env.step(np.array(action))

        # Maintain control rate
        elapsed = time.time() - loop_t0
        time.sleep(max(0.0, dt - elapsed))

class _MockCamera:
    def read(self, img_size=None):
        rgb = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        depth = np.random.randint(0, 65536, size=(480, 640), dtype=np.uint16)
        return rgb, depth

if __name__ == "__main__":
    tyro.cli(main)