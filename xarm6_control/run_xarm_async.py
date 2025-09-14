#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async Diffusion Policy runner for xArm6 (dual RealSense, 7-DoF joint+gripper).

Modes
- Local: background inference thread (double-buffered, prefetch)
- Remote: background websocket client to a policy server returning action chunks

Speed
- Mixed precision, pinned-memory, TF32, warm-up
- Prefetch next action chunk before current chunk finishes
- Keeps your delta-threshold interpolation & step-through safety

Usage:
  # LOCAL (no server)
  python run_diffusion_xarm6_async.py --ckpt /path/to/ckpt --mock False

  # REMOTE (policy server on 127.0.0.1:8765)
  python run_diffusion_xarm6_async.py --ckpt DUMMY --use_remote_policy \
      --policy_server_host 127.0.0.1 --policy_server_port 8765 --mock False
"""
import os, time, copy, math, threading, queue, asyncio
import tyro, torch, dill, hydra, numpy as np
import msgpack_numpy
import websockets
from collections import deque
from typing import Dict, Tuple, Optional

from xarm6_control.xarm_env import XArmRealEnv, MockXArmEnv
from xarm6_control.zmq_core.camera_node import ZMQClientCamera

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from tools.resize_pkl import resize_with_pad

# ---------- Speed knobs ----------
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def to_chw01(img_hwc_uint8: np.ndarray) -> np.ndarray:
    img = img_hwc_uint8.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1)).copy()  # CHW, ensure contiguous

class ObsBuffer:
    def __init__(self, n_obs_steps: int = 2, H: int = 224, W: int = 224):
        self.n = n_obs_steps; self.H, self.W = H, W
        self.base_rgb = deque(maxlen=self.n)
        self.wrist_rgb = deque(maxlen=self.n)
        self.robot_state = deque(maxlen=self.n)

    def push(self, obs: Dict):
        base = resize_with_pad(obs["base_rgb"], self.H, self.W)
        wrist = resize_with_pad(obs["wrist_rgb"], self.H, self.W)
        self.base_rgb.append(to_chw01(base))
        self.wrist_rgb.append(to_chw01(wrist))
        state7 = np.concatenate([obs["joint_position"], obs["gripper_position"]]).astype(np.float32)
        self.robot_state.append(state7)
        while len(self.base_rgb) < self.n:
            self.base_rgb.appendleft(self.base_rgb[0].copy())
            self.wrist_rgb.appendleft(self.wrist_rgb[0].copy())
            self.robot_state.appendleft(self.robot_state[0].copy())

    def ready(self) -> bool:
        return len(self.base_rgb) == self.n

    def snapshot_np(self) -> Dict[str, np.ndarray]:
        return dict(
            base_rgb=np.stack(list(self.base_rgb), axis=0),
            wrist_rgb=np.stack(list(self.wrist_rgb), axis=0),
            robot_state=np.stack(list(self.robot_state), axis=0),
        )

# ---------------- Local policy loader ----------------
def load_diffusion_policy(ckpt_path: str, num_inference_steps: int) -> Tuple[BaseImagePolicy, dict, int, torch.device]:
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

    # Inference iteration budget (latency knob) — defensive setter
    if hasattr(policy, 'set_num_inference_steps'):
        policy.set_num_inference_steps(int(num_inference_steps))
    else:
        setattr(policy, 'num_inference_steps', int(num_inference_steps))

    n_obs_steps = int(getattr(cfg, "n_obs_steps", 2))
    return policy, cfg, n_obs_steps, device

def _to_torch_batch(obs_np: Dict[str, np.ndarray], device: torch.device, channels_last: bool) -> Dict[str, torch.Tensor]:
    out = {}
    pin = (device.type == 'cuda')
    for k, v in obs_np.items():
        t = torch.from_numpy(v).unsqueeze(0)  # (1,T,...)
        if pin:
            t = t.pin_memory()
        t = t.to(device, non_blocking=pin)
        if channels_last and t.dim() == 4:
            t = t.contiguous(memory_format=torch.channels_last)
        out[k] = t
    return out

# ---------------- Local inference worker ----------------
class LocalInferenceWorker(threading.Thread):
    def __init__(self,
                 policy: BaseImagePolicy,
                 device: torch.device,
                 in_q: "queue.Queue[Tuple[int, Dict[str, np.ndarray]]]",
                 out_q: "queue.Queue[Tuple[int, np.ndarray]]",
                 stop_evt: threading.Event,
                 fp16: bool,
                 channels_last: bool):
        super().__init__(daemon=True)
        self.policy = policy
        self.device = device
        self.in_q = in_q
        self.out_q = out_q
        self.stop_evt = stop_evt
        self.fp16 = fp16 and (device.type == 'cuda')
        self.channels_last = channels_last

    def run(self):
        try:
            seq, obs_np = self.in_q.get(timeout=5.0)
        except queue.Empty:
            return
        self._infer(seq, obs_np)

        while not self.stop_evt.is_set():
            try:
                seq, obs_np = self.in_q.get(timeout=0.05)
                while not self.in_q.empty():
                    seq, obs_np = self.in_q.get_nowait()
                self._infer(seq, obs_np)
            except queue.Empty:
                continue

    def _infer(self, seq: int, obs_np: Dict[str, np.ndarray]):
        try:
            obs_t = _to_torch_batch(obs_np, self.device, self.channels_last)
            with torch.inference_mode():
                if self.fp16:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        result = self.policy.predict_action(obs_t)
                else:
                    result = self.policy.predict_action(obs_t)
            action_chunk = result["action"][0].detach().to("cpu").numpy()
            try:
                while not self.out_q.empty():
                    self.out_q.get_nowait()
                self.out_q.put_nowait((seq, action_chunk))
            except queue.Full:
                pass
        except Exception as e:
            print(f"[LOCAL WORKER] Inference error: {e}")

# ---------------- Remote policy worker (websocket) ----------------
class RemotePolicyWorker(threading.Thread):
    """Sends single-step obs to server; receives action chunk."""
    def __init__(self,
                 host: str,
                 port: int,
                 H: int,
                 W: int,
                 in_q: "queue.Queue[Tuple[int, Dict]]",
                 out_q: "queue.Queue[Tuple[int, np.ndarray]]",
                 stop_evt: threading.Event,
                 prompt: Optional[str] = None):
        super().__init__(daemon=True)
        self.uri = f"ws://{host}:{port}"
        self.H, self.W = H, W
        self.in_q = in_q
        self.out_q = out_q
        self.stop_evt = stop_evt
        self.packer = msgpack_numpy.Packer()
        self.prompt = prompt

    def run(self):
        asyncio.run(self._loop())

    async def _loop(self):
        try:
            async with websockets.connect(self.uri, max_size=None) as ws:
                # Try to read metadata frame (non-fatal)
                try:
                    meta = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    print(f"[REMOTE] Connected. Metadata: {msgpack_numpy.unpackb(meta)}")
                except Exception:
                    print("[REMOTE] Connected. No metadata frame.")
                while not self.stop_evt.is_set():
                    # Drain to newest job
                    try:
                        seq, obs = self.in_q.get(timeout=0.05)
                        while not self.in_q.empty():
                            seq, obs = self.in_q.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.005)
                        continue

                    # Build single-step observation (server maintains its own buffer)
                    try:
                        base = resize_with_pad(obs["base_rgb"], self.H, self.W)
                        wrist = resize_with_pad(obs["wrist_rgb"], self.H, self.W)
                        msg = {
                            "base_rgb": base,                # HxWx3 uint8
                            "wrist_rgb": wrist,              # HxWx3 uint8
                            "joint_position": obs["joint_position"].astype(np.float32),
                            "gripper_position": obs["gripper_position"].astype(np.float32),
                        }
                        if self.prompt:
                            msg["prompt"] = self.prompt
                    except KeyError as e:
                        print(f"[REMOTE] Missing key in obs: {e}")
                        continue

                    await ws.send(self.packer.pack(msg))
                    reply = await ws.recv()
                    data = msgpack_numpy.unpackb(reply)
                    actions = np.asarray(data.get("actions", []), dtype=np.float32)
                    if actions.ndim == 1:
                        actions = actions[None, :]
                    # Publish newest only
                    try:
                        while not self.out_q.empty():
                            self.out_q.get_nowait()
                        self.out_q.put_nowait((seq, actions))
                    except queue.Full:
                        pass
        except Exception as e:
            print(f"[REMOTE] Connection/loop error: {e}")

class _MockCamera:
    def read(self, img_size=None):
        rgb = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
        depth = np.random.randint(0, 65536, size=(480, 640), dtype=np.uint16)
        return rgb, depth

def main(
    ckpt: str,
    remote_host: str = "localhost",
    wrist_camera_port: int = 5000,
    base_camera_port: int = 5001,
    max_steps: int = 5000,
    mock: bool = False,
    control_hz: float = 30.0,
    step_through_instructions: bool = False,
    delta_threshold: float = 0.25,           # degrees per joint
    log_dir: str = "/media/acrv/DanielsSSD/Test_sem2/diffusion",
    save: bool = False,
    num_inference_steps: int = 8,            # local latency lever (try 6–12)
    prefetch_margin: int = 4,                # request next chunk when this many steps remain
    fp16: bool = True,                       # enable mixed precision on CUDA
    channels_last: bool = False,             # keep False; 5-D tensors aren't channels-last
    # Remote policy settings
    use_remote_policy: bool = False,
    policy_server_host: str = "127.0.0.1",
    policy_server_port: int = 8765,
    prompt: str = "Pick a ripe, red tomato and drop it in the blue bucket."
):
    # Env & cameras
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

    H = 224; W = 224

    # Policy / worker setup
    policy = cfg = device = None
    n_obs_steps = 2
    in_q_local: "queue.Queue[Tuple[int, Dict]]" = queue.Queue(maxsize=1)
    out_q: "queue.Queue[Tuple[int, np.ndarray]]" = queue.Queue(maxsize=1)
    stop_evt = threading.Event()

    if use_remote_policy:
        worker = RemotePolicyWorker(
            host=policy_server_host, port=policy_server_port,
            H=H, W=W, in_q=in_q_local, out_q=out_q, stop_evt=stop_evt, prompt=prompt
        )
        print(f"[REMOTE POLICY] ws://{policy_server_host}:{policy_server_port}")
        # Remote server maintains its own temporal buffer; we don't need ObsBuffer for gating
        buf = None
    else:
        policy, cfg, n_obs_steps, device = load_diffusion_policy(ckpt, num_inference_steps)
        try:
            policy.reset()
        except Exception:
            pass
        print(f"[LOCAL POLICY] device={device}, n_obs_steps={n_obs_steps}, num_inference_steps={num_inference_steps}")
        buf = ObsBuffer(n_obs_steps=n_obs_steps, H=H, W=W)
        worker = LocalInferenceWorker(policy, device, in_q_local, out_q, stop_evt, fp16, channels_last)

    dt = 1.0 / float(control_hz)

    # Action buffers
    action_chunk: Optional[np.ndarray] = None
    next_action_chunk: Optional[np.ndarray] = None
    action_chunk_len = 0
    actions_done = 0
    seq = 0
    worker_started = False

    for step_idx in range(max_steps):
        loop_t0 = time.time()
        obs = env.get_observation()

        # Push into buffer only for local mode
        if not use_remote_policy:
            buf.push(obs)

        # Prime worker:
        if not worker_started and (use_remote_policy or (buf and buf.ready())):
            try:
                if use_remote_policy:
                    in_q_local.put_nowait((seq, obs))  # single-step obs
                else:
                    in_q_local.put_nowait((seq, (buf.snapshot_np())))
                worker.start()
                worker_started = True
                print("[WORKER] Started and primed.")
            except queue.Full:
                pass

        # Pull any finished inference without blocking (common for local/remote)
        if not out_q.empty():
            try:
                _, produced = out_q.get_nowait()
                if produced is not None and produced.ndim == 2 and produced.shape[1] == 7:
                    next_action_chunk = produced
            except queue.Empty:
                pass

        # If no current chunk, try to adopt next; otherwise (local only) do sync warm-up
        if action_chunk is None:
            if next_action_chunk is not None:
                action_chunk = next_action_chunk; next_action_chunk = None
                action_chunk_len = action_chunk.shape[0]; actions_done = 0
                print(f"[POLICY] Adopted initial chunk: {action_chunk_len} steps")
            else:
                if (not use_remote_policy) and (buf and buf.ready()):
                    obs_np = buf.snapshot_np()
                    with torch.inference_mode():
                        if fp16 and device and device.type == 'cuda':
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                result = policy.predict_action(_to_torch_batch(obs_np, device, channels_last))
                        else:
                            result = policy.predict_action(_to_torch_batch(obs_np, device, channels_last))
                    action_chunk = result["action"][0].detach().cpu().numpy()
                    action_chunk_len = action_chunk.shape[0]; actions_done = 0
                    print(f"[POLICY] Sync warm-up chunk: {action_chunk_len} steps")
                elapsed = time.time() - loop_t0
                if elapsed < dt: time.sleep(dt - elapsed)
                # Also kick the worker with the latest obs/snapshot in case queue is empty
                if worker_started and in_q_local.empty():
                    seq += 1
                    try:
                        if use_remote_policy:
                            in_q_local.put_nowait((seq, obs))
                        else:
                            in_q_local.put_nowait((seq, (buf.snapshot_np())))
                    except queue.Full:
                        pass
                continue

        # Prefetch next chunk when close to end
        remaining = action_chunk_len - actions_done
        if remaining <= max(1, min(prefetch_margin, max(1, action_chunk_len // 2))):
            if worker_started and in_q_local.empty():
                seq += 1
                try:
                    if use_remote_policy:
                        in_q_local.put_nowait((seq, obs))         # single-step
                    else:
                        in_q_local.put_nowait((seq, buf.snapshot_np()))
                except queue.Full:
                    pass

        # --- Guard before indexing the chunk (prevents OOB & stalls) ---
        if action_chunk is not None and actions_done >= action_chunk_len:
            if next_action_chunk is not None:
                action_chunk = next_action_chunk; next_action_chunk = None
                action_chunk_len = action_chunk.shape[0]; actions_done = 0
                print(f"[POLICY] Swapped chunk: {action_chunk_len} steps")
            else:
                elapsed = time.time() - loop_t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                continue

        # Take next action
        action = action_chunk[actions_done].astype(np.float32)
        action[-1] = float(np.clip(action[-1], 0.0, 1.0))  # clamp gripper [0,1]
        actions_done += 1

        # Safety: degrees thresholding
        current_joints_deg = np.degrees(obs["joint_position"][:6])
        action_joints_deg  = np.degrees(action[:6])
        delta_deg = action_joints_deg - current_joints_deg

        if step_through_instructions:
            print(f"\n[STEP {step_idx+1}, ACTION {actions_done}/{action_chunk_len}]")
            print("Current (deg):", np.round(current_joints_deg, 2))
            print("Action  (deg):", np.round(action_joints_deg, 2))
            print("Delta   (deg):", np.round(delta_deg, 2))
            print(f"Gripper pose: {obs['gripper_position']}, Gripper action: {action[-1]:.3f}")
            if np.any(np.abs(delta_deg) > delta_threshold):
                print("⚠️  Warning: large joint delta detected!")
            cmd = input("Press [Enter] to execute, 's' to skip, or 'q' to quit: ").strip().lower()
            if cmd == "q":
                print("Exiting policy execution."); break
            elif cmd == "s":
                print("Skipping this action."); actions_done = action_chunk_len
            else:
                if np.any(np.abs(delta_deg) > delta_threshold):
                    state = np.concatenate([obs["joint_position"], obs["gripper_position"]])
                    interp = env.generate_joint_trajectory(state, action, delta_threshold * math.pi / 180.0)
                    obs = env.step_through_interpolated_trajectory(
                        interp, obs, step_idx, log_dir, control_hz, step_through_instructions, save
                    )
                else:
                    if save: env.save_step_data(log_dir, step_idx, copy.deepcopy(obs), action)
                    env.step(np.array(action))
            elapsed = time.time() - loop_t0
            if elapsed < dt: time.sleep(dt - elapsed)
            continue

        # Non step-through
        if np.any(np.abs(delta_deg) > delta_threshold):
            state = np.concatenate([obs["joint_position"], obs["gripper_position"]])
            interp = env.generate_joint_trajectory(state, action, delta_threshold * math.pi / 180.0)
            obs = env.step_through_interpolated_trajectory(
                interp, obs, step_idx, log_dir, control_hz, step_through_instructions, save
            )
        else:
            if save: env.save_step_data(log_dir, step_idx, copy.deepcopy(obs), action)
            env.step(np.array(action))

        # Keep control rate
        elapsed = time.time() - loop_t0
        if elapsed < dt:
            time.sleep(dt - elapsed)

    stop_evt.set()

if __name__ == "__main__":
    tyro.cli(main)
