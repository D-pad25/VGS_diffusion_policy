#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serve a Diffusion Policy checkpoint over WebSocket (pi0-style), optimized.

What’s new (fast path):
- TF32 on CUDA, cudnn.benchmark, threads=1
- Fewer denoise steps by default (6) via set_num_inference_steps if available
- Optional torch.compile (PyTorch 2.x) with fallback
- AMP (fp16) inference on CUDA
- Pinned-memory + non-blocking tensor transfers
- Warm-up passes to lock fast kernels
- Avoid slow .tolist(); return NumPy (msgpack_numpy handles it)
- WebSocket tuned: compression off, small queue, keepalive pings disabled
- Optional micro-profiling prints

Expected message in (one timestep per msg):
{
  "base_rgb":  HxWx3 uint8,
  "wrist_rgb": HxWx3 uint8,
  "joint_position": (6,) float (rad),
  "gripper_position": (1,) float
}
OR
{
  "base_rgb": ..., "wrist_rgb": ..., "robot_state": (7,) float
}

Response:
{ "actions": np.ndarray[T, 7] }  # action chunk

Usage (HPC):
  (robodiff)$ python -m xarm6_control.serve_diffusion_ws \
      --ckpt "$CKPT" --host 127.0.0.1 --port 8000 --image_size 224 --perf False
"""

import asyncio
import logging
import traceback
import time
from typing import Dict, Tuple, Optional
from collections import deque

import numpy as np
import torch
import dill
import hydra
from omegaconf import OmegaConf

import websockets.asyncio.server
import websockets.frames

# msgpack with numpy support
import msgpack_numpy

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _resize_with_pad(img_hwc_uint8: np.ndarray, H: int = 224, W: int = 224) -> np.ndarray:
    import cv2
    h, w = img_hwc_uint8.shape[:2]
    scale = min(W / w, H / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(
        img_hwc_uint8, (nw, nh),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    )
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    x0 = (W - nw) // 2
    y0 = (H - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


def _to_chw01(img_hwc_uint8: np.ndarray) -> np.ndarray:
    img = img_hwc_uint8.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))  # CHW, float32 in [0,1]


class _ObsBuffer:
    def __init__(self, n_obs_steps: int = 2, H: int = 224, W: int = 224):
        self.n = n_obs_steps
        self.H, self.W = H, W
        self.base_rgb = deque(maxlen=self.n)
        self.wrist_rgb = deque(maxlen=self.n)
        self.robot_state = deque(maxlen=self.n)

    def push_single(self, obs: Dict):
        base = _resize_with_pad(np.asarray(obs["base_rgb"], dtype=np.uint8), self.H, self.W)
        wrist = _resize_with_pad(np.asarray(obs["wrist_rgb"], dtype=np.uint8), self.H, self.W)
        self.base_rgb.append(_to_chw01(base))
        self.wrist_rgb.append(_to_chw01(wrist))

        # Accept either split fields or pre-packed robot_state
        if "robot_state" in obs:
            state7 = np.asarray(obs["robot_state"], dtype=np.float32)
        else:
            state7 = np.concatenate([
                np.asarray(obs["joint_position"], dtype=np.float32),
                np.asarray(obs["gripper_position"], dtype=np.float32)
            ]).astype(np.float32)
        self.robot_state.append(state7)

        # Front-fill until buffer full
        while len(self.base_rgb) < self.n:
            self.base_rgb.appendleft(self.base_rgb[0].copy())
            self.wrist_rgb.appendleft(self.wrist_rgb[0].copy())
            self.robot_state.appendleft(self.robot_state[0].copy())

    def ready(self) -> bool:
        return len(self.base_rgb) == self.n

    def as_torch(self, device: torch.device) -> Dict[str, torch.Tensor]:
        # Cast to float32 now; use pinned memory + non-blocking copies
        pin = (device.type == 'cuda')
        base_np  = np.stack(list(self.base_rgb), axis=0).astype(np.float32)
        wrist_np = np.stack(list(self.wrist_rgb), axis=0).astype(np.float32)
        state_np = np.stack(list(self.robot_state), axis=0).astype(np.float32)

        base  = torch.from_numpy(base_np).unsqueeze(0)   # (1,T,3,H,W)
        wrist = torch.from_numpy(wrist_np).unsqueeze(0)  # (1,T,3,H,W)
        state = torch.from_numpy(state_np).unsqueeze(0)  # (1,T,7)
        if pin:
            base, wrist, state = base.pin_memory(), wrist.pin_memory(), state.pin_memory()

        base  = base.to(device, non_blocking=pin)
        wrist = wrist.to(device, non_blocking=pin)
        state = state.to(device, non_blocking=pin)
        return {"base_rgb": base, "wrist_rgb": wrist, "robot_state": state}


def _extract_actions(result: Dict) -> torch.Tensor:
    a = result.get("action", result.get("action_pred"))
    if isinstance(a, torch.Tensor):
        pass
    else:
        # Some policies return dict/obj; try attribute or first value
        try:
            a = a["action"]
        except Exception:
            raise KeyError("No 'action' or 'action_pred' in policy result")
    # Expect (B,T,A) or (T,A); normalize to (T,A)
    if a.ndim == 3:
        a = a[0]
    return a


def _load_diffusion_policy(ckpt_path: str) -> Tuple[BaseImagePolicy, dict, int]:
    # Global speed toggles
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
    policy = policy.to(device).eval()

    # Fewer denoise steps by default (latency knob)
    try:
        policy.set_num_inference_steps(6)
    except AttributeError:
        policy.num_inference_steps = min(getattr(policy, 'num_inference_steps', 16), 16)

    # Optional compile (PyTorch 2.x). Ignore if unsupported.
    try:
        policy = torch.compile(policy, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass

    n_obs_steps = int(getattr(cfg, "n_obs_steps", 2))
    return policy, cfg, n_obs_steps


class _DiffusionPolicyAdapter:
    """pi0-like interface: .infer(obs) -> {'actions': ndarray(T,7)}"""
    def __init__(self, policy: BaseImagePolicy, n_obs_steps: int, H: int = 224, W: int = 224):
        self.policy = policy
        self.device = next(policy.parameters()).device
        self.buf = _ObsBuffer(n_obs_steps=n_obs_steps, H=H, W=W)

    @torch.inference_mode()
    def infer(self, obs: Dict) -> Dict:
        self.buf.push_single(obs)
        if not self.buf.ready():
            return {"actions": np.empty((0, 7), dtype=np.float32)}  # warm-up returns empty
        model_obs = self.buf.as_torch(self.device)

        if self.device.type == 'cuda':
            with torch.cuda.amp.autocast(dtype=torch.float16):
                result = self.policy.predict_action(model_obs)
        else:
            result = self.policy.predict_action(model_obs)

        actions_t = _extract_actions(result)                         # (T,7) torch
        actions = actions_t.detach().to("cpu", non_blocking=True).numpy()
        return {"actions": actions}


class WebsocketPolicyServer:
    def __init__(
        self,
        policy: BaseImagePolicy,
        n_obs_steps: int,
        host: str = "127.0.0.1",
        port: int = 8000,
        image_size: int = 224,
        metadata: Optional[dict] = None,
        perf: bool = False
    ):
        self._policy = policy
        self._n_obs_steps = n_obs_steps
        self._host = host
        self._port = port
        self._image_size = image_size
        self._metadata = metadata or {}
        self._perf = perf
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        print(f"[SERVER] Listening on ws://{self._host}:{self._port}")
        async with websockets.asyncio.server.serve(
            self._handler, self._host, self._port,
            compression=None, max_size=None, max_queue=1,
            ping_interval=None, ping_timeout=None
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer(use_bin_type=True)
        await websocket.send(packer.pack(self._metadata))

        # Adapter/buffer PER CONNECTION; use configured image_size
        adapter = _DiffusionPolicyAdapter(
            self._policy, n_obs_steps=self._n_obs_steps,
            H=self._image_size, W=self._image_size
        )

        while True:
            try:
                t0 = time.perf_counter()
                raw = await websocket.recv()
                obs = msgpack_numpy.unpackb(raw)
                t1 = time.perf_counter()

                out = adapter.infer(obs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t2 = time.perf_counter()

                await websocket.send(packer.pack(out))
                t3 = time.perf_counter()

                if self._perf:
                    print(f"[PERF] recv={(t1-t0)*1e3:.1f}ms  infer={(t2-t1)*1e3:.1f}ms  send={(t3-t2)*1e3:.1f}ms")
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                tb = traceback.format_exc()
                try:
                    await websocket.send(tb)
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame."
                    )
                finally:
                    raise


def main(ckpt: str, host: str = "0.0.0.0", port: int = 8000, image_size: int = 224, perf: bool = False):
    policy, cfg, n_obs_steps = _load_diffusion_policy(ckpt)
    device = next(policy.parameters()).device
    print(f"[SERVER] Loaded policy on {device}; n_obs_steps={n_obs_steps}")

    # Warm-up to pick fastest kernels (esp. cudnn)
    B, T, C, H, W = 1, n_obs_steps, 3, image_size, image_size
    dummy = {
        "base_rgb":  torch.zeros((B, T, C, H, W), device=device),
        "wrist_rgb": torch.zeros((B, T, C, H, W), device=device),
        "robot_state": torch.zeros((B, T, 7), device=device),
    }
    with torch.inference_mode():
        for _ in range(3):
            _ = policy.predict_action(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    meta = {
        "model": getattr(cfg, "name", "diffusion"),
        "n_obs_steps": n_obs_steps,
        "image_size": image_size
    }
    WebsocketPolicyServer(
        policy, n_obs_steps=n_obs_steps, host=host, port=port,
        image_size=image_size, metadata=meta, perf=perf
    ).serve_forever()


if __name__ == "__main__":
    try:
        import tyro
        tyro.cli(main)
    except Exception:
        import argparse
        p = argparse.ArgumentParser()
        p.add_argument("--ckpt", required=True)
        p.add_argument("--host", default="0.0.0.0")
        p.add_argument("--port", type=int, default=8000)
        p.add_argument("--image_size", type=int, default=224)
        p.add_argument("--perf", action="store_true")
        args = p.parse_args()
        main(**vars(args))
