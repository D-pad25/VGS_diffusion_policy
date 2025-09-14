#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serve a Diffusion Policy checkpoint over WebSocket (pi0-style).
- Loads checkpoint on GPU (if available)
- Keeps a 2-step observation buffer PER CONNECTION
- Expects one timestep per message:
    {
      "base_rgb":  HxWx3 uint8,
      "wrist_rgb": HxWx3 uint8,
      "joint_position": (6,) float (radians),
      "gripper_position": (1,) float
    }
- Responds with: {"actions": [[...7...], ...]}  # action chunk

Usage (HPC):
  (robodiff)$ python -m xarm6_control.serve_diffusion_ws --ckpt /path/to/ckpt --host 127.0.0.1 --port 8765
"""

import asyncio
import logging
import traceback
import numpy as np
import torch
import dill
import hydra
from typing import Dict, Tuple, Optional
from collections import deque
from omegaconf import OmegaConf

import websockets.asyncio.server
import websockets.frames

# msgpack with numpy support (same as pi0)
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
    return np.transpose(img, (2, 0, 1))  # CHW


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
        state7 = np.concatenate([
            np.asarray(obs["joint_position"], dtype=np.float32),
            np.asarray(obs["gripper_position"], dtype=np.float32)
        ]).astype(np.float32)
        self.robot_state.append(state7)

        # front-fill until buffer full
        while len(self.base_rgb) < self.n:
            self.base_rgb.appendleft(self.base_rgb[0].copy())
            self.wrist_rgb.appendleft(self.wrist_rgb[0].copy())
            self.robot_state.appendleft(self.robot_state[0].copy())

    def ready(self) -> bool:
        return len(self.base_rgb) == self.n

    def as_torch(self, device: torch.device) -> Dict[str, torch.Tensor]:
        base  = torch.from_numpy(np.stack(list(self.base_rgb), axis=0)).unsqueeze(0).to(device)   # (1,T,3,H,W)
        wrist = torch.from_numpy(np.stack(list(self.wrist_rgb), axis=0)).unsqueeze(0).to(device)  # (1,T,3,H,W)
        state = torch.from_numpy(np.stack(list(self.robot_state), axis=0)).unsqueeze(0).to(device) # (1,T,7)
        return {"base_rgb": base, "wrist_rgb": wrist, "robot_state": state}


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


class _DiffusionPolicyAdapter:
    """pi0-like interface: .infer(obs) -> {'actions': list}"""
    def __init__(self, policy: BaseImagePolicy, n_obs_steps: int, H: int = 224, W: int = 224):
        self.policy = policy
        self.device = next(policy.parameters()).device
        self.buf = _ObsBuffer(n_obs_steps=n_obs_steps, H=H, W=W)

    @torch.no_grad()
    def infer(self, obs: Dict) -> Dict:
        self.buf.push_single(obs)
        if not self.buf.ready():
            return {"actions": []}  # warm-up
        model_obs = self.buf.as_torch(self.device)
        result = self.policy.predict_action(model_obs)
        actions = result["action"][0].detach().to("cpu").numpy().tolist()
        return {"actions": actions}


class WebsocketPolicyServer:
    def __init__(self, policy: BaseImagePolicy, n_obs_steps: int, host="127.0.0.1", port=8765, metadata: Optional[dict] = None):
        self._policy = policy
        self._n_obs_steps = n_obs_steps
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        print(f"[SERVER] Listening on ws://{self._host}:{self._port}")
        async with websockets.asyncio.server.serve(
            self._handler, self._host, self._port,
            compression=None, max_size=None
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))

        # Adapter/buffer PER CONNECTION
        adapter = _DiffusionPolicyAdapter(self._policy, n_obs_steps=self._n_obs_steps, H=224, W=224)

        while True:
            try:
                obs = msgpack_numpy.unpackb(await websocket.recv())
                action = adapter.infer(obs)
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR,
                                      reason="Internal server error. Traceback included in previous frame.")
                raise


def main(ckpt: str, host: str = "0.0.0.0", port: int = 8000, image_size: int = 224):
    policy, cfg, n_obs_steps = _load_diffusion_policy(ckpt)
    print(f"[SERVER] Loaded policy on {next(policy.parameters()).device}; n_obs_steps={n_obs_steps}")
    meta = {"model": getattr(cfg, "name", "diffusion"), "n_obs_steps": n_obs_steps, "image_size": image_size}
    WebsocketPolicyServer(policy, n_obs_steps=n_obs_steps, host=host, port=port, metadata=meta).serve_forever()


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
        args = p.parse_args()
        main(**vars(args))
