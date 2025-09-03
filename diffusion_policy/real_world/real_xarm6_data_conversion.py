from typing import Sequence, Dict, Optional, Union, Tuple
import os, re, pickle, pathlib, multiprocessing, concurrent.futures
import numpy as np
import zarr
import numcodecs
from numcodecs import Blosc
from tqdm import tqdm

import cv2
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
register_codecs()

def real_data_to_replay_buffer(
        dataset_path: str,
        out_store: Optional[zarr.ABSStore]=None,
        out_resolutions: Union[None, Tuple[int,int], Dict[str,Tuple[int,int]]]=(224, 224),
        lowdim_keys: Optional[Sequence[str]]=None,
        image_keys: Optional[Sequence[str]]=None,
        lowdim_compressor: Optional[numcodecs.abc.Codec]=None,
        image_compressor: Optional[numcodecs.abc.Codec]=None,
        n_decoding_threads: int=multiprocessing.cpu_count(),
        n_encoding_threads: int=multiprocessing.cpu_count(),
        max_inflight_tasks: int=multiprocessing.cpu_count()*5,
        verify_read: bool=True
    ) -> ReplayBuffer:
    """
    Adapted for your dataset layout:
    parent/
      Episode1/
        step1.pkl
        step2.pkl
        ...
      Episode2/
        ...
    Each step.pkl contains keys including:
      'wrist_rgb' (H,W,3) uint8
      'base_rgb'  (H,W,3) uint8
      'joint_positions' (7,) float64  # 6 joints + gripper
      'control' (7,) float64          # action (6 + gripper)

    Writes a zarr replay buffer with arrays:
      'wrist_rgb', 'base_rgb' (T,H,W,3) uint8
      'robot_state' (T,7) float32
      'action' (T,7) float32
      'episode_ends' (E,) int64
      'episode_lengths' (E,) int64
      'timestamp' (T,) float64
    """
    # ---------- Defaults ----------
    if out_store is None:
        out_store = zarr.MemoryStore()
    if image_compressor is None:
        image_compressor = Jpeg2k(level=50)
    if lowdim_compressor is None:
        lowdim_compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    if image_keys is None:
        image_keys = ['wrist_rgb', 'base_rgb']
    # We'll always include these low-dim/meta keys; add extras via lowdim_keys if you want
    base_lowdim_keys = ['robot_state', 'action', 'timestamp', 'episode_ends', 'episode_lengths']
    if lowdim_keys is None:
        lowdim_keys = base_lowdim_keys
    else:
        # ensure required meta keys are present
        for k in base_lowdim_keys:
            if k not in lowdim_keys:
                lowdim_keys = list(lowdim_keys) + [k]

    input_root = pathlib.Path(os.path.expanduser(dataset_path)).resolve()
    assert input_root.is_dir(), f"dataset_path not found: {input_root}"

    # ---------- Enumerate episodes/steps ----------
    ep_dir_re = re.compile(r'episode[_\- ]?(\d+)$', re.IGNORECASE)
    def ep_sort_key(p: pathlib.Path):
        m = ep_dir_re.search(p.name)
        return int(m.group(1)) if m else p.name

    episode_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()],
                          key=ep_sort_key)
    assert len(episode_dirs) > 0, f"No episode dirs found under {input_root}"

    step_re = re.compile(r'step[_\- ]?(\d+)\.pkl$', re.IGNORECASE)
    def step_sort_key(p: pathlib.Path):
        m = step_re.search(p.name)
        return int(m.group(1)) if m else p.name

    # ---------- Pass 1: collect low-dim + meta ----------
    robot_state = []
    actions = []
    episode_lengths = []
    timestamps = []

    t = 0
    for ep in episode_dirs:
        step_files = sorted([p for p in ep.iterdir() if p.is_file() and p.suffix=='.pkl'],
                            key=step_sort_key)
        assert len(step_files) > 0, f"No step .pkl files in {ep}"
        for sf in step_files:
            with open(sf, 'rb') as f:
                d = pickle.load(f)
            # required keys
            qpos = np.asarray(d['joint_positions'], dtype=np.float32)   # (7,)
            act  = np.asarray(d['control'], dtype=np.float32)           # (7,)
            robot_state.append(qpos)
            actions.append(act)
            timestamps.append(float(t))  # simple monotonic timestamp; adjust if you have real times
            t += 1
        episode_lengths.append(len(step_files))

    robot_state = np.stack(robot_state, axis=0)         # (T,7) f32
    actions     = np.stack(actions, axis=0)             # (T,7) f32
    timestamps  = np.asarray(timestamps, dtype=np.float64)
    episode_lengths = np.asarray(episode_lengths, dtype=np.int64)
    episode_ends = np.cumsum(episode_lengths, dtype=np.int64)  # (E,)

    T = robot_state.shape[0]
    E = episode_lengths.shape[0]
    assert actions.shape == (T, 7), f"Expected action (T,7), got {actions.shape}"
    assert robot_state.shape == (T, 7), f"Expected robot_state (T,7), got {robot_state.shape}"

    # ---------- Build a tiny temp store to leverage ReplayBuffer.copy_from_store ----------
    tmp_store = zarr.MemoryStore()
    tmp_root = zarr.group(store=tmp_store)
    tmp_root.create_dataset('robot_state', data=robot_state, chunks=robot_state.shape, compressor=None)
    tmp_root.create_dataset('action', data=actions, chunks=actions.shape, compressor=None)
    tmp_root.create_dataset('timestamp', data=timestamps, chunks=timestamps.shape, compressor=None)
    tmp_root.create_dataset('episode_lengths', data=episode_lengths, chunks=episode_lengths.shape, compressor=None)
    tmp_root.create_dataset('episode_ends', data=episode_ends, chunks=episode_ends.shape, compressor=None)

    # Save low-dim as single-chunk arrays (fast random access; matches original pattern)
    chunks_map = {k: tmp_root[k].shape for k in lowdim_keys if k in tmp_root}
    compressor_map = {k: lowdim_compressor for k in lowdim_keys if k in tmp_root}

    # Copy low-dim/meta into output store and get a ReplayBuffer handle
    out_replay_buffer = ReplayBuffer.copy_from_store(
        src_store=tmp_store,
        store=out_store,
        keys=[k for k in lowdim_keys if k in tmp_root],
        chunks=chunks_map,
        compressors=compressor_map
    )

    # ---------- Prepare/allocate image arrays ----------
    # Determine output resolutions per image key
    def get_out_res(arr_name: str, in_shape_hw: Tuple[int,int]) -> Tuple[int,int]:
        if isinstance(out_resolutions, dict):
            if arr_name in out_resolutions:
                return tuple(out_resolutions[arr_name])
            # fallback to a generic
            return tuple(next(iter(out_resolutions.values())))
        elif out_resolutions is None:
            # keep native resolution
            return (in_shape_hw[1], in_shape_hw[0])   # (W,H)
        else:
            return tuple(out_resolutions)

    # Dry read first image to infer in-res
    with open(sorted([p for p in episode_dirs[0].iterdir() if p.suffix=='.pkl'], key=step_sort_key)[0], 'rb') as f:
        first = pickle.load(f)

    # Create per-key image datasets
    for k in image_keys:
        assert k in first, f"Missing image key '{k}' in step pkl"
        img0 = first[k]
        assert img0.ndim == 3 and img0.shape[-1] == 3, f"Key '{k}' must be HxWx3 uint8, got {img0.shape} {img0.dtype}"
        in_h, in_w = int(img0.shape[0]), int(img0.shape[1])
        out_w, out_h = get_out_res(k, (in_h, in_w))
        # channel-last, chunk by timestep for better compression/IO
        _ = out_replay_buffer.data.require_dataset(
            name=k,
            shape=(T, out_h, out_w, 3),
            chunks=(1, out_h, out_w, 3),
            compressor=image_compressor,
            dtype=np.uint8
        )

    # ---------- Pass 2: write images with optional parallelism ----------
    # We’ll reuse a simple pool to write (global_idx, image_key, frame)
    def put_img(zarr_arr, zarr_idx, img):
        try:
            zarr_arr[zarr_idx] = img
            if verify_read:
                _ = zarr_arr[zarr_idx]
            return True
        except Exception:
            return False

    # Build per-key transforms once per output size
    img_tf_cache = {}  # (k, in_w,in_h,out_w,out_h) -> callable
    def get_tf(k, in_w, in_h, out_w, out_h):
        key = (k, in_w, in_h, out_w, out_h)
        if key not in img_tf_cache:
            img_tf_cache[key] = get_image_transform(
                input_res=(in_w, in_h), output_res=(out_w, out_h), bgr_to_rgb=False
            )
        return img_tf_cache[key]

    with tqdm(total=T*len(image_keys), desc="Encoding images", mininterval=1.0) as pbar:
        futures = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_encoding_threads) as ex:
            gidx = 0
            for ep_idx, ep in enumerate(episode_dirs):
                step_files = sorted([p for p in ep.iterdir() if p.is_file() and p.suffix=='.pkl'],
                                    key=step_sort_key)
                for sf in step_files:
                    with open(sf, 'rb') as f:
                        d = pickle.load(f)
                    for k in image_keys:
                        frame = d[k]
                        in_h, in_w = int(frame.shape[0]), int(frame.shape[1])
                        arr = out_replay_buffer[k]
                        out_h, out_w = arr.shape[1], arr.shape[2]
                        tf = get_tf(k, in_w, in_h, out_w, out_h)
                        img_out = tf(frame)  # uint8, HxWx3

                        if len(futures) >= max_inflight_tasks:
                            done, futures = concurrent.futures.wait(
                                futures, return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for f in done:
                                if not f.result():
                                    raise RuntimeError("Failed to encode image!")
                            pbar.update(len(done))

                        futures.add(ex.submit(put_img, arr, gidx, img_out))
                    gidx += 1

            done, futures = concurrent.futures.wait(futures)
            for f in done:
                if not f.result():
                    raise RuntimeError("Failed to encode image!")
            pbar.update(len(done))

    return out_replay_buffer
