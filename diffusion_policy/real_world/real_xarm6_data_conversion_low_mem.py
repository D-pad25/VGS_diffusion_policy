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
from typing import Callable
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
    Streams PKL episodes → zarr ReplayBuffer with minimal RAM.
    Layout:
      data/
        robot_state (T,7) f32
        action      (T,7) f32
        timestamp   (T,)  f64
        wrist_rgb   (T,H,W,3) u8
        base_rgb    (T,H,W,3) u8
      meta/
        episode_lengths (E,) i64
        episode_ends    (E,) i64
    """
    # ---------- Defaults ----------
    cv2.setNumThreads(1)

    input_root = pathlib.Path(os.path.expanduser(dataset_path)).resolve()
    assert input_root.is_dir(), f"dataset_path not found: {input_root}"

    if out_store is None:
        out_store = zarr.DirectoryStore(str(input_root / "_inline_build.zarr"))

    if image_compressor is None:
        image_compressor = Jpeg2k(level=50)
    if lowdim_compressor is None:
        lowdim_compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    if image_keys is None:
        image_keys = ['wrist_rgb', 'base_rgb']

    if n_encoding_threads is None:
        n_encoding_threads = min(2, multiprocessing.cpu_count())
    else:
        n_encoding_threads = max(1, min(n_encoding_threads, 4))
    max_inflight_tasks = min(max_inflight_tasks, 8)

    base_lowdim_keys = ['robot_state', 'action', 'timestamp', 'episode_ends', 'episode_lengths']
    if lowdim_keys is None:
        lowdim_keys = base_lowdim_keys
    else:
        for k in base_lowdim_keys:
            if k not in lowdim_keys:
                lowdim_keys = list(lowdim_keys) + [k]

    ep_dir_re = re.compile(r'episode[_\- ]?(\d+)$', re.IGNORECASE)
    step_re   = re.compile(r'step[_\- ]?(\d+)\.pkl$', re.IGNORECASE)
    def ep_sort_key(p: pathlib.Path):
        m = ep_dir_re.search(p.name)
        return int(m.group(1)) if m else p.name

    episode_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()], key=ep_sort_key)
    assert len(episode_dirs) > 0, f"No episode dirs found under {input_root}"

    def step_sort_key(p: pathlib.Path):
        m = step_re.search(p.name)
        return int(m.group(1)) if m else p.name

    # PASS A
    episode_lengths = []
    for ep in episode_dirs:
        step_files = sorted([p for p in ep.iterdir() if p.is_file() and p.suffix=='.pkl'], key=step_sort_key)
        assert len(step_files) > 0, f"No step .pkl files in {ep}"
        episode_lengths.append(len(step_files))
    episode_lengths = np.asarray(episode_lengths, dtype=np.int64)
    episode_ends = np.cumsum(episode_lengths, dtype=np.int64)
    T = int(episode_ends[-1])

    # set up tmp shapes/chunks
    tmp_store = zarr.MemoryStore()
    tmp_root = zarr.group(store=tmp_store)
    tmp_data = tmp_root.create_group("data")
    tmp_meta = tmp_root.create_group("meta")

    lowdim_chunk_rows = min(8192, max(1, T))
    tmp_data.create_dataset('robot_state', shape=(T, 7), chunks=(lowdim_chunk_rows, 7),
                            dtype=np.float32, compressor=None)
    tmp_data.create_dataset('action',      shape=(T, 7), chunks=(lowdim_chunk_rows, 7),
                            dtype=np.float32, compressor=None)
    tmp_data.create_dataset('timestamp',   shape=(T,),   chunks=(lowdim_chunk_rows,),
                            dtype=np.float64, compressor=None)
    tmp_meta.create_dataset('episode_lengths', data=episode_lengths,
                            chunks=episode_lengths.shape, dtype=np.int64, compressor=None)
    tmp_meta.create_dataset('episode_ends', data=episode_ends,
                            chunks=episode_ends.shape, dtype=np.int64, compressor=None)

    chunks_map = {
        'robot_state': (lowdim_chunk_rows, 7),
        'action':      (lowdim_chunk_rows, 7),
        'timestamp':   (lowdim_chunk_rows,)
    }
    compressor_map = {k: lowdim_compressor for k in chunks_map.keys()}

    out_replay_buffer = ReplayBuffer.copy_from_store(
        src_store=tmp_store,
        store=out_store,
        keys=['robot_state', 'action', 'timestamp'],
        chunks=chunks_map,
        compressors=compressor_map
    )

    # Ensure meta in output
    root_out = zarr.group(store=out_store)
    meta_out = root_out.require_group("meta")
    data_out = root_out['data']
    if 'episode_lengths' not in meta_out:
        meta_out.require_dataset('episode_lengths', data=episode_lengths,
                                 chunks=episode_lengths.shape, dtype=np.int64, compressor=None)
    if 'episode_ends' not in meta_out:
        meta_out.require_dataset('episode_ends', data=episode_ends,
                                 chunks=episode_ends.shape, dtype=np.int64, compressor=None)

    # allocate images
    def get_out_res(arr_name: str, in_shape_hw: Tuple[int,int]) -> Tuple[int,int]:
        if isinstance(out_resolutions, dict):
            if arr_name in out_resolutions:
                return tuple(out_resolutions[arr_name])
            return tuple(next(iter(out_resolutions.values())))
        elif out_resolutions is None:
            return (in_shape_hw[1], in_shape_hw[0])   # (W,H)
        else:
            return tuple(out_resolutions)

    first_sf = sorted([p for p in episode_dirs[0].iterdir() if p.suffix=='.pkl'], key=step_sort_key)[0]
    with open(first_sf, 'rb') as f:
        first = pickle.load(f)

    for k in image_keys:
        assert k in first, f"Missing image key '{k}' in step pkl"
        img0 = first[k]
        assert img0.ndim == 3 and img0.shape[-1] == 3 and img0.dtype == np.uint8, \
            f"Key '{k}' must be HxWx3 uint8, got {img0.shape} {img0.dtype}"
        in_h, in_w = int(img0.shape[0]), int(img0.shape[1])
        out_w, out_h = get_out_res(k, (in_h, in_w))
        out_replay_buffer.data.require_dataset(
            name=k,
            shape=(T, out_h, out_w, 3),
            chunks=(1, out_h, out_w, 3),
            compressor=image_compressor,
            dtype=np.uint8
        )

    # PASS B
    img_tf_cache: Dict[Tuple[str,int,int,int,int], Callable[[np.ndarray], np.ndarray]] = {}
    def get_tf(k, in_w, in_h, out_w, out_h):
        key = (k, in_w, in_h, out_w, out_h)
        if key not in img_tf_cache:
            img_tf_cache[key] = get_image_transform(
                input_res=(in_w, in_h), output_res=(out_w, out_h), bgr_to_rgb=True  # <- set True if source is BGR
            )
        return img_tf_cache[key]

    robot_state_arr = data_out['robot_state']
    action_arr      = data_out['action']
    timestamp_arr   = data_out['timestamp']
    img_arrays      = {k: data_out[k] for k in image_keys}

    gidx = 0
    with tqdm(total=T, desc="Writing low-dim", mininterval=0.5) as pbar_low:
        for ep, n in zip(episode_dirs, episode_lengths):
            step_files = sorted([p for p in ep.iterdir() if p.is_file() and p.suffix=='.pkl'], key=step_sort_key)
            for sf in step_files:
                with open(sf, 'rb') as f:
                    d = pickle.load(f)
                robot_state_arr[gidx] = np.asarray(d['joint_positions'], dtype=np.float32)
                action_arr[gidx]      = np.asarray(d['control'], dtype=np.float32)
                timestamp_arr[gidx]   = float(gidx)
                gidx += 1
                pbar_low.update(1)

    def put_img(zarr_arr, zarr_idx, img):
        try:
            zarr_arr[zarr_idx] = img
            if verify_read:
                _ = zarr_arr[zarr_idx]
            return True
        except Exception:
            return False

    with tqdm(total=T*len(image_keys), desc="Encoding images", mininterval=1.0) as pbar_img:
        futures = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_encoding_threads) as ex:
            gidx = 0
            for ep, n in zip(episode_dirs, episode_lengths):
                step_files = sorted([p for p in ep.iterdir() if p.is_file() and p.suffix=='.pkl'], key=step_sort_key)
                for sf in step_files:
                    with open(sf, 'rb') as f:
                        d = pickle.load(f)
                    for k, arr in img_arrays.items():
                        frame = d[k]
                        in_h, in_w = int(frame.shape[0]), int(frame.shape[1])
                        out_h, out_w = arr.shape[1], arr.shape[2]
                        tf = get_tf(k, in_w, in_h, out_w, out_h)
                        img_out = tf(frame)
                        if len(futures) >= max_inflight_tasks:
                            done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                            for f in done:
                                if not f.result():
                                    raise RuntimeError("Failed to encode image!")
                            pbar_img.update(len(done))
                        futures.add(ex.submit(put_img, arr, gidx, img_out))
                    gidx += 1
            done, futures = concurrent.futures.wait(futures)
            for f in done:
                if not f.result():
                    raise RuntimeError("Failed to encode image!")
            pbar_img.update(len(done))

    return out_replay_buffer