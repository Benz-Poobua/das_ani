import os
import glob
import time
import numpy as np
import torch
import concurrent.futures
from pathlib import Path
from daspack import DASCoder, Quantizer

# --- Import your project modules ---
from src.utils import load_data, convert_to_tensor, load_config, get_cfg
from src.ani import preprocess, TorchCrossCorrelation
from src.error import rel_frobenius, max_abs_error

def process_file(args):
    """
    Worker function: Handles one file from disk to X-Corr.
    Throttles threads to ensure njobs=4 doesn't thrash the CPU.
    """
    target_file, cfg_params, comp_dir = args
    
    # 1. THREAD MANAGEMENT
    # Keep these to prevent PyTorch CPU ops from spawning uncontrolled threads
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    device = torch.device("cpu")
    
    # 2. INITIALIZE MODULES (Local to process for safety)
    try:
        cc_module = TorchCrossCorrelation(
            mode=cfg_params['mode'], 
            max_lag_samples=cfg_params['max_lag_samples'],
            v1_fft_snap_pow2=cfg_params['v1_fft_snap_pow2'],
            v1_fallback=cfg_params['v1_fallback']
        ).to(device)
        cc_module.eval()
    except Exception as e:
        return {"error": f"CC Init Failed: {str(e)}", "file": target_file}
    
    coder = DASCoder(threads=1)
    
    # 3. LOAD & COMPRESS (Ground Truth)
    target_basename = os.path.basename(target_file)
    comp_file = comp_dir / target_basename.replace(".npz", "_comp.npz")
    
    _, data_raw_orig, _, _, _ = load_data(target_file, mmap=False)
    data_raw_int = np.ascontiguousarray(data_raw_orig, dtype=np.int32)
    
    # Perform Lossless compression
    comp_bytes = coder.encode(data_raw_int, Quantizer.Lossless())
    np.savez(comp_file, compressed_das=np.frombuffer(comp_bytes, dtype=np.uint8), original_shape=data_raw_int.shape)
    
    raw_size = os.path.getsize(target_file)
    comp_size = os.path.getsize(comp_file)

    # Determine segment length based on mode
    npts_seg = cfg_params['npts_seg_v1'] if cfg_params['mode'] == 'v1' else cfg_params['npts_seg']
    num_targets = cfg_params['num_targets']

    # 4. PIPELINE 1: RAW
    t0 = time.perf_counter()
    data_proc = preprocess(
        data_raw_int, cfg_params['fs_raw'], cfg_params['f1'], cfg_params['f2'], 
        cfg_params['decimation'], diff=False, ram_win=cfg_params['ram_win']
    )
    tensor_raw = convert_to_tensor(data_proc, device=device)
    
    valid_len = (tensor_raw.shape[1] // npts_seg) * npts_seg
    d2 = tensor_raw[0:num_targets, :valid_len].contiguous().view(-1, npts_seg)
    d1 = tensor_raw[0:1, :valid_len].contiguous().view(-1, npts_seg).repeat(num_targets, 1)
    
    with torch.inference_mode():
        out_raw = cc_module(d1, d2)
    raw_time = time.perf_counter() - t0

    # 5. PIPELINE 2: COMPRESSED
    t0 = time.perf_counter()
    # Read compressed file back from disk to simulate actual production flow
    with np.load(comp_file) as ld:
        stream_array = ld["compressed_das"]
        orig_shape = tuple(ld["original_shape"])
        
    decoded_raw = coder.decode(stream_array.tobytes()).reshape(orig_shape)
    
    data_proc_comp = preprocess(
        decoded_raw, cfg_params['fs_raw'], cfg_params['f1'], cfg_params['f2'], 
        cfg_params['decimation'], diff=False, ram_win=cfg_params['ram_win']
    )
    tensor_comp = convert_to_tensor(data_proc_comp, device=device)
    
    d2_c = tensor_comp[0:num_targets, :valid_len].contiguous().view(-1, npts_seg)
    d1_c = tensor_comp[0:1, :valid_len].contiguous().view(-1, npts_seg).repeat(num_targets, 1)
    
    with torch.inference_mode():
        out_comp = cc_module(d1_c, d2_c)
    comp_time = time.perf_counter() - t0

    # 6. VERIFY
    err_max = max_abs_error(out_comp.numpy(), out_raw.numpy())
    
    return {
        "file": target_basename,
        "raw_time": raw_time,
        "comp_time": comp_time,
        "raw_size": raw_size,
        "comp_size": comp_size,
        "err_max": err_max
    }

def run_multiprocess_benchmark():
    config_path = "configs/cc.yaml"
    cfg = load_config(config_path)
    
    fs_raw = float(get_cfg(cfg, ["data", "fs_raw"], 250.0))
    decimation = int(get_cfg(cfg, ["preprocess", "decimation"], 1))
    fs_proc = fs_raw / decimation
    
    # Lag and Segment Settings
    max_lag_sec = float(get_cfg(cfg, ["xcorr", "max_lag_sec"], 2.0))
    max_lag_samples = int(round(max_lag_sec * fs_proc))
    
    cfg_params = {
        'fs_raw': fs_raw,
        'f1': float(get_cfg(cfg, ["preprocess", "f1"], 1.0)),
        'f2': float(get_cfg(cfg, ["preprocess", "f2"], 10.0)),
        'decimation': decimation,
        'ram_win': float(get_cfg(cfg, ["preprocess", "ram_win_sec"], 0.0)),
        'npts_seg': int(round(float(get_cfg(cfg, ["xcorr", "xcorr_seg_sec"], 60.0)) * fs_proc)),
        'npts_seg_v1': int(round(float(get_cfg(cfg, ["xcorr", "xcorr_seg_sec_v1"], 60.0)) * fs_proc)),
        'max_lag_samples': max_lag_samples,
        'mode': str(get_cfg(cfg, ["xcorr", "mode"], "v1")).lower(),
        'v1_fft_snap_pow2': bool(get_cfg(cfg, ["xcorr", "v1_fft_snap_pow2"], True)),
        'v1_fallback': str(get_cfg(cfg, ["xcorr", "v1_fallback"], "v1_2M")),
        'num_targets': 10
    }

    data_root = Path(get_cfg(cfg, ["paths", "data_root"], "./data/preprocessed")).expanduser().resolve()
    target_dir = data_root / "20210901"
    comp_dir = data_root / "20210901_compressed"
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    real_files = sorted(glob.glob(f"{target_dir}/*.npz"))[:4] 
    njobs = min(int(get_cfg(cfg, ["runtime", "njobs"], 4)), len(real_files))
    
    print(f"🚀 Launching v1 Benchmark | njobs={njobs} | mode={cfg_params['mode']} | lag={max_lag_sec}s")
    print("="*75)
    
    worker_args = [(f, cfg_params, comp_dir) for f in real_files]
    t_start_wall = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=njobs) as executor:
        results = list(executor.map(process_file, worker_args))

    wall_clock_time = time.perf_counter() - t_start_wall

    total_raw_time = 0.0
    total_comp_time = 0.0
    total_raw_bytes = 0
    total_comp_bytes = 0

    for res in results:
        if "error" in res:
            print(f"❌ {res['file']}: {res['error']}")
            continue
        print(f"📄 {res['file']} | Raw: {res['raw_time']:.3f}s | Comp: {res['comp_time']:.3f}s | MaxErr: {res['err_max']:.1e}")
        total_raw_time += res["raw_time"]
        total_comp_time += res["comp_time"]
        total_raw_bytes += res["raw_size"]
        total_comp_bytes += res["comp_size"]

    print("\n" + "="*75)
    print("🏆 MULTIPROCESSING VERDICT")
    print("="*75)
    print(f"Wall-Clock Time ({len(results)} files in parallel): {wall_clock_time:.4f} sec")
    print(f"Total CPU Time (Raw Pipeline):         {total_raw_time:.4f} sec")
    print(f"Total CPU Time (Comp Pipeline):        {total_comp_time:.4f} sec")
    print(f"Global Compression Ratio:              {(total_raw_bytes/total_comp_bytes):.2f}x smaller")
    print("-" * 75)
    
    if total_comp_time < total_raw_time:
        print(f"✅ WIN: DASPack pipeline used {(total_raw_time - total_comp_time):.4f} sec LESS total CPU time!")
    else:
        print(f"❌ LOSS: Raw pipeline used {(total_comp_time - total_raw_time):.4f} sec LESS total CPU time.")
    print("="*75)

if __name__ == "__main__":
    run_multiprocess_benchmark()