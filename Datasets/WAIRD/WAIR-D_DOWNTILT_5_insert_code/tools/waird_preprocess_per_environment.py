#!/usr/bin/env python3
"""Offline preprocessing: WAIR-D 00001..00300 -> per-environment, per-BS text files.

Output:
  <out>/<env_id>/bs_00.txt ... bs_04.txt

The cluster C++/MPI run never imports Python or reads .npy files.
"""
import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

NUM_BS = 5
NUM_UE = 30


def load_dict(path: Path) -> Dict:
    x = np.load(path, allow_pickle=True)
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        return x.item()
    raise ValueError(f"Expected dictionary-style npy: {path}")


def find_envs(root: Path, start: int, count: int) -> List[Path]:
    envs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "Info.npy").exists())
    if start > 0:
        envs = [p for p in envs if int(p.name) >= start]
    return envs[:count] if count > 0 else envs


def load_geometry(info_path: Path, bs_height: float, ue_height: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    info = np.load(info_path, allow_pickle=True)
    expected = 2 + NUM_BS * NUM_UE * 4
    if info.ndim != 1 or info.size < expected:
        raise ValueError(f"Unsupported Info.npy shape {info.shape} at {info_path}")
    table = info[2:expected].reshape(NUM_BS * NUM_UE, 4)
    bs_xy = np.stack([table[b * NUM_UE, 0:2] for b in range(NUM_BS)])
    ue_xy = np.stack([table[u, 2:4] for u in range(NUM_UE)])
    phi = np.zeros((NUM_BS, NUM_UE), dtype=float)
    for b in range(NUM_BS):
        for u in range(NUM_UE):
            d = max(float(np.linalg.norm(bs_xy[b] - ue_xy[u])), 1e-6)
            phi[b, u] = math.degrees(math.atan2(bs_height - ue_height, d))
    return bs_xy, ue_xy, phi


def build_gains(env: Path, carrier: str, path_name: str, k_count: int,
                bandwidth_hz: float, delay_scale: float, mismatch_policy: str) -> np.ndarray:
    h_dict = load_dict(env / carrier)
    p_dict = load_dict(env / path_name)
    freqs = (np.arange(k_count) - (k_count - 1) / 2.0) * (bandwidth_hz / k_count)
    gains = np.zeros((NUM_BS, NUM_UE, k_count), dtype=float)
    for b in range(NUM_BS):
        for u in range(NUM_UE):
            key = f"bs{b}_ue{u}"
            resp = np.asarray(h_dict[key], dtype=np.complex128).reshape(-1)
            tau = np.asarray(p_dict[key]["taud"], dtype=float).reshape(-1) * delay_scale
            if resp.size != tau.size:
                if mismatch_policy == "error":
                    raise ValueError(f"{env.name}/{key}: H paths={resp.size}, delays={tau.size}; files may not match")
                n = min(resp.size, tau.size)
                resp, tau = resp[:n], tau[:n]
            if resp.size:
                h = np.sum(resp[:, None] * np.exp(-2j * np.pi * tau[:, None] * freqs[None, :]), axis=0)
                gains[b, u] = np.abs(h) ** 2
    return gains


def preprocess_env(env: Path, out_root: Path, args) -> None:
    _, _, phi = load_geometry(env / "Info.npy", args.bs_height, args.ue_height)
    all_band_gains = []
    for carrier in [x.strip() for x in args.carrier_files.split(',') if x.strip()]:
        all_band_gains.append(build_gains(env, carrier, args.path_file, args.subcarriers,
                                          args.bandwidth_mhz * 1e6,
                                          {"s":1.0,"ms":1e-3,"us":1e-6,"ns":1e-9,"ps":1e-12}[args.delay_unit],
                                          args.mismatch_policy))
    gains = np.concatenate(all_band_gains, axis=2)
    mean_gain = gains.mean(axis=2)
    serving = np.argmax(mean_gain, axis=0) if args.association == "strongest" else np.argmax(phi, axis=0)

    positive = gains[gains > 0]
    if args.gain_normalization == "median" and positive.size:
        scale = float(np.median(positive))
    elif args.gain_normalization == "p90" and positive.size:
        scale = float(np.percentile(positive, 90))
    elif args.gain_normalization == "mean" and positive.size:
        scale = float(np.mean(positive))
    else:
        scale = 1.0
    scale = max(scale, 1e-300)
    gains = gains / scale

    out_dir = out_root / env.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for b in range(NUM_BS):
        users = [u for u in range(NUM_UE) if int(serving[u]) == b]
        with (out_dir / f"bs_{b:02d}.txt").open("w", encoding="utf-8") as f:
            f.write(f"# environment={env.name} local_bs={b} gain_scale={scale:.12e}\n")
            f.write("# ue subcarrier serving_bs phi0 phi1 phi2 phi3 phi4 gain0 gain1 gain2 gain3 gain4\n")
            for u in users:
                for k in range(gains.shape[2]):
                    values = [str(u), str(k), str(int(serving[u]))]
                    values += [f"{phi[j,u]:.8f}" for j in range(NUM_BS)]
                    values += [f"{gains[j,u,k]:.12e}" for j in range(NUM_BS)]
                    f.write(" ".join(values) + "\n")
    with (out_dir / "meta.txt").open("w", encoding="utf-8") as f:
        f.write(f"environment={env.name}\n")
        f.write(f"carrier_files={args.carrier_files}\n")
        f.write(f"subcarriers_per_band={args.subcarriers}\n")
        f.write(f"total_subcarriers={gains.shape[2]}\n")
        f.write(f"association={args.association}\n")
        f.write(f"gain_scale={scale:.12e}\n")
        f.write("users_per_bs=" + ",".join(str(int(np.sum(serving == b))) for b in range(NUM_BS)) + "\n")
    print(f"{env.name}: users_per_bs={[int(np.sum(serving == b)) for b in range(NUM_BS)]}, rows={NUM_UE*gains.shape[2]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="WAIRD", help="folder containing 00001..00300")
    ap.add_argument("--out", default="Benchmarks/data/WAIRD_DOWNTILT_5")
    ap.add_argument("--carrier-files", default="H_28_0_G.npy")
    ap.add_argument("--path-file", default="Path.npy")
    ap.add_argument("--start-env", type=int, default=1)
    ap.add_argument("--num-envs", type=int, default=300)
    ap.add_argument("--subcarriers", type=int, default=64)
    ap.add_argument("--bandwidth-mhz", type=float, default=46.08)
    ap.add_argument("--delay-unit", choices=["s","ms","us","ns","ps"], default="ns")
    ap.add_argument("--bs-height", type=float, default=6.0)
    ap.add_argument("--ue-height", type=float, default=1.5)
    ap.add_argument("--association", choices=["strongest","nearest"], default="strongest")
    ap.add_argument("--gain-normalization", choices=["median","p90","mean","none"], default="median")
    ap.add_argument("--mismatch-policy", choices=["error","min"], default="error",
                    help="use min only to diagnose mismatched H/Path files")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    envs = find_envs(root, args.start_env, args.num_envs)
    if not envs:
        raise SystemExit(f"No WAIR-D environment folders found under {root}")
    for env in envs:
        preprocess_env(env, out, args)
    print(f"Done: {len(envs)} environments -> {out}")

if __name__ == "__main__":
    main()
