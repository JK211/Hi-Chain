"""
Export BDQN cache-replacement actions and access trace for NS-3 replay.

NS-3 cannot run PyTorch in-process easily. Workflow:
  1. Train BDQN in Python (same n_nodes as NS-3 run).
  2. Run this script to export greedy-policy actions on cache miss.
  3. Pass CSVs to NS-3:
       --accessTrace=ns3_access_trace.csv
       --rlActions=ns3_rl_actions.csv
       --cachePolicy=rl

Usage:
  cd HiChain_Code
  python train_model007.py   # or use existing checkpoint
  python export_ns3_rl_trace.py --n-nodes 50 --checkpoint results/train/bdqn_policy.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

from BDQN import BDQN
from CacheReplacementEnv import CacheReplacementEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BLOCK_CSV = os.path.join(SCRIPT_DIR, "datasets", "block_frequencies.csv")
DEFAULT_ACCESS_CSV = os.path.join(
    SCRIPT_DIR, "datasets", "block_access_10000blocks_100000steps_phased_mission.csv"
)
OUT_DIR = os.path.join(SCRIPT_DIR, "datasets", "ns3_export")


def export_trace(
    n_nodes: int,
    block_csv: str,
    access_csv: str,
    checkpoint: str | None,
    max_steps: int,
    block_size_mb: int,
    cache_mb: int,
    storage_mb: int,
    output_dir: str,
) -> None:
    block_size = block_size_mb * 1024 * 1024
    cache_per_node = cache_mb * 1024 * 1024
    storage_per_node = storage_mb * 1024 * 1024

    if not os.path.isfile(block_csv):
        raise FileNotFoundError(f"Missing {block_csv}; run data_generator003.py first.")

    if not os.path.isfile(access_csv):
        raise FileNotFoundError(
            f"Missing {access_csv}; run data_generator_mission_phases.py or provide access CSV."
        )

    env = CacheReplacementEnv(
        block_csv=block_csv,
        access_csv=access_csv,
        n_nodes=n_nodes,
        block_size=block_size,
        cache_per_node=cache_per_node,
        storage_per_node=storage_per_node,
    )

    input_dim = env.observation_space.shape[0]
    cache_slots = cache_per_node // block_size
    total_blocks = len(env.network.blocks)

    policy = BDQN(input_dim, cache_slots, total_blocks)
    if checkpoint and os.path.isfile(checkpoint):
        policy.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        policy.eval()
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("No checkpoint — using random-init BDQN (export for pipeline test only).")

    os.makedirs(output_dir, exist_ok=True)
    access_rows = []
    rl_rows = []
    metrics_rows = []

    state = env.reset()
    done = False
    step = 0
    hits = 0
    lat_sum = 0.0

    block_id_list = sorted(env.network.blocks.keys())

    while not done and (max_steps <= 0 or step < max_steps):
        block_id = int(env.access_df.iloc[env.current_step])
        access_rows.append({"step": step, "block_id": block_id})

        with torch.no_grad():
            st = torch.FloatTensor(state).unsqueeze(0)
            q_remove, q_add = policy(st)
            evict_slot = int(torch.argmax(q_remove).item())
            add_idx = int(torch.argmax(q_add).item())
        add_block_id = block_id_list[add_idx % len(block_id_list)]

        # Record action before step (applied on miss inside env)
        rl_rows.append(
            {
                "step": step,
                "evict_slot": evict_slot,
                "add_block_id": add_block_id,
            }
        )

        next_state, reward, done, _ = env.step([evict_slot, add_idx])
        if env.hit_types and env.hit_types[-1] in (1, 2):
            hits += 1
        if env.access_latencies:
            lat_sum += env.access_latencies[-1]

        state = next_state
        step += 1

    access_path = os.path.join(output_dir, "ns3_access_trace.csv")
    rl_path = os.path.join(output_dir, "ns3_rl_actions.csv")
    pd.DataFrame(access_rows).to_csv(access_path, index=False)
    pd.DataFrame(rl_rows).to_csv(rl_path, index=False)

    hit_rate = hits / max(step, 1)
    avg_lat = lat_sum / max(step, 1)
    print(f"Exported {step} steps")
    print(f"  access trace -> {access_path}")
    print(f"  RL actions   -> {rl_path}")
    print(f"Python greedy eval: hit_rate={hit_rate:.4f} avg_latency_ms={avg_lat:.2f}")
    print(f"\nNS-3 example (from ns-3.25 root, after copying scratch .cc):")
    print(
        f'  ./waf --run "scratch/hi-chain-uav-swarm-validation '
        f"--nUavs={n_nodes} --cachePolicy=rl "
        f'--accessTrace={access_path} --rlActions={rl_path}"'
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Export NS-3 RL action trace from BDQN")
    p.add_argument("--n-nodes", type=int, default=50, help="Must match NS-3 --nUavs")
    p.add_argument("--block-csv", default=DEFAULT_BLOCK_CSV)
    p.add_argument("--access-csv", default=DEFAULT_ACCESS_CSV)
    p.add_argument("--checkpoint", default=os.path.join(SCRIPT_DIR, "results", "train", "bdqn_policy.pt"))
    p.add_argument("--max-steps", type=int, default=0, help="0 = full trace")
    p.add_argument("--block-size-mb", type=int, default=1)
    p.add_argument("--cache-mb", type=int, default=20)
    p.add_argument("--storage-mb", type=int, default=1000)
    p.add_argument("--output-dir", default=OUT_DIR)
    args = p.parse_args()

    export_trace(
        n_nodes=args.n_nodes,
        block_csv=args.block_csv,
        access_csv=args.access_csv,
        checkpoint=args.checkpoint if args.checkpoint else None,
        max_steps=args.max_steps,
        block_size_mb=args.block_size_mb,
        cache_mb=args.cache_mb,
        storage_mb=args.storage_mb,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
