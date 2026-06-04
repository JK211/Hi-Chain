"""
Compare total storage of dynamic RS encoding under different Zipf exponents.

8000 blocks, 50-node swarm. Storage model matches compare_of_storage_cost_fig6-9.DynamicEncoding:
  per block fragment_size = BLOCK_SIZE_MB / k, cluster cost = fragment_size * n_nodes.

Zipf exponent s (weights 1/rank^s) — typical ranges in literature (scenario-dependent):
  - ~0.6–1.2: web, P2P, some CDN/cache workloads
  - ~1.0–1.5: sharper hotspots (video/CDN)
  - s < 0.7: flatter, near-uniform; s > 1.5: extreme long tail
  Sensitivity sweeps use ZIPF_LIST below.
"""
import os
import sys
from collections import Counter
from typing import Dict, Tuple

import numpy as np

# Align with compare_of_storage_cost_fig6-9.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_generator003 import BlockDatasetGenerator  # noqa: E402

NUM_BLOCKS = 8000
N_NODES = 50
BLOCK_SIZE_MB = 1.0
# Sweep from flat to heavy tail; adjust for paper scenario
ZIPF_LIST = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8)
RNG_SEED = 42


def dynamic_total_storage_mb(
    frequencies, generator: BlockDatasetGenerator
) -> Tuple[float, np.ndarray]:
    """Accumulate total storage (MB) per DynamicEncoding.store_block logic."""
    ks = np.empty(len(frequencies), dtype=np.int64)
    total = 0.0
    for i, f in enumerate(frequencies):
        k = int(generator.calculate_encoding_k(float(f)))
        k = max(1, min(N_NODES, k))
        ks[i] = k
        fragment = BLOCK_SIZE_MB / k
        total += fragment * N_NODES
    return total, ks


def k_value_counts(ks: np.ndarray) -> Dict[int, int]:
    """Count blocks per k value."""
    return {int(k): int(c) for k, c in Counter(ks.tolist()).items()}


def top_blocks_traffic_share_pct(
    frequencies: np.ndarray, top_fraction: float = 0.1
) -> float:
    """
    Share of total access (0–100) from top top_fraction fraction of blocks by frequency.

    Total access ≈ sum of frequencies; consistent with data_generator003 top_1pct / top_10pct.
    """
    f = np.asarray(frequencies, dtype=np.float64)
    total = float(f.sum())
    if total <= 0:
        return 0.0
    n_top = max(1, len(f) // int(round(1 / top_fraction)))
    order = np.argsort(-f)
    return float(f[order[:n_top]].sum() / total * 100.0)


def print_k_distribution(rows):
    """Print block counts per k for each Zipf s; k × zipf summary table."""
    all_ks = sorted({k for r in rows for k in r["k_counts"]})
    zipf_labels = [f"{r['zipf_s']:.2f}" for r in rows]

    print("\nBlock count per k for each Zipf s:")
    print("-" * 48)
    for r in rows:
        parts = ", ".join(
            f"k={k}:{r['k_counts'][k]}" for k in sorted(r["k_counts"])
        )
        print(f"  zipf_s={r['zipf_s']:.2f}  ({parts})")

    col_w = max(8, max(len(z) for z in zipf_labels) + 2)
    header = f"{'k':>4}" + "".join(f"{z:>{col_w}}" for z in zipf_labels)
    print(f"\nSummary (rows=k, cols=zipf_s, cells=block count):")
    print(header)
    print("-" * len(header))
    for k in all_ks:
        line = f"{k:>4}"
        for r in rows:
            line += f"{r['k_counts'].get(k, 0):>{col_w}}"
        print(line)


def main():
    np.random.seed(RNG_SEED)

    rows = []
    for zipf_s in ZIPF_LIST:
        gen = BlockDatasetGenerator(
            num_blocks=NUM_BLOCKS,
            n_nodes=N_NODES,
            zipf_s=zipf_s,
        )
        freqs = gen.get_block_frequencies()
        total_mb, ks = dynamic_total_storage_mb(freqs, gen)
        top1_traffic_pct = top_blocks_traffic_share_pct(freqs, top_fraction=0.01)
        top10_traffic_pct = top_blocks_traffic_share_pct(freqs, top_fraction=0.1)
        rows.append(
            {
                "zipf_s": zipf_s,
                "total_storage_mb": total_mb,
                "mean_k": float(ks.mean()),
                "min_k": int(ks.min()),
                "max_k": int(ks.max()),
                "k_counts": k_value_counts(ks),
                "top1_traffic_pct": top1_traffic_pct,
                "top10_traffic_pct": top10_traffic_pct,
                "stats": gen.get_frequency_distribution(),
            }
        )

    print(
        f"Setup: blocks={NUM_BLOCKS}, nodes={N_NODES}, block_size={BLOCK_SIZE_MB} MB, "
        f"RNG_SEED={RNG_SEED}\n"
    )
    print(
        f"{'zipf_s':>8}  {'total_MB':>12}  {'mean_k':>8}  {'k_range':>12}  "
        f"{'top1%':>11}  {'top10%':>12}"
    )
    print("-" * 75)
    for r in rows:
        kr = f"{r['min_k']}–{r['max_k']}"
        print(
            f"{r['zipf_s']:>8.2f}  {r['total_storage_mb']:>12.2f}  {r['mean_k']:>8.4f}  "
            f"{kr:>12}  {r['top1_traffic_pct']:>10.2f}%  {r['top10_traffic_pct']:>11.2f}%"
        )

    print(
        "\nNote: top1% / top10% = traffic share of highest-frequency 1% / 10% blocks "
        f"({max(1, NUM_BLOCKS // 100)} / {max(1, NUM_BLOCKS // 10)} blocks) "
        "over sum of all block frequencies"
    )

    print_k_distribution(rows)

    # Optional bar chart (requires matplotlib)
    try:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("viridis")
        labels = [f"{r['zipf_s']:.2f}" for r in rows]
        totals = [r["total_storage_mb"] for r in rows]
        n_bars = len(totals)
        colors = cmap(np.linspace(0.15, 0.9, n_bars))
        fig, ax = plt.subplots(figsize=(max(8.0, 0.9 * n_bars), 4.2))
        ax.bar(labels, totals, color=colors, edgecolor="0.35", linewidth=0.4)
        ax.set_xlabel("Zipf exponent s")
        ax.set_ylabel("Total storage (MB)")
        ax.set_title("Dynamic encoding: total storage vs Zipf exponent\n"
                     f"{NUM_BLOCKS} blocks, {N_NODES} nodes, block={BLOCK_SIZE_MB} MB")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=0)
        out_path = os.path.join(SCRIPT_DIR, "zipf_dynamic_storage_compare.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\nBar chart saved: {out_path}")
    except ImportError:
        print("\nmatplotlib not installed; skipping plot.")


if __name__ == "__main__":
    main()
