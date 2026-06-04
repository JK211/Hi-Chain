# data_generator_mission_phases.py
#
# Reviewer request: access sequence should show "severe popularity shifts over mission phases".
#
# block_frequencies.csv assigns a fixed nominal frequency per block (e.g. id=7502 → 1000).
# This script:
#   Phase 0: weighted sampling over all blocks using CSV block_id→frequency as-is.
#   Phases 1, 2: permute the multiset of frequency values onto block ids (same values,
#   different hotspot locations), then sample. Hotspots shift sharply across phases without
#   changing the overall frequency distribution—only which ids are hot.
#
# Output CSV columns: timestamp, phase, block_id, frequency_per_sec (int); phase is mission stage (from 0).
import os

import numpy as np
import pandas as pd


def load_frequencies_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "block_id" not in df.columns or "frequency_per_sec" not in df.columns:
        raise ValueError("block_frequencies.csv must contain columns: block_id, frequency_per_sec")
    df = df.sort_values("block_id").reset_index(drop=True)
    ids = df["block_id"].to_numpy()
    if not np.array_equal(ids, np.arange(len(df))):
        raise ValueError("block_id must be 0..N-1 consecutive and match row count")
    freq = np.round(df["frequency_per_sec"].to_numpy(dtype=np.float64)).astype(np.int64)
    freq = np.maximum(freq, 1)
    return freq


def build_phase_frequency_rows(freq_base, n_phases, rng):
    """
    Return shape (n_phases, N): nominal frequency per block_id per phase.
    Row 0 = original CSV; other rows = freq_base[perm] with independent random permutations.
    """
    n = len(freq_base)
    rows = np.empty((n_phases, n), dtype=np.int64)
    rows[0] = freq_base
    for p in range(1, n_phases):
        perm = rng.permutation(n)
        rows[p] = freq_base[perm]
    return rows


def sample_by_weights(freq_row, size, rng):
    """Weighted sampling on indices 0..N-1 using freq_row as relative weights."""
    w = freq_row.astype(np.float64)
    s = w.sum()
    if s <= 0:
        p = np.full(len(w), 1.0 / len(w))
    else:
        p = w / s
    return rng.choice(np.arange(len(w), dtype=np.int64), size=size, p=p)


def generate_phased_access_sequence(
    sequence_length=100000,
    frequencies_csv=None,
    output_dir=None,
    num_blocks=10000,
    n_phases=3,
    rng_seed=None,
):
    """
    :param frequencies_csv: block_frequencies.csv; default HiChain_Code/datasets/block_frequencies.csv
    :param output_dir: output directory; default same as frequencies_csv
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if frequencies_csv is None:
        frequencies_csv = os.path.join(base_dir, "datasets", "block_frequencies.csv")
    if output_dir is None:
        output_dir = os.path.dirname(frequencies_csv)

    freq_base = load_frequencies_from_csv(frequencies_csv)
    if len(freq_base) != num_blocks:
        raise ValueError(f"Frequency table length {len(freq_base)} != num_blocks={num_blocks}")

    rng = np.random.default_rng(rng_seed)
    phase_freq = build_phase_frequency_rows(freq_base, n_phases, rng)

    L = int(sequence_length)
    base = L // n_phases
    rem = L % n_phases
    lengths = [base + (1 if i < rem else 0) for i in range(n_phases)]

    block_parts, freq_parts, phase_parts = [], [], []
    for p, plen in enumerate(lengths):
        if plen <= 0:
            continue
        blocks = sample_by_weights(phase_freq[p], plen, rng)
        block_parts.append(blocks)
        freq_parts.append(phase_freq[p, blocks])
        phase_parts.append(np.full(plen, p, dtype=np.int64))

    block_id = np.concatenate(block_parts)
    frequency_per_sec = np.concatenate(freq_parts)
    phase = np.concatenate(phase_parts)
    timestamp = np.arange(L, dtype=np.int64)

    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "phase": phase,
            "block_id": block_id.astype(np.int64),
            "frequency_per_sec": frequency_per_sec.astype(np.int64),
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(
        output_dir,
        f"block_access_{num_blocks}blocks_{L}steps_phased_mission.csv",
    )
    df.to_csv(csv_path, index=False)
    print(f"Access sequence saved: {csv_path}")
    for p in range(n_phases):
        row = phase_freq[p]
        top = int(np.argmax(row))
        print(
            f"  Phase {p}: steps {lengths[p]}, "
            f"hottest block id={top}, freq={row[top]} "
            f"(phase 0 matches CSV; phase>=1 uses permuted hotspots)"
        )
    return csv_path


if __name__ == "__main__":
    generate_phased_access_sequence(sequence_length=100000, rng_seed=42)
