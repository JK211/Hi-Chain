# Hi-Chain: Hierarchical and Intelligent Blockchain Data Storage Scheme for UAV Networks

**Authors**: [Zhe Ren](https://github.com/JK211)

**Keywords**: UAVNet, Blockchain, error-correction coding, hierarchical framework, Markov decision process, reinforcement learning

## Requirements

- Python 3.8+
- torch>=1.10.1
- gym>=0.24.0
- numpy>=1.21.0
- matplotlib>=3.5.0
- pandas>=1.3.0
- scipy>=1.7.0

```bash
pip install torch gym matplotlib pandas numpy scipy tqdm
```

## Description

This repository contains the official implementation and experiment scripts for the paper *Hi-Chain: Hierarchical and Intelligent Blockchain Data Storage Scheme for UAV Networks*. Core code lives under `HiChain_Code/`.

**Recommended order** (first-time setup):

1. `python data_generator003.py` — creates `datasets/block_frequencies.csv` (per-block frequency and k)
2. Run the experiment scripts below as needed

## Project layout (`HiChain_Code/`)

| File | Description |
|------|-------------|
| `AdaptiveRSEncoder.py` | ARSES adaptive RS encoding (Eqs. 18–20: availability constraints + Sigmoid k mapping) |
| `data_generator003.py` | Zipf heavy-tail block-level frequency dataset |
| `compare_of_coding_fig4.py` | Fig. 4: encoding curves for a small set of α/β |
| `compare_of_coding_fig4_extended.py` | Fig. 4 extended: α/β grid sensitivity and Origin export |
| `k_value_cal_fig5.py` | Fig. 5: k-value distribution stats and plots |
| `compare_of_storage_cost_fig6-9.py` | Figs. 6–9: storage overhead comparison and CSV export |
| `zipf_dynamic_storage_test.py` | Sensitivity of dynamic encoding total storage to Zipf exponent |
| `sim_cooperative_recovery_success.py` | Cooperative recovery success rate vs offline rate and k |
| `ns3/scratch/hi-chain-uav-swarm-validation.cc` | **NS-3 v3.25** scratch: hit rate, latency, cache policies (`none`/`lru`/`freq`/`rl`) |
| `ns3/scratch/hi-chain-paper-params.h` | Paper-aligned NS-3 default parameters |
| `export_ns3_rl_trace.py` | Export BDQN actions + access trace CSV for NS-3 `--cachePolicy=rl` |
| `sim_uav_swarm_hit_latency_ns3.py` | Python-only surrogate (not NS-3); use `ns3/` for real simulator code |
| `data_generator_mission_phases.py` | Phased-mission access trace with hotspot shifts |
| `BDQN.py` / `CacheReplacementEnv.py` / `train_model007.py` | BDQN cache replacement training |

Generated figures and CSVs are written next to each script (see `.gitignore`) and are not tracked in git.

---

## Usage

### 1. `data_generator003.py` (dataset generation)

**Purpose**: Generate block-level access frequencies from Zipf weights `1/rank^s`, map them to `[freq_min, freq_max]`, compute per-block k via `AdaptiveRSEncoder`, and write `datasets/block_frequencies.csv`.

**Main parameters** (`BlockDatasetGenerator`):

- `num_blocks`: number of blocks (default 10000; `__main__` example uses 8000)
- `n_nodes`: swarm size (default 100; example uses 50)
- `zipf_s`: Zipf exponent; larger → hotter hotspots
- `freq_min` / `freq_max`: nominal access frequency bounds (per second)

**Run**:

```bash
cd HiChain_Code
python data_generator003.py
```

---

### 2. `AdaptiveRSEncoder.py` (encoding core)

**Purpose**: Paper ARSES implementation.

- **Eq. (18)**: Availability `P(B)` from offline probability `p` and node count `n`; feasible k range `[k_min, k_max]` under `p_max` / `p_min`
- **Eqs. (19–20)**: Normalize raw access frequency and map to k via Sigmoid (high frequency → smaller k → more redundancy)

With `n=50`, k values align with `block_frequencies.csv`.

---

### 3. `compare_of_coding_fig4.py`

**Purpose**: Fig. 4 k / redundancy curves for a few α, β combinations.  
**Output**: `encoding_validation/` (CSVs + Origin plotting guide); optional figures.

```bash
python compare_of_coding_fig4.py
```

---

### 4. `compare_of_coding_fig4_extended.py` (extended)

**Purpose**: Extended **α, β grid** sensitivity on top of Fig. 4.

- `ALPHAS = [1, 3, 5, 7, 9]`, `BETAS = [0.1, 0.3, 0.5, 0.7, 0.9]`
- Subplots 1–2: fixed `β=0.5`, sweep α; subplots 3–4: fixed `α=5`, sweep β
- Timestamped raw/summary CSVs and `Origin_Plotting_Guide_*.txt`
- Figures under `encoding_figures/`

```bash
python compare_of_coding_fig4_extended.py
```

---

### 5. `k_value_cal_fig5.py`

**Purpose**: Load `block_frequencies.csv`, summarize and plot k distribution.  
**Related to**: Fig. 5 dataset analysis.

```bash
python k_value_cal_fig5.py
```

---

### 6. `compare_of_storage_cost_fig6-9.py` (enhanced)

**Purpose**: Compare five storage schemes on total / per-node / per-block cost and capacity limits; dedicated static vs dynamic RS comparison.

**Schemes**: FullNode, LightNode, Sharding, StaticEncoding (min k from CSV), DynamicEncoding (per-block k from CSV).

**Enhancements**:

- Reads `frequency_per_sec` and per-block `k` from `datasets/block_frequencies.csv`
- `REF_NODE_COUNT=50` aligned with default swarm size in data generation
- `STORAGE_HISTORY_RECORD_INTERVAL` thins line-plot samples (default every 1000 blocks)
- `export_data_to_csv()` → `storage_overhead_compare/` for Origin
- Figures: `fig13_combined.png`, `encoding_comparison.png`

```bash
python compare_of_storage_cost_fig6-9.py
```

---

### 7. `zipf_dynamic_storage_test.py` (new)

**Purpose**: Under **8000 blocks, 50 nodes**, sweep Zipf exponents `s`, accumulate dynamic RS total storage (MB) using the same model as `DynamicEncoding`, and report k-tier counts and top1%/top10% traffic share.

**Tunable constants** (top of script): `ZIPF_LIST`, `NUM_BLOCKS`, `N_NODES`, `RNG_SEED`.

```bash
python zipf_dynamic_storage_test.py
```

Optional output: `zipf_dynamic_storage_compare.png`.

---

### 8. `sim_cooperative_recovery_success.py` (new)

**Purpose**: Under independent node offline failures, evaluate **cooperative recovery success rate** `P(online nodes ≥ k)` for each k tier.

- Default `n=50`, offline rates 10%–100% (step 10%)
- Default k list matches typical tiers for n=50 in the dataset
- `--mc 0`: analytic probabilities only; default 200k Monte Carlo snapshots with ~95% CI

```bash
python sim_cooperative_recovery_success.py
python sim_cooperative_recovery_success.py --mc 0
python sim_cooperative_recovery_success.py --n-nodes 50 --k-list 10 15 20
```

---

### 9. NS-3 v3.25 — UAV swarm hit rate, latency & RL cache (`ns3/`)

**Purpose**: C++ for **NS-3.25** — paper defaults in `hi-chain-paper-params.h`, WiFi ad-hoc access model, and cache policies `none` | `lru` | `freq` | `rl`.

**Install** (see [`ns3/README.md`](ns3/README.md)):

```bash
cp ns3/scratch/hi-chain-paper-params.h ns3/scratch/hi-chain-uav-swarm-validation.cc /path/to/ns-3.25/scratch/
cd /path/to/ns-3.25 && ./waf build
./waf --run "scratch/hi-chain-uav-swarm-validation --nUavs=50 --numBlocks=8000"
```

**RL in NS-3** (BDQN stays in Python; NS-3 replays exported CSV):

```bash
cd HiChain_Code
python train_model007.py
python export_ns3_rl_trace.py --n-nodes 50
# then NS-3: --cachePolicy=rl --accessTrace=.../ns3_access_trace.csv --rlActions=.../ns3_rl_actions.csv
```

Use the **same** `--nUavs` / `--n-nodes` as training (state dim `4 + 3*n`).

**Sweep scales**: `ns3/scripts/run_swarm_scale_sweep.sh`

`sim_uav_swarm_hit_latency_ns3.py` is a **Python surrogate** only when NS-3 is unavailable.

---

### 10. `data_generator_mission_phases.py` (new)

**Purpose**: Access traces with **sharp popularity shifts across mission phases** (reviewer scenario).

- Phase 0: weighted sampling using `block_frequencies.csv` id→frequency as-is
- Phases 1, 2: permute the frequency multiset onto block ids, then sample (same distribution, different hotspot locations)
- Output columns: `timestamp`, `phase`, `block_id`, `frequency_per_sec`

```bash
python data_generator_mission_phases.py
```

Default output: `datasets/block_access_10000blocks_100000steps_phased_mission.csv` (name depends on `num_blocks`).

---

### 11. `train_model007.py` + `BDQN.py` + `CacheReplacementEnv.py`

**Purpose**: BDQN-based cache replacement training.  
**Requires**: `datasets/block_frequencies.csv` (and access trace CSV as configured in the script).

```bash
python train_model007.py
```

---

## Data files

| Path | Description |
|------|-------------|
| `HiChain_Code/datasets/block_frequencies.csv` | From `data_generator003.py`; columns `block_id`, `frequency_per_sec`, `k` |
| `HiChain_Code/datasets/block_access_*_phased_mission.csv` | From `data_generator_mission_phases.py` |

---

## Cleanup notes

- Removed `AdaptiveRSEncoder_backup.py` (unused backup)
- Timestamped CSVs under `encoding_validation/` were cleared; re-run Fig. 4 scripts to regenerate
- Generated artifacts are listed in the root `.gitignore`
