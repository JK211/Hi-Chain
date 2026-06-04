# NS-3 v3.25 integration — UAV swarm hit rate, latency & RL cache policy

C++ scratch programs for **NS-3.25** (not standalone Python). Validates **cache hit rate** and **average access latency** over UAV swarms, with optional **BDQN cache replacement** replayed from Python exports.

## Files

| File | Role |
|------|------|
| `scratch/hi-chain-paper-params.h` | Paper-aligned defaults (blocks, cache, RS-k, timeouts) |
| `scratch/hi-chain-uav-swarm-validation.cc` | Main simulation + cache policies |
| `scripts/run_swarm_scale_sweep.sh` | Linux/macOS scale sweep |
| `scripts/run_swarm_scale_sweep.ps1` | Windows scale sweep |

Copy **both** `.h` and `.cc` into `ns-3.25/scratch/`, then `./waf build`.

## Paper parameters

Defaults come from `HiChainPaperParams` (see `hi-chain-paper-params.h`):

- `numBlocks=8000`, `nUavs=50`, `blockSizeMb=1`, `cachePerNodeMb=20`, `storagePerNodeMb=1000`
- `kRecover=20`, `simTime=100s`, `accessIntervalMs=50`

Override on the command line, e.g. `--nUavs=50 --numBlocks=8000`.

Optional block metadata (Zipf frequencies, per-block `k`):

```bash
--blockCsv=/path/to/HiChain_Code/datasets/block_frequencies.csv
```

Generate that CSV with `python data_generator003.py` in `HiChain_Code/`.

## Cache policies (`--cachePolicy`)

| Policy | Meaning |
|--------|---------|
| `none` | No proactive replacement on miss (baseline) |
| `lru` | Evict LRU slot, prefetch requested block |
| `freq` | Evict lowest-frequency cached block, add high-frequency block |
| `rl` | Replay BDQN actions from Python CSV |

Metrics CSV includes a `cache_policy` column for A/B runs.

## RL (BDQN) workflow — why not PyTorch inside NS-3?

NS-3.25 does not embed PyTorch easily. The supported path:

1. **Train** (same `n_nodes` as NS-3 `--nUavs`):

   ```bash
   cd HiChain_Code
   python train_model007.py
   # writes results/train/bdqn_policy.pt
   ```

2. **Export** greedy policy + access trace:

   ```bash
   python export_ns3_rl_trace.py --n-nodes 50
   # writes datasets/ns3_export/ns3_access_trace.csv
   #       datasets/ns3_export/ns3_rl_actions.csv
   ```

3. **Run NS-3** (from `ns-3.25` root):

   ```bash
   ./waf --run "scratch/hi-chain-uav-swarm-validation \
     --nUavs=50 --numBlocks=8000 \
     --cachePolicy=rl \
     --accessTrace=/path/to/ns3_access_trace.csv \
     --rlActions=/path/to/ns3_rl_actions.csv \
     --blockCsv=/path/to/block_frequencies.csv"
   ```

**Important:** BDQN state size is `4 + 3 * n_nodes`. A model trained with `n=10` cannot be used for `n=50` without retraining and re-export.

Access trace should match Python (`block_access_*_phased_mission.csv` or the exported `ns3_access_trace.csv`).

## Compare policies (same trace)

```bash
for p in none lru freq rl; do
  ./waf --run "scratch/hi-chain-uav-swarm-validation --nUavs=50 \
    --cachePolicy=$p --accessTrace=.../ns3_access_trace.csv \
    --rlActions=.../ns3_rl_actions.csv --csvOut=hi-chain-swarm-metrics.csv"
done
```

## Single run / scale sweep

```bash
./waf --run "scratch/hi-chain-uav-swarm-validation --nUavs=50 --simTime=100"
```

```bash
bash Hi-Chain-main/ns3/scripts/run_swarm_scale_sweep.sh /path/to/ns-3.25
```

## Output columns

- `cache_hit_rate` — (local + remote) / total  
- `data_availability_rate` — includes RS recovery  
- `avg_access_latency_ms` — NS-3 event timing  
- `cache_policy`, counts: `local`, `remote`, `recovery`, `miss`

## Architecture (short)

- **Coordinator** (`HiChainCoordinatorApp` on node 0): one global access per step, random UAV (aligned with `CacheReplacementEnv.step()`).
- **Per-UAV app** (`HiChainBlockAccessApp`): UDP 9900, local / remote / recovery / miss.
- On miss (non-local start), optional replacement at ≥90% cache fill; `rl` uses precomputed `(evict_slot, add_block_id)` per step.

## Relation to Python

| Python | NS-3 |
|--------|------|
| `train_model007.py` + `BDQN.py` | Train policy |
| `export_ns3_rl_trace.py` | Export CSV for `--cachePolicy=rl` |
| `sim_uav_swarm_hit_latency_ns3.py` | Fast analytic surrogate (no NS-3) |
| This folder | Real WiFi ad-hoc latency in NS-3.25 |

## Troubleshooting

- **`cachePolicy=rl` exits** — provide both `--accessTrace` and `--rlActions`.
- **Low hit rate vs Python** — WiFi timeouts differ from analytic delays; tune `--remoteTimeoutMs` / mobility in `.cc` or shorten swarm area.
- **Compile errors** — match WiFi API to your NS-3 minor version; enable modules with `./waf configure`.
