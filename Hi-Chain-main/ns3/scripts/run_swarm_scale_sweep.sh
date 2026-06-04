#!/usr/bin/env bash
# Sweep UAV swarm sizes and append metrics to one CSV (run from NS-3 root).
#
# Usage:
#   bash run_swarm_scale_sweep.sh /path/to/ns-3.25
#   bash run_swarm_scale_sweep.sh /path/to/ns-3.25 "10 20 30 40 50" 120

set -euo pipefail

NS3_ROOT="${1:?Usage: $0 /path/to/ns-3.25 [scales] [simTime]}"
SCALES="${2:-10 20 30 40 50}"
SIM_TIME="${3:-100}"
CSV="hi-chain-swarm-metrics.csv"

cd "$NS3_ROOT"
rm -f "$CSV"

for n in $SCALES; do
  echo "=== nUavs=$n ==="
  ./waf --run "scratch/hi-chain-uav-swarm-validation --nUavs=$n --simTime=$SIM_TIME --csvOut=$CSV"
done

echo "Done. Results: $NS3_ROOT/$CSV"
