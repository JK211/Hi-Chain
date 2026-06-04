"""
Cooperative recovery success rate for each encoding k under multiple node offline rates.

Model (one fragment per node; decode when k fragments available):
  - n nodes; each independently offline with probability p_offline;
  - online nodes contribute fragments; success iff online count >= k.

Defaults: n=50, offline rates 10%..100% (step 10%), k tiers from
data_generator003 + AdaptiveRSEncoder(n=50) dataset distribution.

Usage:
  python sim_cooperative_recovery_success.py
  python sim_cooperative_recovery_success.py --mc 0
    Analytic probabilities only (no Monte Carlo).
"""
from __future__ import annotations

import argparse
import math
import random
import sys

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_N = 50
DEFAULT_P_OFFLINE_LIST = tuple(i / 10 for i in range(1, 11))
DEFAULT_K_LIST = (7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20)


def recovery_success_prob(n: int, k: int, p_offline: float) -> float:
    """P(online nodes >= k) with per-node online probability 1 - p_offline."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    q = 1.0 - p_offline
    q = max(0.0, min(1.0, q))
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    term = (1.0 - q) ** n
    cdf = 0.0
    for j in range(0, k):
        cdf += term
        if j < n:
            term = term * (n - j) / (j + 1) * q / (1.0 - q)
    return max(0.0, min(1.0, 1.0 - cdf))


def _one_snapshot_online(n: int, p_on: float, rng: random.Random) -> int:
    """Online node count in one network snapshot (n independent Bernoulli trials)."""
    return sum(1 for _ in range(n) if rng.random() < p_on)


def _fmt_offline(p_off: float) -> str:
    return f"{int(round(p_off * 100)):>3}%"


def _fmt_prob(value: float) -> str:
    return f"{round(value * 100, 2):.2f}%"


def _print_prob_table(
    n: int,
    k_list: list[int],
    p_offline_list: list[float],
    title: str,
    get_prob,
) -> None:
    k_headers = [f"k={k}" for k in k_list]
    col_w = 9
    label_w = max(8, len("Offline"))
    header = f"{'Offline':>{label_w}}  " + "  ".join(f"{h:>{col_w}}" for h in k_headers)
    print(title)
    print(header)
    print("-" * len(header))
    for p_off in p_offline_list:
        row = f"{_fmt_offline(p_off):>{label_w}}  "
        row += "  ".join(f"{_fmt_prob(get_prob(p_off, k)):>{col_w}}" for k in k_list)
        print(row)
    print()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cooperative recovery success vs k and offline rate (independent node failures)"
    )
    p.add_argument("--n-nodes", type=int, default=DEFAULT_N, help="Number of nodes n (default 50)")
    p.add_argument(
        "--p-offline-list",
        type=float,
        nargs="+",
        default=list(DEFAULT_P_OFFLINE_LIST),
        help="Per-node offline probabilities (default 0.1 .. 1.0)",
    )
    p.add_argument(
        "--k-list",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_LIST),
        help="k values to evaluate",
    )
    p.add_argument(
        "--mc",
        type=int,
        default=200_000,
        help="Network snapshots per offline rate; 0 = analytic only",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    n = args.n_nodes
    k_list = args.k_list
    p_offline_list = sorted(set(args.p_offline_list))

    print("Cooperative recovery success (independent node offline)")
    print(f"  Nodes n = {n}")
    print(f"  Offline rates: {', '.join(_fmt_offline(p) for p in p_offline_list)}")
    print("  Success when: online nodes >= k")
    print()

    if args.mc <= 0:
        _print_prob_table(
            n,
            k_list,
            p_offline_list,
            "Analytic success rate",
            lambda p_off, k: recovery_success_prob(n, k, p_off),
        )
        return

    rng = random.Random(args.seed)
    z = 1.96
    print(f"  Monte Carlo: {args.mc} snapshots per offline rate (seed={args.seed})")
    print()

    def exact(p_off: float, k: int) -> float:
        return recovery_success_prob(n, k, p_off)

    _print_prob_table(n, k_list, p_offline_list, "Analytic success rate", exact)

    mc_rates: dict[tuple[float, int], float] = {}
    mc_half: dict[tuple[float, int], float] = {}
    for p_off in p_offline_list:
        p_on = 1.0 - p_off
        hits = {k: 0 for k in k_list}
        for _ in range(args.mc):
            online = _one_snapshot_online(n, p_on, rng)
            for k in k_list:
                if online >= k:
                    hits[k] += 1
        for k in k_list:
            rate = hits[k] / args.mc
            se = math.sqrt(rate * (1.0 - rate) / args.mc)
            mc_rates[(p_off, k)] = rate
            mc_half[(p_off, k)] = z * se

    _print_prob_table(
        n,
        k_list,
        p_offline_list,
        "Simulated success rate",
        lambda p_off, k: mc_rates[(p_off, k)],
    )

    print("Approx. 95% CI half-width (simulation)")
    print(f"{'Offline':>8}  " + "  ".join(f"{'k=' + str(k):>10}" for k in k_list))
    print("-" * (10 + 12 * len(k_list)))
    for p_off in p_offline_list:
        row = f"{_fmt_offline(p_off):>8}  "
        row += "  ".join(f"±{_fmt_prob(mc_half[(p_off, k)])}" for k in k_list)
        print(row)


if __name__ == "__main__":
    main()
