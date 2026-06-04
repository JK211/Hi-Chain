import math
from typing import Tuple, List


def _availability_pb(n: int, k: int, p_offline: float) -> float:
    """
    Eq. (18): P(B) = e^{-λ} * sum_{i=0}^{n-k} λ^i / i!, with λ = p * n.
    p: per-node offline probability; k: fragments required for reconstruction; n: total nodes.
    """
    if k < 1 or k > n:
        return 0.0
    lam = p_offline * n
    m = n - k
    s = 0.0
    term = math.exp(-lam)  # i = 0: e^{-λ} λ^0 / 0!
    for i in range(0, m + 1):
        s += term
        if i < m:
            term = term * lam / (i + 1)
    return s


def _largest_k_meeting_availability(
    n: int, p_offline: float, p_threshold: float
) -> int:
    """
    P(B) is non-increasing in k; scan k from 1 upward and return the largest k
    that still satisfies P(B) >= threshold.
    """
    best = 1
    for k in range(1, n + 1):
        if _availability_pb(n, k, p_offline) >= p_threshold:
            best = k
        else:
            break
    return best


class AdaptiveRSEncoder:
    """ARSES (Adaptive Reed-Solomon Encoding Storage) from the paper."""

    def __init__(
        self,
        n_nodes: int,
        p_offline: float = 0.6,
        p_max: float = 0.99,
        p_min: float = 0.50,
    ):
        """
        :param n_nodes: number of UAV nodes n = |N|
        :param p_offline: mean per-node offline probability p (paper: λ = p·n)
        :param p_max: lower bound on data availability under strict requirements (e.g. 0.95)
        :param p_min: lower bound on data availability under relaxed requirements (e.g. 0.75)
        """
        self.n = n_nodes
        self.p_offline = p_offline
        self.p_max = p_max
        self.p_min = p_min
        self._lambda = p_offline * n_nodes

        # Stricter threshold → smaller feasible k upper bound; looser → larger
        self.k_at_pmax = _largest_k_meeting_availability(n_nodes, p_offline, p_max)
        self.k_at_pmin = _largest_k_meeting_availability(n_nodes, p_offline, p_min)

        if self.k_at_pmin < self.k_at_pmax:
            self.k_at_pmin = self.k_at_pmax

        # Consistent with adaptive_encode: numerically k_min <= k_max
        self.k_min = self.k_at_pmax
        self.k_max = self.k_at_pmin

        print(
            f"ARSES init: n={n_nodes}, p={p_offline}, λ={self._lambda:.4f}, "
            f"p_max={p_max}, p_min={p_min}, "
            f"k(P≥p_max)={self.k_at_pmax}, k(P≥p_min)={self.k_at_pmin} -> "
            f"adaptive k∈[{self.k_min}, {self.k_max}]"
        )

    def adaptive_encode(
        self,
        data: bytes,
        access_frequency: float,
        alpha: float = 9.0,
        beta: float = 0.5,
        f_raw_min: float = 1.0,
        f_raw_max: float = 1000.0,
    ) -> Tuple[int, List[bytes]]:
        """
        Adaptive RS encoding (paper ARSES).

        Eq. (19): f_norm = (f_raw - f_min) / (f_max - f_min)
        Eq. (20) common form: k_h = k_min + (k_max - k_min) / (1 + exp(α(f_norm - β)))
                   (equivalent to k_min + (k_max - k_min)·(1-σ) with σ below)

        This code uses k_h = k_max - (k_max - k_min)·σ, algebraically equivalent to the paper;
        high f_norm → σ→1 → k_h→k_min (more redundancy).
        """
        denom = f_raw_max - f_raw_min
        if denom <= 0:
            f_norm = 0.5
        else:
            f_norm = (access_frequency - f_raw_min) / denom
        f_norm = max(0.0, min(1.0, f_norm))

        # Sigmoid in Eq. (20): σ = 1 / (1 + exp(-α(f_norm - β)))
        sigmoid = 1.0 / (1.0 + math.exp(-alpha * (f_norm - beta)))
        # Equivalent to k_min + (k_max-k_min)*(1-σ) and to k_max - (k_max-k_min)*σ
        k = int(round(self.k_max - (self.k_max - self.k_min) * sigmoid))
        k = max(self.k_min, min(self.k_max, k))
        k = max(1, min(self.n, k))

        segment_size = math.ceil(len(data) / self.n)
        fragments = []
        for i in range(self.n):
            start = i * segment_size
            end = (i + 1) * segment_size
            fragments.append(data[start:end])

        return k, fragments

    def decode_data(self, fragments: List[bytes], k: int) -> bytes:
        valid_fragments = [f for f in fragments if f is not None]
        if len(valid_fragments) >= k:
            return b"".join(valid_fragments[:k])
        raise ValueError("Cannot decode data - insufficient fragments")
