# data_generator003.py
import numpy as np
import pandas as pd
import os
from AdaptiveRSEncoder import AdaptiveRSEncoder


class BlockDatasetGenerator:
    def __init__(self, num_blocks=10000, n_nodes=100, zipf_s=1.15, freq_min=1, freq_max=1000):
        """
        Generate block dataset (block-level access frequency distribution).
        :param num_blocks: total blocks (default 10000)
        :param n_nodes: UAV swarm size
        :param zipf_s: Zipf exponent; larger → hotter hotspots (heavier tail)
        :param freq_min: lower bound of nominal access frequency (per second)
        :param freq_max: upper bound of nominal access frequency (per second)
        """
        self.num_blocks = num_blocks
        self.n_nodes = n_nodes
        self.zipf_s = zipf_s
        self.freq_min = freq_min
        self.freq_max = freq_max

        self.encoder = AdaptiveRSEncoder(n_nodes)

        # Heavy-tail (Zipf) frequencies in [freq_min, freq_max], then shuffled across block_id
        self.base_frequencies = self._generate_heavy_tail_frequencies()

    def _generate_heavy_tail_frequencies(self):
        """
        Weights from Zipf 1/rank^s, linearly mapped to integer frequencies in [freq_min, freq_max].
        Top-ranked blocks near freq_max; long-tail blocks near freq_min.
        """
        ranks = np.arange(1, self.num_blocks + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, self.zipf_s)
        w_min = weights.min()
        w_max = weights.max()
        span = self.freq_max - self.freq_min
        scaled = self.freq_min + (weights - w_min) / (w_max - w_min) * span
        frequencies = np.clip(np.round(scaled), self.freq_min, self.freq_max).astype(np.float64)

        np.random.shuffle(frequencies)
        return frequencies

    def get_block_frequencies(self):
        """Block access frequencies (for environment init during training)."""
        return self.base_frequencies.copy()

    def calculate_encoding_k(self, frequency):
        """
        Encoding k for a given access frequency.
        :param frequency: access frequency (per second)
        :return: k value
        """
        dummy_data = b'0' * 1024  # 1 KB placeholder
        k, _ = self.encoder.adaptive_encode(dummy_data, frequency)
        return k

    def get_frequency_distribution(self):
        """Summary stats of frequency distribution and tail heaviness."""
        f = self.base_frequencies
        total = f.sum()
        order = np.argsort(-f)
        top1 = max(1, self.num_blocks // 100)
        top1_mass = f[order[:top1]].sum() / total * 100.0
        top10_mass = f[order[: max(1, self.num_blocks // 10)]].sum() / total * 100.0
        return {
            'num_blocks': self.num_blocks,
            'freq_min': float(f.min()),
            'freq_max': float(f.max()),
            'freq_mean': float(f.mean()),
            'zipf_s': self.zipf_s,
            'top_1pct_blocks_traffic_share_pct': float(top1_mass),
            'top_10pct_blocks_traffic_share_pct': float(top10_mass),
        }


if __name__ == "__main__":
    generator = BlockDatasetGenerator(num_blocks=8000, n_nodes=50, zipf_s=0.4)

    print("Frequency and heavy-tail statistics:")
    for k, v in generator.get_frequency_distribution().items():
        print(f"  {k}: {v}")

    frequencies = generator.get_block_frequencies()
    k_values = []

    print("Computing encoding k values...")
    for i, freq in enumerate(frequencies):
        k = generator.calculate_encoding_k(freq)
        k_values.append(k)
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(frequencies)} blocks")

    n = len(frequencies)
    freq_df = pd.DataFrame({
        'block_id': range(n),
        'frequency_per_sec': frequencies,
        'k': k_values
    })
    os.makedirs("datasets", exist_ok=True)
    freq_df.to_csv("datasets/block_frequencies.csv", index=False)
    print("Saved block frequencies and encoding info to: datasets/block_frequencies.csv")
