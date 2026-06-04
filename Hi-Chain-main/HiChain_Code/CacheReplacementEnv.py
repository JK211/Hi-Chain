import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gym
from gym import spaces
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from typing import Tuple, Dict, List
from BDQN import BDQN
from AdaptiveRSEncoder import AdaptiveRSEncoder
from collections import deque, OrderedDict, defaultdict
import time

class UAVNode:
    """Simulated UAV node."""

    def __init__(self, node_id: int, cache_capacity: int, storage_capacity: int, network: 'UAVNetwork'):
        self.id = node_id
        self.cache = OrderedDict()  # LRU cache (block_id: data)
        self.encoded_fragments = {}  # Stored encoded fragments (block_id: fragment)
        self.cache_capacity = cache_capacity
        self.storage_capacity = storage_capacity
        self.used_storage = 0
        self.network = network
        self.stats = {
            'hits': 0,
            'misses': 0,
            'local_hits': 0,
            'remote_hits': 0,
            'recovered': 0,
            'access_time': 0,
            'storage_used': 0
        }

    def store_fragment(self, block_id: int, fragment: bytes):
        """Store an encoded fragment."""
        fragment_size = len(fragment)
        if self.used_storage + fragment_size > self.storage_capacity:
            self._free_storage(fragment_size)

        self.encoded_fragments[block_id] = fragment
        self.used_storage += fragment_size
        self.stats['storage_used'] = self.used_storage

    def _free_storage(self, required_size: int):
        """Free storage space."""
        while self.used_storage + required_size > self.storage_capacity and self.encoded_fragments:
            block_id = random.choice(list(self.encoded_fragments.keys()))
            fragment_size = len(self.encoded_fragments[block_id])
            del self.encoded_fragments[block_id]
            self.used_storage -= fragment_size

    def cache_block(self, block_id: int, data: bytes):
        """Cache a block (LRU policy)."""
        data_size = len(data)
        if data_size > self.cache_capacity:
            return False

        while block_id not in self.cache and self._cache_size() + data_size > self.cache_capacity:
            if self.cache:
                oldest_block = next(iter(self.cache))
                self.evict_cache(oldest_block)

        self.cache[block_id] = data
        self.cache.move_to_end(block_id)
        return True

    def evict_cache(self, block_id: int):
        """Remove a block from cache."""
        if block_id in self.cache:
            del self.cache[block_id]

    def _cache_size(self) -> int:
        """Current cache size in bytes."""
        return sum(len(data) for data in self.cache.values())

    def access_block(self, block_id: int) -> Tuple[bool, float, int]:
        """
        Access a block.
        :return: (hit, latency_ms, hit_type: 0=miss, 1=local, 2=remote, 3=recovered)
        """
        start_time = time.perf_counter()
        hit_type = 0  # default miss

        # 1. Local cache
        if block_id in self.cache:
            self.cache.move_to_end(block_id)
            self.stats['hits'] += 1
            self.stats['local_hits'] += 1
            latency = (time.perf_counter() - start_time) * 1000
            self.stats['access_time'] += latency
            return True, latency, 1  # local hit

        # 2. Remote node cache
        remote_hit, remote_latency = self._check_remote_cache(block_id)
        if remote_hit:
            self.stats['hits'] += 1
            self.stats['remote_hits'] += 1
            latency = 50 + random.uniform(0, 30)  # simulated network delay
            time.sleep(latency / 1000)
            latency = (time.perf_counter() - start_time) * 1000
            self.stats['access_time'] += latency
            return True, latency, 2  # remote hit

        # 3. Recover from encoded fragments
        recovered, recovery_latency = self._recover_block(block_id)
        if recovered:
            self.stats['recovered'] += 1
            latency = 100 + random.uniform(0, 50)
            latency = (time.perf_counter() - start_time) * 1000
            self.stats['access_time'] += latency
            return False, latency, 3  # recovered

        # 4. Full miss
        self.stats['misses'] += 1
        latency = (time.perf_counter() - start_time) * 1000
        self.stats['access_time'] += latency
        return False, latency, 0  # miss

    def _check_remote_cache(self, block_id: int) -> Tuple[bool, float]:
        """Query other nodes' caches."""
        start_time = time.perf_counter()
        query_nodes = random.sample(self.network.nodes, len(self.network.nodes))

        for node in query_nodes:
            if node.id == self.id:
                continue

            if block_id in node.cache:
                data = node.cache[block_id]
                self.cache_block(block_id, data)
                transfer_time = len(data) / (1024 * 1024) * 1  # 1 MB/s transfer
                latency = (time.perf_counter() - start_time) * 1000 + transfer_time
                return True, latency

        return False, 0

    def _recover_block(self, block_id: int) -> Tuple[bool, float]:
        """Recover block from encoded fragments."""
        if block_id not in self.network.blocks:
            return False, 0

        start_time = time.perf_counter()
        fragments = []
        k = self.network.blocks[block_id][2]  # encoding k
        collected_fragments = 0

        for node in random.sample(self.network.nodes, len(self.network.nodes)):
            if collected_fragments >= k:
                break

            if block_id in node.encoded_fragments:
                fragments.append(node.encoded_fragments[block_id])
                collected_fragments += 1
            else:
                fragments.append(None)

        if collected_fragments >= k:
            try:
                decoded_data = self.network.encoder.decode_data(fragments, k)
                self.cache_block(block_id, decoded_data)
                transfer_time = sum(len(f) for f in fragments if f is not None) / (1024 * 1024) * 1
                decode_time = len(decoded_data) / (1024 * 1024) * 5
                latency = (time.perf_counter() - start_time) * 1000 + transfer_time + decode_time
                return True, latency
            except ValueError:
                pass

        return False, 0

    def get_stats(self) -> dict:
        """Node statistics."""
        hits = self.stats['hits']
        misses = self.stats['misses']
        total = hits + misses

        return {
            'node_id': self.id,
            'cache_hits': hits,
            'local_hits': self.stats['local_hits'],
            'remote_hits': self.stats['remote_hits'],
            'recovered': self.stats['recovered'],
            'cache_misses': misses,
            'cache_hit_rate': hits / total if total > 0 else 0,
            'avg_access_time': self.stats['access_time'] / total if total > 0 else 0,
            'storage_used': self.stats['storage_used'],
            'storage_utilization': self.stats['storage_used'] / self.storage_capacity
        }


class UAVNetwork:
    """Simulated UAV network."""

    def __init__(self, n_nodes, block_size, cache_per_node, storage_per_node):
        self.n_nodes = n_nodes
        self.encoder = AdaptiveRSEncoder(n_nodes)
        self.nodes = [UAVNode(i, cache_per_node, storage_per_node, self) for i in range(n_nodes)]
        self.blocks: Dict[int, Tuple[bytes, float, int]] = {}  # block_id: (data, access_freq, k)
        self.next_block_id = 1
        self.cache_per_node = cache_per_node
        self.storage_per_node = storage_per_node
        self.default_block_size = block_size
        self.stats = {
            'total_blocks': 0,
            'encoded_blocks': 0,
            'total_data_size': 0,
            'access_counts': defaultdict(int)
        }

    def load_blocks_from_csv(self, csv_file: str):
        """Load block metadata from CSV."""
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            block_id = int(row['block_id'])
            frequency = float(row['frequency_per_sec'])
            data = bytes(random.getrandbits(8) for _ in range(self.default_block_size))
            k, fragments = self.encoder.adaptive_encode(data, frequency)

            self.blocks[block_id] = (data, frequency, k)
            self.stats['total_blocks'] += 1
            self.stats['total_data_size'] += self.default_block_size
            self.stats['encoded_blocks'] += 1

            for i, node in enumerate(self.nodes):
                node.store_fragment(block_id, fragments[i])


class CacheReplacementEnv(gym.Env):
    """RL environment with integrated UAV network."""

    def __init__(self,
                 block_csv: str,
                 access_csv: str,
                 n_nodes=10,
                 block_size=1024 * 1024,  # 1MB
                 cache_per_node=1024 * 1024 * 10,  # 10MB
                 storage_per_node=1024 * 1024 * 50,  # 50MB
                 delay_hit=1.0,
                 delay_miss=10.0,
                 energy_hit=0.1,
                 energy_miss=1.0,
                 test_mode=False):

        # Init UAV network
        self.network = UAVNetwork(
            n_nodes=n_nodes,
            block_size=block_size,
            cache_per_node=cache_per_node,
            storage_per_node=storage_per_node
        )
        self.network.load_blocks_from_csv(block_csv)

        # Load access trace
        self.access_df = pd.read_csv(access_csv)['block_id'].astype(int)
        self.current_step = 0

        # Performance parameters
        self.delay_hit = delay_hit
        self.delay_miss = delay_miss
        self.energy_hit = energy_hit
        self.energy_miss = energy_miss
        self.test_mode = test_mode

        # State: local metrics + per-node network metrics
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(4 + self.network.n_nodes * 3,),  # 4 base + 3 per node
            dtype=np.float32
        )

        # Action: evict slot + block to add
        self.action_space = spaces.MultiDiscrete([
            cache_per_node // block_size,  # evict index
            len(self.network.blocks)  # new block id
        ])

        # Episode statistics
        self.avg_delay = 0.0
        self.hit_count = 0
        self.total_access = 0
        self.energy = 0.0
        self.hit_types = []  # hit type per access
        self.access_latencies = []  # latency per access

    def reset(self):
        """Reset environment."""
        self.current_step = 0
        self.avg_delay = 0.0
        self.hit_count = 0
        self.total_access = 0
        self.energy = 0.0
        self.hit_types = []
        self.access_latencies = []

        # Reset node caches
        for node in self.network.nodes:
            node.cache.clear()
            valid_blocks = list(self.network.blocks.keys())
            cache_blocks = np.random.choice(
                valid_blocks,
                size=min(node.cache_capacity // self.network.default_block_size, len(valid_blocks)),
                replace=False
            )
            for block_id in cache_blocks:
                node.cache_block(block_id, self.network.blocks[block_id][0])

        return self._get_state()

    def _get_state(self):
        """Current observation."""
        if self.current_step >= len(self.access_df):
            current_block = 0
            current_freq = 0.0
        else:
            current_block = int(self.access_df.iloc[self.current_step])
            current_freq = self.network.blocks[current_block][1] if current_block in self.network.blocks else 0.0

        # Per-node hit rate, storage utilization, normalized avg latency
        network_state = []
        for node in self.network.nodes:
            stats = node.get_stats()
            network_state.extend([
                stats['cache_hit_rate'],
                stats['storage_utilization'],
                stats['avg_access_time'] / 100  # normalize
            ])

        return np.concatenate([
            [self.avg_delay / max(self.delay_miss, 1e-6)],
            [self.hit_count / max(self.total_access, 1)],
            [self.energy / max(self.energy_miss, 1e-6)],
            [current_freq],
            np.array(network_state, dtype=np.float32)
        ])

    def step(self, action):
        """Apply action and advance one access."""
        if self.current_step >= len(self.access_df):
            return self._get_state(), 0.0, True, {}

        try:
            # 1. Current access block
            block_to_access = int(self.access_df.iloc[self.current_step])
            freq = self.network.blocks[block_to_access][1] if block_to_access in self.network.blocks else 0.0

            # 2. Random accessing node
            node = random.choice(self.network.nodes)
            hit, latency, hit_type = node.access_block(block_to_access)

            # Record access outcome
            self.hit_types.append(hit_type)
            self.access_latencies.append(latency)

            # 3. Update aggregates
            self.total_access += 1
            self.avg_delay = (self.avg_delay * (self.total_access - 1) + latency) / self.total_access
            if hit:
                self.hit_count += 1
                energy_used = self.energy_hit
            else:
                energy_used = self.energy_miss

            # 4. Cache replacement on miss
            if not hit and action is not None:
                for node in self.network.nodes:
                    if node._cache_size() >= node.cache_capacity * 0.9:  # replace when cache ~full
                        cached_blocks = list(node.cache.keys())
                        if action[0] < len(cached_blocks) and 0 <= action[1] < len(self.network.blocks):
                            block_to_remove = cached_blocks[action[0]]
                            new_block_id = list(self.network.blocks.keys())[action[1]]
                            node.evict_cache(block_to_remove)
                            node.cache_block(new_block_id, self.network.blocks[new_block_id][0])
                            energy_used += 0.5  # replacement energy

            self.energy += energy_used

            # 5. Reward
            reward = 1.0 * hit - 0.2 * latency - 0.1 * energy_used

            # 6. Next step
            self.current_step += 1
            done = self.current_step >= len(self.access_df)

            return self._get_state(), reward, done, {}

        except Exception as e:
            print(f"[Env Error] Step {self.current_step}: {str(e)}")
            return self._get_state(), 0.0, True, {"error": str(e)}

    def get_performance_metrics(self):
        """Aggregate performance metrics."""
        node_stats = [node.get_stats() for node in self.network.nodes]

        total_local_hits = sum(stats['local_hits'] for stats in node_stats)
        total_remote_hits = sum(stats['remote_hits'] for stats in node_stats)
        total_recovered = sum(stats['recovered'] for stats in node_stats)
        total_misses = sum(stats['cache_misses'] for stats in node_stats)
        total_accesses = total_local_hits + total_remote_hits + total_recovered + total_misses

        return {
            'cache_hit_rate': (total_local_hits + total_remote_hits) / total_accesses if total_accesses > 0 else 0,
            'local_hit_rate': total_local_hits / total_accesses if total_accesses > 0 else 0,
            'remote_hit_rate': total_remote_hits / total_accesses if total_accesses > 0 else 0,
            'recovery_rate': total_recovered / total_accesses if total_accesses > 0 else 0,
            'miss_rate': total_misses / total_accesses if total_accesses > 0 else 0,
            'avg_access_time': sum(stats['avg_access_time'] for stats in node_stats) / len(node_stats),
            'avg_storage_util': sum(stats['storage_utilization'] for stats in node_stats) / len(node_stats),
            'hit_types': self.hit_types,
            'access_latencies': self.access_latencies
        }