# Train/test BDQN for cache replacement: which block to evict and which to add per access.

import os

# Work around OpenMP duplicate library issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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
import math
from typing import Tuple, Dict, List
from BDQN import BDQN
from BDQN import ReplayBuffer
from AdaptiveRSEncoder import AdaptiveRSEncoder
from CacheReplacementEnv import CacheReplacementEnv


class Trainer:
    """Trainer with visualization helpers."""

    def __init__(self, env, policy_net, target_net, buffer_capacity=10000,
                 batch_size=64, gamma=0.99, lr=0.001):
        self.env = env
        self.policy_net = policy_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Training metrics
        self.episode_rewards = []
        self.episode_hit_rates = []
        self.episode_delays = []
        self.episode_energies = []

        # Test metrics
        self.test_results = []

    def select_action(self, state, epsilon=0.0):
        """Select action (epsilon=0 → greedy)."""
        if random.random() < epsilon:
            action_remove = random.randint(0, self.env.action_space.nvec[0] - 1)  # evict index
            action_add = random.randint(0, self.env.action_space.nvec[1] - 1)  # block id to add
            return [action_remove, action_add]
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_remove, q_add = self.policy_net(state_tensor)
                return [torch.argmax(q_remove).item(), torch.argmax(q_add).item()]

    def optimize_model(self):
        """Optimize policy network."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions_remove = torch.LongTensor(actions[:, 0])
        actions_add = torch.LongTensor(actions[:, 1])
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_remove, q_add = self.policy_net(states)
        current_q = q_remove.gather(1, actions_remove.unsqueeze(1)) + \
                    q_add.gather(1, actions_add.unsqueeze(1))

        with torch.no_grad():
            next_q_remove, next_q_add = self.target_net(next_states)
            next_q = torch.min(
                torch.max(next_q_remove, dim=1)[0] + torch.max(next_q_add, dim=1)[0],
                torch.tensor(10.0)
            )
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes=500, max_steps=None, epsilon_start=0.9,
              epsilon_end=0.05, epsilon_decay=200):
        """Run training loop."""
        epsilon = epsilon_start
        metrics = {
            'rewards': [],
            'hit_rates': [],
            'delays': [],
            'energies': []
        }

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            step_count = 0

            while not done:
                action = self.select_action(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.push(state, action, reward, next_state, done)
                loss = self.optimize_model()
                state = next_state
                total_reward += reward
                step_count += 1
                if max_steps and step_count >= max_steps:
                    break

            # Record episode metrics
            metrics['rewards'].append(total_reward)
            metrics['hit_rates'].append(self.env.hit_count / self.env.total_access)
            metrics['delays'].append(self.env.avg_delay)
            metrics['energies'].append(self.env.energy)

            # Decay epsilon
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                      np.exp(-1. * episode / epsilon_decay)

            # Sync target network
            if episode % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Log progress
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {total_reward:.1f} | "
                  f"Hit Rate: {metrics['hit_rates'][-1]:.3f} | "
                  f"Avg Delay: {metrics['delays'][-1]:.2f} | "
                  f"Energy: {metrics['energies'][-1]:.2f} | "
                  f"Epsilon: {epsilon:.3f}")

        # Persist training results
        self._save_results(metrics)
        ckpt_dir = os.path.join("results", "train")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "bdqn_policy.pt")
        torch.save(self.policy_net.state_dict(), ckpt_path)
        print(f"Saved BDQN checkpoint: {ckpt_path}")
        return metrics

    def test(self, num_episodes=10):
        """Evaluate policy and build plots."""
        test_metrics = {
            'rewards': [],
            'hit_rates': [],
            'delays': [],
            'energies': [],
            'hit_type_dist': [],
            'latency_dist': [],
            'storage_utils': []
        }

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state, epsilon=0.0)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                total_reward += reward

            # Collect performance metrics
            metrics = self.env.get_performance_metrics()
            test_metrics['rewards'].append(total_reward)
            test_metrics['hit_rates'].append(metrics['cache_hit_rate'])
            test_metrics['delays'].append(metrics['avg_access_time'])
            test_metrics['energies'].append(self.env.energy)
            test_metrics['hit_type_dist'].append(metrics['hit_types'])
            test_metrics['latency_dist'].append(metrics['access_latencies'])
            test_metrics['storage_utils'].append(
                [node.get_stats()['storage_utilization'] for node in self.env.network.nodes])

            # Per-episode storage utilization
            current_storage_utils = [node.get_stats()['storage_utilization'] for node in self.env.network.nodes]
            avg_storage_util = np.mean(current_storage_utils)

            print(f"Test Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {total_reward:.1f} | "
                  f"Hit Rate: {metrics['cache_hit_rate']:.3f} | "
                  f"Avg Delay: {metrics['avg_access_time']:.2f}ms | "
                  f"Energy: {self.env.energy:.2f} | "
                  f"Avg Storage Util: {avg_storage_util:.3f}")

        # Save test results
        self._save_test_results(test_metrics)

        # Storage utilization summary
        all_storage_utils = [util for ep in test_metrics['storage_utils'] for util in ep]
        print(f"\n=== 存储利用率分析 ===")
        print(f"平均存储利用率: {np.mean(all_storage_utils):.3f}")
        print(f"标准差: {np.std(all_storage_utils):.3f}")
        print(f"最小值: {np.min(all_storage_utils):.3f}")
        print(f"最大值: {np.max(all_storage_utils):.3f}")

        # Per-node average storage utilization
        n_nodes = len(test_metrics['storage_utils'][0])
        print(f"\n各节点平均存储利用率:")
        for node_idx in range(n_nodes):
            node_utils = [ep_utils[node_idx] for ep_utils in test_metrics['storage_utils']]
            print(f"节点 {node_idx}: {np.mean(node_utils):.3f} (±{np.std(node_utils):.3f})")

        return test_metrics

    def _save_test_results(self, metrics):
        """Save test metrics and figures."""
        os.makedirs("results/test", exist_ok=True)

        # 1. Save numeric results
        pd.DataFrame({
            'rewards': metrics['rewards'],
            'hit_rates': metrics['hit_rates'],
            'delays': metrics['delays'],
            'energies': metrics['energies']
        }).to_csv("results/test/test_metrics.csv", index=False)

        # 2. Plots
        plt.figure(figsize=(18, 12))

        # Access latency distribution
        plt.subplot(2, 2, 1)
        all_latencies = [lat for ep in metrics['latency_dist'] for lat in ep]
        plt.hist(all_latencies, bins=30, alpha=0.7, color='blue')
        plt.title('Access Latency Distribution')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Frequency')

        # Hit type distribution
        plt.subplot(2, 2, 2)
        hit_types = [ht for ep in metrics['hit_type_dist'] for ht in ep]
        hit_labels = ['Miss', 'Local Hit', 'Remote Hit', 'Recovered']
        hit_counts = [hit_types.count(i) for i in range(4)]
        plt.pie(hit_counts, labels=hit_labels, autopct='%1.1f%%', startangle=90)
        plt.title('Hit Type Distribution')

        # Storage utilization
        plt.subplot(2, 2, 3)
        storage_utils = [util for ep in metrics['storage_utils'] for util in ep]

        # Bar chart per node if utilization is nearly flat
        if len(set([round(util, 2) for util in storage_utils])) <= 3:
            # Mean utilization per node
            n_nodes = len(metrics['storage_utils'][0])
            avg_utils = []
            for node_idx in range(n_nodes):
                node_utils = [ep_utils[node_idx] for ep_utils in metrics['storage_utils']]
                avg_utils.append(np.mean(node_utils))

            bars = plt.bar(range(n_nodes), avg_utils, color='green', alpha=0.7)
            plt.title(f'Average Storage Utilization by Node\n(Range: {min(avg_utils):.3f} - {max(avg_utils):.3f})')
            plt.xlabel('Node ID')
            plt.ylabel('Storage Utilization')
            plt.ylim(0, 1)

            # Value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            # Histogram when utilization varies more
            plt.hist(storage_utils, bins=20, alpha=0.7, color='green')
            plt.title('Storage Utilization Distribution')
            plt.xlabel('Utilization')
            plt.ylabel('Node Count')

        # Annotate summary stats
        plt.figtext(0.02, 0.52,
                    f'Storage Stats:\n'
                    f'Mean: {np.mean(storage_utils):.3f}\n'
                    f'Std: {np.std(storage_utils):.3f}\n'
                    f'Min: {np.min(storage_utils):.3f}\n'
                    f'Max: {np.max(storage_utils):.3f}',
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

        # Latency vs hit type
        plt.subplot(2, 2, 4)
        colors = ['red', 'green', 'blue', 'orange']
        for i in range(4):
            indices = [j for j, ht in enumerate(hit_types) if ht == i]
            plt.scatter(
                [all_latencies[j] for j in indices],
                [hit_types[j] for j in indices],
                c=colors[i], alpha=0.5, label=hit_labels[i]
            )
        plt.title('Latency vs Hit Type')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Hit Type')
        plt.yticks(range(4), hit_labels)
        plt.legend()

        plt.tight_layout()
        plt.savefig('results/test/performance_visualization.png')
        plt.close()
        print("\n测试结果和可视化已保存至 results/test/ 目录")

    def _save_results(self, metrics):
        """Save training metrics and figures."""
        os.makedirs("results/train", exist_ok=True)

        # 1. Save numeric results
        pd.DataFrame({
            'episode': range(len(metrics['rewards'])),
            'rewards': metrics['rewards'],
            'hit_rates': metrics['hit_rates'],
            'delays': metrics['delays'],
            'energies': metrics['energies']
        }).to_csv("results/train/training_metrics.csv", index=False)

        # 2. Plots
        plt.figure(figsize=(12, 8))

        # Reward curve
        plt.subplot(2, 2, 1)
        plt.plot(metrics['rewards'], label='Total Reward')
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()

        # Hit rate curve
        plt.subplot(2, 2, 2)
        plt.plot(metrics['hit_rates'], label='Hit Rate')
        plt.title('Episode Hit Rates')
        plt.xlabel('Episode')
        plt.ylabel('Hit Rate')
        plt.legend()

        # Average delay curve
        plt.subplot(2, 2, 3)
        plt.plot(metrics['delays'], label='Average Delay (ms)')
        plt.title('Episode Average Delays')
        plt.xlabel('Episode')
        plt.ylabel('Delay (ms)')
        plt.legend()

        # Energy curve
        plt.subplot(2, 2, 4)
        plt.plot(metrics['energies'], label='Energy Consumption')
        plt.title('Episode Energy Consumption')
        plt.xlabel('Episode')
        plt.ylabel('Energy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('results/train/training_visualization.png')
        plt.close()
        print("\n训练结果和可视化已保存至 results/train/ 目录")


def main():
    # Hyperparameters
    config = {
        "block_csv": "datasets/block_frequencies.csv",
        "access_csv": "datasets/block_access_10000blocks_100000steps.csv",
        "n_nodes": 10,
        "block_size": 1024 * 1024,  # 1MB
        "cache_per_node": 1024 * 1024 * 20,  # 10MB
        "storage_per_node": 1024 * 1024 * 1000,  # 50MB
        "num_episodes": 200,
        "batch_size": 64,
        "buffer_capacity": 10000,
        "gamma": 0.99,
        "lr": 0.001,
        "test_episodes": 20
    }

    # Environment
    env = CacheReplacementEnv(
        block_csv=config["block_csv"],
        access_csv=config["access_csv"],
        n_nodes=config["n_nodes"],
        block_size=config["block_size"],
        cache_per_node=config["cache_per_node"],
        storage_per_node=config["storage_per_node"]
    )

    # Networks
    policy_net = BDQN(
        input_dim=env.observation_space.shape[0],
        cache_size=config["cache_per_node"] // config["block_size"],
        total_blocks=len(env.network.blocks)
    )
    target_net = BDQN(
        input_dim=env.observation_space.shape[0],
        cache_size=config["cache_per_node"] // config["block_size"],
        total_blocks=len(env.network.blocks)
    )

    # Train
    trainer = Trainer(
        env=env,
        policy_net=policy_net,
        target_net=target_net,
        buffer_capacity=config["buffer_capacity"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        lr=config["lr"]
    )

    print("\n===== 开始训练 =====")
    trainer.train(num_episodes=config["num_episodes"])

    # Test
    print("\n===== 开始测试 =====")
    trainer.test(num_episodes=config["test_episodes"])


if __name__ == "__main__":
    main()