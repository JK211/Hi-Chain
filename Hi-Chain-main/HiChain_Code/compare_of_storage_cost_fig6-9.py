import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

curr_path = os.path.dirname(os.path.abspath(__file__))

# Align with data_generator003.BlockDatasetGenerator defaults (n_nodes, shard size)
REF_NODE_COUNT = 50
NODES_PER_SHARD = 10
BLOCK_SIZE_MB = 1
STORAGE_LIMIT_MB = 10000
# Blocks to simulate: None = all rows in CSV
MAX_BLOCKS_SIM = None
# Line/history CSV: record every N successful writes (+ block 1 and final point)
STORAGE_HISTORY_RECORD_INTERVAL = 1000


class StorageSimulator:
    def __init__(self, node_count=REF_NODE_COUNT, block_size=BLOCK_SIZE_MB, storage_limit=STORAGE_LIMIT_MB):
        self.node_count = node_count
        self.block_size = block_size  # MB
        self.storage_limit = storage_limit  # MB
        self.storage = np.zeros(node_count)
        self.total_storage = 0
        self.storage_history = []

    def reset(self):
        self.storage = np.zeros(self.node_count)
        self.total_storage = 0
        self.storage_history = []

    def store_block(self, block_id):
        raise NotImplementedError

    def update_history(self, block_count, record_interval=STORAGE_HISTORY_RECORD_INTERVAL):
        """Record every record_interval blocks to thin line-plot points."""
        if block_count == 1 or block_count % record_interval == 0:
            self.storage_history.append({
                'block_count': block_count,
                'total': self.total_storage,
                'per_node': np.mean(self.storage),
                'max_node': np.max(self.storage)
            })


class FullNode(StorageSimulator):
    """Full node: each node stores the full block."""

    def store_block(self, block_id, block_count):
        cost = self.block_size * self.node_count
        if np.any(self.storage + self.block_size > self.storage_limit):
            return False

        self.storage += self.block_size
        self.total_storage += cost
        self.update_history(block_count)
        return True


class LightNode(StorageSimulator):
    """Light node: 40% full blocks, 60% headers only."""

    def __init__(self, node_count=REF_NODE_COUNT, block_size=BLOCK_SIZE_MB, storage_limit=STORAGE_LIMIT_MB):
        super().__init__(node_count, block_size, storage_limit)
        self.full_nodes = int(node_count * 0.4)
        self.header_size = 0.01  # MB

    def store_block(self, block_id, block_count):
        full_cost = self.block_size * self.full_nodes
        light_cost = self.header_size * (self.node_count - self.full_nodes)
        total_cost = full_cost + light_cost

        full_storage = self.storage[:self.full_nodes] + self.block_size
        light_storage = self.storage[self.full_nodes:] + self.header_size
        if np.any(full_storage > self.storage_limit) or np.any(light_storage > self.storage_limit):
            return False

        self.storage[:self.full_nodes] += self.block_size
        self.storage[self.full_nodes:] += self.header_size
        self.total_storage += total_cost
        self.update_history(block_count)
        return True


class Sharding(StorageSimulator):
    """Sharding: shards of nodes_per_shard UAVs each."""

    def __init__(self, node_count=REF_NODE_COUNT, block_size=BLOCK_SIZE_MB, storage_limit=STORAGE_LIMIT_MB,
                 nodes_per_shard=NODES_PER_SHARD):
        super().__init__(node_count, block_size, storage_limit)
        self.nodes_per_shard = nodes_per_shard
        self.shards = node_count // nodes_per_shard  # number of shards from node count

    def store_block(self, block_id, block_count):
        shard_id = block_id % self.shards
        start = shard_id * self.nodes_per_shard
        end = start + self.nodes_per_shard
        cost = self.block_size * self.nodes_per_shard

        if np.any(self.storage[start:end] + self.block_size > self.storage_limit):
            return False

        self.storage[start:end] += self.block_size
        self.total_storage += cost
        self.update_history(block_count)
        return True


class StaticEncoding(StorageSimulator):
    """Static RS encoding."""

    def __init__(self, node_count=REF_NODE_COUNT, block_size=BLOCK_SIZE_MB, storage_limit=STORAGE_LIMIT_MB, k=None):
        super().__init__(node_count, block_size, storage_limit)
        self.k = k  # set in main to min k from CSV
        self.fragment_size = None  # computed after set_k

    def set_k(self, k):
        self.k = k
        self.fragment_size = self.block_size / k

    def store_block(self, block_id, block_count):
        if self.fragment_size is None:
            return False  # k not set yet

        cost = self.fragment_size * self.node_count
        if np.any(self.storage + self.fragment_size > self.storage_limit):
            return False

        self.storage += self.fragment_size
        self.total_storage += cost
        self.update_history(block_count)
        return True


class DynamicEncoding(StorageSimulator):
    """Dynamic RS encoding (per-block k from CSV)."""

    def __init__(self, node_count=REF_NODE_COUNT, block_size=BLOCK_SIZE_MB, storage_limit=STORAGE_LIMIT_MB,
                 default_k=5):
        super().__init__(node_count, block_size, storage_limit)
        self.block_k_values = {}  # per-block k values
        self.default_k = max(1, int(round(float(default_k))))

    def set_block_k_values(self, block_k_dict):
        self.block_k_values = block_k_dict

    def store_block(self, block_id, block_count):
        # use per-block k from CSV
        k = self.block_k_values.get(block_id, self.default_k)
        fragment_size = self.block_size / k
        cost = fragment_size * self.node_count

        if np.any(self.storage + fragment_size > self.storage_limit):
            return False

        self.storage += fragment_size
        self.total_storage += cost
        self.update_history(block_count)
        return True


def run_simulation(block_data, schemes, max_blocks=100):
    results = {}
    for name, scheme in schemes.items():
        scheme.reset()
        block_count = 0
        for block_id, freq in sorted(block_data.items(), key=lambda x: x[0]):
            if block_count >= max_blocks:
                break
            if scheme.store_block(block_id, block_count + 1):
                block_count += 1
        interval = STORAGE_HISTORY_RECORD_INTERVAL
        if block_count > 0 and block_count % interval != 0:
            if not scheme.storage_history or scheme.storage_history[-1]['block_count'] != block_count:
                scheme.update_history(block_count)
        results[name] = scheme.storage_history
    return results


def export_data_to_csv(results, min_k, avg_k, output_dir='storage_overhead_compare',
                       ref_nodes=REF_NODE_COUNT, storage_limit_mb=STORAGE_LIMIT_MB,
                       block_size_mb=BLOCK_SIZE_MB, nodes_per_shard=NODES_PER_SHARD):
    """Export experiment data to CSV for Origin replotting."""
    
    # create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")
    
    # 1. Fig 13a total storage (line plot)
    # collect block counts
    block_counts = sorted(list(set([point['block_count'] for scheme_data in results.values() for point in scheme_data])))
    
    # DataFrame: X=block count, Y=total storage per scheme
    fig13a_data = {'Block_Count': block_counts}
    for scheme_name in results.keys():
        scheme_values = []
        for block_count in block_counts:
            # match history point for block count
            value = None
            for point in results[scheme_name]:
                if point['block_count'] == block_count:
                    value = point['total']
                    break
            scheme_values.append(value if value is not None else 0)
        fig13a_data[scheme_name] = scheme_values
    
    fig13a_df = pd.DataFrame(fig13a_data)
    fig13a_df.to_csv(f'{output_dir}/fig13a_total_storage_cost.csv', index=False)
    print(f"图13a总存储开销数据已导出到: {output_dir}/fig13a_total_storage_cost.csv")
    
    # 2. Fig 13b average per-node storage (line plot)
    fig13b_data = {'Block_Count': block_counts}
    for scheme_name in results.keys():
        scheme_values = []
        for block_count in block_counts:
            value = None
            for point in results[scheme_name]:
                if point['block_count'] == block_count:
                    value = point['per_node']
                    break
            scheme_values.append(value if value is not None else 0)
        fig13b_data[scheme_name] = scheme_values
    
    fig13b_df = pd.DataFrame(fig13b_data)
    fig13b_df.to_csv(f'{output_dir}/fig13b_average_storage_per_node.csv', index=False)
    print(f"图13b单个节点平均存储开销数据已导出到: {output_dir}/fig13b_average_storage_per_node.csv")
    
    # 3. Fig 13c storage per block vs drone count (bar)
    drone_counts = np.arange(10, 101, 10)
    fig13c_data = {'Drone_Count': drone_counts}
    
    for scheme_name in results.keys():
        scheme_values = []
        for n in drone_counts:
            if scheme_name == 'FullNode':
                cost = block_size_mb * n
            elif scheme_name == 'LightNode':
                cost = block_size_mb * (n * 0.4) + 0.01 * (n * 0.6)
            elif scheme_name == 'Sharding':
                cost = block_size_mb * nodes_per_shard
            elif scheme_name == 'StaticEncoding':
                k_scaled = min_k * (n / ref_nodes)
                cost = (block_size_mb / k_scaled) * n
            else:  # DynamicEncoding
                k_scaled = avg_k * (n / ref_nodes)
                cost = (block_size_mb / k_scaled) * n
            scheme_values.append(cost)
        fig13c_data[scheme_name] = scheme_values
    
    fig13c_df = pd.DataFrame(fig13c_data)
    fig13c_df.to_csv(f'{output_dir}/fig13c_storage_cost_per_block.csv', index=False)
    print(f"图13c单个区块存储开销数据已导出到: {output_dir}/fig13c_storage_cost_per_block.csv")
    
    # 4. Fig 13d capacity limit vs drone count (bar)
    fig13d_data = {'Drone_Count': drone_counts}
    
    for scheme_name in results.keys():
        scheme_values = []
        for n in drone_counts:
            if scheme_name == 'FullNode':
                cap = storage_limit_mb / block_size_mb
            elif scheme_name == 'LightNode':
                cap = storage_limit_mb / max(1, 0.01)
            elif scheme_name == 'Sharding':
                shards = n // nodes_per_shard
                cap = (storage_limit_mb / block_size_mb) * shards
            elif scheme_name == 'StaticEncoding':
                k_scaled = min_k * (n / ref_nodes)
                cap = storage_limit_mb * k_scaled / block_size_mb
            else:  # DynamicEncoding
                k_scaled = avg_k * (n / ref_nodes)
                cap = storage_limit_mb * k_scaled / block_size_mb
            scheme_values.append(cap)
        fig13d_data[scheme_name] = scheme_values
    
    fig13d_df = pd.DataFrame(fig13d_data)
    fig13d_df.to_csv(f'{output_dir}/fig13d_storage_capacity_limit.csv', index=False)
    print(f"图13d存储容量上限数据已导出到: {output_dir}/fig13d_storage_capacity_limit.csv")
    
    # 5. Encoding comparison (4 CSVs for 4 subplots)
    
    # 5a. Encoding total storage (subplot a)
    encoding_schemes = ['StaticEncoding', 'DynamicEncoding']
    encoding_block_counts = sorted(list(set([point['block_count'] for scheme_name in encoding_schemes if scheme_name in results for point in results[scheme_name]])))
    
    encoding_a_data = {'Block_Count': encoding_block_counts}
    for scheme_name in encoding_schemes:
        if scheme_name in results:
            scheme_values = []
            for block_count in encoding_block_counts:
                value = None
                for point in results[scheme_name]:
                    if point['block_count'] == block_count:
                        value = point['total']
                        break
                scheme_values.append(value if value is not None else 0)
            encoding_a_data[scheme_name] = scheme_values
    
    encoding_a_df = pd.DataFrame(encoding_a_data)
    encoding_a_df.to_csv(f'{output_dir}/encoding_a_total_storage_cost.csv', index=False)
    print(f"编码图a总存储开销对比数据已导出到: {output_dir}/encoding_a_total_storage_cost.csv")
    
    # 5b. Encoding avg per-node storage (subplot b)
    encoding_b_data = {'Block_Count': encoding_block_counts}
    for scheme_name in encoding_schemes:
        if scheme_name in results:
            scheme_values = []
            for block_count in encoding_block_counts:
                value = None
                for point in results[scheme_name]:
                    if point['block_count'] == block_count:
                        value = point['per_node']
                        break
                scheme_values.append(value if value is not None else 0)
            encoding_b_data[scheme_name] = scheme_values
    
    encoding_b_df = pd.DataFrame(encoding_b_data)
    encoding_b_df.to_csv(f'{output_dir}/encoding_b_average_storage_per_node.csv', index=False)
    print(f"编码图b单个节点平均存储开销对比数据已导出到: {output_dir}/encoding_b_average_storage_per_node.csv")
    
    # 5c. Encoding per-block cost vs nodes (subplot c, bar)
    node_counts = np.arange(10, 101, 10)
    encoding_c_data = {'Node_Count': node_counts}
    
    for scheme_name in encoding_schemes:
        scheme_values = []
        for n in node_counts:
            if scheme_name == 'StaticEncoding':
                k_scaled = min_k * (n / ref_nodes)
                cost = (block_size_mb / k_scaled) * n
            else:  # DynamicEncoding
                k_scaled = avg_k * (n / ref_nodes)
                cost = (block_size_mb / k_scaled) * n
            scheme_values.append(cost)
        encoding_c_data[scheme_name] = scheme_values
    
    encoding_c_df = pd.DataFrame(encoding_c_data)
    encoding_c_df.to_csv(f'{output_dir}/encoding_c_storage_cost_per_block.csv', index=False)
    print(f"编码图c单个区块存储开销对比数据已导出到: {output_dir}/encoding_c_storage_cost_per_block.csv")
    
    # 5d. Encoding efficiency (subplot d, bar)
    encoding_d_data = {'Node_Count': node_counts}
    
    for scheme_name in encoding_schemes:
        scheme_values = []
        for n in node_counts:
            if scheme_name == 'StaticEncoding':
                k_scaled = min_k * (n / ref_nodes)
                blocks_per_mb = k_scaled
            else:  # DynamicEncoding
                k_scaled = avg_k * (n / ref_nodes)
                blocks_per_mb = k_scaled
            scheme_values.append(blocks_per_mb)
        encoding_d_data[scheme_name] = scheme_values
    
    encoding_d_df = pd.DataFrame(encoding_d_data)
    encoding_d_df.to_csv(f'{output_dir}/encoding_d_storage_efficiency.csv', index=False)
    print(f"编码图d存储效率对比数据已导出到: {output_dir}/encoding_d_storage_efficiency.csv")
    
    # 6. Experiment parameters
    last_bc = 0
    for _name, hist in results.items():
        if hist:
            last_bc = max(last_bc, hist[-1]['block_count'])
    experiment_info = {
        'Parameter': ['Node_Count', 'Block_Size_MB', 'Storage_Limit_MB', 'Min_K_Value', 'Avg_K_Value',
                      'Ref_Nodes_For_K_Scaling', 'Nodes_Per_Shard', 'Blocks_Recorded_Snapshots', 'Last_Block_Count'],
        'Value': [ref_nodes, block_size_mb, storage_limit_mb, min_k, avg_k, ref_nodes, nodes_per_shard,
                  len(results[list(results.keys())[0]]), last_bc]
    }
    
    info_df = pd.DataFrame(experiment_info)
    info_df.to_csv(f'{output_dir}/experiment_parameters.csv', index=False)
    print(f"实验参数信息已导出到: {output_dir}/experiment_parameters.csv")
    
    print(f"\n所有数据已成功导出到 {output_dir} 文件夹中！")
    print("CSV文件说明:")
    print("- fig13a_total_storage_cost.csv: 图13a总存储开销数据 (线图)")
    print("- fig13b_average_storage_per_node.csv: 图13b单个节点平均存储开销数据 (线图)")
    print("- fig13c_storage_cost_per_block.csv: 图13c单个区块存储开销数据 (柱状图)")
    print("- fig13d_storage_capacity_limit.csv: 图13d存储容量上限数据 (柱状图)")
    print("- encoding_comparison_total_storage.csv: 编码方案对比数据 (线图)")
    print("- experiment_parameters.csv: 实验参数信息")
    print("\n数据格式说明:")
    print("- 每个CSV文件对应一个子图")
    print("- 第一列为X轴数据 (Block_Count 或 Drone_Count)")
    print("- 后续列为各方案的Y轴数据")
    print("- 可直接导入Origin进行绘图")


def plot_results(results, fig_prefix='fig13', min_k=5, avg_k=5.0,
                 ref_nodes=REF_NODE_COUNT, storage_limit_mb=STORAGE_LIMIT_MB,
                 block_size_mb=BLOCK_SIZE_MB, nodes_per_shard=NODES_PER_SHARD):
    schemes = list(results.keys())

    # 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Storage Overhead Comparison in Drone Swarm', fontsize=16, y=0.98)

    # Fig 13a: swarm total storage vs block count
    for name in schemes:
        data = [x['total'] for x in results[name]]
        block_counts = [x['block_count'] for x in results[name]]
        ax1.plot(block_counts, data, label=name, marker='o', markersize=3)
    ax1.set_xlabel('Number of Blocks')
    ax1.set_ylabel('Total Storage Cost (MB)')
    ax1.set_title('(a) Total Storage Cost in Drone Swarm')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Fig 13b: avg storage per drone vs block count
    for name in schemes:
        data = [x['per_node'] for x in results[name]]
        block_counts = [x['block_count'] for x in results[name]]
        ax2.plot(block_counts, data, label=name, marker='s', markersize=3)
    ax2.set_xlabel('Number of Blocks')
    ax2.set_ylabel('Average Storage Cost per Drone (MB)')
    ax2.set_title('(b) Average Storage Cost per Drone')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Fig 13c: storage per block vs drone count (bar)
    drone_counts = np.arange(10, 101, 10)  # drone counts 10..100
    bar_width = 8  # bar width
    x_positions = {}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, name in enumerate(schemes):
        costs = []
        for n in drone_counts:
            if name == 'FullNode':
                cost = block_size_mb * n
            elif name == 'LightNode':
                cost = block_size_mb * (n * 0.4) + 0.01 * (n * 0.6)
            elif name == 'Sharding':
                cost = block_size_mb * nodes_per_shard
            elif name == 'StaticEncoding':
                k_scaled = min_k * (n / ref_nodes)
                cost = (block_size_mb / k_scaled) * n
            else:  # DynamicEncoding
                k_scaled = avg_k * (n / ref_nodes)
                cost = (block_size_mb / k_scaled) * n
            costs.append(cost)

        # x offset per scheme
        offset = (i - len(schemes) / 2 + 0.5) * bar_width / len(schemes)
        x_pos = drone_counts + offset
        ax3.bar(x_pos, costs, width=bar_width / len(schemes), label=name,
                color=colors[i % len(colors)], alpha=0.8)

    ax3.set_xlabel('Number of Drones')
    ax3.set_ylabel('Storage Cost per Block (MB)')
    ax3.set_title('(c) Storage Cost per Block in Swarm')
    ax3.set_xticks(drone_counts)  # show all x ticks
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Fig 13d: capacity limit vs drone count (bar)
    for i, name in enumerate(schemes):
        capacities = []
        for n in drone_counts:
            if name == 'FullNode':
                cap = storage_limit_mb / block_size_mb
            elif name == 'LightNode':
                cap = storage_limit_mb / max(1, 0.01)
            elif name == 'Sharding':
                shards = n // nodes_per_shard
                cap = (storage_limit_mb / block_size_mb) * shards
            elif name == 'StaticEncoding':
                k_scaled = min_k * (n / ref_nodes)
                cap = storage_limit_mb * k_scaled / block_size_mb
            else:  # DynamicEncoding
                k_scaled = avg_k * (n / ref_nodes)
                cap = storage_limit_mb * k_scaled / block_size_mb
            capacities.append(cap)

        # x offset per scheme
        offset = (i - len(schemes) / 2 + 0.5) * bar_width / len(schemes)
        x_pos = drone_counts + offset
        ax4.bar(x_pos, capacities, width=bar_width / len(schemes), label=name,
                color=colors[i % len(colors)], alpha=0.8)

    ax4.set_xlabel('Number of Drones')
    ax4.set_ylabel('Number of Blocks')
    ax4.set_title('(d) Drone Storage Capacity Limit')
    ax4.set_xticks(drone_counts)  # show all x ticks
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # tighten layout
    plt.tight_layout()

    # save combined figure
    plt.savefig(f'{fig_prefix}_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"组合图已保存为 {fig_prefix}_combined.png")


def plot_encoding_comparison(results, min_k=5, avg_k=5.0, fig_prefix='encoding_comparison',
                               ref_nodes=REF_NODE_COUNT, block_size_mb=BLOCK_SIZE_MB):
    """Compare static vs dynamic encoding only."""

    # encoding schemes only
    encoding_results = {
        'Static Encoding': results.get('StaticEncoding', []),
        'Dynamic Encoding': results.get('DynamicEncoding', [])
    }

    # 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Static vs Dynamic Encoding Comparison', fontsize=16, y=0.98)

    colors = {'Static Encoding': '#2E86AB', 'Dynamic Encoding': '#A23B72'}
    markers = {'Static Encoding': 'o', 'Dynamic Encoding': 's'}

    # subplot 1: total storage
    for name, color in colors.items():
        if encoding_results[name]:
            data = [x['total'] for x in encoding_results[name]]
            block_counts = [x['block_count'] for x in encoding_results[name]]
            ax1.plot(block_counts, data, label=name, color=color,
                     marker=markers[name], markersize=4, linewidth=2)
    ax1.set_xlabel('Number of Blocks')
    ax1.set_ylabel('Total Storage Cost (MB)')
    ax1.set_title('(a) Total Storage Cost Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # subplot 2: avg per-node storage
    for name, color in colors.items():
        if encoding_results[name]:
            data = [x['per_node'] for x in encoding_results[name]]
            block_counts = [x['block_count'] for x in encoding_results[name]]
            ax2.plot(block_counts, data, label=name, color=color,
                     marker=markers[name], markersize=4, linewidth=2)
    ax2.set_xlabel('Number of Blocks')
    ax2.set_ylabel('Average Storage Cost per Node (MB)')
    ax2.set_title('(b) Average Storage Cost per Node')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # subplot 3: per-block cost vs nodes (bar)
    node_counts = np.arange(10, 101, 10)
    bar_width = 8

    for i, (name, color) in enumerate(colors.items()):
        costs = []
        for n in node_counts:
            if name == 'Static Encoding':
                k_scaled = min_k * (n / ref_nodes)
                cost = (block_size_mb / k_scaled) * n
            else:  # Dynamic Encoding
                k_scaled = avg_k * (n / ref_nodes)
                cost = (block_size_mb / k_scaled) * n
            costs.append(cost)

        # x offset for bars
        offset = (i - len(colors) / 2 + 0.5) * bar_width / len(colors)
        x_pos = node_counts + offset
        ax3.bar(x_pos, costs, width=bar_width / len(colors), label=name,
                color=color, alpha=0.8)

    ax3.set_xlabel('Number of Nodes')
    ax3.set_ylabel('Storage Cost per Block (MB)')
    ax3.set_title('(c) Storage Cost per Block vs Node Count')
    ax3.set_xticks(node_counts)  # show all x ticks
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # subplot 4: blocks per MB (bar)
    for i, (name, color) in enumerate(colors.items()):
        efficiencies = []
        for n in node_counts:
            if name == 'Static Encoding':
                k_scaled = min_k * (n / ref_nodes)
                blocks_per_mb = k_scaled
            else:  # Dynamic Encoding
                k_scaled = avg_k * (n / ref_nodes)
                blocks_per_mb = k_scaled
            efficiencies.append(blocks_per_mb)

        # x offset for bars
        offset = (i - len(colors) / 2 + 0.5) * bar_width / len(colors)
        x_pos = node_counts + offset
        ax4.bar(x_pos, efficiencies, width=bar_width / len(colors), label=name,
                color=color, alpha=0.8)

    ax4.set_xlabel('Number of Nodes')
    ax4.set_ylabel('Blocks per MB (Compression Efficiency)')
    ax4.set_title('(d) Storage Efficiency Comparison')
    ax4.set_xticks(node_counts)  # show all x ticks
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # tighten layout
    plt.tight_layout()

    # save figure
    plt.savefig(f'{fig_prefix}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"编码对比图已保存为 {fig_prefix}.png")
    print(f"Static Encoding k值: {min_k}")
    print(f"Dynamic Encoding 平均k值: {avg_k:.2f}")


if __name__ == "__main__":
    # Load blocks from data_generator003 block_frequencies.csv
    csv_path = os.path.join(curr_path, 'datasets', 'block_frequencies.csv')
    try:
        block_data_df = pd.read_csv(csv_path)

        # {block_id: frequency} dict
        block_dict = {}
        block_k_dict = {}
        
        # build dicts from DataFrame
        if isinstance(block_data_df, pd.DataFrame):
            for _, row in block_data_df.iterrows():
                block_id = int(row['block_id'])
                frequency = float(row['frequency_per_sec'])
                k_value = int(round(float(row['k'])))
                block_dict[block_id] = frequency
                block_k_dict[block_id] = k_value

            # min/mean k for encoding schemes
            k_values = block_data_df['k'].to_numpy()
            min_k = int(np.min(k_values))
            avg_k = float(np.mean(k_values))
        else:
            raise ValueError("Failed to read CSV as DataFrame")

    except FileNotFoundError:
        print(f"CSV 未找到: {csv_path}，使用占位数据（请先运行 data_generator003.py）")
        n_placeholder = min(100, REF_NODE_COUNT)
        block_dict = {i: 1.0 for i in range(n_placeholder)}
        block_k_dict = {i: 5 for i in range(n_placeholder)}
        min_k = 5
        avg_k = 5.0

    max_blocks = len(block_dict) if MAX_BLOCKS_SIM is None else min(int(MAX_BLOCKS_SIM), len(block_dict))

    # five storage schemes
    static_encoding = StaticEncoding()
    static_encoding.set_k(min_k)  # min k from CSV

    dynamic_encoding = DynamicEncoding(default_k=avg_k)
    dynamic_encoding.set_block_k_values(block_k_dict)

    schemes = {
        'FullNode': FullNode(),
        'LightNode': LightNode(),
        'Sharding': Sharding(),
        'StaticEncoding': static_encoding,
        'DynamicEncoding': dynamic_encoding
    }

    results = run_simulation(block_dict, schemes, max_blocks=max_blocks)

    # export CSV
    export_data_to_csv(results, min_k, avg_k)

    # plot all schemes
    plot_results(results, min_k=min_k, avg_k=avg_k)

    # plot encoding comparison
    plot_encoding_comparison(results, min_k=min_k, avg_k=avg_k)

    print("实验完成！")
    print("所有方案对比图已保存为fig13_combined.png")
    print("编码方案对比图已保存为encoding_comparison.png")
    print(f"使用的最小k值: {min_k}")
    print(f"使用的平均k值: {avg_k:.2f}")
    done_bc = 0
    if results:
        h0 = next(iter(results.values()))
        if h0:
            done_bc = h0[-1]['block_count']
    print(f"CSV 区块条数: {len(block_dict)}；本仿真成功写入区块数: {done_bc}（上限 max_blocks={max_blocks}）")
