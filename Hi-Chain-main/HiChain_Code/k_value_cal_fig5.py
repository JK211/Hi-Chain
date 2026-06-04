# visualize_k_distribution.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

# Legacy CJK font support for matplotlib labels
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_export_folder():
    """Create export folder."""
    folder_name = 'dataset_k_distribute'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"创建文件夹: {folder_name}")
    return folder_name

def export_k_distribution_data(df, export_folder):
    """Export k distribution to CSV."""
    try:
        # k distribution
        k_values = df['k'].values
        unique_k_values = np.unique(k_values)
        k_counts = Counter(k_values)
        
        # build k distribution table
        k_distribution_data = []
        for k in sorted(unique_k_values):
            count = k_counts[k]
            percentage = count / len(k_values) * 100
            k_distribution_data.append({
                'K值': k,
                '区块数量': count,
                '百分比': round(percentage, 2)
            })
        
        # write CSV
        k_distribution_df = pd.DataFrame(k_distribution_data)
        k_distribution_file = os.path.join(export_folder, 'k_distribution_data.csv')
        k_distribution_df.to_csv(k_distribution_file, index=False, encoding='utf-8-sig')
        print(f"K值分布数据已导出到: {k_distribution_file}")
        
        return k_distribution_df
        
    except Exception as e:
        print(f"导出K值分布数据时发生错误: {e}")
        return None

def export_frequency_k_relationship(df, export_folder):
    """Export frequency–k relationship to CSV."""
    try:
        # frequency vs k
        freq_k_data = df.groupby('frequency_per_sec')['k'].first().reset_index()
        freq_k_data = freq_k_data.sort_values('frequency_per_sec')
        
        # block count per frequency bin
        freq_k_data['区块数量'] = freq_k_data['frequency_per_sec'].apply(
            lambda freq: len(df[df['frequency_per_sec'] == freq])
        )
        
        # rename for Origin
        freq_k_data = freq_k_data.rename(columns={
            'frequency_per_sec': '访问频率_次每秒',
            'k': '对应的K值',
            '区块数量': '区块数量'
        })
        
        # write CSV
        freq_k_file = os.path.join(export_folder, 'frequency_k_relationship.csv')
        freq_k_data.to_csv(freq_k_file, index=False, encoding='utf-8-sig')
        print(f"频率-K值关系数据已导出到: {freq_k_file}")
        
        return freq_k_data
        
    except Exception as e:
        print(f"导出频率-K值关系数据时发生错误: {e}")
        return None

def export_detailed_block_data(df, export_folder):
    """Export detailed block table to CSV."""
    try:
        # all original columns
        detailed_data = df.copy()
        
        # rename existing columns for Origin
        rename_map = {
            'block_id': '区块ID',
            'frequency_per_sec': '访问频率_次每秒',
            'k': 'K值',
        }
        if 'timestamp' in detailed_data.columns:
            rename_map['timestamp'] = '时间戳'
        detailed_data = detailed_data.rename(columns=rename_map)
        
        # write CSV
        detailed_file = os.path.join(export_folder, 'detailed_block_data.csv')
        detailed_data.to_csv(detailed_file, index=False, encoding='utf-8-sig')
        print(f"详细区块数据已导出到: {detailed_file}")
        
        return detailed_data
        
    except Exception as e:
        print(f"导出详细区块数据时发生错误: {e}")
        return None

def analyze_k_distribution():
    """Analyze and visualize k distribution."""
    try:
        # load CSV
        df = pd.read_csv('datasets/block_frequencies.csv')
        print("数据加载成功！")
        print(f"总区块数: {len(df)}")
        print("\n数据预览:")
        print(df.head(10))
        
        # export folder
        export_folder = create_export_folder()
        
        # export CSVs
        print(f"\n=== 开始导出数据到 {export_folder} 文件夹 ===")
        
        # 1. k distribution
        k_distribution_df = export_k_distribution_data(df, export_folder)
        
        # 2. frequency–k
        freq_k_df = export_frequency_k_relationship(df, export_folder)
        
        # 3. detailed blocks
        detailed_df = export_detailed_block_data(df, export_folder)
        
        print(f"\n=== 数据导出完成 ===")
        print(f"所有数据文件已保存到 {export_folder} 文件夹中")
        print("可在Origin中使用以下文件进行绘图:")
        print("1. k_distribution_data.csv - K值分布数据")
        print("2. frequency_k_relationship.csv - 频率与K值关系数据")
        print("3. detailed_block_data.csv - 详细区块数据")
        
        # k distribution
        k_values = df['k'].values
        unique_k_values = np.unique(k_values)
        k_counts = Counter(k_values)
        
        print(f"\n=== K值分布统计 ===")
        print(f"K值范围: {min(k_values)} - {max(k_values)}")
        print(f"不同K值数量: {len(unique_k_values)}")
        print("\n各K值对应的区块数量:")
        for k in sorted(unique_k_values):
            count = k_counts[k]
            percentage = count / len(k_values) * 100
            print(f"K={k}: {count}个区块 ({percentage:.1f}%)")
        
        # frequency vs k summary (Zipf bins; print head/tail only)
        print(f"\n=== 频率与K值关系（摘要） ===")
        uniq_freq = np.sort(df['frequency_per_sec'].unique())
        n_freq = len(uniq_freq)
        print(f"不同频率档位数: {n_freq}（频率范围约 {uniq_freq.min():.0f}–{uniq_freq.max():.0f} 次/s）")
        freq_k_pairs = (
            df.groupby('frequency_per_sec', as_index=False)
            .agg(k=('k', 'first'), n_blocks=('block_id', 'count'))
            .sort_values('frequency_per_sec')
        )
        head = freq_k_pairs.head(5)
        tail = freq_k_pairs.tail(5)
        print("最低频率若干档: ")
        for _, row in head.iterrows():
            print(f"  频率 {row['frequency_per_sec']:.0f} 次/s -> K={row['k']} ({int(row['n_blocks'])} 个区块)")
        print("最高频率若干档: ")
        for _, row in tail.iterrows():
            print(f"  频率 {row['frequency_per_sec']:.0f} 次/s -> K={row['k']} ({int(row['n_blocks'])} 个区块)")
        
        # figures: subplots 1 and 3
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. k distribution bar chart
        sorted_k_values = sorted(unique_k_values)
        k_block_counts = [k_counts[k] for k in sorted_k_values]
        
        # even bar spacing
        x_positions = range(len(sorted_k_values))
        bars1 = ax1.bar(x_positions, k_block_counts, alpha=0.7, color='skyblue', edgecolor='black', width=0.8)
        ax1.set_xlabel('K值')
        ax1.set_ylabel('区块数量')
        ax1.set_title('K值分布条形图')
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels([f'K={k}' for k in sorted_k_values])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # bar value labels
        for bar, count in zip(bars1, k_block_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(k_block_counts)*0.01, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # 3. frequency vs k (Zipf tiers from data_generator003);
        #    line/scatter per unique frequency tier.
        freq_k_data = (
            df.groupby('frequency_per_sec', as_index=False)
            .agg(k=('k', 'first'))
            .sort_values('frequency_per_sec')
        )
        fx = freq_k_data['frequency_per_sec'].values
        ky = freq_k_data['k'].values
        n_pts = len(freq_k_data)
        if n_pts <= 40:
            if len(fx) > 1:
                bar_w = max(0.5, 0.85 * np.min(np.diff(fx)))
            else:
                bar_w = 0.8
            ax3.bar(fx, ky, width=bar_w, align='center', alpha=0.75, color='steelblue', edgecolor='black')
        else:
            ax3.plot(fx, ky, '-', color='steelblue', linewidth=1.2, alpha=0.9)
            ax3.scatter(fx, ky, s=8, c='darkblue', alpha=0.6, zorder=3)
        ax3.set_xlabel('访问频率 (次/s)')
        ax3.set_ylabel('对应的K值')
        ax3.set_title('频率与K值对应关系（按唯一频率档）')
        if n_pts <= 40:
            ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('datasets/k_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n可视化图表已保存至: datasets/k_distribution_analysis.png")
        
        return df
        
    except FileNotFoundError:
        print("错误: 找不到 datasets/block_frequencies.csv 文件")
        print("请先运行 data_generator003.py 生成数据")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None

if __name__ == "__main__":
    df = analyze_k_distribution() 