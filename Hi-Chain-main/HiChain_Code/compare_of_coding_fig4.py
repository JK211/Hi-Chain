import os

# Work around OpenMP duplicate library issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Any
from AdaptiveRSEncoder import AdaptiveRSEncoder
import matplotlib

matplotlib.rc("font", family='YouYuan')


def test_adaptive_rs_encoding():
    """
    Test adaptive RS encoding over alpha/beta.
    100 blocks, access frequency 1..1000.
    """
    # encoder, 50 UAV nodes
    n_nodes = 50
    encoder = AdaptiveRSEncoder(n_nodes=n_nodes)

    # 100 blocks, frequencies 1..1000
    n_blocks = 100
    access_frequencies = np.linspace(1, 1000, n_blocks)

    # parameter combinations to test
    # subplots 1-2: beta=0.5, alpha in 1,5,9
    alpha_beta_combinations = [
        (1, 0.5), (5, 0.5), (9, 0.5),  # subplots 1-2
        (5, 0.1), (5, 0.5), (5, 0.9)   # subplots 3-4
    ]

    # accumulate results
    all_results = []

    print("开始测试自适应RS编码...")
    print(f"无人机节点数: {n_nodes}")
    print(f"测试区块数: {n_blocks}")
    print(f"访问频率范围: {access_frequencies[0]:.2f} - {access_frequencies[-1]:.2f}")
    print(f"k_min: {encoder.k_min}, k_max: {encoder.k_max}")
    print("-" * 60)

    for alpha, beta in alpha_beta_combinations:
        print(f"测试参数: alpha={alpha}, beta={beta}")
        
        for i, freq in enumerate(access_frequencies):
            # mock block data (~1 KB)
            block_data = f"Block_{i:03d}_data_".encode() * 100  # ~1 KB

            # adaptive_encode with given alpha, beta
            k_value, fragments = encoder.adaptive_encode(block_data, freq, alpha=alpha, beta=beta)

            # redundancy ratio
            redundancy = (n_nodes - k_value) / n_nodes

            # append row
            all_results.append({
                'block_id': i,
                'access_frequency': freq,
                'alpha': alpha,
                'beta': beta,
                'k_value': k_value,
                'redundancy': redundancy,
                'fragments_count': len(fragments)
            })

    # to DataFrame
    df = pd.DataFrame(all_results)

    # summary stats
    print("\n" + "=" * 60)
    print("测试结果统计:")
    print("=" * 60)
    
    # skip if empty
    if not df.empty:
        k_values = df['k_value']
        redundancy_values = df['redundancy']
        print(f"k值范围: {k_values.min():.0f} - {k_values.max():.0f}")
        print(f"平均k值: {k_values.mean():.2f}")
        print(f"冗余度范围: {redundancy_values.min():.2f} - {redundancy_values.max():.2f}")
        print(f"平均冗余度: {redundancy_values.mean():.2f}")
    else:
        print("警告：没有数据可供分析")

    # plot
    plot_results(df, n_nodes)

    return df


def plot_results(df, n_nodes):
    """Plot test results."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('自适应RS编码参数对比分析', fontsize=16, fontweight='bold')

    # subplots 1-2: beta=0.5
    beta_fixed = 0.5
    alphas = [1, 5, 9]
    colors = ['blue', 'red', 'green']
    
    # access frequency vs k (beta=0.5)
    for i, alpha in enumerate(alphas):
        mask = (df['alpha'] == alpha) & (df['beta'] == beta_fixed)
        subset = df[mask]
        if not subset.empty:
            ax1.plot(subset['access_frequency'], subset['k_value'], 
                    color=colors[i], linewidth=2, alpha=0.7, 
                    label=f'α={alpha}, β={beta_fixed}')
            ax1.scatter(subset['access_frequency'], subset['k_value'], 
                       c=colors[i], s=20, alpha=0.6)
    
    ax1.set_xlabel('访问频率')
    ax1.set_ylabel('k值 (所需片段数)')
    ax1.set_title('访问频率与k值的关系 (固定β=0.5)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # access frequency vs redundancy (beta=0.5)
    for i, alpha in enumerate(alphas):
        mask = (df['alpha'] == alpha) & (df['beta'] == beta_fixed)
        subset = df[mask]
        if not subset.empty:
            ax2.plot(subset['access_frequency'], subset['redundancy'], 
                    color=colors[i], linewidth=2, alpha=0.7, 
                    label=f'α={alpha}, β={beta_fixed}')
            ax2.scatter(subset['access_frequency'], subset['redundancy'], 
                       c=colors[i], s=20, alpha=0.6)
    
    ax2.set_xlabel('访问频率')
    ax2.set_ylabel('冗余度')
    ax2.set_title('访问频率与冗余度的关系 (固定β=0.5)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # subplots 3-4: alpha=5
    alpha_fixed = 5
    betas = [0.1, 0.5, 0.9]
    colors_beta = ['purple', 'orange', 'brown']
    
    # access frequency vs k (alpha=5)
    for i, beta in enumerate(betas):
        mask = (df['alpha'] == alpha_fixed) & (df['beta'] == beta)
        subset = df[mask]
        if not subset.empty:
            ax3.plot(subset['access_frequency'], subset['k_value'], 
                    color=colors_beta[i], linewidth=2, alpha=0.7, 
                    label=f'α={alpha_fixed}, β={beta}')
            ax3.scatter(subset['access_frequency'], subset['k_value'], 
                       c=colors_beta[i], s=20, alpha=0.6)
    
    ax3.set_xlabel('访问频率')
    ax3.set_ylabel('k值 (所需片段数)')
    ax3.set_title('访问频率与k值的关系 (固定α=5)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # access frequency vs redundancy (alpha=5)
    for i, beta in enumerate(betas):
        mask = (df['alpha'] == alpha_fixed) & (df['beta'] == beta)
        subset = df[mask]
        if not subset.empty:
            ax4.plot(subset['access_frequency'], subset['redundancy'], 
                    color=colors_beta[i], linewidth=2, alpha=0.7, 
                    label=f'α={alpha_fixed}, β={beta}')
            ax4.scatter(subset['access_frequency'], subset['redundancy'], 
                       c=colors_beta[i], s=20, alpha=0.6)
    
    ax4.set_xlabel('访问频率')
    ax4.set_ylabel('冗余度')
    ax4.set_title('访问频率与冗余度的关系 (固定α=5)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()


def detailed_analysis(df):
    """Detailed analysis of encoding results."""
    print("\n" + "=" * 60)
    print("详细分析报告")
    print("=" * 60)

    # per (alpha, beta)
    param_combinations = df[['alpha', 'beta']].drop_duplicates()
    
    for _, row in param_combinations.iterrows():
        alpha, beta = row['alpha'], row['beta']
        mask = (df['alpha'] == alpha) & (df['beta'] == beta)
        subset = df[mask]
        
        if not subset.empty:
            print(f"\n参数组合 α={alpha}, β={beta}:")
            print(f"  k值范围: {subset['k_value'].min()} - {subset['k_value'].max()}")
            print(f"  平均k值: {subset['k_value'].mean():.2f}")
            print(f"  冗余度范围: {subset['redundancy'].min():.2f} - {subset['redundancy'].max():.2f}")
            print(f"  平均冗余度: {subset['redundancy'].mean():.2f}")


def export_data_for_origin(df, n_nodes):
    """
    Export CSVs for Origin replotting.
    """
    import datetime
    import os
    
    # timestamp suffix
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # output folder
    output_folder = "encoding_validation"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"✓ 创建文件夹: {output_folder}")
    
    print("\n" + "=" * 60)
    print("导出数据到CSV文件 (Origin绘图专用)")
    print("=" * 60)
    
    # 1. full raw data
    raw_data_file = os.path.join(output_folder, f'adaptive_rs_raw_data_{timestamp}.csv')
    df.to_csv(raw_data_file, index=False, encoding='utf-8-sig')
    print(f"✓ 完整原始数据已保存到: {raw_data_file}")
    
    # 2. per (alpha, beta) CSV
    param_combinations = df[['alpha', 'beta']].drop_duplicates()
    
    for _, row in param_combinations.iterrows():
        alpha, beta = row['alpha'], row['beta']
        mask = (df['alpha'] == alpha) & (df['beta'] == beta)
        subset = df[mask].copy()
        
        if not subset.empty:
            # sort by frequency
            subset = subset.sort_values('access_frequency')
            
            # output path
            filename = os.path.join(output_folder, f'alpha_{alpha}_beta_{beta}_{timestamp}.csv')
            
            # columns for Origin
            export_data = subset[['access_frequency', 'k_value', 'redundancy']].copy()
            export_data.columns = ['Access_Frequency', 'K_Value', 'Redundancy']
            
            # CSV comment header
            with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
                f.write(f"# Parameters: alpha={alpha}, beta={beta}\n")
                f.write(f"# UAV nodes: {n_nodes}\n")
                f.write(f"# Data points: {len(subset)}\n")
                f.write(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("#\n")
                export_data.to_csv(f, index=False)
            
            print(f"✓ 参数组合 α={alpha}, β={beta} 数据已保存到: {filename}")
    
    # 3. summary across parameter sets
    summary_file = os.path.join(output_folder, f'adaptive_rs_summary_{timestamp}.csv')
    
    # one column group per (alpha, beta)
    summary_data = []
    access_freqs = sorted(df['access_frequency'].unique())
    
    for freq in access_freqs:
        row_data = {'Access_Frequency': freq}
        
        for _, param_row in param_combinations.iterrows():
            alpha, beta = param_row['alpha'], param_row['beta']
            mask = (df['alpha'] == alpha) & (df['beta'] == beta) & (df['access_frequency'] == freq)
            subset = df[mask]
            
            if not subset.empty:
                row_data[f'K_Value_alpha{alpha}_beta{beta}'] = subset.iloc[0]['k_value']
                row_data[f'Redundancy_alpha{alpha}_beta{beta}'] = subset.iloc[0]['redundancy']
            else:
                row_data[f'K_Value_alpha{alpha}_beta{beta}'] = None
                row_data[f'Redundancy_alpha{alpha}_beta{beta}'] = None
        
        summary_data.append(row_data)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
    print(f"✓ 汇总对比数据已保存到: {summary_file}")
    
    # 4. statistics file
    stats_file = os.path.join(output_folder, f'adaptive_rs_statistics_{timestamp}.csv')
    stats_data = []
    
    for _, param_row in param_combinations.iterrows():
        alpha, beta = param_row['alpha'], param_row['beta']
        mask = (df['alpha'] == alpha) & (df['beta'] == beta)
        subset = df[mask]
        
        if not subset.empty:
            stats_data.append({
                'Alpha': alpha,
                'Beta': beta,
                'Min_K_Value': subset['k_value'].min(),
                'Max_K_Value': subset['k_value'].max(),
                'Mean_K_Value': subset['k_value'].mean(),
                'Std_K_Value': subset['k_value'].std(),
                'Min_Redundancy': subset['redundancy'].min(),
                'Max_Redundancy': subset['redundancy'].max(),
                'Mean_Redundancy': subset['redundancy'].mean(),
                'Std_Redundancy': subset['redundancy'].std(),
                'Data_Points': len(subset)
            })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"✓ 统计信息已保存到: {stats_file}")
    
    # 5. Origin plotting guide
    readme_file = os.path.join(output_folder, f'Origin_Plotting_Guide_{timestamp}.txt')
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write("Adaptive RS encoding data - Origin plotting guide\n")
        f.write("=" * 50 + "\n\n")
        f.write("Generated files:\n")
        f.write("1. adaptive_rs_raw_data_*.csv - full raw data\n")
        f.write("2. alpha_*_beta_*.csv - per-parameter-set CSVs\n")
        f.write("3. adaptive_rs_summary_*.csv - summary across parameter sets\n")
        f.write("4. adaptive_rs_statistics_*.csv - statistics\n\n")
        
        f.write("Origin plotting tips:\n")
        f.write("- Use alpha_*_beta_*.csv for multi-line plots\n")
        f.write("- X-axis: Access_Frequency\n")
        f.write("- Y-axis: K_Value or Redundancy\n")
        f.write("- Use color/line style to distinguish parameter sets\n\n")
        
        f.write("Parameter sets:\n")
        for _, param_row in param_combinations.iterrows():
            alpha, beta = param_row['alpha'], param_row['beta']
            f.write(f"- α={alpha}, β={beta}\n")
        
        f.write(f"\nExperiment setup:\n")
        f.write(f"- UAV nodes: {n_nodes}\n")
        f.write(f"- Blocks tested: {len(df['block_id'].unique())}\n")
        f.write(f"- Access frequency range: {df['access_frequency'].min():.2f} - {df['access_frequency'].max():.2f}\n")
    
    print(f"✓ Origin绘图指南已保存到: {readme_file}")
    
    print(f"\n所有文件已导出完成！时间戳: {timestamp}")
    print(f"文件保存位置: {os.path.abspath(output_folder)}")
    print("建议使用 alpha_*_beta_*.csv 文件在Origin中创建多线图。")


if __name__ == "__main__":
    # run test
    results_df = test_adaptive_rs_encoding()

    # detailed analysis
    detailed_analysis(results_df)

    # export for Origin
    export_data_for_origin(results_df, n_nodes=50)

    print("\n测试完成!")
