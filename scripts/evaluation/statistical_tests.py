import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns

def load_results_for_statistical_test(results_dir, dataset, metric='accuracy'):
    model_files = {
        'Teacher': f'{dataset}_multimodal-teacher_evaluation_results.csv',
        'Student': f'{dataset}_multimodal-student_evaluation_results.csv',
        'MobileNetV3': f'{dataset}_baseline-mobilenetv3_evaluation_results.csv',
        'ShuffleNet': f'{dataset}_baseline-shufflenet_evaluation_results.csv',
        'AlexNet': f'{dataset}_baseline-alexnet_evaluation_results.csv',
        'EfficientNet': f'{dataset}_baseline-efficientnet-lite_evaluation_results.csv',
        'ResNet50': f'{dataset}_baseline-resnet50_evaluation_results.csv'
    }
    
    data = {}
    for model_name, filename in model_files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if metric in df.columns:
                data[model_name] = df[metric].values
            else:
                print(f"警告: {filepath} 中找不到指标 {metric}")
        else:
            print(f"警告: 找不到结果文件 {filepath}")
    
    return data

def perform_anova_test(data):
    models = list(data.keys())
    values = [data[model] for model in models]
    
    f_stat, p_value = stats.f_oneway(*values)
    
    print("ANOVA 检验结果:")
    print(f"F统计量: {f_stat:.4f}")
    print(f"P值: {p_value:.4f}")
    
    if p_value < 0.05:
        print("结果: 模型间存在显著差异 (p < 0.05)")
    else:
        print("结果: 模型间无显著差异 (p >= 0.05)")
    
    return f_stat, p_value

def perform_posthoc_test(data, test_type='tukey'):
    models = list(data.keys())
    values = []
    groups = []
    
    for i, model in enumerate(models):
        values.extend(data[model])
        groups.extend([model] * len(data[model]))
    
    df = pd.DataFrame({'value': values, 'group': groups})
    
    if test_type == 'tukey':
        posthoc_result = sp.posthoc_tukey(df, val_col='value', group_col='group')
    elif test_type == 'dunn':
        posthoc_result = sp.posthoc_dunn(df, val_col='value', group_col='group')
    else:
        raise ValueError(f"不支持的检验类型: {test_type}")
    
    print(f"\n{test_type.title()} 事后检验结果:")
    print(posthoc_result)
    
    return posthoc_result

def perform_wilcoxon_test(data, model1, model2):
    if model1 not in data or model2 not in data:
        print(f"错误: 找不到模型 {model1} 或 {model2}")
        return None
    
    stat, p_value = stats.wilcoxon(data[model1], data[model2])
    
    print(f"\nWilcoxon 符号秩检验: {model1} vs {model2}")
    print(f"统计量: {stat:.4f}")
    print(f"P值: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"结果: {model1} 和 {model2} 存在显著差异 (p < 0.05)")
    else:
        print(f"结果: {model1} 和 {model2} 无显著差异 (p >= 0.05)")
    
    return stat, p_value

def create_statistical_plots(data, output_dir):
    models = list(data.keys())
    values = []
    groups = []
    
    for model in models:
        values.extend(data[model])
        groups.extend([model] * len(data[model]))
    
    df = pd.DataFrame({'Value': values, 'Model': groups})
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='Model', y='Value')
    plt.title('Model Performance Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.violinplot(data=df, x='Model', y='Value')
    plt.title('Model Performance Distribution (Violin Plot)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_violinplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_effect_size(data, model1, model2):
    if model1 not in data or model2 not in data:
        print(f"错误: 找不到模型 {model1} 或 {model2}")
        return None
    
    cohen_d = (np.mean(data[model1]) - np.mean(data[model2])) / np.sqrt(
        (np.std(data[model1], ddof=1)**2 + np.std(data[model2], ddof=1)**2) / 2
    )
    
    print(f"\n效应量分析: {model1} vs {model2}")
    print(f"Cohen's d: {cohen_d:.4f}")
    
    if abs(cohen_d) < 0.2:
        print("效应量: 很小")
    elif abs(cohen_d) < 0.5:
        print("效应量: 小")
    elif abs(cohen_d) < 0.8:
        print("效应量: 中等")
    else:
        print("效应量: 大")
    
    return cohen_d

def main():
    parser = argparse.ArgumentParser(description='统计检验分析')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cicids2017', 'cicids2018', 'toniot'],
                       help='数据集名称')
    parser.add_argument('--results-dir', type=str, default='./',
                       help='结果文件目录')
    parser.add_argument('--output-dir', type=str, default='./statistical_analysis',
                       help='输出目录')
    parser.add_argument('--metric', type=str, default='accuracy',
                       choices=['accuracy', 'precision', 'recall', 'f1', 'avg_inference_time'],
                       help='要分析的指标')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    data = load_results_for_statistical_test(args.results_dir, args.dataset, args.metric)
    
    if not data:
        print("错误: 没有找到有效的结果数据进行统计检验")
        return
    
    print(f"数据集: {args.dataset}")
    print(f"分析指标: {args.metric}")
    print(f"找到的模型: {list(data.keys())}")
    
    create_statistical_plots(data, args.output_dir)
    
    f_stat, p_value = perform_anova_test(data)
    
    if p_value < 0.05:
        print("\n进行事后检验...")
        posthoc_result = perform_posthoc_test(data, test_type='tukey')
        
        posthoc_result.to_csv(os.path.join(args.output_dir, 'posthoc_results.csv'))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(posthoc_result < 0.05, annot=posthoc_result, fmt=".3f", cmap="RdYlGn_r")
        plt.title('Posthoc Test Results (p-values)')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'posthoc_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("\n进行教师-学生模型比较...")
    if 'Teacher' in data and 'Student' in data:
        wilcoxon_stat, wilcoxon_p = perform_wilcoxon_test(data, 'Teacher', 'Student')
        effect_size = calculate_effect_size(data, 'Teacher', 'Student')
    
    print("\n进行最佳基准模型比较...")
    baseline_models = [model for model in data.keys() if model not in ['Teacher', 'Student']]
    if baseline_models:
        best_baseline = max(baseline_models, key=lambda x: np.mean(data[x]))
        print(f"最佳基准模型: {best_baseline} (平均 {args.metric}: {np.mean(data[best_baseline]):.4f})")
        
        if 'Student' in data:
            print(f"\n学生模型 vs 最佳基准模型 ({best_baseline}):")
            wilcoxon_stat, wilcoxon_p = perform_wilcoxon_test(data, 'Student', best_baseline)
            effect_size = calculate_effect_size(data, 'Student', best_baseline)
    
    summary_stats = {}
    for model, values in data.items():
        summary_stats[model] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
    
    summary_df = pd.DataFrame.from_dict(summary_stats, orient='index')
    summary_df.to_csv(os.path.join(args.output_dir, 'descriptive_statistics.csv'))
    
    print(f"\n描述性统计:")
    print(summary_df)
    
    with open(os.path.join(args.output_dir, 'statistical_analysis_summary.txt'), 'w') as f:
        f.write(f"Statistical Analysis Summary\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Metric: {args.metric}\n")
        f.write(f"ANOVA F-statistic: {f_stat:.4f}\n")
        f.write(f"ANOVA p-value: {p_value:.4f}\n")
        f.write(f"Significant difference: {p_value < 0.05}\n")
    
    print(f"\n统计检验完成! 结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()