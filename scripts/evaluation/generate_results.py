import os
import sys
import yaml
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_evaluation_results(results_dir, dataset, model_types):
    results = {}
    for model_type in model_types:
        csv_path = os.path.join(results_dir, f'{dataset}_{model_type}_evaluation_results.csv')
        if os.path.exists(csv_path):
            results[model_type] = pd.read_csv(csv_path)
    return results

def calculate_summary_statistics(results):
    summary = {}
    for model_type, df in results.items():
        summary[model_type] = {
            'accuracy_mean': np.mean(df['accuracy']),
            'accuracy_std': np.std(df['accuracy']),
            'precision_mean': np.mean(df['precision']),
            'precision_std': np.std(df['precision']),
            'recall_mean': np.mean(df['recall']),
            'recall_std': np.std(df['recall']),
            'f1_mean': np.mean(df['f1']),
            'f1_std': np.std(df['f1']),
            'inference_time_mean': np.mean(df['avg_inference_time']),
            'inference_time_std': np.std(df['avg_inference_time']),
            'model_params_mean': np.mean(df['model_params']) if 'model_params' in df.columns else 0
        }
    return summary

def create_comparison_plot(summary, metric, title, save_path):
    models = list(summary.keys())
    means = [summary[model][f'{metric}_mean'] for model in models]
    stds = [summary[model][f'{metric}_std'] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, means, yerr=stds, capsize=5, alpha=0.7)
    plt.title(title)
    plt.ylabel(metric.capitalize())
    plt.xticks(rotation=45)
    
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{mean:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_latex_table(summary, output_path):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'inference_time']
    latex_lines = []
    
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Model Performance Comparison}")
    latex_lines.append("\\begin{tabular}{l" + "c" * len(metrics) + "}")
    latex_lines.append("\\hline")
    
    header = "Model & " + " & ".join([m.capitalize() for m in metrics]) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\hline")
    
    for model, stats in summary.items():
        row = model.replace('_', ' ').title()
        for metric in metrics:
            if metric == 'inference_time':
                row += f" & {stats[f'{metric}_mean']:.2f} ± {stats[f'{metric}_std']:.2f} ms"
            else:
                row += f" & {stats[f'{metric}_mean']:.2f} ± {stats[f'{metric}_std']:.2f}%"
        row += " \\\\"
        latex_lines.append(row)
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))

def main():
    parser = argparse.ArgumentParser(description='生成实验结果报告')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cicids2017', 'cicids2018', 'toniot', 'all'],
                       help='数据集名称')
    parser.add_argument('--results-dir', type=str, default='./',
                       help='结果文件目录')
    parser.add_argument('--output-dir', type=str, default='./reports',
                       help='输出目录')
    parser.add_argument('--config', type=str, default='../../config/experiments.yaml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(args.output_dir, f'report_{timestamp}')
    os.makedirs(report_dir, exist_ok=True)
    
    model_types = [
        'multimodal-teacher',
        'multimodal-student', 
        'baseline-mobilenetv3',
        'baseline-shufflenet',
        'baseline-alexnet',
        'baseline-efficientnet-lite',
        'baseline-resnet50'
    ]
    
    datasets = [args.dataset] if args.dataset != 'all' else ['cicids2017', 'cicids2018', 'toniot']
    
    all_summaries = {}
    
    for dataset in datasets:
        print(f"处理数据集: {dataset}")
        
        results = load_evaluation_results(args.results_dir, dataset, model_types)
        if not results:
            print(f"未找到 {dataset} 的结果文件")
            continue
        
        summary = calculate_summary_statistics(results)
        all_summaries[dataset] = summary
        
        dataset_report_dir = os.path.join(report_dir, dataset)
        os.makedirs(dataset_report_dir, exist_ok=True)
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'inference_time']
        for metric in metrics_to_plot:
            create_comparison_plot(
                summary, 
                metric, 
                f'{dataset} - {metric.capitalize()} Comparison',
                os.path.join(dataset_report_dir, f'{metric}_comparison.png')
            )
        
        generate_latex_table(summary, os.path.join(dataset_report_dir, 'latex_table.tex'))
        
        with open(os.path.join(dataset_report_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        summary_df = pd.DataFrame.from_dict(summary, orient='index')
        summary_df.to_csv(os.path.join(dataset_report_dir, 'summary.csv'))
        
        print(f"{dataset} 报告生成完成")
    
    if all_summaries:
        cross_dataset_comparison = {}
        for dataset, summary in all_summaries.items():
            for model_type, stats in summary.items():
                key = f"{dataset}_{model_type}"
                cross_dataset_comparison[key] = stats
        
        cross_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in cross_metrics:
            plt.figure(figsize=(12, 8))
            models = list(cross_dataset_comparison.keys())
            means = [cross_dataset_comparison[model][f'{metric}_mean'] for model in models]
            stds = [cross_dataset_comparison[model][f'{metric}_std'] for model in models]
            
            bars = plt.bar(range(len(models)), means, yerr=stds, capsize=5, alpha=0.7)
            plt.title(f'Cross-Dataset {metric.capitalize()} Comparison')
            plt.ylabel(metric.capitalize())
            plt.xticks(range(len(models)), models, rotation=45, ha='right')
            
            for bar, mean in zip(bars, means):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{mean:.2f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, f'cross_dataset_{metric}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        with open(os.path.join(report_dir, 'cross_dataset_summary.json'), 'w') as f:
            json.dump(all_summaries, f, indent=2)
        
        print(f"\n综合报告生成完成: {report_dir}")
        
        final_summary = []
        for dataset, summary in all_summaries.items():
            for model_type, stats in summary.items():
                final_summary.append({
                    'Dataset': dataset,
                    'Model': model_type,
                    'Accuracy': f"{stats['accuracy_mean']:.2f} ± {stats['accuracy_std']:.2f}",
                    'Precision': f"{stats['precision_mean']:.2f} ± {stats['precision_std']:.2f}",
                    'Recall': f"{stats['recall_mean']:.2f} ± {stats['recall_std']:.2f}",
                    'F1-Score': f"{stats['f1_mean']:.2f} ± {stats['f1_std']:.2f}",
                    'Inference_Time_ms': f"{stats['inference_time_mean']:.4f} ± {stats['inference_time_std']:.4f}",
                    'Model_Params': f"{stats['model_params_mean'] / 1e6:.2f}M" if stats['model_params_mean'] > 0 else "N/A"
                })
        
        final_df = pd.DataFrame(final_summary)
        final_df.to_csv(os.path.join(report_dir, 'final_summary.csv'), index=False)
        
        print("\n最终汇总表格:")
        print(final_df.to_string(index=False))

if __name__ == "__main__":
    main()