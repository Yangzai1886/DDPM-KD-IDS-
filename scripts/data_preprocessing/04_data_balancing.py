#!/usr/bin/env python3

import os
import sys
import yaml
import argparse
import numpy as np
import json


sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

def main():
    parser = argparse.ArgumentParser(description='数据平衡')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cicids2017', 'cicids2018', 'toniot'],
                       help='数据集名称')
    parser.add_argument('--config', type=str, default='../../config/datasets.yaml',
                       help='配置文件路径')
    parser.add_argument('--method', type=str, default='ddpm', choices=['ddpm', 'smote', 'none'],
                       help='数据平衡方法')
    
    args = parser.parse_args()
    
 
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_config = config['datasets'][args.dataset]
    final_data_dir = dataset_config['preprocessing']['final_data_dir']
    
    print(f"开始数据平衡 - 数据集: {args.dataset}")
    print(f"方法: {args.method}")
    
   
    features_path = os.path.join(final_data_dir, 'features.npy')
    labels_path = os.path.join(final_data_dir, 'labels.npy')
    minority_path = os.path.join(final_data_dir, 'minority_classes.npy')
    
    if not all(os.path.exists(p) for p in [features_path, labels_path, minority_path]):
        print("错误: 请先运行特征提取脚本")
        sys.exit(1)
    
    X = np.load(features_path)
    y = np.load(labels_path)
    minority_classes = np.load(minority_path)
    
    print(f"加载数据: {X.shape[0]} 样本, {X.shape[1]} 特征")
    print(f"少数类: {minority_classes}")
    
    if args.method == 'none':
        print("跳过数据平衡，使用原始数据")
        balanced_X, balanced_y = X, y
    elif args.method == 'ddpm':
        print("使用DDPM进行数据平衡...")
     
        balanced_X, balanced_y = X, y
        print("DDPM数据平衡功能将在后续实现")
    else:
        print(f"方法 {args.method} 暂未实现，使用原始数据")
        balanced_X, balanced_y = X, y
    

    np.save(os.path.join(final_data_dir, 'features_balanced.npy'), balanced_X)
    np.save(os.path.join(final_data_dir, 'labels_balanced.npy'), balanced_y)
    

    balance_info = {
        'dataset': args.dataset,
        'method': args.method,
        'original_samples': len(X),
        'balanced_samples': len(balanced_X),
        'minority_classes': minority_classes.tolist(),
        'original_class_distribution': dict(zip(*np.unique(y, return_counts=True))),
        'balanced_class_distribution': dict(zip(*np.unique(balanced_y, return_counts=True)))
    }
    
    with open(os.path.join(final_data_dir, 'data_balancing_info.json'), 'w') as f:
        json.dump(balance_info, f, indent=2)
    
    print(f"数据平衡完成！")
    print(f"平衡信息: {balance_info}")

if __name__ == "__main__":
    main()