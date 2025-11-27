#!/usr/bin/env python3

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd


sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from data.preprocessing import DataPreprocessor

def main():
    parser = argparse.ArgumentParser(description='特征分组')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cicids2017', 'cicids2018', 'toniot'],
                       help='数据集名称')
    parser.add_argument('--config', type=str, default='../../config/datasets.yaml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_config = config['datasets'][args.dataset]
    csv_path = dataset_config['csv_path']
    
    print(f"开始特征分组 - 数据集: {args.dataset}")
    
    
    preprocessor_config = {
        'img_size': config.get('preprocessing', {}).get('image_generation', {}).get('image_size', 32),
        'save_dendrogram': True
    }
    preprocessor = DataPreprocessor(preprocessor_config)
    
    
    features, labels, df = preprocessor.load_data(csv_path)
    
    
    groups, selected_indices = preprocessor.optimized_feature_grouping(features, labels)
    
    
    output_dir = dataset_config['preprocessing']['final_data_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'feature_groups.npy'), groups)
    np.save(os.path.join(output_dir, 'selected_feature_indices.npy'), selected_indices)
    
    
    group_info = {
        'total_groups': len(groups),
        'features_per_group': [len(g) for g in groups],
        'selected_features_count': len(selected_indices),
        'feature_groups': {f'group_{i}': g.tolist() for i, g in enumerate(groups)}
    }
    
    with open(os.path.join(output_dir, 'feature_grouping_info.json'), 'w') as f:
        import json
        json.dump(group_info, f, indent=2)
    
    print(f"特征分组完成！")
    print(f"分组信息: {group_info}")

if __name__ == "__main__":
    main()