#!/usr/bin/env python3

import os
import sys
import yaml
import argparse
import json


sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from data.preprocessing import DataPreprocessor

def main():
    parser = argparse.ArgumentParser(description='图像生成')
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
    output_dir = dataset_config['preprocessing']['img_output_dir']
    
    print(f"开始图像生成 - 数据集: {args.dataset}")
    print(f"输出目录: {output_dir}")
  
    preprocessor_config = {
        'img_size': config.get('preprocessing', {}).get('image_generation', {}).get('image_size', 32),
        'save_dendrogram': False
    }
    preprocessor = DataPreprocessor(preprocessor_config)
    
    
    image_shape, selected_indices = preprocessor.generate_rgb_images(csv_path, output_dir)
    
  
    gen_info = {
        'dataset': args.dataset,
        'image_shape': image_shape,
        'image_size': preprocessor_config['img_size'],
        'selected_features_count': len(selected_indices),
        'output_directory': output_dir
    }
    
    info_path = os.path.join(output_dir, 'image_generation_info.json')
    with open(info_path, 'w') as f:
        json.dump(gen_info, f, indent=2)
    
    print(f"图像生成完成！")
    print(f"生成信息: {gen_info}")

if __name__ == "__main__":
    main()