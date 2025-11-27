#!/usr/bin/env python3

import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from data.preprocessing import DataPreprocessor

def load_and_preprocess(csv_path):

    print(f"正在预处理数据: {csv_path}")
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, -1].values
    features = df.iloc[:, :-1]
    

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    
 
    class_counts = pd.Series(y_encoded).value_counts()
    minority_mask = class_counts < 1300
    minority_classes_encoded = class_counts[minority_mask].index.tolist()
    
    print(f"类别分布: {dict(class_counts)}")
    print(f"少数类编码: {minority_classes_encoded}")
    
   
    cont_features = features.select_dtypes(include=np.number)
    cat_features = features.select_dtypes(exclude=np.number)
    
    scaler = StandardScaler()
    cont_scaled = scaler.fit_transform(cont_features) if not cont_features.empty else np.empty((len(df), 0))
    
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_encoded = ohe.fit_transform(cat_features) if not cat_features.empty else np.empty((len(df), 0))
    
    X = np.hstack([cont_scaled, cat_encoded])
    
    print(f"预处理完成: {X.shape[0]} 样本, {X.shape[1]} 特征")
    
    return X, y_encoded, minority_classes_encoded, scaler, ohe, le, df

def main():
    parser = argparse.ArgumentParser(description='特征提取')
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
    
    print(f"开始处理数据集: {args.dataset}")
    print(f"CSV路径: {csv_path}")
    
  
    X, y, minority_classes, scaler, ohe, le, df = load_and_preprocess(csv_path)
    
   
    output_dir = dataset_config['preprocessing']['final_data_dir']
    os.makedirs(output_dir, exist_ok=True)
    
  
    np.save(os.path.join(output_dir, 'features.npy'), X)
    np.save(os.path.join(output_dir, 'labels.npy'), y)
    np.save(os.path.join(output_dir, 'minority_classes.npy'), minority_classes)
    
   
    import joblib
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))
    joblib.dump(ohe, os.path.join(output_dir, 'ohe.joblib')) 
    joblib.dump(le, os.path.join(output_dir, 'label_encoder.joblib'))
    
  
    info = {
        'dataset': args.dataset,
        'original_samples': len(df),
        'features_count': X.shape[1],
        'classes_count': len(np.unique(y)),
        'minority_classes': minority_classes,
        'class_distribution': dict(pd.Series(y).value_counts())
    }
    
    with open(os.path.join(output_dir, 'data_info.json'), 'w') as f:
        import json
        json.dump(info, f, indent=2)
    
    print(f"特征提取完成！结果保存在: {output_dir}")
    print(f"数据信息: {info}")

if __name__ == "__main__":
    main()