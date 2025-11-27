#!/usr/bin/env python3
import os
import sys
import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold


sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from models.multimodal import MultiModalIDS, DistilledStudentModel
from data.datasets import MultiModalDataset
from training.trainers import MultimodalTrainer, DistillationTrainer

def setup_multimodal_experiment(config, dataset_name):
   

    data_dir = config['datasets'][dataset_name]['preprocessing']['final_data_dir']
    
    features = np.load(os.path.join(data_dir, 'features_balanced.npy'))
    labels = np.load(os.path.join(data_dir, 'labels_balanced.npy'))
    
    import joblib
    le = joblib.load(os.path.join(data_dir, 'label_encoder.joblib'))
    class_names = le.classes_
    num_classes = len(class_names)
    feature_dim = features.shape[1]
    
    print(f"数据集: {dataset_name}")
    print(f"特征维度: {feature_dim}, 类别数: {num_classes}")
    print(f"类别名称: {class_names}")
    
    return features, labels, num_classes, feature_dim, class_names, le

def main():
    parser = argparse.ArgumentParser(description='多模态模型训练')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cicids2017', 'cicids2018', 'toniot'],
                       help='数据集名称')
    parser.add_argument('--config', type=str, default='../../config/experiments.yaml',
                       help='实验配置文件路径')
    parser.add_argument('--data-config', type=str, default='../../config/datasets.yaml',
                       help='数据集配置文件路径')
    parser.add_argument('--fold', type=int, default=0,
                       help='交叉验证折数（0表示运行所有折）')
    parser.add_argument('--skip-teacher', action='store_true',
                       help='跳过教师模型训练')
    parser.add_argument('--skip-distillation', action='store_true',
                       help='跳过知识蒸馏')
    
    args = parser.parse_args()
    
  
    with open(args.config, 'r') as f:
        exp_config = yaml.safe_load(f)
    
    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
  
    features, labels, num_classes, feature_dim, class_names, le = setup_multimodal_experiment(
        data_config, args.dataset
    )
    

    from torchvision import transforms
    image_transform = transforms.Compose([
        transforms.Resize((exp_config['experiment_config']['img_size'], 
                          exp_config['experiment_config']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
  
    image_dir = data_config['datasets'][args.dataset]['preprocessing']['final_data_dir']
    full_dataset = MultiModalDataset(
        image_dir=image_dir,
        features=features,
        labels=labels,
        label_encoder=le,
        transform=image_transform
    )
    
    valid_labels = full_dataset.get_valid_labels()
    

    k_folds = exp_config['experiment_config']['k_folds']
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    folds_to_run = [args.fold] if args.fold > 0 else range(1, k_folds + 1)
    
    for fold in folds_to_run:
        print(f"\n{'='*60}")
        print(f"开始第 {fold} 折训练")
        print(f"{'='*60}")
        
        train_idx, test_idx = list(skf.split(range(len(full_dataset)), valid_labels))[fold-1]
        
      
        train_indices, val_indices = train_test_split(
            train_idx, test_size=0.1, stratify=valid_labels[train_idx], random_state=42
        )
        
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_idx)
        
     
        batch_size = exp_config['experiment_config']['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
   
        train_config = {
            'model_save_dir': data_config['datasets'][args.dataset]['preprocessing']['model_save_dir'],
            'learning_rate': exp_config['experiment_config']['optimizer']['lr'],
            'weight_decay': exp_config['experiment_config']['optimizer']['weight_decay']
        }
        
 
        if not args.skip_teacher:
            print(f"\n--- 训练教师模型 ---")
            teacher_model = MultiModalIDS(num_classes, feature_dim).to(device)
            
            teacher_trainer = MultimodalTrainer(
                teacher_model, train_loader, val_loader, train_config, args.dataset, fold
            )
            teacher_model = teacher_trainer.train(
                num_epochs=exp_config['experiment_config']['teacher_epochs']
            )
            

            teacher_params = teacher_model.get_parameter_count()
            print(f"教师模型参数总量: {teacher_params:.2f}M")
        else:

            model_path = os.path.join(
                train_config['model_save_dir'], 
                f'best_teacher_model_fold_{fold}.pth'
            )
            if os.path.exists(model_path):
                teacher_model = MultiModalIDS(num_classes, feature_dim).to(device)
                teacher_model.load_state_dict(torch.load(model_path))
                teacher_params = teacher_model.get_parameter_count()
                print(f"加载预训练教师模型，参数总量: {teacher_params:.2f}M")
            else:
                print(f"错误: 找不到预训练教师模型 {model_path}")
                continue
        

        if not args.skip_distillation:
            print(f"\n--- 知识蒸馏训练 ---")
            student_model = DistilledStudentModel(
                num_classes, feature_dim, 
                image_size=exp_config['experiment_config']['img_size']
            ).to(device)
            
 
            student_params = student_model.get_parameter_count()
            print(f"学生模型参数总量: {student_params:.2f}M")
            print(f"学生模型是教师模型的 {student_params / teacher_params * 100:.2f}%")
            
            distillation_config = {
                'model_save_dir': train_config['model_save_dir'],
                'learning_rate': exp_config['experiment_config']['optimizer']['lr'],
                'temperature': exp_config['experiment_config']['temp'],
                'alpha': exp_config['experiment_config']['alpha']
            }
            
            distillation_trainer = DistillationTrainer(
                teacher_model, student_model, train_loader, val_loader, 
                distillation_config, args.dataset, fold
            )
            student_model = distillation_trainer.train(
                num_epochs=exp_config['experiment_config']['student_epochs']
            )
        
        print(f"\n✅ 第 {fold} 折训练完成")
    
    print(f"\n🎉 所有训练任务完成！")

if __name__ == "__main__":
    main()