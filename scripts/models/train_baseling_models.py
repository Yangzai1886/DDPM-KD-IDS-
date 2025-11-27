#!/usr/bin/env python3
import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from models.baselines import initialize_model
from data.datasets import CustomDataset

def main():
    parser = argparse.ArgumentParser(description='基准模型训练')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cicids2017', 'cicids2018', 'toniot'],
                       help='数据集名称')
    parser.add_argument('--config', type=str, default='../../config/experiments.yaml',
                       help='实验配置文件路径')
    parser.add_argument('--data-config', type=str, default='../../config/datasets.yaml',
                       help='数据集配置文件路径')
    parser.add_argument('--model', type=str, default='all',
                       choices=['mobilenetv3', 'shufflenet', 'alexnet', 'efficientnet-lite', 'resnet50', 'all'],
                       help='要训练的模型')
    
    args = parser.parse_args()
    
 
    with open(args.config, 'r') as f:
        exp_config = yaml.safe_load(f)
    
    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)
    
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
   
    from torchvision import transforms
    image_transform = transforms.Compose([
        transforms.Resize((exp_config['experiment_config']['img_size'], 
                          exp_config['experiment_config']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
  
    image_dir = data_config['datasets'][args.dataset]['preprocessing']['final_data_dir']
    dataset = CustomDataset(image_dir, transform=image_transform)
    
    models_to_train = []
    if args.model == 'all':
        models_to_train = ['mobilenetv3', 'shufflenet', 'alexnet', 'efficientnet-lite', 'resnet50']
    else:
        models_to_train = [args.model]
    
   
    k_folds = exp_config['experiment_config']['k_folds']
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for model_name in models_to_train:
        print(f"\n{'='*60}")
        print(f"训练 {model_name} 模型 - 数据集: {args.dataset}")
        print(f"{'='*60}")
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset)), dataset.labels)):
            print(f"\n--- 第 {fold + 1} 折 ---")
            
      
            from sklearn.model_selection import train_test_split
            train_indices, val_indices = train_test_split(
                train_idx, test_size=0.1, stratify=[dataset.labels[i] for i in train_idx], random_state=42
            )
            
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            test_subset = Subset(dataset, test_idx)
            
        
            batch_size = exp_config['experiment_config']['batch_size']
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size)
            test_loader = DataLoader(test_subset, batch_size=batch_size)
            
            print(f"训练集: {len(train_subset)}, 验证集: {len(val_subset)}, 测试集: {len(test_subset)}")
            
       
            num_classes = len(dataset.class_names)
            model = initialize_model(model_name, num_classes)
            model = model.to(device)
            
         
            total_params = sum(p.numel() for p in model.parameters())
            print(f"{model_name} 参数量: {total_params / 1e6:.2f}M")
            
        
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
        
            best_acc = 0.0
            num_epochs = exp_config['experiment_config'].get('baseline_epochs', 20)
            
            for epoch in range(num_epochs):
               
                model.train()
                running_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                
             
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (preds == labels).sum().item()
                
                val_acc = 100 * correct / total
                epoch_loss = running_loss / len(train_loader)
                
                print(f'Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}%')
                
              
                if val_acc > best_acc:
                    best_acc = val_acc
                    model_save_dir = data_config['datasets'][args.dataset]['preprocessing']['model_save_dir']
                    os.makedirs(model_save_dir, exist_ok=True)
                    torch.save(model.state_dict(), 
                              os.path.join(model_save_dir, f'best_{model_name}_fold_{fold+1}.pth'))
            
     
            fold_results.append(best_acc)
            print(f"第 {fold + 1} 折最佳验证准确率: {best_acc:.2f}%")
        
       
        print(f"\n{model_name} 5折交叉验证结果:")
        print(f"平均准确率: {np.mean(fold_results):.2f}% ± {np.std(fold_results):.2f}%")
        for i, acc in enumerate(fold_results):
            print(f"折 {i+1}: {acc:.2f}%")
    
    print(f"\n🎉 所有基准模型训练完成！")

if __name__ == "__main__":
    main()