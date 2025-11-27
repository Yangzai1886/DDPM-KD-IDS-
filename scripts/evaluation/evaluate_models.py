import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import time
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from models.multimodal import MultiModalIDS, DistilledStudentModel
from models.baselines import initialize_model
from data.datasets import MultiModalDataset, CustomDataset

def plot_confusion_matrix(cm, classes, title, save_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_multimodal_model(model, test_loader, class_names, dataset_name, model_type, fold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for images, features, labels in test_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)

            start_time = time.perf_counter()
            outputs = model(images, features)
            _, preds = torch.max(outputs, 1)
            end_time = time.perf_counter()

            inference_times.append((end_time - start_time) * 1000)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * accuracy_score(all_labels, all_preds)
    precision = 100 * precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = 100 * recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = 100 * f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    print(f"\n{dataset_name} - {model_type} - Fold {fold}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1-Score: {f1:.2f}%")

    print('Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)

    plot_confusion_matrix(cm, class_names,
                         title=f'{dataset_name} {model_type} Confusion Matrix Fold {fold}',
                         save_path=f'{dataset_name}_{model_type}_fold_{fold}_cm.png')

    print('Inference Time (ms):')
    print(f'Average: {avg_inference_time:.4f}ms')
    print(f'Std: {std_inference_time:.4f}ms')
    print(f'Min: {np.min(inference_times):.4f}ms')
    print(f'Max: {np.max(inference_times):.4f}ms')

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_time': avg_inference_time,
        'std_inference_time': std_inference_time,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

def evaluate_baseline_model(model, test_loader, class_names, dataset_name, model_name, fold):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            start_time = time.perf_counter()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            end_time = time.perf_counter()

            inference_times.append((end_time - start_time) * 1000)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * accuracy_score(all_labels, all_preds)
    precision = 100 * precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = 100 * recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = 100 * f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    print(f"\n{dataset_name} - {model_name} - Fold {fold}")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1-Score: {f1:.2f}%")

    print('Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion Matrix:')
    print(cm)

    plot_confusion_matrix(cm, class_names,
                         title=f'{dataset_name} {model_name} Confusion Matrix Fold {fold}',
                         save_path=f'{dataset_name}_{model_name}_fold_{fold}_cm.png')

    print('Inference Time (ms):')
    print(f'Average: {avg_inference_time:.4f}ms')
    print(f'Std: {std_inference_time:.4f}ms')
    print(f'Min: {np.min(inference_times):.4f}ms')
    print(f'Max: {np.max(inference_times):.4f}ms')

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_inference_time': avg_inference_time,
        'std_inference_time': std_inference_time,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

def benchmark_model(model, input_size=(1, 3, 224, 224), repetitions=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    inputs = torch.randn(input_size).to(device)

    for _ in range(10):
        _ = model(inputs)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repetitions):
        _ = model(inputs)
    torch.cuda.synchronize()
    avg_time = (time.time() - start) * 1000 / repetitions
    
    return avg_time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='模型评估')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cicids2017', 'cicids2018', 'toniot'],
                       help='数据集名称')
    parser.add_argument('--config', type=str, default='../../config/experiments.yaml',
                       help='实验配置文件路径')
    parser.add_argument('--data-config', type=str, default='../../config/datasets.yaml',
                       help='数据集配置文件路径')
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['multimodal-teacher', 'multimodal-student', 'baseline'],
                       help='评估的模型类型')
    parser.add_argument('--baseline-model', type=str, default='resnet50',
                       choices=['mobilenetv3', 'shufflenet', 'alexnet', 'efficientnet-lite', 'resnet50'],
                       help='基准模型名称')
    parser.add_argument('--fold', type=int, default=0,
                       help='交叉验证折数（0表示评估所有折）')
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        exp_config = yaml.safe_load(f)

    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    data_dir = data_config['datasets'][args.dataset]['preprocessing']['final_data_dir']
    features = np.load(os.path.join(data_dir, 'features_balanced.npy'))
    labels = np.load(os.path.join(data_dir, 'labels_balanced.npy'))

    import joblib
    le = joblib.load(os.path.join(data_dir, 'label_encoder.joblib'))
    class_names = le.classes_
    num_classes = len(class_names)
    feature_dim = features.shape[1]

    from torchvision import transforms
    image_transform = transforms.Compose([
        transforms.Resize((exp_config['experiment_config']['img_size'], 
                          exp_config['experiment_config']['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_dir = data_config['datasets'][args.dataset]['preprocessing']['final_data_dir']
    
    if args.model_type in ['multimodal-teacher', 'multimodal-student']:
        full_dataset = MultiModalDataset(
            image_dir=image_dir,
            features=features,
            labels=labels,
            label_encoder=le,
            transform=image_transform
        )
        valid_labels = full_dataset.get_valid_labels()
    else:
        full_dataset = CustomDataset(image_dir, transform=image_transform)
        valid_labels = full_dataset.labels

    k_folds = exp_config['experiment_config']['k_folds']
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    folds_to_evaluate = [args.fold] if args.fold > 0 else range(1, k_folds + 1)
    
    all_results = []
    
    for fold in folds_to_evaluate:
        print(f"\nEvaluating Fold {fold}")
        
        train_idx, test_idx = list(skf.split(range(len(full_dataset)), valid_labels))[fold-1]
        test_dataset = Subset(full_dataset, test_idx)
        test_loader = DataLoader(test_dataset, batch_size=exp_config['experiment_config']['batch_size'])
        
        model_save_dir = data_config['datasets'][args.dataset]['preprocessing']['model_save_dir']
        
        if args.model_type == 'multimodal-teacher':
            model = MultiModalIDS(num_classes, feature_dim)
            model_path = os.path.join(model_save_dir, f'best_teacher_model_fold_{fold}.pth')
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
                model_params = count_parameters(model)
                print(f"教师模型参数: {model_params / 1e6:.2f}M")
                
                result = evaluate_multimodal_model(model, test_loader, class_names, args.dataset, "Teacher", fold)
                result['model_params'] = model_params
                all_results.append(result)
            else:
                print(f"教师模型文件不存在: {model_path}")
                
        elif args.model_type == 'multimodal-student':
            model = DistilledStudentModel(num_classes, feature_dim, 
                                        image_size=exp_config['experiment_config']['img_size'])
            model_path = os.path.join(model_save_dir, f'best_student_model_fold_{fold}.pth')
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
                model_params = count_parameters(model)
                print(f"学生模型参数: {model_params / 1e6:.2f}M")
                
                result = evaluate_multimodal_model(model, test_loader, class_names, args.dataset, "Student", fold)
                result['model_params'] = model_params
                all_results.append(result)
            else:
                print(f"学生模型文件不存在: {model_path}")
                
        elif args.model_type == 'baseline':
            model = initialize_model(args.baseline_model, num_classes)
            model_path = os.path.join(model_save_dir, f'best_{args.baseline_model}_fold_{fold}.pth')
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
                model_params = count_parameters(model)
                print(f"{args.baseline_model} 参数: {model_params / 1e6:.2f}M")
                
                result = evaluate_baseline_model(model, test_loader, class_names, args.dataset, args.baseline_model, fold)
                result['model_params'] = model_params
                all_results.append(result)
            else:
                print(f"基准模型文件不存在: {model_path}")
    
    if all_results:
        print(f"\n{args.dataset} - {args.model_type} 总体评估结果:")
        accuracies = [r['accuracy'] for r in all_results]
        precisions = [r['precision'] for r in all_results]
        recalls = [r['recall'] for r in all_results]
        f1_scores = [r['f1'] for r in all_results]
        inference_times = [r['avg_inference_time'] for r in all_results]
        
        print(f"准确率: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
        print(f"精确率: {np.mean(precisions):.2f}% ± {np.std(precisions):.2f}%")
        print(f"召回率: {np.mean(recalls):.2f}% ± {np.std(recalls):.2f}%")
        print(f"F1分数: {np.mean(f1_scores):.2f}% ± {np.std(f1_scores):.2f}%")
        print(f"推理时间: {np.mean(inference_times):.4f}ms ± {np.std(inference_times):.4f}ms")
        
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f'{args.dataset}_{args.model_type}_evaluation_results.csv', index=False)

if __name__ == "__main__":
    main()