import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold
import time
import joblib

from .base_experiment import BaseExperiment, ExperimentLogger
from src.models.baselines import initialize_model
from src.data.datasets import CustomDataset
from src.data.transforms import create_transforms
from src.utils.metrics import ModelEvaluator, EarlyStopping
from src.utils.visualization import create_visualization_suite

class BaselineExperiment(BaseExperiment):
    def __init__(self, config, dataset_name, baseline_models=None):
        super().__init__(config, f"baseline_{dataset_name}")
        self.dataset_name = dataset_name
        self.baseline_models = baseline_models or [
            'mobilenetv3', 'shufflenet', 'alexnet', 'efficientnet-lite', 'resnet50'
        ]
        self.logger = ExperimentLogger(os.path.join(self.exp_dir, 'logs'))
        self.viz_suite = create_visualization_suite(self.config)
        
        self.setup_data()
    
    def setup_data(self):
        self.logger.log_info(f"Setting up data for {self.dataset_name}")
        
        data_config = self.config['datasets'][self.dataset_name]
        data_dir = data_config['preprocessing']['final_data_dir']
        
        self.transforms = create_transforms(self.config)
        
        self.full_dataset = CustomDataset(data_dir, transform=self.transforms['test'])
        self.num_classes = len(self.full_dataset.class_names)
        
        self.logger.log_info(f"Data setup complete: {len(self.full_dataset)} samples, {self.num_classes} classes")
    
    def setup_models(self):
        pass
    
    def run(self):
        self.logger.log_info("Starting baseline experiment")
        self.save_config()
        
        start_time = time.time()
        
        k_folds = self.config['experiment_config']['k_folds']
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        all_results = {}
        
        for model_name in self.baseline_models:
            self.logger.log_info(f"Training {model_name}")
            model_results = self._train_model(model_name, skf, k_folds)
            all_results[model_name] = model_results
        
        self._analyze_results(all_results)
        
        total_time = time.time() - start_time
        self.results['total_training_time'] = total_time
        self.results['model_results'] = all_results
        
        self.create_summary()
        self.save_results()
        
        self.logger.log_info(f"Experiment completed in {total_time:.2f} seconds")
        
        return self.results
    
    def _train_model(self, model_name, skf, k_folds):
        model_results = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(self.full_dataset)), self.full_dataset.labels)):
            fold += 1
            self.logger.log_info(f"Training {model_name} - Fold {fold}/{k_folds}")
            
            fold_result = self._run_fold(model_name, fold, train_idx, test_idx)
            model_results.append(fold_result)
        
        return model_results
    
    def _run_fold(self, model_name, fold, train_idx, test_idx):
        train_indices, val_indices = train_test_split(
            train_idx, test_size=0.1, 
            stratify=[self.full_dataset.labels[i] for i in train_idx], 
            random_state=42
        )
        
        train_subset = Subset(self.full_dataset, train_indices)
        val_subset = Subset(self.full_dataset, val_indices)
        test_subset = Subset(self.full_dataset, test_idx)
        
        batch_size = self.config['experiment_config']['batch_size']
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size)
        test_loader = DataLoader(test_subset, batch_size=batch_size)
        
        self.logger.log_info(f"{model_name} Fold {fold}: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}")
        
        model = initialize_model(model_name, self.num_classes)
        model = model.to(self.device)
        
        trained_model = self._train_single_model(model, train_loader, val_loader, model_name, fold)
        test_results = self._evaluate_model(trained_model, test_loader, model_name)
        
        self._save_model(trained_model, model_name, fold)
        
        return test_results
    
    def _train_single_model(self, model, train_loader, val_loader, model_name, fold):
        import torch.nn as nn
        import torch.optim as optim
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        early_stopping = EarlyStopping(patience=5, restore_best=True)
        
        num_epochs = self.config['experiment_config'].get('baseline_epochs', 20)
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            val_accuracy = self._validate_model(model, val_loader)
            epoch_loss = running_loss / len(train_loader)
            
            self.logger.log_info(f"{model_name} Fold {fold} Epoch {epoch+1}: Loss={epoch_loss:.4f}, Val Acc={val_accuracy:.4f}")
            
            early_stopping(val_accuracy, model)
            if early_stopping.early_stop:
                self.logger.log_info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        if early_stopping.restore_best:
            early_stopping.restore_model(model)
        
        return model
    
    def _validate_model(self, model, val_loader):
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def _evaluate_model(self, model, test_loader, model_name):
        evaluator = ModelEvaluator(model, self.device, self.num_classes)
        metrics = evaluator.evaluate_model(test_loader, 'baseline')
        
        complexity = self._calculate_model_complexity(model)
        metrics.update(complexity)
        
        self.logger.log_info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def _calculate_model_complexity(self, model):
        from src.utils.metrics import calculate_model_complexity
        
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        input_size = (1, 3, 224, 224)
        complexity = calculate_model_complexity(model, input_size)
        
        return complexity
    
    def _save_model(self, model, model_name, fold):
        model_dir = os.path.join(self.exp_dir, 'models')
        torch.save(model.state_dict(), os.path.join(model_dir, f'{model_name}_fold_{fold}.pth'))
    
    def _analyze_results(self, all_results):
        self.logger.log_info("Analyzing baseline results...")
        
        summary_data = []
        
        for model_name, results in all_results.items():
            accuracies = [r['accuracy'] for r in results]
            avg_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            
            summary_data.append({
                'model': model_name,
                'accuracy_mean': avg_acc,
                'accuracy_std': std_acc,
                'precision_mean': np.mean([r['precision'] for r in results]),
                'precision_std': np.std([r['precision'] for r in results]),
                'recall_mean': np.mean([r['recall'] for r in results]),
                'recall_std': np.std([r['recall'] for r in results]),
                'f1_mean': np.mean([r['f1_score'] for r in results]),
                'f1_std': np.std([r['f1_score'] for r in results]),
                'inference_time_mean': np.mean([r['inference_stats']['mean'] for r in results]),
                'inference_time_std': np.std([r['inference_stats']['mean'] for r in results]),
                'parameters_millions': np.mean([r['parameters_millions'] for r in results])
            })
            
            self.logger.log_info(f"{model_name}: {avg_acc:.4f} ± {std_acc:.4f}")
        
        self.results['summary'] = summary_data
        self._create_comparison_plots(summary_data)
    
    def _create_comparison_plots(self, summary_data):
        import pandas as pd
        
        df = pd.DataFrame(summary_data)
        self.viz_suite['comparison'].plot_model_comparison(
            df, ['accuracy', 'precision', 'recall', 'f1'],
            filename=f'{self.dataset_name}_baseline_comparison.png'
        )
        
        self.viz_suite['comparison'].create_performance_table(
            df, filename=f'{self.dataset_name}_baseline_performance.html'
        )