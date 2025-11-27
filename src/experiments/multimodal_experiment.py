import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold
import time
import joblib

from .base_experiment import BaseExperiment, ExperimentLogger
from src.models.multimodal import MultiModalIDS, DistilledStudentModel
from src.models.distillation import KnowledgeDistiller, DistillationLoss
from src.data.datasets import MultiModalDataset
from src.data.transforms import create_transforms
from src.training.trainers import MultimodalTrainer, DistillationTrainer
from src.utils.metrics import ModelEvaluator, EarlyStopping
from src.utils.visualization import create_visualization_suite

class MultimodalExperiment(BaseExperiment):
    def __init__(self, config, dataset_name):
        super().__init__(config, f"multimodal_{dataset_name}")
        self.dataset_name = dataset_name
        self.logger = ExperimentLogger(os.path.join(self.exp_dir, 'logs'))
        self.viz_suite = create_visualization_suite(self.config)
        
        self.setup_data()
        self.setup_models()
    
    def setup_data(self):
        self.logger.log_info(f"Setting up data for {self.dataset_name}")
        
        data_config = self.config['datasets'][self.dataset_name]
        data_dir = data_config['preprocessing']['final_data_dir']
        
        self.features = np.load(os.path.join(data_dir, 'features_balanced.npy'))
        self.labels = np.load(os.path.join(data_dir, 'labels_balanced.npy'))
        
        self.le = joblib.load(os.path.join(data_dir, 'label_encoder.joblib'))
        self.class_names = self.le.classes_
        self.num_classes = len(self.class_names)
        self.feature_dim = self.features.shape[1]
        
        self.transforms = create_transforms(self.config)
        
        self.full_dataset = MultiModalDataset(
            image_dir=data_dir,
            features=self.features,
            labels=self.labels,
            label_encoder=self.le,
            transform=self.transforms['test']
        )
        
        self.valid_labels = self.full_dataset.get_valid_labels()
        
        self.logger.log_info(f"Data setup complete: {len(self.full_dataset)} samples, {self.num_classes} classes")
    
    def setup_models(self):
        self.logger.log_info("Setting up models")
        
        self.teacher_model = MultiModalIDS(self.num_classes, self.feature_dim)
        self.student_model = DistilledStudentModel(
            self.num_classes, self.feature_dim,
            image_size=self.config['experiment_config']['img_size']
        )
        
        teacher_params = self.teacher_model.get_parameter_count()
        student_params = self.student_model.get_parameter_count()
        
        self.logger.log_info(f"Teacher model parameters: {teacher_params:.2f}M")
        self.logger.log_info(f"Student model parameters: {student_params:.2f}M")
        self.logger.log_info(f"Compression ratio: {student_params/teacher_params*100:.2f}%")
    
    def run(self):
        self.logger.log_info("Starting multimodal experiment")
        self.save_config()
        
        start_time = time.time()
        
        k_folds = self.config['experiment_config']['k_folds']
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        teacher_results = []
        student_results = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(self.full_dataset)), self.valid_labels)):
            fold += 1
            self.logger.log_info(f"Starting fold {fold}/{k_folds}")
            
            fold_results = self._run_fold(fold, train_idx, test_idx)
            teacher_results.append(fold_results['teacher'])
            student_results.append(fold_results['student'])
            
            self.logger.log_info(f"Fold {fold} completed")
        
        self._analyze_results(teacher_results, student_results)
        
        total_time = time.time() - start_time
        self.results['total_training_time'] = total_time
        self.results['teacher_results'] = teacher_results
        self.results['student_results'] = student_results
        
        self.create_summary()
        self.save_results()
        
        self.logger.log_info(f"Experiment completed in {total_time:.2f} seconds")
        
        return self.results
    
    def _run_fold(self, fold, train_idx, test_idx):
        train_indices, val_indices = train_test_split(
            train_idx, test_size=0.1, stratify=self.valid_labels[train_idx], random_state=42
        )
        
        train_dataset = Subset(self.full_dataset, train_indices)
        val_dataset = Subset(self.full_dataset, val_indices)
        test_dataset = Subset(self.full_dataset, test_idx)
        
        batch_size = self.config['experiment_config']['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        self.logger.log_info(f"Fold {fold}: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        train_config = {
            'model_save_dir': os.path.join(self.exp_dir, 'models'),
            'learning_rate': self.config['experiment_config']['optimizer']['lr'],
            'weight_decay': self.config['experiment_config']['optimizer']['weight_decay']
        }
        
        teacher_model = self._train_teacher_model(train_loader, val_loader, train_config, fold)
        teacher_results = self._evaluate_model(teacher_model, test_loader, f"teacher_fold_{fold}")
        
        student_model = self._train_student_model(
            teacher_model, train_loader, val_loader, train_config, fold
        )
        student_results = self._evaluate_model(student_model, test_loader, f"student_fold_{fold}")
        
        self._save_fold_models(teacher_model, student_model, fold)
        
        return {
            'teacher': teacher_results,
            'student': student_results
        }
    
    def _train_teacher_model(self, train_loader, val_loader, config, fold):
        self.logger.log_info(f"Training teacher model for fold {fold}")
        
        teacher_model = MultiModalIDS(self.num_classes, self.feature_dim).to(self.device)
        
        trainer = MultimodalTrainer(
            teacher_model, train_loader, val_loader, config, self.dataset_name, fold
        )
        
        teacher_model = trainer.train(
            num_epochs=self.config['experiment_config']['teacher_epochs']
        )
        
        return teacher_model
    
    def _train_student_model(self, teacher_model, train_loader, val_loader, config, fold):
        self.logger.log_info(f"Training student model for fold {fold}")
        
        student_model = DistilledStudentModel(
            self.num_classes, self.feature_dim,
            image_size=self.config['experiment_config']['img_size']
        ).to(self.device)
        
        distillation_config = {
            'model_save_dir': config['model_save_dir'],
            'learning_rate': config['learning_rate'],
            'temperature': self.config['experiment_config']['temp'],
            'alpha': self.config['experiment_config']['alpha']
        }
        
        trainer = DistillationTrainer(
            teacher_model, student_model, train_loader, val_loader,
            distillation_config, self.dataset_name, fold
        )
        
        student_model = trainer.train(
            num_epochs=self.config['experiment_config']['student_epochs']
        )
        
        return student_model
    
    def _evaluate_model(self, model, test_loader, model_name):
        evaluator = ModelEvaluator(model, self.device, self.num_classes)
        metrics = evaluator.evaluate_model(test_loader, 'multimodal')
        
        complexity = self._calculate_model_complexity(model)
        metrics.update(complexity)
        
        self.logger.log_info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def _calculate_model_complexity(self, model):
        from src.utils.metrics import calculate_model_complexity
        
        if hasattr(model, 'get_parameter_count'):
            params = model.get_parameter_count() * 1e6
        else:
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        input_size = (1, 3, self.config['experiment_config']['img_size'], 
                     self.config['experiment_config']['img_size'])
        complexity = calculate_model_complexity(model, input_size)
        
        return complexity
    
    def _save_fold_models(self, teacher_model, student_model, fold):
        model_dir = os.path.join(self.exp_dir, 'models')
        
        torch.save(teacher_model.state_dict(), 
                  os.path.join(model_dir, f'teacher_fold_{fold}.pth'))
        torch.save(student_model.state_dict(),
                  os.path.join(model_dir, f'student_fold_{fold}.pth'))
    
    def _analyze_results(self, teacher_results, student_results):
        self.logger.log_info("Analyzing results...")
        
        teacher_accuracies = [r['accuracy'] for r in teacher_results]
        student_accuracies = [r['accuracy'] for r in student_results]
        
        teacher_avg_acc = np.mean(teacher_accuracies)
        student_avg_acc = np.mean(student_accuracies)
        
        teacher_std_acc = np.std(teacher_accuracies)
        student_std_acc = np.std(student_accuracies)
        
        self.results['teacher_accuracy_mean'] = teacher_avg_acc
        self.results['teacher_accuracy_std'] = teacher_std_acc
        self.results['student_accuracy_mean'] = student_avg_acc
        self.results['student_accuracy_std'] = student_std_acc
        self.results['accuracy_difference'] = teacher_avg_acc - student_avg_acc
        
        self.logger.log_info(f"Teacher Accuracy: {teacher_avg_acc:.4f} ± {teacher_std_acc:.4f}")
        self.logger.log_info(f"Student Accuracy: {student_avg_acc:.4f} ± {student_std_acc:.4f}")
        self.logger.log_info(f"Accuracy Difference: {self.results['accuracy_difference']:.4f}")
        
        self._create_comparison_plots(teacher_results, student_results)
    
    def _create_comparison_plots(self, teacher_results, student_results):
        import pandas as pd
        
        comparison_data = []
        for i, (teacher, student) in enumerate(zip(teacher_results, student_results)):
            comparison_data.append({
                'Fold': i+1,
                'Model': 'Teacher',
                'Accuracy': teacher['accuracy'],
                'Precision': teacher['precision'],
                'Recall': teacher['recall'],
                'F1': teacher['f1_score']
            })
            comparison_data.append({
                'Fold': i+1,
                'Model': 'Student', 
                'Accuracy': student['accuracy'],
                'Precision': student['precision'],
                'Recall': student['recall'],
                'F1': student['f1_score']
            })
        
        df = pd.DataFrame(comparison_data)
        self.viz_suite['comparison'].plot_model_comparison(
            df, ['accuracy', 'precision', 'recall', 'f1'],
            filename=f'{self.dataset_name}_comparison.png'
        )