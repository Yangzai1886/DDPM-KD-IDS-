import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import time

class MetricsCalculator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.inference_times = []
    
    def update(self, predictions, targets, inference_time=None):
        self.predictions.extend(predictions.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        if inference_time is not None:
            self.inference_times.append(inference_time)
    
    def compute_accuracy(self):
        return accuracy_score(self.targets, self.predictions)
    
    def compute_precision(self, average='macro'):
        return precision_score(self.targets, self.predictions, average=average, zero_division=0)
    
    def compute_recall(self, average='macro'):
        return recall_score(self.targets, self.predictions, average=average, zero_division=0)
    
    def compute_f1(self, average='macro'):
        return f1_score(self.targets, self.predictions, average=average, zero_division=0)
    
    def compute_confusion_matrix(self):
        return confusion_matrix(self.targets, self.predictions)
    
    def compute_classification_report(self, target_names=None):
        return classification_report(self.targets, self.predictions, 
                                   target_names=target_names, output_dict=True)
    
    def compute_inference_stats(self):
        if not self.inference_times:
            return {}
        times = np.array(self.inference_times)
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times)
        }
    
    def compute_all_metrics(self, target_names=None):
        accuracy = self.compute_accuracy()
        precision = self.compute_precision()
        recall = self.compute_recall()
        f1 = self.compute_f1()
        cm = self.compute_confusion_matrix()
        report = self.compute_classification_report(target_names)
        inference_stats = self.compute_inference_stats()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'inference_stats': inference_stats
        }

class ModelEvaluator:
    def __init__(self, model, device, num_classes):
        self.model = model
        self.device = device
        self.metrics = MetricsCalculator(num_classes)
    
    def evaluate_model(self, dataloader, model_type='multimodal'):
        self.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for batch in dataloader:
                start_time = time.perf_counter()
                
                if model_type == 'multimodal':
                    images, features, targets = batch
                    images, features, targets = images.to(self.device), features.to(self.device), targets.to(self.device)
                    outputs = self.model(images, features)
                else:
                    images, targets = batch
                    images, targets = images.to(self.device), targets.to(self.device)
                    outputs = self.model(images)
                
                end_time = time.perf_counter()
                inference_time = (end_time - start_time) * 1000
                
                _, predictions = torch.max(outputs, 1)
                self.metrics.update(predictions, targets, inference_time)
        
        return self.metrics.compute_all_metrics()

def calculate_model_complexity(model, input_size=(1, 3, 32, 32)):
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def count_flops(model, input_size):
        from thop import profile
        dummy_input = torch.randn(input_size)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops, params
    
    try:
        flops, params = count_flops(model, input_size)
    except:
        flops = 0
        params = count_parameters(model)
    
    return {
        'parameters': params,
        'flops': flops,
        'parameters_millions': params / 1e6,
        'flops_billions': flops / 1e9
    }

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
    
    def save_checkpoint(self, model):
        if self.restore_best:
            self.best_state = model.state_dict().copy()
    
    def restore_model(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)

def compute_roc_auc(predictions, targets, num_classes):
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import label_binarize
    
    targets_bin = label_binarize(targets, classes=range(num_classes))
    
    if len(predictions.shape) == 1:
        predictions_proba = torch.softmax(torch.tensor(predictions), dim=1).numpy()
    else:
        predictions_proba = predictions
    
    try:
        auc_score = roc_auc_score(targets_bin, predictions_proba, average='macro', multi_class='ovr')
        fpr = {}
        tpr = {}
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(targets_bin[:, i], predictions_proba[:, i])
        return auc_score, fpr, tpr
    except:
        return 0.0, {}, {}