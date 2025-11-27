import os
import yaml
import torch
import numpy as np
import random
from datetime import datetime
import json
import pandas as pd
from abc import ABC, abstractmethod

class BaseExperiment(ABC):
    def __init__(self, config, experiment_name):
        self.config = config
        self.experiment_name = experiment_name
        self.device = self._setup_device()
        self.set_seeds()
        self.setup_directories()
        self.results = {}
        
    def _setup_device(self):
        device_config = self.config.get('device', 'cuda')
        if device_config == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_config == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def set_seeds(self):
        seed = self.config.get('seeds', {}).get('training', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def setup_directories(self):
        base_dir = self.config.get('results', {}).get('save_dir', 'results')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(base_dir, f"{self.experiment_name}_{timestamp}")
        
        directories = [
            self.exp_dir,
            os.path.join(self.exp_dir, 'models'),
            os.path.join(self.exp_dir, 'logs'),
            os.path.join(self.exp_dir, 'visualizations'),
            os.path.join(self.exp_dir, 'metrics')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_config(self):
        config_path = os.path.join(self.exp_dir, 'experiment_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def save_results(self):
        results_path = os.path.join(self.exp_dir, 'experiment_results.json')
        
        json_ready_results = {}
        for key, value in self.results.items():
            if isinstance(value, (np.ndarray, np.generic)):
                json_ready_results[key] = value.tolist()
            elif isinstance(value, (int, float, str, bool, list, dict)):
                json_ready_results[key] = value
            else:
                json_ready_results[key] = str(value)
        
        with open(results_path, 'w') as f:
            json.dump(json_ready_results, f, indent=2)
    
    def log_metrics(self, metrics, epoch=None):
        if epoch is not None:
            log_entry = {'epoch': epoch, **metrics}
        else:
            log_entry = metrics
        
        log_path = os.path.join(self.exp_dir, 'training_log.jsonl')
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def create_summary(self):
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'config': self.config,
            'results_summary': self._create_results_summary()
        }
        
        summary_path = os.path.join(self.exp_dir, 'experiment_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def _create_results_summary(self):
        return {
            'total_models_trained': len(self.results.get('models', [])),
            'best_accuracy': self.results.get('best_accuracy', 0),
            'total_training_time': self.results.get('total_training_time', 0)
        }
    
    @abstractmethod
    def setup_data(self):
        pass
    
    @abstractmethod
    def setup_models(self):
        pass
    
    @abstractmethod
    def run(self):
        pass
    
    def cleanup(self):
        if hasattr(self, 'models'):
            for model in self.models:
                if hasattr(model, 'cpu'):
                    model.cpu()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        if exc_type is not None:
            print(f"Experiment ended with error: {exc_type.__name__}: {exc_val}")
        return False

class ExperimentLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.console_log = os.path.join(log_dir, 'console.log')
        self.metrics_log = os.path.join(log_dir, 'metrics.csv')
        
        self._setup_logging()
    
    def _setup_logging(self):
        import logging
        self.logger = logging.getLogger('ExperimentLogger')
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(self.console_log)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_info(self, message):
        self.logger.info(message)
    
    def log_warning(self, message):
        self.logger.warning(message)
    
    def log_error(self, message):
        self.logger.error(message)
    
    def log_metrics(self, metrics_dict):
        df = pd.DataFrame([metrics_dict])
        if not os.path.exists(self.metrics_log):
            df.to_csv(self.metrics_log, index=False)
        else:
            df.to_csv(self.metrics_log, mode='a', header=False, index=False)