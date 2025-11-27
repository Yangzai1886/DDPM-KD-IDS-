import yaml
import os
import argparse
from typing import Dict, Any

class Config:
    def __init__(self, config_dict: Dict[str, Any] = None):
        if config_dict is None:
            config_dict = {}
        self._config = config_dict
    
    def __getattr__(self, name):
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Config has no attribute '{name}'")
    
    def __getitem__(self, key):
        return self._config[key]
    
    def __setitem__(self, key, value):
        self._config[key] = value
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    
    def update(self, other_dict):
        self._config.update(other_dict)
    
    def to_dict(self):
        return self._config.copy()

def load_config(config_path: str) -> Config:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)

def save_config(config: Config, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False)

def merge_configs(base_config: Config, override_config: Config) -> Config:
    merged_dict = _deep_merge(base_config.to_dict(), override_config.to_dict())
    return Config(merged_dict)

def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def setup_experiment_config(args):
    base_config = load_config('config/experiments.yaml')
    dataset_config = load_config('config/datasets.yaml')
    model_config = load_config('config/model_architectures.yaml')
    
    experiment_config = Config({
        'experiment': base_config.to_dict(),
        'datasets': dataset_config.to_dict(),
        'models': model_config.to_dict(),
        'runtime': {
            'dataset': args.dataset,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'seed': getattr(args, 'seed', 42)
        }
    })
    
    return experiment_config

def parse_training_args():
    parser = argparse.ArgumentParser(description='Training Configuration')
    
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['cicids2017', 'cicids2018', 'toniot'],
                       help='Dataset name')
    parser.add_argument('--config', type=str, default='config/experiments.yaml',
                       help='Experiment config file path')
    parser.add_argument('--data-config', type=str, default='config/datasets.yaml',
                       help='Dataset config file path')
    parser.add_argument('--model-config', type=str, default='config/model_architectures.yaml',
                       help='Model config file path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='Log directory')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Model save directory')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return args

def get_model_config(config: Config, model_type: str):
    models_config = config.get('models', {})
    return Config(models_config.get(model_type, {}))

def get_dataset_config(config: Config, dataset_name: str):
    datasets_config = config.get('datasets', {})
    return Config(datasets_config.get('datasets', {}).get(dataset_name, {}))