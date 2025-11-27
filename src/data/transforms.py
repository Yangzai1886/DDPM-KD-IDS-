import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import numpy as np

class CustomTransform:
    def __init__(self, config):
        self.config = config
        self.img_size = config.get('img_size', 32)
        
    def get_train_transform(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def get_val_transform(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def get_test_transform(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class FeatureTransform:
    def __init__(self, config):
        self.config = config
        
    def normalize_features(self, features):
        if self.config.get('feature_normalization', 'standard') == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            return scaler.fit_transform(features)
        elif self.config.get('feature_normalization', 'standard') == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            return scaler.fit_transform(features)
        else:
            return features
    
    def apply_feature_noise(self, features, noise_level=0.01):
        if self.config.get('feature_augmentation', False):
            noise = np.random.normal(0, noise_level, features.shape)
            return features + noise
        return features

class MultimodalTransform:
    def __init__(self, image_transform, feature_transform):
        self.image_transform = image_transform
        self.feature_transform = feature_transform
    
    def __call__(self, image, features):
        image = self.image_transform(image)
        features = torch.tensor(features, dtype=torch.float32)
        return image, features

def create_transforms(config):
    custom_transform = CustomTransform(config)
    feature_transform = FeatureTransform(config)
    
    train_transform = custom_transform.get_train_transform()
    val_transform = custom_transform.get_val_transform()
    test_transform = custom_transform.get_test_transform()
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': test_transform,
        'feature': feature_transform
    }