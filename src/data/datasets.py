import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch

class MultiModalDataset(Dataset):
    
    
    def __init__(self, image_dir, features, labels, label_encoder, transform=None):
        self.image_dir = image_dir
        self.features = features
        self.labels = labels
        self.transform = transform
        self.label_encoder = label_encoder
        
       
        self.valid_labels = []
        
        
        self.class_names = label_encoder.classes_
        
       
        self.samples = []
        class_counters = {cls: 0 for cls in self.class_names}
        
       
        class_samples = {}
        for cls in self.class_names:
            class_dir = os.path.join(image_dir, cls)
            if os.path.exists(class_dir):
                class_samples[cls] = sorted([
                    f for f in os.listdir(class_dir)
                    if f.endswith('.png') and (f.startswith('sample_') or f.startswith('generated_'))
                ])
            else:
                class_samples[cls] = []
        
        
        for idx, label_idx in enumerate(labels):
            class_name = self.class_names[label_idx]
            
            if class_name not in class_samples or not class_samples[class_name]:
                continue
            
           
            if class_counters[class_name] < len(class_samples[class_name]):
                self.valid_labels.append(label_idx)
                img_name = class_samples[class_name][class_counters[class_name]]
                img_path = os.path.join(image_dir, class_name, img_name)
                self.samples.append((img_path, features[idx], label_idx))
                class_counters[class_name] += 1
    
    def get_valid_labels(self):
        
        return np.array(self.valid_labels)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        img_path, feature, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class CustomDataset(Dataset):
  
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_names = sorted(os.listdir(root_dir))
        self.file_paths = []
        self.labels = []
        
        for label_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                self.file_paths.append(os.path.join(class_dir, file))
                self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, label