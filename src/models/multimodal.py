import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalIDS(nn.Module):
    
    
    def __init__(self, num_classes, feature_dim):
        super().__init__()
        
       
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
            self.InvertedResidual(16, 32, stride=2),
            self.InvertedResidual(32, 64, stride=2),
            self.InvertedResidual(64, 128, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
       
        self.feature_branch = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    class InvertedResidual(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            hidden_ch = in_ch * 4
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, 1),
                nn.BatchNorm2d(hidden_ch),
                nn.Hardswish(),
                nn.Conv2d(hidden_ch, hidden_ch, 3, stride=stride, padding=1, groups=hidden_ch),
                nn.BatchNorm2d(hidden_ch),
                nn.Hardswish(),
                nn.Conv2d(hidden_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch)
            )
            
        def forward(self, x):
            return x + self.block(x) if x.shape == self.block(x).shape else self.block(x)
    
    def forward(self, image_input, feature_input):
        image_features = self.image_branch(image_input)
        feature_features = self.feature_branch(feature_input)
        fused = torch.cat((image_features, feature_features), dim=1)
        return self.classifier(fused)
    
    def get_parameter_count(self):"
        total_params = sum(p.numel() for p in self.parameters())
        return total_params / 1e6


class DistilledStudentModel(nn.Module):
    
    def __init__(self, num_classes, feature_dim, image_size=32):
        super().__init__()

        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1, groups=8),  
            nn.Conv2d(8, 16, 1),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
    
        self.feature_branch = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU()
        )
        
       
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size, image_size)
            dummy_output = self.image_branch(dummy_input)
            image_out_dim = dummy_output.shape[1]
        
       
        self.classifier = nn.Sequential(
            nn.Linear(image_out_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, image_input, feature_input):
        img_feat = self.image_branch(image_input)
        tab_feat = self.feature_branch(feature_input)
        fused = torch.cat((img_feat, tab_feat), dim=1)
        return self.classifier(fused)
    
    def get_parameter_count(self):
        
        total_params = sum(p.numel() for p in self.parameters())
        return total_params / 1e6