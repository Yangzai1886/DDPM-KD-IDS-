import torch
import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

def initialize_model(model_name, num_classes):
    
    if model_name == 'mobilenetv3':
        model = models.mobilenet_v3_large(pretrained=True)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    
    elif model_name == 'shufflenet':
        model = models.shufflenet_v2_x1_0(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    
    elif model_name == 'efficientnet-lite':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_classes)
        return model
    
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
    
    else:
        raise ValueError(f"不支持的模型: {model_name}")