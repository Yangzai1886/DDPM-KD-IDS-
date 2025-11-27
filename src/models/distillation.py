import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        loss_ce = self.ce_loss(student_logits, labels)
        
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        loss_kl = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        total_loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kl
        return total_loss, loss_ce, loss_kl

class AttentionTransfer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, student_features, teacher_features):
        s_attention = F.normalize(student_features.pow(2).mean(1).view(student_features.size(0), -1))
        t_attention = F.normalize(teacher_features.pow(2).mean(1).view(teacher_features.size(0), -1))
        
        loss = F.mse_loss(s_attention, t_attention)
        return loss

class SimilarityPreservation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, student_features, teacher_features):
        s_similarity = torch.mm(student_features, student_features.t())
        t_similarity = torch.mm(teacher_features, teacher_features.t())
        
        loss = F.mse_loss(s_similarity, t_similarity)
        return loss

class MultiHeadDistillation(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss()
        self.attention_transfer = AttentionTransfer()
        self.similarity_preservation = SimilarityPreservation()
    
    def forward(self, student_logits, teacher_logits, labels, 
                student_features=None, teacher_features=None):
        loss_ce = self.ce_loss(student_logits, labels)
        
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        loss_kl = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)
        
        total_loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kl
        
        if student_features is not None and teacher_features is not None:
            loss_at = self.attention_transfer(student_features, teacher_features)
            loss_sp = self.similarity_preservation(student_features, teacher_features)
            
            total_loss += self.beta * loss_at + self.gamma * loss_sp
        
        return total_loss

class ProgressiveDistillation:
    def __init__(self, total_epochs, initial_temp=5.0, final_temp=1.0):
        self.total_epochs = total_epochs
        self.initial_temp = initial_temp
        self.final_temp = final_temp
    
    def get_temperature(self, epoch):
        progress = epoch / self.total_epochs
        temperature = self.initial_temp - (self.initial_temp - self.final_temp) * progress
        return max(temperature, self.final_temp)

class KnowledgeDistiller:
    def __init__(self, teacher_model, student_model, config):
        self.teacher = teacher_model
        self.student = student_model
        self.config = config
        
        self.distillation_loss = DistillationLoss(
            temperature=config.get('temperature', 3.0),
            alpha=config.get('alpha', 0.5)
        )
        
        self.optimizer = torch.optim.Adam(
            student_model.parameters(),
            lr=config.get('learning_rate', 0.001)
        )
    
    def distill_batch(self, images, features, labels):
        self.teacher.eval()
        self.student.train()
        
        with torch.no_grad():
            teacher_logits = self.teacher(images, features)
        
        student_logits = self.student(images, features)
        
        loss, loss_ce, loss_kl = self.distillation_loss(
            student_logits, teacher_logits, labels
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), loss_ce.item(), loss_kl.item()
    
    def get_distillation_metrics(self, student_logits, teacher_logits, labels):
        with torch.no_grad():
            student_preds = torch.argmax(student_logits, dim=1)
            teacher_preds = torch.argmax(teacher_logits, dim=1)
            
            agreement = (student_preds == teacher_preds).float().mean()
            student_accuracy = (student_preds == labels).float().mean()
            teacher_accuracy = (teacher_preds == labels).float().mean()
            
        return {
            'agreement': agreement.item(),
            'student_accuracy': student_accuracy.item(),
            'teacher_accuracy': teacher_accuracy.item()
        }