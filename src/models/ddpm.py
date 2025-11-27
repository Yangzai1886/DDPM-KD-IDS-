import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler

class TrafficUNet(nn.Module):

    
    def __init__(self, num_classes=5):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        self.class_embed = nn.Embedding(num_classes, 128)
        self.condition_fuse = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        self.conv_in = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        

        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        

        self.mid = MidBlock(512)
        

        self.up1 = UpBlock(512, 256, skip_channels=256)
        self.up2 = UpBlock(256, 128, skip_channels=128)
        self.up3 = UpBlock(128, 64, skip_channels=64)
        
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, padding=1)
    
    def forward(self, x, t, class_labels):
        t_embed = self.time_embed(t.unsqueeze(-1).float())
        c_embed = self.class_embed(class_labels)
        combined = torch.cat([t_embed, c_embed], dim=1)
        condition = self.condition_fuse(combined)
        
        x1 = self.conv_in(x)
        x2 = self.down1(x1, condition)
        x3 = self.down2(x2, condition)
        x4 = self.down3(x3, condition)
        
        x_mid = self.mid(x4, condition)
        
        x = self.up1(x_mid, x3, condition)
        x = self.up2(x, x2, condition)
        x = self.up3(x, x1, condition)
        
        return self.conv_out(x)


class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU()
        )
        self.time_proj = nn.Linear(128, out_c)
    
    def forward(self, x, t_embed):
        t = self.time_proj(t_embed).view(-1, self.time_proj.out_features, 1, 1)
        return self.conv(x) + t


class MidBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU()
        )
        self.time_proj = nn.Linear(128, channels)
    
    def forward(self, x, t_embed):
        t = self.time_proj(t_embed).view(-1, self.time_proj.out_features, 1, 1)
        return self.conv(x) + t


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(out_c + skip_channels, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU()
        )
        self.time_proj = nn.Linear(128, out_c)
    
    def forward(self, x, skip, t_embed):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        t = self.time_proj(t_embed).view(-1, self.time_proj.out_features, 1, 1)
        return self.conv(x) + t


class TimeEmbedding(nn.Module):

    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        pos_enc = torch.zeros(len(t), self.dim, device=device)
        pos_enc[:, 0::2] = torch.sin(t.unsqueeze(1) * inv_freq.unsqueeze(0))
        pos_enc[:, 1::2] = torch.cos(t.unsqueeze(1) * inv_freq.unsqueeze(0))
        return pos_enc


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, t_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.t_embed = nn.Linear(t_dim, output_dim)
        self.act = nn.Mish()
        self.shortcut = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x, t):
        shortcut = self.shortcut(x)
        h = self.fc(x) + self.t_embed(t)
        return self.act(h) + shortcut


class ConditionalDDPM(nn.Module):

    
    def __init__(self, feat_dim, t_dim=64):
        super().__init__()
        self.t_embedder = TimeEmbedding(t_dim)
        self.input_layer = nn.Linear(feat_dim, 128)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(128, 128, t_dim),
            ResidualBlock(128, 128, t_dim)
        ])
        self.output_layer = nn.Linear(128, feat_dim)
    
    def forward(self, x, t):
        t_embed = self.t_embedder(t)
        h = self.input_layer(x)
        for block in self.res_blocks:
            h = block(h, t_embed)
        return self.output_layer(h)