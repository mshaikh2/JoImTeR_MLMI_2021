import os
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_img import ImageEncoder

# new image encoder classification model, using SA
class ImageEncoder_Classification(nn.Module):
    def __init__(self, num_class=14, encoder_path=None, pretrained=False, cfg=None):
        super(ImageEncoder_Classification, self).__init__()
        # batchsize, timesteps, 1
        self.pretrained_encoder = ImageEncoder(output_channels=cfg.hidden_dim)
        if pretrained:
            print('Load image encoder from:', encoder_path)
            state_dict = torch.load(encoder_path, map_location='cpu')
            if 'model' in state_dict.keys():
                self.pretrained_encoder.load_state_dict(state_dict['model'])
            else:
                self.pretrained_encoder.load_state_dict(state_dict)
        
#         self.soft_attention = SoftAttention(in_groups=1, m_heads=16, in_channels=512)
        
        self.gap = nn.AdaptiveAvgPool2d(1) # gap = GlobalAveragePooling
        self.projection_region = nn.Linear(cfg.hidden_dim, 256)
        self.projection_global = nn.Linear(cfg.hidden_dim, 256)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(512, 256)
        self.do = nn.Dropout(0.5)
        self.output = nn.Linear(256, num_class) # 8 classes including no findings
        
    def forward(self, x):
        f_r, f_g, attn_maps_l4, learnable_scalar_l4, attn_maps_l5, learnable_scalar_l5 = self.pretrained_encoder(x) # N, 512, 16, 16; N, 512
#         f_r,attn_maps = self.soft_attention(f_r)
        g_p = torch.squeeze(self.gap(f_r)) # N, 512
        f_r = self.relu(self.projection_region(self.do(g_p))) # N, 256
        f_g = self.relu(self.projection_global(self.do(f_g))) # N, 256
#         print(f_r.shape,f_g.shape)
        net = torch.cat((f_r,f_g), dim=-1) #  N, 512 + 512 (1024 --> 512 --> 256 --> 14)
        net = self.relu(self.fc1(self.do(net)))  # N, 256
        out = self.output(self.do(net)) # N, 14
        return out, attn_maps_l4, learnable_scalar_l4, attn_maps_l5, learnable_scalar_l5
