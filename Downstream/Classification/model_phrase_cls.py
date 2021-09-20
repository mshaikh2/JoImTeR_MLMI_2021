import os
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SoftAttention(nn.Module):
    def __init__(self,in_groups,m_heads,in_channels):
        super(SoftAttention, self).__init__()
        
        self.learnable_scalar = nn.Parameter(torch.rand(1))
        self.conv3d = nn.Conv3d(in_channels=in_groups,out_channels=m_heads,kernel_size=(in_channels,1,1), stride=(in_channels,1,1))
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.softmax = nn.Softmax(-1)
    def forward(self, x):
#         print('x.shape:',x)
        h,w = x.shape[-2],x.shape[-1]
        c = torch.unsqueeze(x,1)
#         print('c.shape:',c)
        c = self.conv3d(c)
        c = self.lrelu(c)
#         print('c.shape relu:',c)
        c = c.squeeze(2)
        c = c.view(c.shape[0],c.shape[1],h*w)
#         print('c.shape h*w:',c)
        c = self.softmax(c)
#         print('c.shape sfmx:',c)
        c = c.view(c.shape[0],c.shape[1],h,w)
#         print('c.shape:',c)
        attn_maps = torch.unsqueeze(c.sum(1),1)
#         print('attn_maps.shape:',attn_maps)
        importance = x*attn_maps
        out = x + importance*self.learnable_scalar.expand_as(importance)
#         print('out.shape:',out)
        return out, attn_maps, self.learnable_scalar

class BasicBlock(nn.Module):
    """
    A wrapup of a residual block
    """
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

   
        

class ImageEncoder(nn.Module):
    """
    The image embedder that is implemented as a residual network 
    """

    def __init__(self, output_channels=512, layers=[2,2,2,2,2,2], block=BasicBlock, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, 
                 norm_layer=None):
        super(ImageEncoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1
        # if replace_stride_with_dilation is None:
        #     # each element in the tuple indicates if we should replace
        #     # the 2x2 stride with a dilated convolution instead
        #     replace_stride_with_dilation = [False, False, False]
        # if len(replace_stride_with_dilation) != 3:
        #     raise ValueError("replace_stride_with_dilation should be None "
        #                      "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=4, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.sa_l4 = SoftAttention(in_groups=1,m_heads=16,in_channels=128) # save some GPU memory
        self.sa_l5 = SoftAttention(in_groups=1,m_heads=16,in_channels=256)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 256, layers[4], stride=2)
        self.layer6 = self._make_layer(block, 512, layers[5], stride=2)
        # self.layer7 = self._make_layer(block, 512, layers[6], stride=2)
        self.region = conv1x1(256, output_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((2,2))
        self.global_feats = nn.Linear(2048, output_channels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, 
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) # batch_size, 8, 512, 512
        # x = self.maxpool(x)

        x = self.layer1(x) # batch_size, 16, 256, 256
        x = self.layer2(x) # batch_size, 32, 128, 128
        x = self.layer3(x) # batch_size, 64, 64, 64
        x = self.layer4(x) # batch_size, 128, 32, 32
        x, attn_maps_l4, learnable_scalar_l4 = self.sa_l4(x) # batch_size, 128, 32, 32
#         region_feat = self.region(x) # batch_size, 512, 32, 32
        x = self.layer5(x) # batch_size, 256, 16, 16
        x, attn_maps_l5, learnable_scalar_l5 = self.sa_l5(x) # batch_size, 256, 16, 16
        region_feat = self.region(x) # batch_size, 512, 16, 16
        x = self.layer6(x) # batch_size, 512, 8, 8
        # x = self.layer7(x) # batch_size, 192, 4, 4
        
        x = self.avgpool(x)
        z = torch.flatten(x, 1) # batch_size, 2048
        # outputs = (z,)
        global_feat = self.global_feats(z)
        # outputs += (logits,)
        # outputs += (lowerlevel_img_feat,)
#         attn_maps_l4, learnable_scalar_l4 = attn_maps_l5, learnable_scalar_l5 # complement to the output number
        return region_feat, global_feat, attn_maps_l4, learnable_scalar_l4, attn_maps_l5, learnable_scalar_l5


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
        net = torch.cat((f_r,f_g), dim=-1) #  N, 512 + 512 (1024 --> 512 --> 256 --> 14)
        net = self.relu(self.fc1(self.do(net)))  # N, 256
        out = self.output(self.do(net)) # N, 14
        return out, attn_maps_l4, learnable_scalar_l4, attn_maps_l5, learnable_scalar_l5

    
################ Transformer: Text Encoder ############
class TextEncoder(nn.Module):
    def __init__(self, bert_config, output_channels):
        super(TextEncoder, self).__init__()
        # batchsize, timesteps, 1
        self.bert = BertModel(bert_config, add_pooling_layer=False)
        # batchsize, timesteps, 512
        self.bigram = nn.Conv1d(in_channels=output_channels, out_channels=output_channels, kernel_size=2)
        self.GELU = nn.GELU()
        self.norm = nn.LayerNorm(bert_config.hidden_size, eps=bert_config.layer_norm_eps)
#         self.bigram_learnable_scalar = nn.Parameter(torch.rand(1), requires_grad = True)
        self.trigram = nn.Conv1d(in_channels=output_channels, out_channels=output_channels, kernel_size=3)
#         self.trigram_learnable_scalar = nn.Parameter(torch.rand(1), requires_grad = True)
#         self.trigram = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=3)
        self.sent_fc = nn.Linear(output_channels, output_channels)
        self.word_fc = nn.Linear(output_channels, output_channels)
        self.bigram_fc = nn.Linear(output_channels, output_channels)
        self.trigram_fc = nn.Linear(output_channels, output_channels)
        
    def forward(self, x, mask):
        word_feat = self.bert(input_ids=x, attention_mask=mask)[0] # [B, L, C]  
        sent_feat = (word_feat * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1) # masked average pooling
        sent_feat = self.sent_fc(sent_feat) # global feature
        
#         word_feat = word_feat # for the input of conv layer, [B, C, L]
        
        bigram = self.norm(self.GELU(self.bigram(word_feat.transpose(1,2))).transpose(1,2)) # conv1d expects [B, C, L], --> [B, L-1, C]
#         print('bigram.shape:',bigram.shape)
        trigram = self.norm(self.GELU(self.trigram(word_feat.transpose(1,2))).transpose(1,2)) # conv1d expects [B, C, L], --> [B, L-2, C]
#         print('trigram.shape:',bigram.shape)
           
        bigram_feat = self.bigram_fc(bigram) # [B, L, C]
        trigram_feat = self.trigram_fc(trigram) # [B, L-1, C]
        word_feat = self.word_fc(word_feat) # [B, L-2, C]
        return word_feat, bigram_feat, trigram_feat, sent_feat
    
    
# class TextEncoderMLM(nn.Module):
#     def __init__(self, bert_config, output_channels=512):
#         super(TextEncoderMLM, self).__init__()
        
#         print('Bert encoder with MaskedLMhead.')
        
#         self.bert = BertModel(bert_config, add_pooling_layer=False)
# #         self.sent = nn.Linear(bert_config.hidden_size, output_channels)
#         self.transform = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
#         self.act = nn.GELU()
#         self.LayerNorm = nn.LayerNorm(bert_config.hidden_size, eps=bert_config.layer_norm_eps)
#         self.mlm = nn.Linear(bert_config.hidden_size, bert_config.vocab_size - 1) ## classification layer for MLM
        
#     def forward(self, x, mask):
#         word_feat = self.bert(input_ids=x, attention_mask=mask)[0]
#         pred = self.mlm(self.LayerNorm(self.act(self.transform(word_feat)))) # [B, timestep, 512] --> [B, timestep, 8410]

#         # for CLS feat as sent_feat 
#         # sent_feat = word_feat[:,0,:] 
    
#         # for masked mean sent_feat
# #         sent_feat = (word_feat * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1)
    
# #         sent_feat = word_feat.mean(1)
# #         sent_feat = self.sent(sent_feat)
#         return pred.transpose(1,2) # word_feat.transpose(1,2), sent_feat




class ImageText_Classification(nn.Module):
    def __init__(self, num_class=14, img_encoder_path=None, txt_encoder_path=None, pretrained=False, cfg=None, bert_config=None):
        super(ImageText_Classification, self).__init__()
        # batchsize, timesteps, 1
        self.img_encoder = ImageEncoder(output_channels=cfg.hidden_dim)
        self.txt_encoder = TextEncoder(bert_config=bert_config, output_channels=cfg.hidden_dim)
        
        if cfg.pretrained_text_encoder_path != '':
            print('Initiate text encoder from MLM pretrained parameters from:', cfg.pretrained_text_encoder_path)
            state_dict = torch.load(cfg.pretrained_text_encoder_path, map_location='cpu')
            self.txt_encoder.load_state_dict(state_dict['model'], strict=False)
#         print(misskey)
#         print(unknowkey)

        if pretrained:
            print('Load image encoder from:', img_encoder_path)
            state_dict = torch.load(img_encoder_path, map_location='cpu')
            if 'model' in state_dict.keys():
                self.img_encoder.load_state_dict(state_dict['model'])
            else:
                self.img_encoder.load_state_dict(state_dict)
                
            print('Load text encoder from:', txt_encoder_path)
            state_dict = torch.load(txt_encoder_path, map_location='cpu')
            if 'model' in state_dict.keys():
                self.txt_encoder.load_state_dict(state_dict['model'])
            else:
                self.txt_encoder.load_state_dict(state_dict)
        
#         self.soft_attention = SoftAttention(in_groups=1, m_heads=16, in_channels=512)
        ## image feature input ##
        self.gap = nn.AdaptiveAvgPool2d(1) # gap = GlobalAveragePooling, pooling region features into one feature
        self.proj_region = nn.Linear(cfg.hidden_dim, 256) # projection pooled region feature, 512 --> 256
        self.proj_global = nn.Linear(cfg.hidden_dim, 256) # projection global image feature, 512 --> 256
        ## text feature input ##
        self.proj_sent = nn.Linear(cfg.hidden_dim, 256) # projection global sentence feature, 512 --> 256
        self.proj_word = nn.Linear(cfg.hidden_dim, 256) # projection masked averaged words/phrases feature, 512 --> 256
        
        self.relu = nn.ReLU(inplace=True) # activation function
        self.fc1 = nn.Linear(1024, 512) # FC1 after concat image + text projected features, (4*256)
        self.fc2 = nn.Linear(512, 256) # FC2
        self.do = nn.Dropout(0.5) # dropout layer
        self.output = nn.Linear(256, num_class) # 14 classes including no findings
        
    def forward(self, x, t, mask):
        ## image input ##
        f_r, f_g, attn_maps_l4, learnable_scalar_l4, attn_maps_l5, learnable_scalar_l5 = self.img_encoder(x) # N, 512, 16, 16; N, 512
#         f_r,attn_maps = self.soft_attention(f_r)
        g_p = torch.squeeze(self.gap(f_r)) # N, 512
        f_r = self.relu(self.proj_region(self.do(g_p))) # N, 256, projection region feature
        f_g = self.relu(self.proj_global(self.do(f_g))) # N, 256, projection global feature
        ## text input ##
        f_w, f_b, f_t, f_s = self.txt_encoder(t, mask) # N, 160, 512; N, 159, 512; N, 158, 512; N, 512
        f_w_avg = (f_w * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1) # [N, 512] masked average pooling from word_feat
        f_b_avg = (f_b * mask[:,:-1].unsqueeze(-1)).sum(1) / mask[:,:-1].sum(1).unsqueeze(-1) # [N, 512] masked average pooling from bigram_feat
        f_t_avg = (f_t * mask[:,:-2].unsqueeze(-1)).sum(1) / mask[:,:-2].sum(1).unsqueeze(-1) # [N, 512] masked average pooling from trigram_feat
        f_wp = (f_w_avg + f_b_avg + f_t_avg) / 3.0 # merge 3 level word/phrase features into one feature [N, 512]
        f_wp = self.relu(self.proj_word(self.do(f_wp))) # N, 256, projection word feature
        f_s = self.relu(self.proj_sent(self.do(f_s))) # N, 256, projection sent feature
        ## classification head ##
        net = torch.cat((f_g, f_r, f_s, f_wp), dim=-1) #  N, 512 + 512 (1024 --> 512 --> 256 --> 14)
        net = self.relu(self.fc1(self.do(net)))  # N, 512
        net = self.relu(self.fc2(self.do(net)))  # N, 256
        out = self.output(self.do(net)) # N, 14
        return out, attn_maps_l4, learnable_scalar_l4, attn_maps_l5, learnable_scalar_l5
    

### add new code here ###
class Text_Classification(nn.Module):
    def __init__(self, num_class=14, txt_encoder_path=None, pretrained=False, cfg=None, bert_config=None):
        super(Text_Classification, self).__init__()
        # batchsize, timesteps, 1
        self.txt_encoder = TextEncoder(bert_config=bert_config, output_channels=cfg.hidden_dim)
        
        if cfg.pretrained_text_encoder_path != '':
            print('Initiate text encoder from MLM pretrained parameters from:', cfg.pretrained_text_encoder_path)
            state_dict = torch.load(cfg.pretrained_text_encoder_path, map_location='cpu')
            self.txt_encoder.load_state_dict(state_dict['model'], strict=False)
#         print(misskey)
#         print(unknowkey)

        if pretrained:
            state_dict = torch.load(txt_encoder_path, map_location='cpu')
            if 'model' in state_dict.keys():
                self.txt_encoder.load_state_dict(state_dict['model'])
            else:
                self.txt_encoder.load_state_dict(state_dict)
            print('Load text encoder from:', txt_encoder_path)
        
        ## text feature input ##
        self.proj_sent = nn.Linear(cfg.hidden_dim, 256) # projection global sentence feature, 512 --> 256
        self.proj_word = nn.Linear(cfg.hidden_dim, 256) # projection masked averaged words/phrases feature, 512 --> 256
        
        self.relu = nn.ReLU(inplace=True) # activation function
#         self.fc1 = nn.Linear(1024, 512) # FC1 after concat image + text projected features, (4*256)
        self.fc1 = nn.Linear(512, 256) # FC2
        self.do = nn.Dropout(0.5) # dropout layer
        self.output = nn.Linear(256, num_class) # 14 classes including no findings
        
    def forward(self, t, mask):
        ## text input ##
        f_w, f_b, f_t, f_s = self.txt_encoder(t, mask) # N, 160, 512; N, 159, 512; N, 158, 512; N, 512
        f_w_avg = (f_w * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1) # [N, 512] masked average pooling from word_feat
        f_b_avg = (f_b * mask[:,:-1].unsqueeze(-1)).sum(1) / mask[:,:-1].sum(1).unsqueeze(-1) # [N, 512] masked average pooling from bigram_feat
        f_t_avg = (f_t * mask[:,:-2].unsqueeze(-1)).sum(1) / mask[:,:-2].sum(1).unsqueeze(-1) # [N, 512] masked average pooling from trigram_feat
        f_wp = (f_w_avg + f_b_avg + f_t_avg) / 3.0 # merge 3 level word/phrase features into one feature [N, 512]
        f_wp = self.relu(self.proj_word(self.do(f_wp))) # N, 256, projection word feature
        f_s = self.relu(self.proj_sent(self.do(f_s))) # N, 256, projection sent feature
        ## classification head ##
        net = torch.cat((f_s, f_wp), dim=-1) #  N, 512 (512 --> 256 --> 14)
#         net = self.relu(self.fc1(self.do(net)))  # N, 512
        net = self.relu(self.fc1(self.do(net)))  # N, 256
        out = self.output(self.do(net)) # N, 14
        return out