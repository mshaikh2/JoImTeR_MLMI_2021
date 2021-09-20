import os,sys
sys.path.append('..')
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch.autograd import Variable

from transformers import BertConfig

from misc.config import Config

class MLMHead(nn.Module):
    def __init__(self, bert_config):
        super(MLMHead, self).__init__()
        self.transform = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.act = nn.GELU()
        self.LayerNorm = nn.LayerNorm(bert_config.hidden_size, eps=bert_config.layer_norm_eps)
        self.decoder = nn.Linear(bert_config.hidden_size, bert_config.vocab_size - 1) # classification layer for MLM
        
    def forward(self, seq_feat):
        pred = self.decoder(self.LayerNorm(self.act(self.transform(seq_feat)))) # [B, L, 512] --> [B, L, vocab-1]
        return pred
    
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class JointTransformer(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, bert_config, nlayers, output_channels, dropout=0.5):
        super(JointTransformer, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        
        encoder_layers = TransformerEncoderLayer(bert_config.hidden_size #512
                                                 , bert_config.num_attention_heads #8
                                                 , bert_config.intermediate_size #2048
                                                 , dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.mlm =  MLMHead(bert_config)
        self.avgpool = nn.AdaptiveAvgPool2d((2,2))
        self.sent_fc = nn.Linear(output_channels, output_channels) # projection layer for sent
        self.word_fc = nn.Linear(output_channels, output_channels) # projection layer for word
        
        self.global_image_fc = nn.Linear(2048, output_channels) # projection layer for images features after joint transformer
        self.region = conv1x1(512, output_channels) # output of tranformer is 512 channels hence input to conv1x1 is 512
        
    def forward(self, image_features, text_features, mask, task = 'itm'):
        h,w = image_features.shape[2], image_features.shape[3]
        batch_size, EmbSize = image_features.shape[0], image_features.shape[1]
#         print(text_features.is_contiguous(),',',image_features.is_contiguous())
#         print('image features shape:',(batch_size, EmbSize,h,w))
        image_features =  image_features.view(batch_size, EmbSize, -1).contiguous().permute(2,0,1).contiguous() # from [B, EmbSize, 16x16] to [Seq_Len_Img, B, EmbSize]
#         print(text_features.is_contiguous(),',',image_features.is_contiguous())
#         print('image features shape for transformer input:',image_features.shape)
        image_features_seq_len = image_features.shape[0]
        text_features = text_features.permute(2,0,1).contiguous() # from [B, EmbDim, Seqlen] to [Seq_Len_Txt, B, EmbSize]
#         print(text_features.is_contiguous(),',',image_features.is_contiguous())
#         print('text features shape for transformer input:',text_features.shape)
        combined_features = torch.cat([image_features,text_features], dim = 0)
#         print(text_features.is_contiguous(),',',image_features.is_contiguous(),combined_features.is_contiguous())
#         print('combined features shape for transformer input:',combined_features.shape)
        transformed_feats = self.transformer_encoder(combined_features)  # [Seq_Len_Img + Seq_Len_Txt, batchsize, feats_size]
#         print('transformed features shape for transformer input:',transformed_feats.shape)
        text_feats = transformed_feats[image_features_seq_len:,:,:].permute(1,0,2).contiguous()
#         print(text_feats.is_contiguous())
#         print('post transformed text feats shape:',text_feats.shape)
        if task == 'mlm':
            return self.forward_mlm(text_feats)
        elif task == 'itm':
            return self.forward_itm(transformed_feats[:image_features_seq_len,:,:].permute(1,2,0).view(batch_size, EmbSize, h, w)
                                    ,transformed_feats[image_features_seq_len:,:,:].permute(1,0,2), mask)
        else:
            raise ValueError('invalid task for text encoder')
        
    def forward_mlm(self, seq_feat):
        
        pred = self.mlm(seq_feat) # [B, L, vocab-1]
        
        return pred.transpose(1,2) # need transpose pred for loss_fn
        
    def forward_itm(self, vis_feat, seq_feat, mask):
        #-----------------------------------------
        region_feat = self.region(vis_feat) # [B, C, H, W]
        #-----------------------------------------
        x = self.avgpool(vis_feat)
        z = torch.flatten(x, 1) # batch_size, 2048
        global_feat = self.global_image_fc(z) # [B, C]
        #-----------------------------------------
        sent_feat = self.sent_fc(seq_feat[:,0]) # [B, 0th [CLS] token]
        #-----------------------------------------
        word_feat = self.word_fc(seq_feat) # [B, L, C]
        
        return region_feat, global_feat, word_feat.transpose(1,2), sent_feat



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
class TextEncoderMLM(nn.Module):
    def __init__(self, bert_config, output_channels=512):
        super(TextEncoderMLM, self).__init__()
        
        print('Bert encoder with MaskedLMhead.')
        
        self.bert = BertModel(bert_config, add_pooling_layer=False)
#         self.sent = nn.Linear(bert_config.hidden_size, output_channels)
        self.transform = nn.Linear(bert_config.hidden_size, bert_config.hidden_size)
        self.act = nn.GELU()
        self.LayerNorm = nn.LayerNorm(bert_config.hidden_size, eps=bert_config.layer_norm_eps)
        self.mlm = nn.Linear(bert_config.hidden_size, bert_config.vocab_size - 1) ## classification layer for MLM
        
    def forward(self, x, mask):
        word_feat = self.bert(input_ids=x, attention_mask=mask)[0]
        return word_feat # word_feat.transpose(1,2), sent_feat



class Text_Classification(nn.Module):
    def __init__(self, num_class=14, txt_encoder_path=None, pretrained=False, cfg=None, bert_config=None):
        super(Text_Classification, self).__init__()
        # batchsize, timesteps, 1
        self.txt_encoder = TextEncoderMLM(bert_config=bert_config, output_channels=cfg.hidden_dim)
        
        if cfg.pretrained_text_encoder_path != '':
            print('Initiate text encoder from MLM pretrained parameters from:', cfg.pretrained_text_encoder_path)
            state_dict = torch.load(cfg.pretrained_text_encoder_path, map_location='cpu')
            self.txt_encoder.load_state_dict(state_dict['model'], strict=False)

        ## text feature input ##
#         self.proj_sent = nn.Linear(cfg.hidden_dim, 256) # projection global sentence feature, 512 --> 256
        self.proj_word = nn.Linear(cfg.hidden_dim, 512) # projection masked averaged words/phrases feature, 512 --> 256
        
        self.relu = nn.ReLU(inplace=True) # activation function
#         self.fc1 = nn.Linear(1024, 512) # FC1 after concat image + text projected features, (4*256)
        self.fc1 = nn.Linear(512, 256) # FC2
        self.fc2 = nn.Linear(256, 128) # FC2
        self.do = nn.Dropout(0.5) # dropout layer
        self.output = nn.Linear(128, num_class) # 14 classes including no findings
        
    def forward(self, t, mask):
        ## text input ##
        f_w = self.txt_encoder(t, mask) # N, 160, 512; N, 159, 512; N, 158, 512; N, 512
        f_w_avg = (f_w * mask.unsqueeze(-1)).sum(1) / mask.sum(1).unsqueeze(-1) # [N, 512] masked average pooling from word_feat
#         f_b_avg = (f_b * mask[:,:-1].unsqueeze(-1)).sum(1) / mask[:,:-1].sum(1).unsqueeze(-1) # [N, 512] masked average pooling from bigram_feat
#         f_t_avg = (f_t * mask[:,:-2].unsqueeze(-1)).sum(1) / mask[:,:-2].sum(1).unsqueeze(-1) # [N, 512] masked average pooling from trigram_feat
#         f_wp = (f_w_avg + f_b_avg + f_t_avg) / 3.0 # merge 3 level word/phrase features into one feature [N, 512]
        f_wp = self.relu(self.proj_word(self.do(f_w_avg))) # N, 256, projection word feature
#         f_s = self.relu(self.proj_sent(self.do(f_s))) # N, 256, projection sent feature
        ## classification head ##
#         net = torch.cat((f_s, f_wp), dim=-1) #  N, 512 (512 --> 256 --> 14)
#         net = self.relu(self.fc1(self.do(net)))  # N, 512
        net = self.relu(self.fc1(self.do(f_wp)))  # N, 256
        net = self.relu(self.fc2(self.do(net)))  # N, 128
        out = self.output(self.do(net)) # N, 14
        return out