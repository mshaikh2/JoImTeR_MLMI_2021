import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torchvision as tv

import os, sys
# import pandas as pd
import numpy as np
import pickle
# from PIL import Image


class MimicDataset(Dataset):
    def __init__(self, root, dataset, max_length, mode='train', log_dir='test'):
        super().__init__()
        
        self.root = root #save dir
#         self.transform = transform
        self.mode = mode

        self.neg_label = dataset['neg_label']
        self.manual_neg_label_with_zeros = dataset['manual_neg_label_with_zeros']

        self.classes = dataset['label'] # multi-label one-hot vector
        self.datadict = dataset['image'] # uid: {text:text, filenames:[filename]}
        if self.mode == 'train':
            self.keys = dataset['split']['train'] # uid list
        elif self.mode == 'val':
            self.keys = np.concatenate([dataset['split']['val1'], dataset['split']['val2']]) # uid list
        elif self.mode == 'test':
            self.keys = dataset['split']['test'] # uid list
            
        elif self.mode == 'manual_gt_with_zeros':
            self.keys = dataset['split']['manual_with_zeros'] # uid list
        
        self.idx2word = dataset['idx2word']
        self.idx2word[8410] = '[MASK]'
        self.word2idx = dataset['word2idx']
        self.word2idx['[MASK]'] = 8410
        self.__sep_id__ = dataset['word2idx']['[SEP]']
        self.vocab_size = len(dataset['word2idx'])
        self.max_length = max_length + 1
#         self.__mask_id__ = 8410 # [MASK] token id
        
        ## classification params
        self.class_to_idx = {
            'Atelectasis': 0, 'Cardiomegaly': 1, 'Consolidation': 2, 'Edema': 3, 'Enlarged Cardiomediastinum': 4, 
            'Fracture': 5, 'Lung Lesion': 6, 'Lung Opacity': 7, 'No Finding': 8, 'Pleural Effusion': 9, 
            'Pleural Other': 10, 'Pneumonia': 11, 'Pneumothorax': 12, 'Support Devices': 13
        }
        
        self.idx_to_class = {
            0:'Atelectasis', 1:'Cardiomegaly', 2:'Consolidation', 3:'Edema', 4:'Enlarged Cardiomediastinum', 
            5:'Fracture', 6:'Lung Lesion', 7:'Lung Opacity', 8:'No Finding', 9:'Pleural Effusion', 
            10:'Pleural Other', 11:'Pneumonia', 12:'Pneumothorax', 13:'Support Devices'
        }
        
        self.num_classes = 14
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        uid = self.keys[idx]
        
        
        if self.mode == 'manual_gt_with_zeros':
            if self.manual_neg_label_with_zeros[uid] is None:
                return None
        else:
            if self.neg_label[uid] is None:
                return None
        
#         classes = torch.tensor(self.classes[uid]).float()

        if self.mode == 'manual_gt_with_zeros':
            classes = torch.tensor(self.manual_neg_label_with_zeros[uid]).float()
        else:
            classes = torch.tensor(self.neg_label[uid]).float()
        
        ## load text input ##
        max_len_array = np.zeros(self.max_length, dtype='int')
        cap_mask = np.zeros(self.max_length, dtype='int')
        caption = np.array(self.datadict[uid]['token_ids'])
        if len(caption)<=self.max_length:
            cap_mask[:len(caption)] = 1
            max_len_array[:len(caption)] = caption
        else:
            cap_mask[:] = 1
            max_len_array = caption[:self.max_length]
            max_len_array[-1] = self.__sep_id__
#         caption = max_len_array
        cap_mask = cap_mask.astype(bool)
        cap_lens = cap_mask.sum(-1)
        
        return max_len_array, cap_mask, classes, uid, cap_lens
        

def build_dataset(mode='train', cfg=None, out_dir=None):
    data_dir = cfg.dataset_root
    img_dir = os.path.join(data_dir, 'physionet.org/files/', 'mimic-cxr-jpg/2.0.0/')
    with open(os.path.join(data_dir, 'lm_reports/class_label_mit_v3.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    with open(os.path.join(data_dir,'lm_reports/mimic_dataset_mit_normalized.pkl'),'rb') as f2:
        dataset_token = pickle.load(f2)
    dataset['word2idx'] = dataset_token['word2idx'] # copy the token dicts to the dataset
    dataset['idx2word'] = dataset_token['idx2word']
    
    if mode == 'train':
        data = MimicDataset(img_dir, dataset, 
                               max_length=cfg.max_length, mode=mode, 
                               log_dir=out_dir)
        return data

    elif mode == 'val':
        data = MimicDataset(img_dir, dataset, 
                               max_length=cfg.max_length, mode=mode, 
                               log_dir=out_dir)
        return data
    
    elif mode == 'test':
        data = MimicDataset(img_dir, dataset, 
                               max_length=cfg.max_length, mode=mode, 
                               log_dir=out_dir)
        return data
    
    elif mode == 'manual_gt_with_zeros':
        data = MimicDataset(img_dir, dataset, 
                               max_length=cfg.max_length, mode=mode, 
                               log_dir=out_dir)
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")