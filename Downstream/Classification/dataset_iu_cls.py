import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torchvision as tv

import os,sys
import pandas as pd
import numpy as np
import pickle
from PIL import Image
# import spacy
# import scispacy
from tqdm import tqdm 
# from nltk.tokenize import RegexpTokenizer
# from transformers import BertTokenizer


MAX_DIM = 2048

train_transform = tv.transforms.Compose([
    tv.transforms.RandomRotation(15), # rotation will cost 0.1s for each 10 images
    tv.transforms.RandomCrop(MAX_DIM, pad_if_needed=True), # 0.6s for each 10 images
    tv.transforms.ColorJitter(brightness=[0.5, 1.8] # colorjitter will cost 0.32s for each 10 images
                              , contrast=[0.5, 1.8]
                              , saturation=[0.5, 1.8]),
#     RandomTranslateCrop(MAX_DIM), # used for skimage.io, slower
    tv.transforms.ToTensor(), 
#     ConvertTensor(), # used for skimage.io
    tv.transforms.Normalize(0.5, 0.5)
])

val_transform = tv.transforms.Compose([
    tv.transforms.CenterCrop(MAX_DIM),
#     CenterCrop(MAX_DIM), # used for skimage.io
    tv.transforms.ToTensor(),
#     ConvertTensor(), # used for skimage.io    
    tv.transforms.Normalize(0.5, 0.5)
])

class IUDataset(Dataset):
    def __init__(self, root, dataset, max_length, transform=val_transform, mode='test', log_dir='test'):
        super().__init__()
        
        self.root = root #save dir
        self.transform = transform
        self.mode = mode

        self.classes = dataset['classes'] # multi-label list
        self.datadict = dataset['data_dict'] # uid: {text:text, filenames:[filename]}
        if self.mode == 'train':
            self.keys = dataset['data_split']['train_uids'] # uid list
        elif self.mode == 'val':
            self.keys = dataset['data_split']['val_uids'] # uid list
        elif self.mode == 'test':
            self.keys = dataset['data_split']['test_uids'] # uid list
        elif self.mode == 'few1':
            self.keys = dataset['few_shot']['fewshot1'] # uid list
        elif self.mode == 'few5':
            self.keys = dataset['few_shot']['fewshot5'] # uid list
        elif self.mode == 'test1':
            self.keys = dataset['few_shot']['test1'] # uid list
        elif self.mode == 'test5':
            self.keys = dataset['few_shot']['test5'] # uid list
        
        self.idx2word = dataset['idx2word']
        self.idx2word[8410] = '[MASK]'
        self.word2idx = dataset['word2idx']
        self.word2idx['[MASK]'] = 8410
        self.__sep_id__ = dataset['word2idx']['[SEP]']
        self.vocab_size = len(dataset['word2idx'])
        self.max_length = max_length + 1
#         self.__mask_id__ = 8410 # [MASK] token id
        
#         ## classification params
#         self.class_to_idx = {
#             'Atelectasis': 0, 'Cardiomegaly': 1, 'Consolidation': 2, 'Edema': 3, 'Enlarged Cardiomediastinum': 4, 
#             'Fracture': 5, 'Lung Lesion': 6, 'Lung Opacity': 7, 'No Finding': 8, 'Pleural Effusion': 9, 
#             'Pleural Other': 10, 'Pneumonia': 11, 'Pneumothorax': 12, 'Support Devices': 13
#         }
        
#         self.idx_to_class = {
#             0:'Atelectasis', 1:'Cardiomegaly', 2:'Consolidation', 3:'Edema', 4:'Enlarged Cardiomediastinum', 
#             5:'Fracture', 6:'Lung Lesion', 7:'Lung Opacity', 8:'No Finding', 9:'Pleural Effusion', 
#             10:'Pleural Other', 11:'Pneumonia', 12:'Pneumothorax', 13:'Support Devices'
#         }

        self.class_to_idx = {'no finding':8
                            ,'edema':3
                            ,'consolidation':2
                            ,'pneumonia':11
                            ,'pneumothorax':12
                            ,'atelectasis':0
                            ,'cardiomegaly':1
                            ,'effusion':9}
        
        self.idx_to_class = {0:'atelectasis'
                             ,1:'cardiomegaly'
                             ,9:'effusion'
                             ,3:'edema'
                             ,2:'consolidation'
                             ,11:'pneumonia'
                             ,12:'pneumothorax'
                             ,8:'no finding'}
        
        self.num_classes = 14
        

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        uid = self.keys[idx]
        
        image_id = np.random.choice(self.datadict[uid]['filenames'])# get one file name randomly
        image_path = os.path.join(self.root, image_id) #original used 'jpg', try 'png'

        try:
            with Image.open(image_path) as img: 
                if self.transform:
                    image = self.transform(img)

        except Exception as ex:
#             print(ex)
            with open(self.err_log, 'a+') as f:
                f.write('%s\nERR_IMG %s\n' % (ex, image_path))
            return None ## return None, collate_fn will ignore this broken sample
        
        classes = torch.tensor([self.class_to_idx[x] for x in self.classes[uid]])
        y_onehot = torch.FloatTensor(self.num_classes).zero_()
        y_onehot.scatter_(0, classes, 1)
        
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
        
        return image, max_len_array, cap_mask, y_onehot, uid, cap_lens
        

def build_dataset(mode='test', cfg=None, out_dir=None):
    data_dir = cfg.dataset_root
    img_dir = os.path.join(data_dir, 'images', 'images_normalized')
    with open(os.path.join(data_dir, 'cleaned_dataset_v3.pickle'), 'rb') as f:
        dataset = pickle.load(f)
    if mode == 'train':
        data = IUDataset(img_dir, dataset, 
                               max_length=cfg.max_length, mode=mode, 
                               transform=train_transform, log_dir=out_dir)
        return data
    
    elif mode == 'val':
        data = IUDataset(img_dir, dataset, 
                               max_length=cfg.max_length, mode=mode, 
                               transform=val_transform, log_dir=out_dir)
        return data
    
    elif mode == 'test':
        data = IUDataset(img_dir, dataset, 
                               max_length=cfg.max_length, mode=mode, 
                               transform=val_transform, log_dir=out_dir)
        return data
    
    elif mode == 'few1':
        data = IUDataset(img_dir, dataset, 
                               max_length=cfg.max_length, mode=mode, 
                               transform=train_transform, log_dir=out_dir)
        return data
    
    elif mode == 'few5':
        data = IUDataset(img_dir, dataset, 
                               max_length=cfg.max_length, mode=mode, 
                               transform=train_transform, log_dir=out_dir)
        return data
    
    elif mode == 'test1':
        data = IUDataset(img_dir, dataset, 
                               max_length=cfg.max_length, mode=mode, 
                               transform=val_transform, log_dir=out_dir)
        return data
    
    elif mode == 'test5':
        data = IUDataset(img_dir, dataset, 
                               max_length=cfg.max_length, mode=mode, 
                               transform=val_transform, log_dir=out_dir)
        return data
    
    else:
        raise NotImplementedError(f"{mode} not supported")
        
## collate_fn for handling None type item due to image corruption ##
## This will return batch size - broken image number ##
# def collate_fn_ignore_none(batch):
#     batch = list(filter(lambda x: x is not None, batch))
#     return torch.utils.data.dataloader.default_collate(batch)