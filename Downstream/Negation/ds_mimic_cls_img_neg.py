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
from PIL import Image


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


class MimicDataset(Dataset):
    def __init__(self, root, dataset, max_length, transform=train_transform, mode='train', log_dir='test'):
        super().__init__()
        
        self.root = root #save dir
        self.transform = transform
        self.mode = mode

        self.neg_label = dataset['neg_label']
        self.classes = dataset['label'] # multi-label one-hot vector
        self.datadict = dataset['image'] # uid: {text:text, filenames:[filename]}
        if self.mode == 'train':
            self.keys = dataset['split']['train'] # uid list
        elif self.mode == 'val':
            self.keys = np.concatenate([dataset['split']['val1'], dataset['split']['val2']]) # uid list
        elif self.mode == 'test':
            self.keys = dataset['split']['test'] # uid list
            
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
        if log_dir is not None:
            self.err_log = os.path.join(log_dir, 'err.log') # create error log
            if not os.path.exists(self.err_log):
                with open(self.err_log, 'w') as f:
                    f.write('Epoch 0:\n\n')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        uid = self.keys[idx]
        
        if self.neg_label[uid] is None:
            return None
        
        image_id = np.random.choice(self.datadict[uid]['filenames'])# get one file name randomly
        image_path = os.path.join(self.root, image_id.replace('dcm','jpg')) #original used 'jpg', try 'png'

        try:
            with Image.open(image_path) as img: 
                if self.transform:
                    image = self.transform(img)

        except Exception as ex:
            with open(self.err_log, 'a+') as f:
                f.write('%s\nERR_IMG %s\n' % (ex, image_path))
            return None ## return None, collate_fn will ignore this broken sample
        
        
        classes = torch.tensor(self.neg_label[uid]).float()
        
        return image, classes, uid
        

def build_dataset(mode='train', cfg=None, out_dir=None):
    data_dir = cfg.dataset_root
    img_dir = os.path.join(data_dir, 'physionet.org/files/', 'mimic-cxr-jpg/2.0.0/')
                           
    with open(os.path.join(data_dir, 'lm_reports/class_label_mit_v3.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    if mode == 'train':
        data = MimicDataset(img_dir, dataset, 
                               max_length=cfg.max_length, mode=mode, 
                               transform=train_transform, log_dir=out_dir)
        return data

    elif mode == 'val':
        data = MimicDataset(img_dir, dataset, 
                               max_length=cfg.max_length, mode=mode, 
                               transform=val_transform, log_dir=out_dir)
        return data
    
    elif mode == 'test':
        data = MimicDataset(img_dir, dataset, 
                               max_length=cfg.max_length, mode=mode, 
                               transform=val_transform, log_dir=out_dir)
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")