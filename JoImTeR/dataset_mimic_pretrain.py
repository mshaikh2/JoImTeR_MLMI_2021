import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torchvision as tv

import os, sys
import numpy as np
import pickle
from PIL import Image

MAX_DIM = 2048

train_transform = tv.transforms.Compose([
    tv.transforms.RandomRotation(15),  # rotation will cost 0.1s for each 10 images
    tv.transforms.RandomCrop(MAX_DIM, pad_if_needed=True),  # 0.6s for each 10 images
    tv.transforms.ColorJitter(brightness=[0.5, 1.8]  # colorjitter will cost 0.32s for each 10 images
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


################# image-text-matching ####################       
class MimicDataset(Dataset):
    def __init__(self, root, dataset, max_length, transform=train_transform, mode='train', log_dir='test'):
        super().__init__()

        self.root = root  # save dir
        self.transform = transform
        self.mode = mode
        self.datadict = dataset['data_dict']  # uid: {text:text, filenames:[filename]}
        #         self.classes = dataset['label'] # multi-label one-hot vector
        #         self.datadict = dataset['image'] # uid: {text:text, filenames:[filename]}
        if self.mode == 'train':
            self.keys = dataset['data_split']['train_uids']  # uid list
        elif self.mode == 'val':
            self.keys = dataset['data_split']['val_uids']  # uid list
        elif self.mode == 'test':
            self.keys = dataset['data_split']['test_uids']  # uid list

        self.idx2word = dataset['idx2word']
        self.idx2word[8410] = '[MASK]'
        self.word2ids = dataset['word2idx']
        self.word2ids['[MASK]'] = 8410
        self.__sep_id__ = dataset['word2idx']['[SEP]']
        self.vocab_size = len(dataset['word2idx'])
        self.max_length = max_length + 1
        self.__mask_id__ = 8410  # [MASK] token id

        self.err_log = os.path.join(log_dir, 'err.log')  # create error log
        if not os.path.exists(self.err_log):
            with open(self.err_log, 'w') as f:
                f.write('Epoch 0:\n\n')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        uid = self.keys[idx]

        image_id = np.random.choice(self.datadict[uid]['filenames'])  # get one file name randomly
        image_path = os.path.join(self.root, image_id.replace('dcm', 'jpg'))  # original used 'jpg', try 'png'

        try:
            with Image.open(image_path) as img:
                if self.transform:
                    image = self.transform(img)

        except Exception as ex:
            with open(self.err_log, 'a+') as f:
                f.write('%s\nERR_IMG %s\n' % (ex, image_path))
            #             print(ex)
            #             print(image_path)
            return None  ## return None, collate_fn will ignore this broken sample

        max_len_array = np.zeros(self.max_length, dtype='int')
        cap_mask = np.zeros(self.max_length, dtype='int')
        caption = np.array(self.datadict[uid]['token_ids'])
        if len(caption) <= self.max_length:
            cap_mask[:len(caption)] = 1
            max_len_array[:len(caption)] = caption
        else:
            cap_mask[:] = 1
            max_len_array = caption[:self.max_length]
            max_len_array[-1] = self.__sep_id__
        cap_mask = cap_mask.astype(bool)
        cap_lens = cap_mask.sum(-1)
        return image, max_len_array, cap_mask, uid, cap_lens


def build_dataset_itm(mode='train', cfg=None, out_dir=None):
    data_dir = cfg.dataset_root
    img_dir = os.path.join(data_dir, 'physionet.org/files/', 'mimic-cxr-jpg/2.0.0/')
    with open(os.path.join(data_dir, 'lm_reports/mimic_dataset_mit_normalized.pkl'), 'rb') as f:
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


################## masked language modeling ##################
class MimicDatasetMLM(Dataset):
    def __init__(self, root, dataset, max_length, transform=train_transform, mode='train'):
        super().__init__()

        self.root = root  # save dir
        self.transform = transform
        self.mode = mode
        self.datadict = dataset['data_dict']  # uid: {text:text, filenames:[filename]}
        #         self.classes = dataset['label'] # multi-label one-hot vector
        #         self.datadict = dataset['image'] # uid: {text:text, filenames:[filename]}
        if self.mode == 'train':
            self.keys = dataset['data_split']['train_uids']  # uid list
        elif self.mode == 'val':
            self.keys = dataset['data_split']['val_uids']  # uid list
        elif self.mode == 'test':
            self.keys = dataset['data_split']['test_uids']  # uid list

        self.idx2word = dataset['idx2word']
        self.idx2word[8410] = '[MASK]'
        self.word2ids = dataset['word2idx']
        self.word2ids['[MASK]'] = 8410
        self.__sep_id__ = dataset['word2idx']['[SEP]']
        self.vocab_size = len(dataset['word2idx'])
        self.max_length = max_length + 1
        self.__mask_id__ = 8410  # [MASK] token id

    def __len__(self):
        return len(self.keys)

    def mask_input(self, encoded_texts):  # used for MLM
        # 15% BERT masking
        inp_mask = np.random.rand(*encoded_texts.shape) < 0.15
        # Do not mask special tokens
        inp_mask[encoded_texts <= 3] = False
        # Set targets to -1 by default, it means ignore
        labels = -1 * np.ones(encoded_texts.shape, dtype=int)
        # Set labels for masked tokens
        labels[inp_mask] = encoded_texts[inp_mask]

        # Prepare input
        encoded_texts_masked = np.copy(encoded_texts)
        # Set input to [MASK] which is the last token for the 90% of tokens
        # This means leaving 10% unchanged
        inp_mask_2mask = inp_mask & (np.random.rand(*encoded_texts.shape) < 0.90)
        encoded_texts_masked[
            inp_mask_2mask
        ] = self.__mask_id__  # mask token is the last in the dict

        # Set 10% to a random token
        inp_mask_2random = inp_mask_2mask & (np.random.rand(*encoded_texts.shape) < 1 / 9)
        encoded_texts_masked[inp_mask_2random] = np.random.randint(
            4, self.__mask_id__, inp_mask_2random.sum()
        )

        # Prepare sample_weights to pass to .fit() method
        #         sample_weights = np.ones(labels.shape)
        #         sample_weights[labels == -1] = 0

        # y_labels would be same as encoded_texts i.e input tokens
        #         y_labels = np.copy(encoded_texts)

        return encoded_texts_masked, labels  # , sample_weights

    def __getitem__(self, idx):
        uid = self.keys[idx]
        max_len_array = np.zeros(self.max_length, dtype='int')
        cap_mask = np.zeros(self.max_length, dtype='int')
        caption_unmask = np.array(self.datadict[uid]['token_ids'])

        ### Mask input here ###
        caption, label = self.mask_input(caption_unmask)  ## caption is masked 15% tokens
        label_full = -1 * np.ones(self.max_length, dtype='int')

        ### padding ###
        if len(caption) <= self.max_length:
            label_full[:len(label)] = label  ## used for MLM
            cap_mask[:len(caption)] = 1
            max_len_array[:len(caption)] = caption
        else:
            label_full = label[:self.max_length]
            label_full[-1] = -1
            cap_mask[:] = 1
            max_len_array = caption[:self.max_length]
            max_len_array[-1] = self.__sep_id__

        #         caption = max_len_array
        cap_mask = cap_mask.astype(bool)
        cap_lens = cap_mask.sum(-1)

        return max_len_array, cap_mask, label_full, uid, cap_lens


def build_dataset_mlm(mode='train', cfg=None):
    data_dir = cfg.dataset_root
    img_dir = os.path.join(data_dir, 'physionet.org/files/', 'mimic-cxr-jpg/2.0.0/')
    with open(os.path.join(data_dir, 'lm_reports/mimic_dataset_mit_normalized.pkl'), 'rb') as f:
        dataset = pickle.load(f)

    if mode == 'train':
        data = MimicDatasetMLM(img_dir, dataset,
                               max_length=cfg.max_length, mode=mode,
                               transform=train_transform)
        return data

    elif mode == 'val':
        data = MimicDatasetMLM(img_dir, dataset,
                               max_length=cfg.max_length, mode=mode,
                               transform=val_transform)
        return data

    elif mode == 'test':
        data = MimicDatasetMLM(img_dir, dataset,
                               max_length=cfg.max_length, mode=mode,
                               transform=val_transform)
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")
