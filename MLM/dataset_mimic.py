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

# import spacy
# import scispacy
# from tqdm import tqdm

# from skimage import io
# import scipy.ndimage as ndimage
# from math import floor, ceil
# from nltk.tokenize import RegexpTokenizer
# from transformers import BertTokenizer

# nlp = spacy.load('en_core_sci_md')


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


###### image-text-matching #######        
class MimicDataset(Dataset):
    def __init__(self, root, dataset, max_length, transform=train_transform, mode='train', log_dir='test'):
        super().__init__()

        self.root = root  # save dir
        self.transform = transform
        self.mode = mode
        #         vocab='allenai/scibert_scivocab_uncased'
        #         self.tokenizer = BertTokenizer.from_pretrained(vocab, do_lower=True)

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
        #         self.__mask_id__ = 8410 # [MASK] token id

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
                #             image = load_image(image_path) # use skimage.io to open images
                #             if image.size[0] < MAX_DIM or image.size[1] < MAX_DIM:
                #                 image = tv.transforms.Resize(MAX_DIM)(image)
                #                 # print(image.size,':', self.datadict[uid]['filenames'])
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
        #         caption = normalize_report(caption)['sentences']
        #         caption_encoded = self.tokenizer.encode_plus(
        #                                     caption, max_length=self.max_length, padding='max_length',
        #                                     return_attention_mask=True, return_token_type_ids=False, truncation=True)

        #         caption = np.array(caption_encoded['input_ids'])
        #         cap_mask = np.array(caption_encoded['attention_mask']).astype(bool)
        #         caption = max_len_array
        cap_mask = cap_mask.astype(bool)
        cap_lens = cap_mask.sum(-1)
        return image, max_len_array, cap_mask, uid, cap_lens


def build_dataset(mode='train', cfg=None, out_dir=None):
    data_dir = cfg.dataset_root
    img_dir = os.path.join(data_dir, 'physionet.org/files/', 'mimic-cxr-jpg/2.0.0/')
    with open(os.path.join(data_dir, 'lm_reports/mimic_dataset_mit_normalized.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    #     with open(os.path.join(data_dir,'lm_reports/class_label_mit.pkl'),'rb') as f:
    #         dataset = pickle.load(f)
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


class MimicDatasetMLM(Dataset):
    def __init__(self, root, dataset, max_length, transform=train_transform, mode='train', log_dir='test'):
        super().__init__()

        self.root = root  # save dir
        self.transform = transform
        self.mode = mode
        #         vocab='allenai/scibert_scivocab_uncased'
        #         self.tokenizer = BertTokenizer.from_pretrained(vocab, do_lower=True)

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

    #         self.err_log = os.path.join(log_dir, 'err.log') # create error log
    #         if not os.path.exists(self.err_log):
    #             with open(self.err_log, 'w') as f:
    #                 f.write('Epoch 0:\n\n')

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

        #         image_id = np.random.choice(self.datadict[uid]['filenames'])# get one file name randomly
        #         image_path = os.path.join(self.root, image_id.replace('dcm','jpg')) #original used 'jpg', try 'png'

        #         try:
        #             with Image.open(image_path) as img:
        # #             image = load_image(image_path) # use skimage.io to open images
        # #             if image.size[0] < MAX_DIM or image.size[1] < MAX_DIM:
        # #                 image = tv.transforms.Resize(MAX_DIM)(image)
        # #                 # print(image.size,':', self.datadict[uid]['filenames'])
        #                 if self.transform:
        #                     image = self.transform(img)

        #         except Exception as ex:
        #             with open(self.err_log, 'a+') as f:
        #                 f.write('%s\nERR_IMG %s\n' % (ex, image_path))
        # #             print(ex)
        # #             print(image_path)
        #             return None ## return None, collate_fn will ignore this broken sample

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
## custom functions for transforms using skimage and scipy

# class RandomTranslateCrop(object):
#     """Translate and crop the image in a sample.
#     Args:
#         output_size (tuple or int): Desired output size. 
#         If int, square crop is made.
#     """

#     def __init__(self, output_size, shift_mean=0,
#                  shift_std=200, rotation_mean=0, rotation_std=20):
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size
#         self.shift_mean = shift_mean
#         self.shift_std = shift_std
#         self.rotation_mean = rotation_mean
#         self.rotation_std = rotation_std

#     def __call__(self, image):
#         image = self.__translate_2Dimage(image)
#         #image = self.__rotate_2Dimage(image)
#         h, w = image.shape[0:2]
#         new_h, new_w = self.output_size

#         if new_h>h or new_w>w:
#             raise ValueError('This image needs to be padded!')

#         top = floor((h - new_h) / 2)
#         down = top + new_h
#         left = floor((w - new_w) / 2)
#         right = left + new_w

#         return image[top:down, left:right]

#     def __translate_2Dimage(self, image):
#         'Translate 2D images as data augmentation'
#         h, w = image.shape[0:2]
#         h_output, w_output = self.output_size[0:2]

#         # Generate random Gaussian numbers for image shift as data augmentation
#         shift_h = int(np.random.normal(self.shift_mean, self.shift_std))
#         shift_w = int(np.random.normal(self.shift_mean, self.shift_std))
#         if abs(shift_h) > 2 * self.shift_std:
#             shift_h = 0
#         if abs(shift_w) > 2 * self.shift_std:
#             shift_w = 0

#         # Pad the 2D image
#         pad_h_length = max(0, float(h_output - h))
#         pad_h_length_1 = floor(pad_h_length / 2) + 4  # 4 is extra padding
#         pad_h_length_2 = floor(pad_h_length / 2) + 4  # 4 is extra padding
#         pad_h_length_1 = pad_h_length_1 + max(shift_h , 0)
#         pad_h_length_2 = pad_h_length_2 + max(-shift_h , 0)

#         pad_w_length = max(0, float(w_output - w))
#         pad_w_length_1 = floor(pad_w_length / 2) + 4  # 4 is extra padding
#         pad_w_length_2 = floor(pad_w_length / 2) + 4  # 4 is extra padding
#         pad_w_length_1 = pad_w_length_1 + max(shift_w , 0)
#         pad_w_length_2 = pad_w_length_2 + max(-shift_w , 0)

#         image = np.pad(image, ((pad_h_length_1, pad_h_length_2), (pad_w_length_1, pad_w_length_2)),
#                        'constant', constant_values=((0, 0), (0, 0)))

#         return image

#     def __rotate_2Dimage(self, image):
#         'Rotate 2D images as data augmentation'

#         # Generate a random Gaussian number for image rotation angle as data augmentation
#         angle = np.random.normal(self.rotation_mean, self.rotation_std)
#         if abs(angle) > 2 * self.rotation_std:
#             angle = 0
#             return image

#         return ndimage.rotate(image, angle)


# class CenterCrop(object):
#     """Crop randomly the image in a sample.
#     Args:
#         output_size (tuple or int): Desired output size. 
#         If int, square crop is made.
#     """

#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size

#     def __call__(self, image):
#         image = self.__pad_2Dimage(image)
#         h, w = image.shape[0:2]
#         new_h, new_w = self.output_size

#         if new_h>h or new_w>w:
#             raise ValueError('This image needs to be padded!')

#         top = floor((h - new_h) / 2)
#         down = top + new_h
#         left = floor((w - new_w) / 2)
#         right = left + new_w

#         return image[top:down, left:right]

#     def __pad_2Dimage(self, image):
#         'Pad 2D images to match output_size'
#         h, w = image.shape[0:2]
#         h_output, w_output = self.output_size[0:2]

#         pad_h_length = max(0, float(h_output - h))
#         pad_h_length_1 = floor(pad_h_length / 2) + 4  # 4 is extra padding
#         pad_h_length_2 = floor(pad_h_length / 2) + 4  # 4 is extra padding

#         pad_w_length = max(0, float(w_output - w))
#         pad_w_length_1 = floor(pad_w_length / 2) + 4  # 4 is extra padding
#         pad_w_length_2 = floor(pad_w_length / 2) + 4  # 4 is extra padding

#         image = np.pad(image, ((pad_h_length_1, pad_h_length_2), (pad_w_length_1, pad_w_length_2)),
#                        'constant', constant_values=((0, 0), (0, 0)))

#         return image


# class ConvertTensor(object):
#     """Convert the numpy image to Tensor.
#     """
#     def __call__(self, image):
#         image = torch.tensor(image)
#         image = image.unsqueeze(0)
#         return image

# # Load an .npy or .png image using skimage.io
# def load_image(img_path):
#     if img_path[-3:] == 'npy':
#         image = np.load(img_path)
#     if img_path[-3:] == 'png' or img_path[-3:] == 'jpg':
#         image = io.imread(img_path)
#         image = image.astype(np.float32)
#         image = image/np.max(image)
#     return image

############ main dataset code ############

# def remove_whitespace(line):
#     return str(" ".join(line.split()).strip())

# def list_to_string(sentence):
#     return " ".join(sentence)

# def normalize_report(row):
#     report = row
#     report_sentences = nlp(report)
#     new_report_sentences = []
#     for sentence in report_sentences.sents:
#         index_to_keep_dict = {} # index: {keep that token or not, replace_with}
#         for index in range(0, len(sentence)):
#             token = sentence[index]
#             if index < len(sentence) - 1:
#                 next_token = sentence[index + 1]
#                 if token.is_punct and next_token.is_punct and token.text.strip() == next_token.text.strip():
#                     # when it is the same type of punctuation
#                     index_to_keep_dict[index] = {'keep': False, 'replace_with': None}
#                     continue
#             if token.like_num:
#                 index_to_keep_dict[index] = {'keep': True, 'replace_with': 'NUMBER'}
#             else:
#                 index_to_keep_dict[index] = {'keep': True, 'replace_with': None}
#         # generate a new sentence based on this replacement
#         new_sentence = []
#         for index in range(0, len(sentence)):
#             token = sentence[index]
#             if not index_to_keep_dict[index]['keep']:
#                 continue # don't append when there is a double punctuation happening
#             if index_to_keep_dict[index]['replace_with'] is not None:
#                 new_sentence.append(index_to_keep_dict[index]['replace_with'])
#                 continue
#             new_sentence.append(token.text)
#         s = list_to_string(new_sentence).strip()
#         s = s.replace('DEID', '')
#         s = remove_whitespace(s)
#         new_report_sentences.append(s)
#     return {'sentences': ' '.join(new_report_sentences).replace(',','').replace('.','')}
