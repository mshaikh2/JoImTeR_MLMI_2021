from __future__ import print_function
from six.moves import range
import os
import time
import numpy as np
import sys
import re
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image
import datetime
import dateutil.tz
import shutil
from misc.utils import mkdir_p
# from datasets import prepare_data
from model import TextEncoder, ImageEncoder, TextEncoderMLM
# from InceptionScore import calculate_inception_score

from misc.losses import sent_loss, words_loss

from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader

import math
from tqdm import tqdm
import timeit
# from catr.engine import train_one_epoch, evaluate
from misc.config import Config
from transformers import BertConfig  # , BertTokenizer

# from nltk.tokenize import RegexpTokenizer


cfg = Config()  # initialize catr config here


# tokenizer = BertTokenizer.from_pretrained(cfg.vocab, do_lower=True)
# retokenizer = BertTokenizer.from_pretrained("catr/damsm_vocab.txt", do_lower=True)
# # reg_tokenizer = RegexpTokenizer(r'\w+')
# frozen_list_image_encoder = ['Conv2d_1a_3x3','Conv2d_2a_3x3','Conv2d_2b_3x3','Conv2d_3b_1x1','Conv2d_4a_3x3']

# ################# Joint Image Text Representation (JoImTeR) learning task############################ #
class JoImTeR(object):
    def __init__(self, output_dir, data_loader, dataloader_val):
        if cfg.TRAIN:
            self.model_dir = os.path.join(output_dir, 'Model')
            #             self.image_dir = os.path.join(output_dir, 'Image')
            mkdir_p(self.model_dir)
        #             mkdir_p(self.image_dir)

        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.batch_size = data_loader.batch_size
        self.val_batch_size = dataloader_val.batch_size
        self.max_epoch = cfg.epochs
        self.snapshot_interval = cfg.snapshot_interval

        self.data_loader = data_loader
        self.dataloader_val = dataloader_val
        #         self.log_file = os.path.join(output_dir, 'err.log')
        #         print('Write broken images to %s.' % self.log_file)
        self.num_batches = len(self.data_loader)
        self.bert_config = BertConfig(vocab_size=data_loader.dataset.vocab_size, hidden_size=512, num_hidden_layers=3,
                                      num_attention_heads=8, intermediate_size=2048, hidden_act='gelu',
                                      hidden_dropout_prob=cfg.hidden_dropout_prob,
                                      attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
                                      max_position_embeddings=512, layer_norm_eps=1e-12,
                                      initializer_range=0.02, type_vocab_size=2, pad_token_id=0)

    def build_models(self):
        # ################### encoders ################################# #      
        #         image_encoder = ImageEncoder(output_channels=cfg.hidden_dim)
        #         for p in image_encoder.parameters(): # make image encoder grad on
        #             p.requires_grad = True

        text_encoder = TextEncoderMLM(bert_config=self.bert_config, output_channels=cfg.hidden_dim)
        for p in text_encoder.parameters():  # make text encoder grad on
            p.requires_grad = True

        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
        #             image_encoder = image_encoder.cuda()

        #         optimizerI = torch.optim.AdamW(image_encoder.parameters()
        #                                            , lr=cfg.lr
        #                                            , weight_decay=cfg.weight_decay)
        optimizerT = torch.optim.AdamW(text_encoder.parameters()
                                       , lr=cfg.lr
                                       , weight_decay=cfg.weight_decay)

        ### lr_schedular: use stepwise schedular for finetuning ###
        epoch = 0
        if cfg.text_encoder_path != '':
            epoch = re.findall(r'\d+', os.path.basename(cfg.text_encoder_path))[-1]
            epoch = int(epoch) + 1
            print('From checkpoint epoch %d' % epoch)

        epochs = self.max_epoch
        start_epoch = 0  # normaly its the step 0
        drop_epoch = cfg.lr_drop
        batch_step = self.num_batches
        start_step = start_epoch * batch_step  # normaly its the step 0
        drop_step = drop_epoch * batch_step  # where to start drop lr
        total_steps = (epochs - start_epoch) * batch_step

        lambda1 = lambda step: min((total_steps - step) / (total_steps - drop_step), 1)  # stepwise linear decay
        #         lambda1 = lambda step: 0.3 ** (step / batch_step) # stepwise

        if cfg.scheduler_step:
            #             lr_schedulerI = torch.optim.lr_scheduler.LambdaLR(optimizerI, lr_lambda=lambda1)
            lr_schedulerT = torch.optim.lr_scheduler.LambdaLR(optimizerT, lr_lambda=lambda1)
        else:
            #             lr_schedulerI = torch.optim.lr_scheduler.StepLR(optimizerI, cfg.lr_drop, gamma=cfg.lr_gamma)
            lr_schedulerT = torch.optim.lr_scheduler.StepLR(optimizerT, cfg.lr_drop, gamma=cfg.lr_gamma)
        ##### load checkpoints if exists #########

        if cfg.text_encoder_path != '':
            #             img_encoder_path = cfg.text_encoder_path.replace('text_encoder', 'image_encoder')
            #             print('Load image encoder checkpoint from:', img_encoder_path)
            #             state_dict = torch.load(img_encoder_path, map_location='cpu')
            #             if 'model' in state_dict.keys():
            #                 image_encoder.load_state_dict(state_dict['model'])
            #                 optimizerI.load_state_dict(state_dict['optimizer'])
            #                 if cfg.scheduler_init:
            #                     lr_schedulerI.load_state_dict(state_dict['lr_scheduler'])
            #             else:
            #                 image_encoder.load_state_dict(state_dict)

            text_encoder_path = cfg.text_encoder_path
            print('Load text encoder checkpoint from:', text_encoder_path)
            state_dict = torch.load(text_encoder_path, map_location='cpu')
            if 'model' in state_dict.keys():
                text_encoder.load_state_dict(state_dict['model'])
                optimizerT.load_state_dict(state_dict['optimizer'])
                epoch = state_dict['epoch']
                val_loss = state_dict['val_loss']
                if cfg.scheduler_init:
                    lr_schedulerT.load_state_dict(state_dict['lr_scheduler'])
            else:
                text_encoder.load_state_dict(state_dict)
        # ########################################################### #

        #         return [text_encoder, image_encoder, epoch, optimizerI, optimizerT, lr_schedulerI, lr_schedulerT]
        return [text_encoder, optimizerT]

    #     def save_model(self, image_encoder, text_encoder, optimizerI, optimizerT, lr_schedulerI, lr_schedulerT, epoch):
    def save_model(self, text_encoder, optimizerT, epoch, best_val_loss):

        # save image encoder model here
        #         torch.save({
        #             'model': image_encoder.state_dict(),
        #             'optimizer': optimizerI.state_dict(),
        #             'lr_scheduler': lr_schedulerI.state_dict(),
        #         }, '%s/image_encoder%d.pth' % (self.model_dir, epoch))

        # save text encoder model here
        torch.save({
            'model': text_encoder.state_dict(),
            'optimizer': optimizerT.state_dict(),
            #             'lr_scheduler': lr_schedulerT.state_dict(),
            'epoch': epoch,
            'val_loss': best_val_loss,
        }, '%s/text_encoder.pth' % self.model_dir)

    def train(self):

        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        #     LAMBDA_FT,LAMBDA_FI,LAMBDA_DAMSM=01,50,10
        tb_dir = '../tensorboard/{0}_{1}_{2}'.format(cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        mkdir_p(tb_dir)
        tbw = SummaryWriter(log_dir=tb_dir)  # Tensorboard logging

        ####### init models ########
        #         text_encoder, image_encoder, start_epoch, optimizerI, optimizerT, lr_schedulerI, lr_schedulerT = self.build_models()
        text_encoder, optimizerT = self.build_models()
        start_epoch = 0
        #         print('Starting epoch %d' % start_epoch)

        ##### init data #############################
        ## due to the broken image collate_fn, batch size could change. Move this inside step loop
        #         labels = Variable(torch.LongTensor(range(self.batch_size))) # used for matching loss
        #         if cfg.CUDA:
        #             labels = labels.cuda()
        #         batch_size = self.batch_size
        #         match_labels = self.prepare_labels() # deprecated.
        ##################################################################

        text_encoder.train()
        #         image_encoder.train()
        mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        if cfg.CUDA:
            mlm_loss_fn = mlm_loss_fn.cuda()

        ###############################################################

        tensorboard_step = 0
        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches

        best_val_loss = 10000.0  ## used for saving model checkpoints

        #### print lambdas ###
        #         print('LAMBDA_GEN:{0},LAMBDA_CAP:{1},LAMBDA_FT:{2},LAMBDA_FI:{3},LAMBDA_DAMSM:{4}'.format(cfg.TRAIN.SMOOTH.LAMBDA_GEN
        #                                                                                                   ,cfg.TRAIN.SMOOTH.LAMBDA_CAP
        #                                                                                                   ,cfg.TRAIN.SMOOTH.LAMBDA_FT
        #                                                                                                   ,cfg.TRAIN.SMOOTH.LAMBDA_FI
        #                                                                                                   ,cfg.TRAIN.SMOOTH.LAMBDA_DAMSM))
        for epoch in range(start_epoch, self.max_epoch):

            ##### set everything to trainable ####
            text_encoder.train()
            #             image_encoder.train()
            ####################################

            ####### init loss variables ############          
            #             s_total_loss0 = 0
            #             s_total_loss1 = 0
            #             w_total_loss0 = 0
            #             w_total_loss1 = 0
            #             total_damsm_loss = 0
            total_mlm_loss = 0

            ####### print out lr of each optimizer before training starts, make sure lrs are correct #########
            print('[epoch: %d] Learning rates: lr_t %.4e'
                  % (epoch, optimizerT.param_groups[0]['lr']))

            #########################################################################################

            start_t = time.time()

            data_iter = iter(self.data_loader)
            #             step = 0
            pbar = tqdm(range(self.num_batches))
            ## training each epoch 
            for step in pbar:
                #                 break
                ### use stepwise linear schedule here ###
                #                 if cfg.scheduler_step:
                #                     lr_schedulerI.step()
                #                     lr_schedulerT.step()
                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                captions, masks, labels, class_ids, cap_lens = data_iter.next()
                class_ids = class_ids.numpy()
                ## due to the broken image collate_fn, batch size could change. Move labels inside step loop
                #                 batch_size = imgs.shape[0] # actual batch size could be different from self.batch_size
                #                 labels = torch.LongTensor(range(batch_size)) # used for matching loss
                #                 match_labels = self.prepare_labels(batch_size) # used for matching loss

                if cfg.CUDA:
                    captions, masks, labels, cap_lens = captions.cuda(), masks.cuda(), labels.cuda(), cap_lens.cuda()
                #                     labels = labels.cuda()
                # add images, image masks, captions, caption masks for catr model

                ################## feedforward damsm model ##################
                #                 text_encoder.zero_grad()
                optimizerT.zero_grad()
                # hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                preds = text_encoder(captions, masks)

                mlm_loss = mlm_loss_fn(preds, labels)
                total_mlm_loss += mlm_loss.item()
                mlm_loss.backward()
                torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), cfg.clip_max_norm)
                optimizerT.step()

                ##################### loss values for each step #########################################
                tbw.add_scalar('Pretrain_step/mlm_step_loss', mlm_loss.item(), step + epoch * self.num_batches)

                ################################################################################################    

                ############ tqdm descriptions showing running average loss in terminal ##############################
                pbar.set_description('lr: %.4e, mlm %.4f'
                                     % (optimizerT.param_groups[0]['lr'],
                                        total_mlm_loss / (step + 1)))

                ######################################################################################################
                ##########################################################
            v_mlm_loss, acc = self.evaluate(text_encoder)
            print('[epoch: %d] val_mlm_loss: %.4f, accuracy: %.4f' % (epoch, v_mlm_loss, acc))
            ### val losses ###
            tbw.add_scalar('Pretrain_val_step/val_mlm_loss', v_mlm_loss, epoch)
            tbw.add_scalar('Pretrain_val_step/val_mlm_acc', acc, epoch)

            #             if not cfg.scheduler_step:
            #                 lr_schedulerI.step()
            #                 lr_schedulerT.step()

            end_t = time.time()

            #             if epoch % cfg.snapshot_interval == 0:
            if v_mlm_loss < best_val_loss:
                best_val_loss = v_mlm_loss
                self.save_model(text_encoder, optimizerT, epoch, best_val_loss)
                print(
                    'New checkpoint is saved. Epoch: %d, best_val_loss: %.5f, acc: %.5f' % (epoch, best_val_loss, acc))

    #             broken_images = [] # find all the broken images within this epoch
    #             with open(self.log_file, 'r') as f:
    #                 lines = f.readlines()
    #                 for line in reversed(lines):
    #                     if 'ERR_IMG' in line:
    #                         broken_images.append(line[8:-1])
    #                     if 'Epoch' in line:
    #                         last_epoch = int(re.findall('\d+', line)[-1])
    #                         break
    #             #### copy broken images from original storage here ####
    #             for tar in broken_images:
    #                 org = tar.replace('SStor1/zhanghex', 'MyDataStor2')
    #                 shutil.copyfile(org, tar)
    #             #######################################################
    #             with open(self.log_file, 'a+') as f: # start a new epoch record
    #                 f.write('\nEpoch %d:\n\n' % (last_epoch + 1))

    #         self.save_model(image_encoder, text_encoder, optimizerI, optimizerT, lr_schedulerI, lr_schedulerT, epoch)

    @torch.no_grad()
    def evaluate(self, trx_model):
        #         cnn_model.eval()
        trx_model.eval()
        #         s_total_loss = 0
        #         w_total_loss = 0
        total_mlm_loss = 0
        ### add caption criterion here. #####
        ## same reason as train(), batch_size could change when image broken
        #         labels = Variable(torch.LongTensor(range(batch_size))) # used for matching loss
        #         if cfg.CUDA:
        #             labels = labels.cuda()
        #####################################
        total_t_num = 0
        total_num = 0

        val_data_iter = iter(self.dataloader_val)
        for step in tqdm(range(len(val_data_iter)), leave=False):  #
            captions, masks, labels, class_ids, cap_lens = val_data_iter.next()
            class_ids = class_ids.numpy()
            #             batch_size = real_imgs.shape[0]
            #             labels = torch.LongTensor(range(batch_size)) # used for matching loss
            if cfg.CUDA:
                captions, masks, labels, cap_lens = captions.cuda(), masks.cuda(), labels.cuda(), cap_lens.cuda()
            #                 labels = labels.cuda()
            #             words_features, sent_code = cnn_model(real_imgs)
            preds = trx_model(captions, masks)
            mlm_loss = nn.functional.cross_entropy(preds, labels, ignore_index=-1, reduction='mean')
            total_mlm_loss += mlm_loss.item()

            #### get the accuracy of the prediction ####
            preds_cpu = preds.data.cpu().numpy()
            preds_cpu = np.argmax(preds_cpu, axis=1)
            labels_cpu = labels.cpu().numpy()
            masked_flag = labels_cpu != -1
            total_t_num += np.sum(preds_cpu[masked_flag] == labels_cpu[masked_flag])
            total_num += np.sum(masked_flag)

        #             w_loss0, w_loss1, _ = words_loss(words_features, words_embs[:,:,1:], labels,
        #                                                 cap_lens-1, class_ids, batch_size)
        #             w_total_loss += (w_loss0 + w_loss1).item()

        #             s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
        #             s_total_loss += (s_loss0 + s_loss1).item()
        v_mlm_loss = total_mlm_loss / (step + 1)
        #         s_cur_loss = s_total_loss / (step+1)
        #         w_cur_loss = w_total_loss / (step+1)
        acc = total_t_num / total_num

        idx2word = self.data_loader.dataset.idx2word
        p = np.random.randint(0, labels_cpu.shape[0], 1)[0]
        caption = captions.cpu().numpy()
        cap_mask = masks.cpu().numpy()
        print('ground truth : ', labels_cpu[p, masked_flag[p]])
        print('masked input : ', caption[p, masked_flag[p]])
        print('prediction   : ', preds_cpu[p, masked_flag[p]])
        gt_token = np.copy(caption[p])
        gt_token[masked_flag[p]] = labels_cpu[p, masked_flag[p]]
        print('original text: ', ' '.join([idx2word[x.item()] for x in gt_token[cap_mask[p]]]))
        print('input text   : ', ' '.join([idx2word[x.item()] for x in caption[p, cap_mask[p]]]))
        pr_token = np.copy(caption[p])
        pr_token[masked_flag[p]] = preds_cpu[p, masked_flag[p]]
        print('output text  : ', ' '.join([idx2word[x.item()] for x in pr_token[cap_mask[p]]]))

        return v_mlm_loss, acc  # s_cur_loss, w_cur_loss