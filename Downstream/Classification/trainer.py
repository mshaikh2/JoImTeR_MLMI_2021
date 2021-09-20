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
from model import TextEncoder, ImageEncoder
# from InceptionScore import calculate_inception_score

from misc.losses import sent_loss, words_loss

from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader

import math
from tqdm import tqdm
import timeit
# from catr.engine import train_one_epoch, evaluate
from misc.config import Config
from transformers import BertConfig #, BertTokenizer
# from nltk.tokenize import RegexpTokenizer


cfg = Config() # initialize catr config here
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
        self.log_file = os.path.join(output_dir, 'err.log')
        print('Write broken images to %s.' % self.log_file)
        self.num_batches = len(self.data_loader)
        self.bert_config = BertConfig(vocab_size=data_loader.dataset.vocab_size, hidden_size=512, num_hidden_layers=3,
                    num_attention_heads=8, intermediate_size=2048, hidden_act='gelu',
                    hidden_dropout_prob=cfg.hidden_dropout_prob, attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
                    max_position_embeddings=512, layer_norm_eps=1e-12,
                    initializer_range=0.02, type_vocab_size=2, pad_token_id=0)

    def build_models(self):
        # ################### encoders ################################# #      
        image_encoder = ImageEncoder(output_channels=cfg.hidden_dim)
        for p in image_encoder.parameters(): # make image encoder grad on
            p.requires_grad = True 
            
        text_encoder = TextEncoder(bert_config = self.bert_config, output_channels=cfg.hidden_dim)
        for p in text_encoder.parameters(): # make text encoder grad on
            p.requires_grad = True
            
        if cfg.CUDA:
            text_encoder = text_encoder.cuda()
            image_encoder = image_encoder.cuda()
            
        optimizerI = torch.optim.AdamW(image_encoder.parameters()
                                           , lr=cfg.lr
                                           , weight_decay=cfg.weight_decay)       
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
        start_epoch = 0 # normaly its the step 0
        drop_epoch = cfg.lr_drop
        batch_step = self.num_batches
        start_step = start_epoch * batch_step # normaly its the step 0
        drop_step = drop_epoch * batch_step # where to start drop lr
        total_steps = (epochs - start_epoch) * batch_step
        
        lambda1 = lambda step: min((total_steps - step) / (total_steps - drop_step), 1) # stepwise linear decay
#         lambda1 = lambda step: 0.3 ** (step / batch_step) # stepwise
        
        if cfg.scheduler_step: 
            lr_schedulerI = torch.optim.lr_scheduler.LambdaLR(optimizerI, lr_lambda=lambda1)
            lr_schedulerT = torch.optim.lr_scheduler.LambdaLR(optimizerT, lr_lambda=lambda1)
        else:
            lr_schedulerI = torch.optim.lr_scheduler.StepLR(optimizerI, cfg.lr_drop, gamma=cfg.lr_gamma)
            lr_schedulerT = torch.optim.lr_scheduler.StepLR(optimizerT, cfg.lr_drop, gamma=cfg.lr_gamma)
        ##### load checkpoints if exists #########
        
        if cfg.text_encoder_path != '':
            img_encoder_path = cfg.text_encoder_path.replace('text_encoder', 'image_encoder')
            print('Load image encoder checkpoint from:', img_encoder_path)
            state_dict = torch.load(img_encoder_path, map_location='cpu')
            if 'model' in state_dict.keys():
                image_encoder.load_state_dict(state_dict['model'])
                optimizerI.load_state_dict(state_dict['optimizer'])
                if cfg.scheduler_init:
                    lr_schedulerI.load_state_dict(state_dict['lr_scheduler'])
            else:
                image_encoder.load_state_dict(state_dict)
            
            text_encoder_path = cfg.text_encoder_path
            print('Load text encoder checkpoint from:', text_encoder_path)
            state_dict = torch.load(text_encoder_path, map_location='cpu')
            if 'model' in state_dict.keys():
                text_encoder.load_state_dict(state_dict['model'])
                optimizerT.load_state_dict(state_dict['optimizer'])
                if cfg.scheduler_init:
                    lr_schedulerT.load_state_dict(state_dict['lr_scheduler'])
            else:
                text_encoder.load_state_dict(state_dict)
        # ########################################################### #

        return [text_encoder, image_encoder, epoch, optimizerI, optimizerT, lr_schedulerI, lr_schedulerT]

    def save_model(self, image_encoder, text_encoder, optimizerI, optimizerT, lr_schedulerI, lr_schedulerT, epoch):
       
        # save image encoder model here
        torch.save({
            'model': image_encoder.state_dict(),
            'optimizer': optimizerI.state_dict(),
            'lr_scheduler': lr_schedulerI.state_dict(),
        }, '%s/image_encoder%d.pth' % (self.model_dir, epoch))

        
        # save text encoder model here
        torch.save({
            'model': text_encoder.state_dict(),
            'optimizer': optimizerT.state_dict(),
            'lr_scheduler': lr_schedulerT.state_dict(),
        }, '%s/text_encoder%d.pth' % (self.model_dir, epoch))
            
    def train(self):
        
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        #     LAMBDA_FT,LAMBDA_FI,LAMBDA_DAMSM=01,50,10
        tb_dir = '../tensorboard/{0}_{1}_{2}'.format(cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        mkdir_p(tb_dir)
        tbw = SummaryWriter(log_dir=tb_dir) # Tensorboard logging

        
        ####### init models ########
        text_encoder, image_encoder, start_epoch, optimizerI, optimizerT, lr_schedulerI, lr_schedulerT = self.build_models()
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
        image_encoder.train()

        ###############################################################

        tensorboard_step = 0
        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches
        
        #### print lambdas ###
#         print('LAMBDA_GEN:{0},LAMBDA_CAP:{1},LAMBDA_FT:{2},LAMBDA_FI:{3},LAMBDA_DAMSM:{4}'.format(cfg.TRAIN.SMOOTH.LAMBDA_GEN
#                                                                                                   ,cfg.TRAIN.SMOOTH.LAMBDA_CAP
#                                                                                                   ,cfg.TRAIN.SMOOTH.LAMBDA_FT
#                                                                                                   ,cfg.TRAIN.SMOOTH.LAMBDA_FI                           
#                                                                                                   ,cfg.TRAIN.SMOOTH.LAMBDA_DAMSM))  
        for epoch in range(start_epoch, self.max_epoch):
            
            ##### set everything to trainable ####
            text_encoder.train()
            image_encoder.train()
            ####################################
            
            ####### init loss variables ############          
            s_total_loss0 = 0
            s_total_loss1 = 0
            w_total_loss0 = 0
            w_total_loss1 = 0
            total_damsm_loss = 0
            
            ####### print out lr of each optimizer before training starts, make sure lrs are correct #########
            print('[epoch: %d] Learning rates: lr_i %.4e, lr_t %.4e' 
                 % (epoch, optimizerI.param_groups[0]['lr'], optimizerT.param_groups[0]['lr']))
                     
            #########################################################################################
            
            start_t = time.time()

            data_iter = iter(self.data_loader)
#             step = 0
            pbar = tqdm(range(self.num_batches))
            ## training each epoch 
            for step in pbar: 
                ### use stepwise linear schedule here ###
                if cfg.scheduler_step: 
                    lr_schedulerI.step()
                    lr_schedulerT.step()
                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                imgs, captions, masks, class_ids, cap_lens = data_iter.next()
                class_ids = class_ids.numpy()
                ## due to the broken image collate_fn, batch size could change. Move labels inside step loop
                batch_size = imgs.shape[0] # actual batch size could be different from self.batch_size
                labels = torch.LongTensor(range(batch_size)) # used for matching loss
#                 match_labels = self.prepare_labels(batch_size) # used for matching loss
                
                if cfg.CUDA:
                    imgs, captions, masks, cap_lens = imgs.cuda(), captions.cuda(), masks.cuda(), cap_lens.cuda()
                    labels = labels.cuda()
                # add images, image masks, captions, caption masks for catr model
                
                ################## feedforward damsm model ##################
#                 image_encoder.zero_grad() # image/text encoders zero_grad here
#                 text_encoder.zero_grad()
                optimizerI.zero_grad()
                optimizerT.zero_grad()
                
                words_features, sent_code = image_encoder(imgs) # input images to image encoder, feedforward
                nef, att_sze = words_features.size(1), words_features.size(2)
                # hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, masks) 
                
                #### damsm losses
                w_loss0, w_loss1, _ = words_loss(words_features, words_embs[:,:,1:], labels, cap_lens-1, class_ids, batch_size)
                w_total_loss0 += w_loss0.item()
                w_total_loss1 += w_loss1.item()
                damsm_loss = w_loss0 + w_loss1
                
                s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
                s_total_loss0 += s_loss0.item()
                s_total_loss1 += s_loss1.item()
                damsm_loss += s_loss0 + s_loss1
                
                total_damsm_loss += damsm_loss.item()
                ############################################################################
                
                damsm_loss.backward()
                    
                torch.nn.utils.clip_grad_norm_(image_encoder.parameters(), cfg.clip_max_norm)                    
                optimizerI.step()
                
                torch.nn.utils.clip_grad_norm_(text_encoder.parameters(), cfg.clip_max_norm)
                optimizerT.step()
                ##################### loss values for each step #########################################
                ## damsm ##
                tbw.add_scalar('Train_step/train_w_step_loss0', float(w_loss0.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/train_s_step_loss0', float(s_loss0.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/train_w_step_loss1', float(w_loss1.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/train_s_step_loss1', float(s_loss1.item()), step + epoch * self.num_batches)
                tbw.add_scalar('Train_step/train_damsm_step_loss', float(damsm_loss.item()), step + epoch * self.num_batches)

                ################################################################################################    
                
                ############ tqdm descriptions showing running average loss in terminal ##############################
                pbar.set_description('lr: %.4e %.4e, damsm %.4f, w_loss: %.4f, s_loss: %.4f' 
                                     % ( optimizerI.param_groups[0]['lr'], optimizerT.param_groups[0]['lr'],
                                         total_damsm_loss / (step+1), 
                                        (w_total_loss0 + w_total_loss1) / (step+1), 
                                        (s_total_loss0 + s_total_loss1) / (step+1) ) )
                               
                ######################################################################################################
                ##########################################################
            v_s_cur_loss, v_w_cur_loss = self.evaluate(image_encoder, text_encoder)
            print('[epoch: %d] val_w_loss: %.4f, val_s_loss: %.4f' % (epoch, v_w_cur_loss, v_s_cur_loss))
            ### val losses ###
            tbw.add_scalar('Val_step/val_w_loss', float(v_w_cur_loss), epoch)
            tbw.add_scalar('Val_step/val_s_loss', float(v_s_cur_loss), epoch)
            
            if not cfg.scheduler_step:
                lr_schedulerI.step()
                lr_schedulerT.step()
            
            end_t = time.time()
            
            if epoch % cfg.snapshot_interval == 0:
                self.save_model(image_encoder, text_encoder, optimizerI, optimizerT, lr_schedulerI, lr_schedulerT, epoch)
                
            broken_images = [] # find all the broken images within this epoch
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if 'ERR_IMG' in line:
                        broken_images.append(line[8:-1]) 
                    if 'Epoch' in line:
                        last_epoch = int(re.findall('\d+', line)[-1])
                        break
            #### copy broken images from original storage here ####
            for tar in broken_images:
                org = tar.replace('SStor1/zhanghex', 'MyDataStor2')
                shutil.copyfile(org, tar)
            #######################################################
            with open(self.log_file, 'a+') as f: # start a new epoch record
                f.write('\nEpoch %d:\n\n' % (last_epoch + 1))

        self.save_model(image_encoder, text_encoder, optimizerI, optimizerT, lr_schedulerI, lr_schedulerT, epoch)                
                

    @torch.no_grad()
    def evaluate(self, cnn_model, trx_model):
        cnn_model.eval()
        trx_model.eval()
        s_total_loss = 0
        w_total_loss = 0
        ### add caption criterion here. #####
        ## same reason as train(), batch_size could change when image broken
#         labels = Variable(torch.LongTensor(range(batch_size))) # used for matching loss
#         if cfg.CUDA:
#             labels = labels.cuda()
        #####################################
        
        val_data_iter = iter(self.dataloader_val)
        for step in tqdm(range(len(val_data_iter)), leave=False):  #
            real_imgs, captions, masks, class_ids, cap_lens = val_data_iter.next()
            class_ids = class_ids.numpy()
            batch_size = real_imgs.shape[0]
            labels = torch.LongTensor(range(batch_size)) # used for matching loss
            if cfg.CUDA:
                real_imgs, captions, masks, cap_lens = real_imgs.cuda(), captions.cuda(), masks.cuda(), cap_lens.cuda()
                labels = labels.cuda()
            words_features, sent_code = cnn_model(real_imgs)
            words_embs, sent_emb = trx_model(captions, masks)
            w_loss0, w_loss1, _ = words_loss(words_features, words_embs[:,:,1:], labels,
                                                cap_lens-1, class_ids, batch_size)
            w_total_loss += (w_loss0 + w_loss1).item()

            s_loss0, s_loss1 = sent_loss(sent_code, sent_emb, labels, class_ids, batch_size)
            s_total_loss += (s_loss0 + s_loss1).item()

        s_cur_loss = s_total_loss / (step+1)
        w_cur_loss = w_total_loss / (step+1)
        return s_cur_loss, w_cur_loss
        
      
    @torch.no_grad()
    def save_img_results(self, imgs, attention_maps):
        # Save images
        for i in range(len(attention_maps)):
            if len(imgs) > 1:
                img = imgs[i + 1].detach().cpu()
                lr_img = imgs[i].detach().cpu()
            else:
                img = imgs[0].detach().cpu()
                lr_img = None
            attn_maps = attention_maps[i]
            att_sze = attn_maps.size(2)
            img_set, _ = \
                build_super_images(img, captions, self.ixtoword,
                                   attn_maps, att_sze, lr_imgs=lr_img, batch_size = self.batch_size)
            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/G_%s_%d_%d.png'\
                    % (self.image_dir, name, gen_iterations, i)
                im.save(fullpath)

        # for i in range(len(netsD)):
        i = -1
        img = fake_imgs[i].detach()
        region_features, _ = image_encoder(img)
        att_sze = region_features.size(2)
        _, _, att_maps = words_loss(region_features.detach(),
                                    words_embs.detach(),
                                    None, cap_lens,
                                    None, self.batch_size)
        img_set, _ = \
            build_super_images(fake_imgs[i].detach().cpu(),
                               captions, self.ixtoword, att_maps, att_sze, batch_size = self.batch_size)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/D_%s_%d.png'\
                % (self.image_dir, name, gen_iterations)
            im.save(fullpath)