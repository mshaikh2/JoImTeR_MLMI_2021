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
from model_cls_phr_neg import Text_Classification
from sklearn.metrics import roc_auc_score

from torch.utils.tensorboard import SummaryWriter

import math
from tqdm import tqdm
import timeit
from misc.config import Config
from transformers import BertConfig

cfg = Config() # initialize catr config here

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
        
        ###### calculate classwise loss weights ######
#         Q_c = np.array([43590, 42520, 10207, 25470, 6760, 4189, 5931, 48778, 72528, 51348, 1894, 15677, 9927, 63100]) # training set class number
#         Q = 217246.0
        
        Q_c = np.array([751, 11169, 7665, 20839, 3523, 937, 644, 3375, 72303, 20825, 89, 14965, 37114, 2857]) # training set class number, no finding counted highest to reduce weight to 0
        Q = 72303.0
        self.class_weight_vector = torch.tensor((Q - Q_c) / Q)
#         pos_weights = torch.tensor(72528.0 / Q_c) # no upper bound
        pos_weights = torch.tensor(np.minimum(72303.0 / Q_c, 5.0)) # use NoFindings as Neg baseline, pos_weight cannot set too large. 
#         pos_weights = torch.tensor((Q - Q_c) / Q_c) # adjust pos_weight for each class, might be too large for small class. 
#         pos_weights = torch.tensor(np.minimum((Q - Q_c) / Q_c, 4.0)) # adjust pos_weight for each class, set upperbound. 
        self.criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight = pos_weights)

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
                    hidden_dropout_prob=cfg.hidden_dropout_prob, attention_probs_dropout_prob=cfg.attention_probs_dropout_prob,
                    max_position_embeddings=512, layer_norm_eps=1e-12,
                    initializer_range=0.02, type_vocab_size=2, pad_token_id=0)
        
        if cfg.CUDA:
            pos_weights = pos_weights.cuda()
            self.class_weight_vector = self.class_weight_vector.cuda()
            self.criterion = self.criterion.cuda()

 def build_models(self):
        # ################### encoders ################################# #      
        main_model = Text_Classification(num_class=self.data_loader.dataset.num_classes,
                                                 txt_encoder_path=cfg.init_text_encoder_path, 
                                                 pretrained=cfg.pretrained, cfg=cfg, bert_config=self.bert_config)
#         print(main_model)
        
        for p in main_model.parameters(): # make all image encoder grad on
            p.requires_grad = True
        
        classification_parameters = [] # parameters only for new layers (classification layers)
        for n, p in main_model.named_parameters():
            if 'txt_encoder' not in n:
                print(n)
                classification_parameters.append(p)
        optim_params = [{'params':classification_parameters, 'lr':cfg.lr}]
        
        if cfg.freeze_backbone:
            main_model.txt_encoder.eval() ## turn off BN and Dropout
            for p in main_model.txt_encoder.parameters(): 
                p.requires_grad = False
        else:
            optim_params.append({'params':main_model.txt_encoder.parameters(), 'lr':cfg.lr_backbone})

        if cfg.CUDA:
            main_model = main_model.cuda()
            
        optimizerI = torch.optim.AdamW(optim_params, weight_decay=cfg.weight_decay)
        
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
#             lr_schedulerT = torch.optim.lr_scheduler.LambdaLR(optimizerT, lr_lambda=lambda1)
        else:
            lr_schedulerI = torch.optim.lr_scheduler.StepLR(optimizerI, cfg.lr_drop, gamma=cfg.lr_gamma)
#             lr_schedulerT = torch.optim.lr_scheduler.StepLR(optimizerT, cfg.lr_drop, gamma=cfg.lr_gamma)
        ##### load checkpoints if exists #########
        
        if cfg.text_encoder_path != '':
            model_path = cfg.text_encoder_path #.replace('text_encoder', 'main_model')
            print('Load the classification model checkpoint from:', model_path)
            state_dict = torch.load(model_path, map_location='cpu')
            if 'model' in state_dict.keys():
                main_model.load_state_dict(state_dict['model'])
                optimizerI.load_state_dict(state_dict['optimizer'])
                if cfg.scheduler_init:
                    lr_schedulerI.load_state_dict(state_dict['lr_scheduler'])
            else:
                main_model.load_state_dict(state_dict)
        # ########################################################### #

        return [main_model, epoch, optimizerI, lr_schedulerI]

    def save_model(self, main_model, optimizerI, lr_schedulerI, epoch):
       
        # save image encoder model here
        torch.save({
            'model': main_model.state_dict(),
            'optimizer': optimizerI.state_dict(),
            'lr_scheduler': lr_schedulerI.state_dict(),
        }, '%s/Txt_class_model%d.pth' % (self.model_dir, epoch))
            
    def train(self):
        
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        #     LAMBDA_FT,LAMBDA_FI,LAMBDA_DAMSM=01,50,10
        tb_dir = '../../tensorboard/{0}_{1}_{2}'.format(cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        mkdir_p(tb_dir)
        tbw = SummaryWriter(log_dir=tb_dir) # Tensorboard logging

        ####### init models ########
        main_model, start_epoch, optimizerI, lr_schedulerI = self.build_models()
main_model, start_epoch, optimizerI, lr_schedulerI = self.build_models()
#         print('Starting epoch %d' % start_epoch)
        ##################################################################
        main_model.train()
        if cfg.freeze_backbone:
#             main_model.img_encoder.eval() ## turn off BN and Dropout
            main_model.txt_encoder.eval() ## turn off BN and Dropout
        ###############################################################

        tensorboard_step = 0
        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches 
        for epoch in range(start_epoch, self.max_epoch):
            
            ##### set everything to trainable ####
            main_model.train()
            if cfg.freeze_backbone:
#                 main_model.img_encoder.eval() ## turn off BN and Dropout
                main_model.txt_encoder.eval() ## turn off BN and Dropout
            ####################################
            
            ####### init loss variables ############          
            total_bce_loss_epoch = 0.0
            
            ####### print out lr of each optimizer before training starts, make sure lrs are correct #########
            if cfg.freeze_backbone:
                print('[epoch: %d] Learning rates: lr_c %.4e, freeze backbone' % (epoch, optimizerI.param_groups[0]['lr']))
            else: 
                print('[epoch: %d] Learning rates: lr_c %.4e, lr_b %.4e' 
                     % (epoch, optimizerI.param_groups[0]['lr'], optimizerI.param_groups[1]['lr']))
                     
            #########################################################################################
            
            start_t = time.time()

            data_iter = iter(self.data_loader)
#             step = 0
            pbar = tqdm(range(self.num_batches))
            ## training each epoch 
            for step in pbar: 
#                 break
                ### use stepwise linear schedule here ###
                if cfg.scheduler_step: 
                    lr_schedulerI.step()
                ######################################################
                # (1) Prepare training data and Compute text embeddings
                ######################################################
                captions, cap_masks, classes, uids, cap_lens = data_iter.next()
                ## due to the broken image collate_fn, batch size could change. Move labels inside step loop
                batch_size = captions.shape[0] # actual batch size could be different from self.batch_size
               
                if cfg.CUDA:
                    captions, cap_masks, classes = captions.cuda(), cap_masks.cuda(), classes.cuda()
                
                ################## feedforward damsm model ##################
                main_model.zero_grad() # image encoder zero_grad here
#                 optimizerI.zero_grad()
                
                y_pred = main_model(captions, cap_masks) # feedforward
                bce_loss = self.criterion(y_pred, classes)
                bce_loss = bce_loss * self.class_weight_vector
                bce_loss = bce_loss.mean()
                total_bce_loss_epoch += bce_loss.item()
                ############################################################################
                
                bce_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(main_model.parameters(), cfg.clip_max_norm)                    
                optimizerI.step()
                ##################### loss values for each step #########################################
                ## bce loss ##
                tbw.add_scalar('Train_step/bce_loss', bce_loss.item(), step + epoch * self.num_batches)
                ################################################################################################    
                
                ############ tqdm descriptions showing running average loss in terminal ##############################
                if cfg.freeze_backbone:
                    pbar.set_description('lr: %.4e, freeze backbone, bce_loss %.4f' 
                                         % ( optimizerI.param_groups[0]['lr'], 
                                             total_bce_loss_epoch / (step+1) ) )
                else:
                    pbar.set_description('lr: %.4e, %.4e, bce_loss %.4f' 
                                         % ( optimizerI.param_groups[0]['lr'], optimizerI.param_groups[1]['lr'],
                                             total_bce_loss_epoch / (step+1) ) )

                ######################################################################################################
                ##########################################################
            v_loss, auc_scores = self.evaluate(main_model)
#             print('[epoch: %d] val_loss: %.4f' % (epoch, v_loss))
#             print(auc_scores)
            ### val losses ###
            tbw.add_scalar('Val_step/val_bce_loss', v_loss, epoch)
            for idx in range(len(auc_scores)):
                tbw.add_scalar('Val_step/{0}'.format(self.data_loader.dataset.idx_to_class[idx]), auc_scores[idx], epoch)
            
            if not cfg.scheduler_step:
                lr_schedulerI.step()
            
            end_t = time.time()
            
            if epoch % cfg.snapshot_interval == 0:
                self.save_model(main_model, optimizerI, lr_schedulerI, epoch)
                


    @torch.no_grad()
    def evaluate(self, cnn_model):
        cnn_model.eval()
        
        total_bce_loss_epoch=0.0
        val_data_iter = iter(self.dataloader_val)
        y_preds = []
        y_trues = []
        class_auc = []
        #####################################
        for step in tqdm(range(len(val_data_iter)), leave=False):  
            real_imgs, classes, uids = val_data_iter.next()
            if cfg.CUDA:
                real_imgs, classes = real_imgs.cuda(), classes.cuda()
            
            y_pred, _, _, _, _ = cnn_model(real_imgs)
            y_pred_sigmoid = torch.sigmoid(y_pred)
            
            bce_loss = self.criterion(y_pred, classes)
            bce_loss = bce_loss * self.class_weight_vector
            bce_loss = bce_loss.mean()
            
            total_bce_loss_epoch += bce_loss.item()
            y_preds.append(y_pred_sigmoid.detach().cpu().numpy())
            y_trues.append(classes.detach().cpu().numpy())
            
#             if step == 5: break

        y_preds = np.concatenate(y_preds,axis=0)
        y_trues = np.concatenate(y_trues,axis=0)
        for i in range(y_preds.shape[-1]):
            if i==8: # No Finding
                class_auc.append(0)
            else:
                class_auc.append(roc_auc_score(y_trues[:,i],y_preds[:,i]))
        
        v_cur_loss = total_bce_loss_epoch / (step+1)
        return v_cur_loss, class_auc
        
      
