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
from model_phrase_cls import ImageEncoder_Classification
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
        Q_c = np.array([43590, 42520, 10207, 25470, 6760, 4189, 5931, 48778, 72528, 51348, 1894, 15677, 9927, 63100]) # training set class number
        Q = 217246.0
        self.class_weight_vector = torch.tensor((Q - Q_c) / Q)
#         pos_weights = torch.tensor(72528.0 / Q_c) # no upper bound
        pos_weights = torch.tensor(np.minimum(72528.0 / Q_c, 5.0)) # use NoFindings as Neg baseline, pos_weight cannot set too large. 
#         pos_weights = torch.tensor((Q - Q_c) / Q_c) # adjust pos_weight for each class, might be too large for small class. 
#         pos_weights = torch.tensor(np.minimum((Q - Q_c) / Q_c, 4.0)) # adjust pos_weight for each class, set upperbound. 
        self.criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight = pos_weights)

        self.batch_size = data_loader.batch_size
        self.val_batch_size = dataloader_val.batch_size
        self.max_epoch = cfg.epochs
        self.snapshot_interval = cfg.snapshot_interval

        self.data_loader = data_loader
        self.dataloader_val = dataloader_val
        self.log_file = os.path.join(output_dir, 'err.log')
        print('Write broken images to %s.' % self.log_file)
        self.num_batches = len(self.data_loader)
        
        if cfg.CUDA:
            pos_weights = pos_weights.cuda()
            self.class_weight_vector = self.class_weight_vector.cuda()
            self.criterion = self.criterion.cuda()

    def build_models(self):
        # ################### encoders ################################# #      
        image_encoder = ImageEncoder_Classification(num_class=self.data_loader.dataset.num_classes, 
                                                    encoder_path=cfg.init_image_encoder_path, pretrained=cfg.pretrained, cfg = cfg)
        
        for p in image_encoder.parameters(): # make all image encoder grad on
            p.requires_grad = True
        
        classification_parameters = [] # parameters only for new layers (classification layers)
        for n, p in image_encoder.named_parameters():
            if 'pretrained_encoder' not in n:
                classification_parameters.append(p)
        optim_params = [{'params':classification_parameters, 'lr':cfg.lr}]
        
        if cfg.freeze_backbone:
            image_encoder.pretrained_encoder.eval() ## turn off BN and Dropout
            for p in image_encoder.pretrained_encoder.parameters(): 
                p.requires_grad = False
        else:
            optim_params.append({'params':image_encoder.pretrained_encoder.parameters(), 'lr':cfg.lr_backbone})

        if cfg.CUDA:
            image_encoder = image_encoder.cuda()
            
        optimizerI = torch.optim.Adam(optim_params, weight_decay=cfg.weight_decay)
        
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
            img_encoder_path = cfg.text_encoder_path #.replace('text_encoder', 'image_encoder')
            print('Load image encoder checkpoint from:', img_encoder_path)
            state_dict = torch.load(img_encoder_path, map_location='cpu')
            if 'model' in state_dict.keys():
                image_encoder.load_state_dict(state_dict['model'])
                optimizerI.load_state_dict(state_dict['optimizer'])
                if cfg.scheduler_init:
                    lr_schedulerI.load_state_dict(state_dict['lr_scheduler'])
            else:
                image_encoder.load_state_dict(state_dict)
        # ########################################################### #

        return [image_encoder, epoch, optimizerI, lr_schedulerI]

    def save_model(self, image_encoder, optimizerI, lr_schedulerI, epoch):
       
        # save image encoder model here
        torch.save({
            'model': image_encoder.state_dict(),
            'optimizer': optimizerI.state_dict(),
            'lr_scheduler': lr_schedulerI.state_dict(),
        }, '%s/image_encoder%d.pth' % (self.model_dir, epoch))
            
    def train(self):
        
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
        #     LAMBDA_FT,LAMBDA_FI,LAMBDA_DAMSM=01,50,10
        tb_dir = '../tensorboard/{0}_{1}_{2}'.format(cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        mkdir_p(tb_dir)
        tbw = SummaryWriter(log_dir=tb_dir) # Tensorboard logging

        ####### init models ########
        image_encoder, start_epoch, optimizerI, lr_schedulerI = self.build_models()
#         print('Starting epoch %d' % start_epoch)
        ##################################################################
        image_encoder.train()
        if cfg.freeze_backbone:
            image_encoder.pretrained_encoder.eval() ## turn off BN and Dropout
        ###############################################################

        tensorboard_step = 0
        gen_iterations = 0
        # gen_iterations = start_epoch * self.num_batches 
        for epoch in range(start_epoch, self.max_epoch):
            
            ##### set everything to trainable ####
            image_encoder.train()
            if cfg.freeze_backbone:
                image_encoder.pretrained_encoder.eval() ## turn off BN and Dropout
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
                imgs, classes, uids = data_iter.next()
                ## due to the broken image collate_fn, batch size could change. Move labels inside step loop
                batch_size = imgs.shape[0] # actual batch size could be different from self.batch_size
               
                if cfg.CUDA:
                    imgs, classes = imgs.cuda(), classes.cuda()
                
                ################## feedforward damsm model ##################
                image_encoder.zero_grad() # image encoder zero_grad here
#                 optimizerI.zero_grad()
                
                y_pred, _, _, _, _ = image_encoder(imgs) # input images to image encoder, feedforward
                bce_loss = self.criterion(y_pred, classes)
                bce_loss = bce_loss * self.class_weight_vector
                bce_loss = bce_loss.mean()
                total_bce_loss_epoch += bce_loss.item()
                ############################################################################
                
                bce_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(image_encoder.parameters(), cfg.clip_max_norm)                    
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
            v_loss, auc = self.evaluate(image_encoder)
            print('[epoch: %d] val_loss: %.4f' % (epoch, v_loss))
            
            for idx in range(len(auc)):
                print('%s: %.4f' % (self.data_loader.dataset.idx_to_class[idx], auc[idx]))
#             print(auc_scores)
            avg= np.mean(np.array(auc)[[0,1,2,3,4,5,6,7,9,10,11,12,13]])
            print('Avg: %.4f' % avg)

            # weight = np.array([679, 808, 191, 659, 132, 78, 108, 974, 539, 990, 52, 309, 94, 1061]) # weight for hold_out_set
            # weight = np.array([958, 997, 233, 600, 174, 79, 168, 1134, 1636, 1243, 31, 361, 234, 1566]) # weight for val+test set
            # weight = np.array([566, 575, 131, 354, 110, 33, 105, 650, 1017, 710, 28, 227, 124, 881]) # weight for val2 set
            weight = np.array([566, 575, 131, 354, 110, 33, 105, 650, 710, 28, 227, 124, 881]) # weight for val2 set

            weight = weight / weight.sum()
            wavg = np.array(auc)[[0,1,2,3,4,5,6,7,9,10,11,12,13]] @ weight
            print('wAvg: %.4f' % wavg)
            ### val losses ###
            tbw.add_scalar('Val_step/val_bce_loss', v_loss, epoch)
            for idx in range(len(auc)):
                tbw.add_scalar('Val_step/{0}'.format(self.data_loader.dataset.idx_to_class[idx]), auc[idx], epoch)
            
            if not cfg.scheduler_step:
                lr_schedulerI.step()
            
            end_t = time.time()
            
            if epoch % cfg.snapshot_interval == 0:
                self.save_model(image_encoder, optimizerI, lr_schedulerI, epoch)
                
            broken_images = [] # find all the broken images within this epoch
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if 'ERR_IMG' in line:
                        broken_images.append(line[8:-1]) 
                    if 'Epoch' in line:
                        last_epoch = int(re.findall('\d+', line)[-1])
                        break
#             #### copy broken images from original storage here ####
#             for tar in broken_images:
#                 org = tar.replace('SStor1/zhanghex', 'MyDataStor2')
#                 shutil.copyfile(org, tar)
            #######################################################
            with open(self.log_file, 'a+') as f: # start a new epoch record
                f.write('\nEpoch %d:\n\n' % (last_epoch + 1))

#         self.save_model(image_encoder, text_encoder, optimizerI, optimizerT, lr_schedulerI, lr_schedulerT, epoch)                
                

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
            class_auc.append(roc_auc_score(y_trues[:,i],y_preds[:,i]))
        
        v_cur_loss = total_bce_loss_epoch / (step+1)
        return v_cur_loss, class_auc
        
      
