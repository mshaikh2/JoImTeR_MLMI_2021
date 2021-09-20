from __future__ import print_function

import os
import sys

sys.path.append('..')
# dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
# sys.path.append(dir_path)

from misc.config import Config
from dataset_mimic_pretrain import build_dataset_itm, build_dataset_mlm
from trainer_pretrain import JoImTeR as trainer

import json
import time
import random
# import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
# import pandas as pd
import torch
import torchvision.transforms as transforms
# import pickle
from misc.utils import mkdir_p

cfg = Config()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a JoImTeR network')
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    args = parser.parse_args()
    return args


## collate_fn for handling None type item due to image corruption ##
## This will return batch size - broken image number ##
def collate_fn_ignore_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


if __name__ == "__main__":
    args = parse_args()

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    torch.manual_seed(cfg.seed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(cfg.seed)

    ########################################

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    #     LAMBDA_FT,LAMBDA_FI,LAMBDA_DAMSM=01,50,10
    output_dir = '../output/%s_%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    mkdir_p(output_dir)

    ### save cfg to log file ###
    cfg_log = os.path.join(output_dir, 'cfg.log')
    with open(cfg_log, 'w') as f:
        json.dump(cfg.__dict__, f, indent=4)

    ### build training dataloader for itm ###
    data_set_itm = build_dataset_itm('train', cfg, output_dir)
    print('ITM training set %d is loaded.' % len(data_set_itm))
    train_loader_itm = torch.utils.data.DataLoader(
        data_set_itm, batch_size=cfg.batch_size,
        collate_fn=collate_fn_ignore_none, drop_last=True,
        shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    ### build training dataloader for mlm ###
    data_set_mlm = build_dataset_mlm('train', cfg)
    print('MLM training set %d is loaded.' % len(data_set_mlm))
    train_loader_mlm = torch.utils.data.DataLoader(
        data_set_mlm, batch_size=cfg.batch_size,
        collate_fn=collate_fn_ignore_none, drop_last=True,
        shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    ### build testing dataloader for itm ###
    val_data_set_itm = build_dataset_itm('val', cfg, output_dir)
    print('Validation set %d is loaded.' % len(val_data_set_itm))
    val_loader_itm = torch.utils.data.DataLoader(
        val_data_set_itm, batch_size=cfg.val_batch_size,
        collate_fn=collate_fn_ignore_none, drop_last=False,
        shuffle=False, num_workers=2, pin_memory=True)

    ### build testing dataloader for mlm ###
    val_data_set_mlm = build_dataset_mlm('val', cfg)
    print('Validation set %d is loaded.' % len(val_data_set_mlm))
    val_loader_mlm = torch.utils.data.DataLoader(
        val_data_set_mlm, batch_size=cfg.val_batch_size,
        collate_fn=collate_fn_ignore_none, drop_last=False,
        shuffle=False, num_workers=2, pin_memory=True)

    print('Vocab size is %d.' % data_set_itm.vocab_size)

    # Define models and go to train/evaluate
    algo = trainer(output_dir, train_loader_itm, train_loader_mlm, val_loader_itm, val_loader_mlm)

    start_t = time.time()

    algo.train()

    end_t = time.time()
    print('Total time for training:', end_t - start_t)
