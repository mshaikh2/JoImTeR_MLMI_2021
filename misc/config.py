#### used in main project, do not change ####
class Config(object):
    def __init__(self):
        #         self.task = 'damsm_triplet_mlm: hidden_dropout_prob=0.5: joint train from scratch; joint training losses; joint losses'# 'finetune' # label for log

        self.task = 'negation classification: text encoder : using phrase : from scratch point'
        self.scheduler_init = True  # whether to load lr_scheduler.state_dict
        self.scheduler_step = False  # whether to update lr stepwise (if not, then use epochwise)

        self.multitasksampling = 0.3  # 0.3 for ITM, 0.7 for MLM

        self.DATASET_NAME = 'MIMIC'
        #         self.dataset_root = '/media/MyDataStor2/MIMIC-CXR/'
        #         self.dataset_root = '/media/SStor1/zhanghex/MIMIC-CXR/'
        self.dataset_root = '/media/My1TBSSD1/MIMIC-CXR'  # on mmserver
        self.CUDA = True
        self.snapshot_interval = 1

        # if the training stops / or one has to continue from a checkpoint then this variable is supposed to be populated
        self.text_encoder_path = ''  # '../output/MIMIC_pretrain_2021_05_04_23_03_10/Model/text_encoder30.pth'
        self.joint_encoder_path = ''
        #         self.CONFIG_NAME = 'joint_pretrain'
        #         self.CONFIG_NAME = 'neg_img_cls'
        self.CONFIG_NAME = 'neg_txt_cls_fs'
        self.DATA_DIR = '../'
        self.TRAIN = True
        self.GPU_ID = 1
        self.GAMMA1 = 4.0
        self.GAMMA2 = 5.0
        self.GAMMA3 = 10.0
        self.sent_margin = 0.5
        self.word_margin = 0.5
        self.LAMBDA_DAMSM = 1.0
        self.LAMBDA_TRIPLET = 2.0

        self.clip_max_norm = 1.5
        self.max_length = 159
        self.hidden_dim = 512  # embedding dim
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        # Learning Rates
        self.lr = 1e-4  # 1e-3 not converge; 5e-5 slow converge.

        # Epochs
        self.epochs = 35
        self.lr_drop = 20
        self.lr_gamma = 0.1
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Basic
        self.seed = 42
        self.batch_size = 128  # 32
        self.val_batch_size = 100  # 50
        self.num_workers = 6

        # JointModel
        self.nlayers = 1

        # -----------------------------------
        self.pretrained_text_encoder_path = ''
        self.init_text_encoder_path = ''  # '/media/My1TBSSD1/MICCAI2021/output/MIMIC_pretrain_2021_05_04_23_03_10/Model/text_encoder28.pth'
        self.init_image_encoder_path = ''  # '/media/My1TBSSD1/MICCAI2021/output/MIMIC_pretrain_2021_05_04_23_03_10/Model/image_encoder28.pth'
        self.pretrained = False
        self.freeze_backbone = False
        self.lr_backbone = 1e-4


# pretrained
# if using a BERT pretrained on MLM then this variable is supposed to be updated with the path of the pth file
#         self.pretrained_MLM_text_encoder_path = '../output/MIMIC_pretrain_2021_04_13_21_03_07/Model/text_encoder26.pth'
8
