#### used in main project, do not change ####
class Config(object):
    def __init__(self):
        self.task = 'I-T_classification_phrase_finetune: use NoFindings as N for pos_weight, set upperbound as 5'

        self.scheduler_init = False  # whether to load lr_scheduler.state_dict
        self.scheduler_step = False  # whether to update lr stepwise (if not, then use epochwise)

        self.pretrained = True  # whether to initialize the classification encoders with pretrained matching model
        self.pretrained_text_encoder_path = ''  # '/media/My1TBSSD1/IPMI2021/output/MIMIC_mlm_2021_02_24_19_58_21/Model/text_encoder.pth' # load MLM pretrained model, used for matching pretraining, not use for classification!

        self.init_text_encoder_path = '/media/My1TBSSD1/IPMI2021/output/MIMIC_phrase_2021_02_26_23_25_51/Model/text_encoder28.pth'
        self.init_image_encoder_path = '/media/My1TBSSD1/IPMI2021/output/MIMIC_phrase_2021_02_26_23_25_51/Model/image_encoder28.pth'  # the pretrained checkpoint used for initialize the image encoder backbone
        self.freeze_backbone = False  # whether to freeze the pretrained backbone

        self.DATASET_NAME = 'MIMIC'
        #         self.dataset_root = '/media/MyDataStor2/MIMIC-CXR/'
        #         self.dataset_root = '/media/SStor1/zhanghex/MIMIC-CXR/'
        self.dataset_root = '/media/My1TBSSD1/MIMIC-CXR'  # on mmserver
        self.CUDA = True
        self.snapshot_interval = 1
        self.text_encoder_path = ''  # '/media/SStor1/zhanghex/IPMI2021/output/MIMIC_triplet_st_2020_12_08_06_00_26/Model/text_encoder17.pth' ## used for loading checkpoint for the current model, both matching and classification can use.
        self.CONFIG_NAME = 'class_it_ft_new'
        self.DATA_DIR = '../'
        self.TRAIN = True
        self.GPU_ID = 1
        self.GAMMA1 = 4.0  ## test 1,1,2, slow converge
        self.GAMMA2 = 5.0
        self.GAMMA3 = 10.0
        self.sent_margin = 0.5
        self.word_margin = 0.5
        self.LAMBDA_DAMSM = 1.0
        self.LAMBDA_TRIPLET = 2.0

        self.clip_max_norm = 2.0
        self.max_length = 159
        self.hidden_dim = 512  # embedding dim
        self.hidden_dropout_prob = 0.5
        self.attention_probs_dropout_prob = 0.1
        # Learning Rates
        self.lr = 1e-4  # 1e-3 not converge; 5e-5 slow converge.
        self.lr_backbone = 1e-5

        # Epochs
        self.epochs = 20
        self.lr_drop = 10
        self.lr_gamma = 0.1
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Basic
        self.seed = 42
        self.batch_size = 32  # --> 24
        self.val_batch_size = 50
        self.num_workers = 6
