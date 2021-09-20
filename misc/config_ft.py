#### used in main project, do not change ####
class Config(object):
    def __init__(self):
        self.task = 'finetune'  # 'finetune' # label for log
        self.scheduler_init = True  # whether to load lr_scheduler.state_dict
        self.scheduler_step = True  # whether to update lr stepwise (if not, then use epochwise)

        self.DATASET_NAME = 'MIMIC'
        #         self.dataset_root = '/media/MyDataStor2/MIMIC-CXR/'
        self.dataset_root = '/media/SStor1/zhanghex/MIMIC-CXR/'
        self.CUDA = True
        self.snapshot_interval = 1
        self.text_encoder_path = '/media/SStor1/zhanghex/IPMI2021/output/MIMIC_test_2020_12_04_19_34_46/Model/text_encoder30.pth'
        self.CONFIG_NAME = 'test'
        self.DATA_DIR = '../'
        self.TRAIN = True
        self.GPU_ID = 3
        self.GAMMA1 = 4.0
        self.GAMMA2 = 5.0
        self.GAMMA3 = 10.0
        self.clip_max_norm = 1.0
        self.max_length = 256
        self.hidden_dim = 512  # embedding dim
        self.hidden_dropout_prob = 0.5
        self.attention_probs_dropout_prob = 0.1
        # Learning Rates
        self.lr = 5e-5

        # Epochs
        self.epochs = 35
        self.lr_drop = 30
        self.lr_gamma = 0.1
        self.start_epoch = 0
        self.weight_decay = 1e-4

        # Basic
        self.seed = 42
        self.batch_size = 32
        self.val_batch_size = 50
        self.num_workers = 6
