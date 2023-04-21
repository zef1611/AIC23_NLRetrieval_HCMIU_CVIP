from yacs.config import CfgNode as CN

_C = CN()

# DATA process related configurations.
_C.DATA = CN()
_C.DATA.DATA_DIR = "/mnt/ssd8tb/huy/track2"
_C.DATA.ROOT_DIR = "/home/synh/workspace/huy/AIC23_NLRetrieval_HCMIU_CVIP"
_C.DATA.SIZE = 244
_C.DATA.CROP_AREA = 1.0  ## new_w = CROP_AREA * old_w

# Original Data
_C.DATA.CITYFLOW_PATH = "./data/AIC23_Track2_NL_Retrieval/data"
_C.DATA.TRAIN_ORIGINAL_JSON_PATH = "./data/json/original_data/train-tracks.json"
_C.DATA.TEST_TRACKS_JSON_PATH = "./data/json/original_data/test-tracks.json"
_C.DATA.TRAIN_JSON_PATH = "./data/json/train-tracks.json"
_C.DATA.EVAL_JSON_PATH = "./data/json/val-s1.json"
_C.DATA.TEST_JSON_PATH = "./data/json/original_data/test-queries.json"
_C.DATA.CHECKPOINT_PATH = "./data/checkpoints"
_C.DATA.LOG_PATH = "./data/logs"
_C.DATA.EMBEDDING_PATH = "./data/embeddings"
_C.DATA.TEST_OUTPUT_PATH = "./data/output/testset"
_C.DATA.VAL_OUTPUT_PATH = "./data/output/valset"
_C.DATA.RESUME = None

# Model specific configurations.
_C.MODEL = CN()
_C.MODEL.TYPE = "retrieval"
_C.MODEL.TRAINER = "Baseline"
_C.MODEL.NAME = "CLIP"
_C.MODEL.IMG_ENCODER = "CLIP"
_C.MODEL.LANG_ENCODER = "CLIP"
_C.MODEL.CLIP_TYPE = "CLIP"
_C.MODEL.BERT_TYPE = "ROBERTA"
_C.MODEL.BERT_NAME = "roberta-large"
_C.MODEL.NUM_CLASSES = 2155
_C.MODEL.EMBED_DIM = 2048
_C.MODEL.ONLY_CROP = False
_C.MODEL.FINETUNE = False
_C.MODEL.FREEZE_VISUAL_ENCODER = False
_C.MODEL.VISUAL_ID_LOSS = True
_C.MODEL.LANG_ID_LOSS = True
_C.MODEL.SCENE_ID_LOSS = False
_C.MODEL.CAMERA_ID_LOSS = False

# Model specific configurations.
_C.LOSS = CN()
_C.LOSS.INFONCE = True
_C.LOSS.CIRCLE_LOSS = False
_C.LOSS.METRIC_WEIGHT = 1.0
_C.LOSS.LOSS_SCALE = 80
_C.LOSS.LOSS_MARGIN = 0.4
# Training configurations
_C.TRAIN = CN()
_C.TRAIN.ONE_EPOCH_REPEAT = 30
_C.TRAIN.EPOCH = 40
_C.TRAIN.EARLYSTOPPING = 5
_C.TRAIN.SAVE_EPOCH = 5
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.NUM_WORKERS = 6
_C.TRAIN.PRINT_FREQ = 20
_C.TRAIN.SCHEDULER = "WarmupLR"
_C.TRAIN.LR = CN()
_C.TRAIN.LR.STEPS = [4, 8]
_C.TRAIN.LR.BASE_LR = 1e-2
_C.TRAIN.LR.MIN_LR = 1e-7
_C.TRAIN.LR.WARMUP_EPOCH = 40
_C.TRAIN.LR.WARMUP_FACTOR = 0.01
_C.TRAIN.LR.WARMUP_METHOD = "linear"
_C.TRAIN.LR.DELAY = 8
_C.TRAIN.LR.GAMMA = 0.1
_C.TRAIN.LR.WEIGHT_DECAY = 1e-4
_C.TRAIN.SEED = 1611
_C.TRAIN.EVAL = True
_C.TRAIN.RESUME_FROM = None

# Test configurations
_C.TEST = CN()
_C.TEST.INFERENCE_FROM = None
_C.TEST.QUERY_JSON_PATH = "./data/json/test-queries.json"
_C.TEST.ID_JSON = "./data/json/recognition/color_id.json"
_C.TEST.BATCH_SIZE = 64
_C.TEST.FEAT_IDX = 2
_C.TEST.NUM_WORKERS = 32


def get_default_config():
    return _C.clone()
