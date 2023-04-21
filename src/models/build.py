
from .retrieval_models import *
from .recognition_models import *

supported_models = ["CLIP", "CLIPv2", "CLIP_recognition"]

def build_model(cfg):
    if cfg.MODEL.NAME == "CLIP":
        model = CLIP(cfg.MODEL)
    elif cfg.MODEL.NAME == "CLIPv2":
        model = CLIP_Extended_Feature_V2(cfg.MODEL)
    elif cfg.MODEL.NAME == "CLIP_recognition":
        model = CLIP_recognition(cfg.MODEL)
    else:
        assert cfg.MODEL.NAME in supported_models, f"unsupported model {cfg.MODEL.NAME}"

    return model