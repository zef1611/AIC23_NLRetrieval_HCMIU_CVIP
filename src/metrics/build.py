from .criterions import *

supported_losses = ["infoNCE", "CircleLoss"]

def build_loss(cfg):
    loss_dict = dict()
    if cfg.LOSS.INFONCE: 
        loss_dict["infoNCE"] = infoNCE()
    if cfg.LOSS.CIRCLE_LOSS:
        loss_dict["CircleLoss"] = CircleLoss(m=cfg.LOSS.LOSS_MARGIN, gamma=cfg.LOSS.LOSS_SCALE)
    return loss_dict