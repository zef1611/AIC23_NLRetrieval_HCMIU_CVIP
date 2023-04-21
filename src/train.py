import json
import os
import argparse
import torch
import torch.multiprocessing
from tqdm import tqdm
from configs import get_default_config
from models.backbones.CLIP import clip
from utils.utils import AverageMeter,accuracy,ProgressMeter,WriteLog, set_seed
import torchvision
from models.backbones import open_clip
import time
import torch.nn.functional as F
from collections import OrderedDict
from metrics.mrr import SimilarityToMRR
from transformers import BertTokenizer, RobertaTokenizer, DebertaV2Tokenizer
from collections import OrderedDict
from solver.build import build_optimizer
from models import build_model
from datasets import build_dataloader
from metrics import build_loss
import numpy as np


def make_configurations():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--resume', type = bool, default=False, help='resume from checkpoint')
    parser.add_argument('--config', default="baseline.yaml", type=str,
                        help='config_file')
    parser.add_argument('--name', default="baseline", type=str,
                        help='experiments')
    args = parser.parse_args()

    cfg = get_default_config()

    cfg.merge_from_file(os.path.join(cfg.DATA.ROOT_DIR, 'configs', args.config))

    return args, cfg

best_top1_eval = 0.
start_epoch = 0
global_step = 0
best_top1 = 0.
best_loss = 12.
early_stopping = 0

args, cfg = make_configurations()

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((cfg.DATA.SIZE,cfg.DATA.SIZE)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
set_seed(cfg.TRAIN.SEED)
use_cuda = True

trainloader = build_dataloader(is_train=True, cfg=cfg)
model = build_model(cfg)
optimizer, scheduler = build_optimizer(cfg, model, trainloader)
loss_list = build_loss(cfg)
CKPT_SAVE_DIR = os.path.join(cfg.DATA.DATA_DIR, cfg.DATA.CHECKPOINT_PATH, args.name)
LOG_SAVE_DIR = os.path.join(cfg.DATA.DATA_DIR, cfg.DATA.LOG_PATH)
os.makedirs(CKPT_SAVE_DIR,exist_ok = True)
os.makedirs(LOG_SAVE_DIR,exist_ok = True)

print(cfg)

if cfg.MODEL.FINETUNE:
    if cfg.MODEL.FINETUNE:
        print("Fine-tuning model from weight %s"%cfg.TRAIN.RESUME_FROM)
    else:
        print("Resume training model from weight %s"%cfg.TRAIN.RESUME_FROM)
    RESUME_DIR = os.path.join(cfg.DATA.DATA_DIR, cfg.TRAIN.RESUME_FROM)
    checkpoint = torch.load(RESUME_DIR)
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:] # remove `module.`
        if "id_cls" in name:
            continue
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict = False)
if cfg.MODEL.LANG_ENCODER == "BERT":
    if cfg.MODEL.BERT_TYPE == "BERT":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif cfg.MODEL.BERT_TYPE == "ROBERTA":
        tokenizer = RobertaTokenizer.from_pretrained(cfg.MODEL.BERT_NAME)
    elif cfg.MODEL.BERT_TYPE == "DEBERTA":
        tokenizer = DebertaV2Tokenizer.from_pretrained(cfg.MODEL.BERT_NAME)

if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
model.train()
writeLog = WriteLog(cfg=cfg, save_path=LOG_SAVE_DIR, isFirstTime=start_epoch==0, _type='TRAIN')

for epoch in range(start_epoch,cfg.TRAIN.EPOCH+1):
    if early_stopping == cfg.TRAIN.EARLYSTOPPING:
        print("Early stopping due to model diverge after {} consecutive divergence epochs".format(cfg.TRAIN.EARLYSTOPPING))
        break
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    top1_acc = AverageMeter('Acc@1', ':6.2f')
    top5_acc = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(trainloader)*cfg.TRAIN.ONE_EPOCH_REPEAT,
        [batch_time, data_time, losses, top1_acc, top5_acc],
        prefix="Epoch: [{}]".format(epoch),
        writeLog= writeLog)
    end = time.time()
    for tmp in range(cfg.TRAIN.ONE_EPOCH_REPEAT):
        for batch_idx, batch in enumerate(trainloader):
            crop, bg, text, id_car = batch["crop"], batch["frame"], batch["text"], batch["idx"]
            data_time.update(time.time() - end)
            global_step +=1   

            if cfg.MODEL.LANG_ENCODER == "CLIP":
                if cfg.MODEL.CLIP_TYPE== "open_clip":
                    tokenizer = open_clip.get_tokenizer('ViT-B-32')
                    tokens = tokenizer(text)
                else:
                    tokens = clip.tokenize(text)
                pairs,logit_scale,cls_logits = model(crop.cuda(), bg.cuda(), tokens.cuda(), None, None)
            elif cfg.MODEL.LANG_ENCODER == "BERT":
                tokens = tokenizer.batch_encode_plus(text, padding='longest',
                                                            return_tensors='pt')           
                pairs,logit_scale,cls_logits = model(crop.cuda(), bg.cuda(), None, tokens['input_ids'].cuda(),tokens['attention_mask'].cuda())
            logit_scale = logit_scale.mean().exp()
            total_loss = 0 
            for pair in pairs:
                visual_embeds, lang_embeds = pair
                sim_i_2_t = torch.matmul(torch.mul(logit_scale, visual_embeds), torch.t(lang_embeds))
                sim_t_2_i = sim_i_2_t.t()
                for loss_name in loss_list:
                    if loss_name == 'infoNCE':
                        total_loss += loss_list[loss_name](sim_i_2_t, sim_t_2_i, crop.size(0))
                    if loss_name == 'CircleLoss':
                        total_loss += loss_list[loss_name](torch.cat(pair), torch.cat([id_car, id_car]).long().cuda())
            for cls_logit in cls_logits:
                total_loss += 0.5*F.cross_entropy(cls_logit, id_car.long().cuda())

            acc1, acc5 = accuracy(sim_t_2_i, torch.arange(crop.size(0)).cuda(), topk=(1, 5))
            losses.update(total_loss.item(), crop.size(0))
            top1_acc.update(acc1[0], crop.size(0))
            top5_acc.update(acc5[0], crop.size(0))

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.clamp_(model.module.logit_scale.data, max=np.log(100))
            batch_time.update(time.time() - end)
            end = time.time()
            if scheduler is not None:
                scheduler.step(epoch)
            if batch_idx % cfg.TRAIN.PRINT_FREQ == 0:
                progress.display(global_step%(len(trainloader)*30))

CHECKPOINT_FILE = os.path.join(CKPT_SAVE_DIR,"checkpoint_last.pth"%epoch)
torch.save(
    {"epoch": epoch, 
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_loss": best_loss,
        "best_top1_eval": best_top1_eval,
        "best_top1": best_top1}, CHECKPOINT_FILE) 