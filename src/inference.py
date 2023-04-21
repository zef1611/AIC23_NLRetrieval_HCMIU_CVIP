import json
import os
import argparse
import torch
import torch.multiprocessing
from tqdm import tqdm
from configs import get_default_config
from models.backbones.CLIP import clip
from models.backbones import open_clip
import torch.nn.functional as F
from utils.utils import set_seed
from collections import OrderedDict
from models import build_model
from datasets import build_dataloader
import numpy as np

def get_mean_img_feats(tracks_ids, img_feats):
    mean_gallery = []
    for k in tracks_ids:
        tmp = []
        if k not in img_feats:
            continue
        for fid in img_feats[k]:
            tmp.append(img_feats[k][fid])
        tmp = np.vstack(tmp)
        tmp = np.mean(tmp, 0)
        mean_gallery.append(tmp)
    mean_gallery = np.vstack(mean_gallery)
    return mean_gallery

def make_configurations():
    parser = argparse.ArgumentParser(description='AICT5 Training')
    parser.add_argument('--resume', type = bool, default=False, help='resume from checkpoint')
    parser.add_argument('--config', default="baseline.yaml", type=str,
                        help='config_file')
    parser.add_argument('--name', default="baseline", type=str,
                        help='experiments')
    args = parser.parse_args()

    cfg = get_default_config()

    cfg.merge_from_file(os.path.join(cfg.DATA.ROOT_DIR, 'configs', args.config))

    return args, cfg

args, cfg = make_configurations()

set_seed(cfg.TRAIN.SEED)
use_cuda = True

testloader = build_dataloader(is_train=False, cfg=cfg)
model = build_model(cfg)
CKPT_SAVE_DIR = os.path.join(cfg.DATA.DATA_DIR, cfg.DATA.CHECKPOINT_PATH, args.name)
LOG_SAVE_DIR = os.path.join(cfg.DATA.DATA_DIR, cfg.DATA.LOG_PATH)

os.makedirs(CKPT_SAVE_DIR,exist_ok = True)
os.makedirs(LOG_SAVE_DIR,exist_ok = True)

print(cfg)

INFERENCE_DIR = os.path.join(cfg.DATA.DATA_DIR, cfg.TEST.INFERENCE_FROM)

print("Inference model from weight %s"%INFERENCE_DIR)
    
checkpoint = torch.load(INFERENCE_DIR)
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[7:] # remove `module.`
    # print(name)
    new_state_dict[name] = v
model.load_state_dict(new_state_dict, strict=False)

if use_cuda:
    model.cuda()
model.eval()

if cfg.MODEL.TYPE == "recognition":
    ID_JSON_DIR = os.path.join(cfg.DATA.ROOT_DIR, cfg.TEST.ID_JSON)
    JSON_DATASET_DIR = os.path.join(cfg.DATA.ROOT_DIR, cfg.DATA.TEST_TRACKS_JSON_PATH)
    with open(ID_JSON_DIR) as f:
        recognition_ids = json.load(f)

    with open(JSON_DATASET_DIR) as f:
        tracks = json.load(f)

    uuids = list(tracks.keys())
    indexs = list(range(len(uuids)))

    colors = list(recognition_ids.keys())
    color_ids = list(range(len(recognition_ids)))
    out = dict()
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(testloader)):
            crop,frame,track_id,frames_id = batch["crop"], batch["frame"], batch["track_id"], batch["frames_id"] 
            cls_logit = model(crop.cuda(), frame.cuda())
            predict = cls_logit.argmax(1)
            for  i in range(len(track_id)):
                if track_id[i] not in out:
                    out[track_id[i]]=[]
                out[track_id[i]].append(int(predict[i].data.cpu().numpy()))

    count = 0
    correct_dict = dict()
    for i in indexs:
        id = uuids[i]
        pred = np.bincount(out[id]).argmax()
        correct_dict[id] = tracks[id]
        correct_dict[id]["id"] = int(pred)


    with open(f"./data/json/recognition/{args.name}.json", "w") as f:
        json.dump(correct_dict, f,indent=4)
elif cfg.MODEL.TYPE == "retrieval":
    with open(cfg.DATA.TEST_TRACKS_JSON_PATH) as f:
        tracks = json.load(f)

    with open(cfg.TEST.QUERY_JSON_PATH) as f:
        queries = json.load(f)
    queries_ids = list(queries.keys())

    tracks_ids = list(tracks.keys())
    print(f"Using features number: {cfg.TEST.FEAT_IDX}")
    visual_embeds = dict()
    #Visual process
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(testloader)):
            crop,frame,track_id,frames_id = batch["crop"], batch["frame"], batch["track_id"], batch["frames_id"] 
            vis_embeds = model.visual_forward(crop.cuda(),frame.cuda())
            if cfg.MODEL.NAME != "CLIP":
                vis_embed = vis_embeds[cfg.TEST.FEAT_IDX]
            for  i in range(len(track_id)):
                if track_id[i] not in visual_embeds:
                    visual_embeds[track_id[i]]=dict()
                visual_embeds[track_id[i]][frames_id[i].item()] = vis_embed[i,:].data.cpu().numpy()

    #Text process


    textual_embeds = dict()
    with torch.no_grad():
        for q_id in tqdm(queries.keys()):
            # print(queries[q_id]['nl'])
            if cfg.MODEL.LANG_ENCODER == "CLIP":
                if cfg.MODEL.CLIP_TYPE== "open_clip":
                    tokenizer = open_clip.get_tokenizer('ViT-B-32')
                    tokens = tokenizer(queries[q_id]['nl'])
                else:
                    tokens = clip.tokenize(queries[q_id]['nl'])
                lang_embeds = model.text_forward(tokens.cuda(), None, None)
            elif cfg.MODEL.LANG_ENCODER == "BERT":
                tokens = tokenizer.batch_encode_plus(queries[q_id]['nl'], padding='longest',
                                                            return_tensors='pt')               
                lang_embeds = model.text_forward(None, tokens['input_ids'].cuda(),tokens['attention_mask'].cuda())
            lang_embed = lang_embeds[cfg.TEST.FEAT_IDX]
            textual_embeds[q_id] = lang_embed.data.cpu().numpy()

    print(len(textual_embeds))
    nlp_feats = textual_embeds
    img_feats = [get_mean_img_feats(tracks_ids, visual_embeds)]
    results = dict()
    weights = 1
    for query in queries_ids:
        score = 0.
        q = nlp_feats[query]
        score += np.mean(np.matmul(q, img_feats[0].T), 0)
        index = np.argsort(score)[::-1]
        results[query]=[]
        for i in index:
            results[query].append(tracks_ids[i])


    with open(f"./data/json/retrieval/{args.name}.json", "w") as f:
        json.dump(results, f, indent=4)


