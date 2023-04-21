import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F

def get_mean_feats2(img_feats, tracks_ids):
    mean_gallery = []
    for k in tracks_ids:
        mean_gallery.append(img_feats[(k,)])
    mean_gallery = np.vstack(mean_gallery)
    mean_gallery = torch.from_numpy(mean_gallery)
    mean_gallery = F.normalize(mean_gallery, p=2, dim=-1).numpy()
    return mean_gallery

class SimilarityToMRR(object):
    def __init__(self,track_json_path):
        with open(track_json_path) as f:
            tracks = json.load(f)
        self.tracks = tracks
        self.tracks_ids = list(tracks.keys())
        self.img_feats = dict()
        self.nlp_feats = dict()

    def get_mean_img_feats(self, img_feats):
        mean_gallery = []
        for k in self.tracks_ids:
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

    def calculuate_similarity(self):
        # print(len(self.img_feats.keys()))
        nlp_feats = self.nlp_feats
        # print("img num: ", len(self.img_feats))
        # print("nlp num: ", len(self.nlp_feats))
        img_feats = [self.get_mean_img_feats(self.img_feats)]
        results = dict()
        weights = 1
        for query in self.tracks_ids:
            if query not in self.img_feats:
                continue
            score = 0.
            # for i in range(len(nlp_feats)):
            q = nlp_feats[query]
            score += np.mean(np.matmul(q, img_feats[0].T), 0)
            index = np.argsort(score)[::-1]
            results[query]=[]
            for i in index:
                results[query].append(self.tracks_ids[i])
        return results

    def calculate_mrr(self, results):
        recall_5 = 0.
        recall_10 = 0.
        mrr = 0.
        if len(results) == 0:
            return mrr, recall_5, recall_10
        for query in results:
            result = results[query]
            target = query
            try:
                rank = result.index(target)
            except ValueError:
                rank = len(results.keys())
            rank += 1
            if rank < 10:
                recall_10 += 1
            if rank < 5:
                recall_5 += 1
            mrr += 1.0 / rank
        recall_5 /= len(results)
        recall_10 /= len(results)
        mrr /= len(results)
        return mrr, recall_5, recall_10

    def get_mrr(self):
        return self.calculate_mrr(self.calculuate_similarity())

        
    def update_img_feats(self,visual_embeds):
        self.img_feats.update(visual_embeds)

    def update_nlp_feats(self,textual_embeds):
        self.nlp_feats.update(textual_embeds)

    def reset(self):
        self.img_feats = dict()
        self.nlp_feats = dict()
        