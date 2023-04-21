import json
import random
import copy

with open("./data/json/dataclean_v1/train_standard.json") as f:
    tracks = json.load(f)
with open("./data/json/pseudo_labels/pseudo_label_testset.json") as f:
    tracks_v2 = json.load(f)

for track_id in tracks_v2:
	tracks[track_id] = tracks_v2[track_id]
        
print(len(tracks))

with open("./data/json/pseudo_labels/train_standard_merge_pseudo_testset.json", "w") as f:
	json.dump(tracks, f,indent=4)
