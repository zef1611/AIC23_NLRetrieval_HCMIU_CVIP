import json
import random

'''
Generate direction & color & type pseudo labels 
'''

with open("./data/json/recognition/direction_id.json") as f:
    direction_id_dict = json.load(f)

directions = list(direction_id_dict.keys())
direction_ids = list(range(len(direction_id_dict)))

with open("./data/json/recognition/type_id.json") as f:
	type_id_dict = json.load(f)

types = list(type_id_dict.keys())
type_ids = list(range(len(type_id_dict)))

with open("./data/json/recognition/color_id.json") as f:
	color_id_dict = json.load(f)

colors = list(color_id_dict.keys())
color_ids = list(range(len(color_id_dict)))

with open("./data/json/recognition/test-tracks-color.json") as f:
	color_pseudo = json.load(f)

with open("./data/json/recognition/test-tracks-direction.json") as f:
	direction_pseudo = json.load(f)

with open("./data/json/recognition/test-tracks-type.json") as f:
	type_pseudo = json.load(f)

with open("./data/json/original_data/test-tracks.json") as f:
    tracks = json.load(f)

pseudo_dataset = dict()
for track in tracks:
	pseudo_dataset[track] = tracks[track]
	color_text = colors[color_pseudo[track]["id"]]
	type_text = types[type_pseudo[track]["id"]]
	direction_text = directions[direction_pseudo[track]["id"]]
	tracks[track]["nl"] = []
	text = f"{color_text} {type_text} {direction_text}"
	tracks[track]["nl"].append(text)
	tracks[track]["nl"].append(text)
	tracks[track]["nl"].append(text)

print(len(pseudo_dataset.keys()))

with open("./data/json/pseudo_labels/pseudo_label_testset.json", "w") as f:
	json.dump(pseudo_dataset, f,indent=4)
