import json
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F


def default_loader(path):
	return Image.open(path).convert('RGB')

class CityFlowNLDataset(Dataset):
	def __init__(self, data_cfg, json_path, transform = None, Random = True, type = None, finetune = False):
		"""
		Dataset for training.
		:param data_cfg: CfgNode for CityFlow NL.
		:json_path: str
		"""
		self.data_cfg = data_cfg.clone()
		self.crop_area = data_cfg.CROP_AREA
		self.dataset_dir = os.path.join(self.data_cfg.DATA_DIR, self.data_cfg.CITYFLOW_PATH)
		self.json_dir = os.path.join(self.data_cfg.ROOT_DIR, json_path)
		self.random = Random
		self.finetune = finetune
		self.type = type
		with open(self.json_dir) as f:
			tracks = json.load(f)

			if self.type == "train":
				print("Loading json for training from: %s " %self.json_dir)
			else:
				print("Loading json for eval-training from: %s " %self.json_dir)

		self.list_of_uuids = list(tracks.keys())
		self.list_of_tracks = list(tracks.values())
		self.transform = transform
		self.all_indexs = list(range(len(self.list_of_uuids)))
		print("[%s] Total data: %.2d tracks " %(type, len(self.all_indexs)))

	def __len__(self):
		return len(self.all_indexs)

	def __getitem__(self, index):

		tmp_index = self.all_indexs[index]
		track = self.list_of_tracks[tmp_index]
		if self.random:
			nl_idx = int(random.uniform(0, len(track["nl"])))
			frame_idx = int(random.uniform(0, len(track["frames"])))
		else:
			nl_idx = 1
			frame_idx = 0
		if self.finetune:
			nl_idx = 0
		text = track["nl"][nl_idx]

		frame_path = os.path.join(self.dataset_dir, track["frames"][frame_idx])
		frame = default_loader(frame_path)
		box = track["boxes"][frame_idx]
		if self.crop_area == 1.6666667:
			box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
		else:
			box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))
		
			
		crop = frame.crop(box)
		if self.transform is not None:
			crop = self.transform(crop)
			frame = self.transform(frame)
		data = {
                "crop": crop,
                "frame": frame,
                "text": text,
                "idx": tmp_index,
            }
		return data

class CityFlowNLInferenceDataset(Dataset):
	def __init__(self, data_cfg, transform = None, type ='test'):
		"""Dataset for evaluation. Loading tracks instead of frames."""
		self.data_cfg = data_cfg.clone()
		self.crop_area = data_cfg.CROP_AREA
		self.dataset_dir = os.path.join(self.data_cfg.DATA_DIR, self.data_cfg.CITYFLOW_PATH)
		self.transform = transform
		self.type = type
		
		if type == 'test':
			test_data_json_dir = os.path.join(self.data_cfg.ROOT_DIR, self.data_cfg.TEST_TRACKS_JSON_PATH)
			with open(test_data_json_dir) as f:
				print("Loading Testset from %s " %self.data_cfg.TEST_TRACKS_JSON_PATH)
				tracks = json.load(f)
		
		else:
			eval_data_json_dir = os.path.join(self.data_cfg.ROOT_DIR, self.data_cfg.EVAL_JSON_PATH)
			with open(eval_data_json_dir) as f:
				print("Loading Validation set %s " %eval_data_json_dir)
				tracks = json.load(f)
		self.list_of_uuids = list(tracks.keys())
		self.list_of_tracks = list(tracks.values())
		self.list_of_crops = list()
		for track_id_index,track in enumerate(self.list_of_tracks):
			for frame_idx, frame in enumerate(track["frames"]):
				frame_path = os.path.join(self.dataset_dir, frame)
				box = track["boxes"][frame_idx]
				crop = {"frame": frame_path, "frames_id":frame_idx,"track_id": self.list_of_uuids[track_id_index], "box": box}
				self.list_of_crops.append(crop)
		print("[%s] Total data: %.2d tracks " %(type, len(self.list_of_uuids)))
	def __len__(self):
		return len(self.list_of_crops)

	def __getitem__(self, index):
		track = self.list_of_crops[index]
		frame_path = track["frame"]

		frame = default_loader(frame_path)
		box = track["box"]
		if self.crop_area == 1.6666667:
			box = (int(box[0]-box[2]/3.),int(box[1]-box[3]/3.),int(box[0]+4*box[2]/3.),int(box[1]+4*box[3]/3.))
		else:
			box = (int(box[0]-(self.crop_area-1)*box[2]/2.),int(box[1]-(self.crop_area-1)*box[3]/2),int(box[0]+(self.crop_area+1)*box[2]/2.),int(box[1]+(self.crop_area+1)*box[3]/2.))
		

		crop = frame.crop(box)
		if self.transform is not None:
			crop = self.transform(crop)
			frame = self.transform(frame)
		
		data = {
			"crop": crop,
			"frame": frame,
			"track_id": track["track_id"],
			"frames_id": track["frames_id"],
		}
		return data

