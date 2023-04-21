import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import CircleLoss

class infoNCE(nn.Module):

	def __init__(self,):
		super(infoNCE, self).__init__()

	def forward(self, sim_i_2_t, sim_t_2_i, batch_num):
		"""
		Args:
			visual_embeds: 
			lang_embeds: 
			logit_scale:
		"""
		loss_i_2_t = F.cross_entropy(sim_i_2_t, torch.arange(batch_num).cuda())
		loss_t_2_i = F.cross_entropy(sim_t_2_i, torch.arange(batch_num).cuda())
		loss = (loss_t_2_i+loss_i_2_t)/2
		return loss
	