import torch.nn as nn
import clip
import torch
import numpy as np

AVAILABLE_MODELS = (
	"RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14""ViT-L/14@336px",
)


class CLIP(nn.Module):

	def __init__(self, config):
		super(CLIP, self).__init__()

		self.config = config

		assert self.config.model.image_caption_encoder.model_name in AVAILABLE_MODELS

		self.backbone, _ = clip.load(
			self.config.model.image_caption_encoder.model_name, download_root=self.config.experiment.cache_dir
		)

		# relearn logit scale for smaller training setup
		if self.config.criterion.temperature:

			self.backbone.logit_scale = nn.Parameter(
				torch.ones([]) * np.log(1 / self.config.criterion.temperature),
				requires_grad=self.config.criterion.tune_temperature
			)

	def forward(self, images, tokens, *kwargs):
		"""

		:param images:
		:param tokens:
		:param kwargs:
		:return:
		"""

		z_images = self.backbone.encode_image(images)
		z_captions = self.backbone.encode_text(tokens)

		logit_scale = self.backbone.logit_scale.exp()

		return z_images, z_captions, logit_scale
