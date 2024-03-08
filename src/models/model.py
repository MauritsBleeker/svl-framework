import logging
import torch.nn as nn
from models.encoders.clip import CLIP
from models.encoders.vse import VSE
from models.encoders.base_model import BaseModel
from models.projection_layer import PredictionLayer
from models.target_decoder import TargetDecoder
from utils.utils import get_device
from utils.vocab import Vocabulary


class Model(nn.Module):

	def __init__(self, config):
		"""
		:param config:
		"""
		super(Model, self).__init__()

		self.config = config

		self.model = self.get_model()
		self.device = None

		self.requires_grad = self.image_encoder_requires_grad()

		if 'target_decoder' in self.config.model and self.config.model.target_decoder.decode_target:

			self.target_decoder = TargetDecoder(
				in_features=self.config.model.image_caption_encoder.embed_dim,
				hidden_features=self.config.model.target_decoder.hidden_features,
				reconstruction_dim=self.config.model.target_decoder.reconstruction_dim
			)

	def __repr__(self):
		return f'Model: {self.config.model.image_caption_encoder.name}'

	def to_device(self, device):
		"""
		:param device:
		:return:
		"""

		self.device = device
		self.model.to(device)
		self.model.float()

		if 'target_decoder' in self.config.model and self.config.model.target_decoder.decode_target:

			self.target_decoder.to(device)


	def forward(self, images, tokens, length=None):
		"""
		:param images:
		:param tokens:
		:param length: only for VSE++
		:return:
		"""

		image_tokens_logist = caption_tokens_logist = reconstructions = None

		z_images, z_captions, logit_scale = self.model(
			images,
			tokens,
			length
		)

		z_images = z_images / z_images.norm(dim=-1, keepdim=True)
		z_captions = z_captions / z_captions.norm(dim=-1, keepdim=True)

		if 'target_decoder' in self.config.model and self.config.model.target_decoder.decode_target:

			reconstructions = self.target_decoder(z_captions)

		return z_images, z_captions, logit_scale, reconstructions

	def return_parameters(self):
		"""
		:return:
		"""

		parameters = []
		parameters += list(self.model.parameters())

		if 'target_decoder' in self.config.model and self.config.model.target_decoder.decode_target:

			parameters += list(self.target_decoder.parameters())

		return parameters

	def image_encoder_requires_grad(self):
		"""

		:param requires_grad:
		:return:
		"""

		requires_grad = self.config.model.image_caption_encoder.train_img_encoder

		if self.config.model.image_caption_encoder.name == 'clip':
			for name, param in list(self.model.named_parameters()):
				if 'visual' in name:
					param.requires_grad = requires_grad
		elif self.config.model.image_caption_encoder.name == 'VSE' or self.config.model.image_caption_encoder.name == "BASE":
			for param in self.model.image_encoder.cnn.parameters():
				param.requires_grad = requires_grad
		else:
			raise NotImplementedError

		print('Image encoder parameters: requires_grad is switched to: ', requires_grad)

	def get_model(self):
		"""
		Return image-caption encoder model
		:return:
		"""

		if self.config.model.image_caption_encoder.name == 'clip':
			return CLIP(config=self.config)
		elif self.config.model.image_caption_encoder.name == 'VSE':
			return VSE(config=self.config)
		elif self.config.model.image_caption_encoder.name == 'BASE':
			return BaseModel(config=self.config)
		else:
			raise NotImplementedError
