import torch
import torch.nn as nn
from torch.nn import functional as F


class InfoNCE(nn.Module):

	def __init__(self, device, use_ifm=False, epsilon=0.1):

		super(InfoNCE, self).__init__()

		self.device = device
		self.use_ifm = use_ifm
		self.epsilon = epsilon

	def forward(self, z_images, z_captions, logit_scale):
		"""

		:param z_images: tensor with image latents
		:param z_captions: tensor with caption latents
		:param logit_scale: scaler to scale the logits
		:return: loss value
		"""

		logits_per_img = z_images @ z_captions.t()

		if self.use_ifm:

			ifm_logits_per_img = self.imf_logist(logits_per_img)

			ifm_logits_per_img = logit_scale * ifm_logits_per_img

			ifm_logist_per_caption = ifm_logits_per_img.t()

			imf_loss = self.cross_entropy(ifm_logits_per_img, ifm_logist_per_caption)

		logits_per_img = logit_scale * logits_per_img

		logits_per_caption = logits_per_img.t()

		loss = self.cross_entropy(logits_per_img, logits_per_caption)

		if self.use_ifm:
			return (loss + imf_loss) / 2

		return loss

	def cross_entropy(self, logits_per_image, logits_per_caption):
		"""

		:param logits_per_image: image scores
		:param logits_per_caption: caption scores
		:return: cross entropy loss
		"""

		num_logits = logits_per_image.shape[0]

		labels = torch.arange(num_logits, device=self.device, dtype=torch.long)

		total_loss = (
				 F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_caption, labels)
		 ) / 2

		return total_loss

	def imf_logist(self, logist):
		"""

		:param logist:
		:return:
		"""

		batch_size = logist.shape[0]
		matrix = torch.ones(batch_size, batch_size, device=self.device)
		matrix[torch.eye(batch_size) == 0] = self.epsilon
		matrix[torch.eye(batch_size) > 0.5] = - self.epsilon
		logist = logist + matrix

		return logist