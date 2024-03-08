import torch
import pickle
import os
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from PIL import Image
from utils.transform import get_image_transform
from utils.shortcut_sampler import ShortcutSampler
from utils.vocab import load_vocab, get_tokenizer, get_vocab


class Dataset(Dataset):

	def __init__(self, config, split, json_file):
		"""
		:param config: config class
		:param split: train, val, or test
		:param json_file: json file to load annoations
		"""
		super().__init__()

		assert split in set(['train', 'val', 'test'])

		self.config = config
		self.split = split

		self.json_file = json_file

		self.tokenizer = get_tokenizer(config)
		self.vocab = get_vocab(config, self.tokenizer)
		self.img_normalize, self.img_transform = get_image_transform(config=config, split=split)

		self.vocab_size = len(self.vocab)
		self.config.dataset.vocab_size = self.vocab_size

		self.captions = {}
		self.images = {}
		self.caption_ids = None

		self._load_annotations_from_json()

		if self.config.shortcuts.use_shortcuts:

			self.shortcut_sampler = ShortcutSampler(
				config,
				eval=(self.split == 'val' or self.split == 'test'),
				n_images=len(self.caption_ids) // self.config.dataset.captions_per_image
			)

	def __len__(self):

		return len(self.caption_ids)

	def __repr__(self):

		return f'Dataset: {self.json_file["dataset"]}'

	def _load_annotations_from_json(self):
		"""
		:return:
		"""

		for shortcut_id, image in enumerate(self.json_file['images']):
			if image['split'] == self.split or (image['split'] == 'restval' and self.split == 'train'):

				self.images[image['imgid']] = {
					'filename': image['filename'],
					'sentids': image['sentids'],
					'shortcut_id': shortcut_id
				}

				for sentence in image['sentences'][:self.config.dataset.captions_per_image]:
					self.captions[sentence['sentid']] = sentence
					self.captions[sentence['sentid']]['shortcut_id'] = shortcut_id

		self.caption_ids = list(self.captions.keys())

	def _get_token_distribution(self, tokens, normalize=False):
		"""
		:param tokens
		:param normalize
		:return:
		"""
		dist = np.zeros(self.vocab_size)

		counts = Counter(tokens)

		if normalize:
			dist[list(counts.keys())] = list(counts.values())
			dist = torch.Tensor(dist / sum(counts.values()))
		else:
			dist[list(counts.keys())] = 1
			dist = torch.Tensor(dist)

		return dist

	def __getitem__(self, idx):
		"""
		:param idx:
		:return:
		"""

		caption_id = self.caption_ids[idx]
		caption = self.captions[caption_id]
		raw_caption = caption['raw']

		image_id = caption['imgid']
		image = Image.open(os.path.join(self.config.dataset.img_path, self.images[image_id]['filename']))\
			.convert('RGB')

		assert self.images[image_id]['shortcut_id'] == caption['shortcut_id']

		image = self.img_transform(image)

		if self.config.shortcuts.use_shortcuts:
			image, raw_caption, shortcut_id = self.shortcut_sampler.sample_shortcut(
				image=image,
				caption=raw_caption,
				shortcut_id=caption['shortcut_id']
			)

		caption, dist = self.tokenize(raw_caption=raw_caption)

		image = self.img_normalize(image)

		latent_target = None

		return caption, image, dist, caption_id, image_id, raw_caption, idx, latent_target

	def tokenize(self, raw_caption):
		"""

		:param raw_caption:
		:return:
		"""
		if self.config.model.image_caption_encoder.name == "clip":

			tokens, caption = self.tokenizer_clip(raw_caption=raw_caption)

		elif self.config.model.image_caption_encoder.name == "VSE" or self.config.model.image_caption_encoder.name == "BASE":

			tokens, caption = self.tokenizer_vse(raw_caption=raw_caption)

		else:
			raise NotImplementedError

		dist = self._get_token_distribution(tokens)

		return caption, dist

	def tokenizer_clip(self, raw_caption):
		"""

		:param raw_caption:
		:return:
		"""

		tokens = self.tokenizer.encode(raw_caption)

		caption = torch.Tensor(
			[self.tokenizer.encoder["<|startoftext|>"]] + tokens + [self.tokenizer.encoder["<|endoftext|>"]]
		)
		return tokens, caption

	def tokenizer_vse(self, raw_caption):
		"""

		:param raw_caption:
		:return:
		"""
		
		tokens = self.tokenizer(
			str(raw_caption).lower()
		)

		tokens = [self.vocab(token) for token in tokens]

		caption = list()
		caption.append(self.vocab('<start>'))
		caption.extend(tokens)
		caption.append(self.vocab('<end>'))

		return tokens, torch.Tensor(caption)


def collate_fn(data, sort_captions=False):
	"""

	:param data: tuple with the data for training/batch
	:param sort_captions: If true, sort captions based on length
	:return:
	"""

	if sort_captions:
		data.sort(key=lambda x: len(x[0]), reverse=True)

	captions, images, dists, caption_ids, image_ids, raw_captions, idxs, latent_targets = zip(*data)

	cap_lengths = [len(cap) for cap in captions]

	max_caption_length = max(cap_lengths) if sort_captions else 77

	images = torch.stack(images, 0) if images[0] != None else None

	dists = torch.stack(dists, 0)

	tokens = torch.zeros(len(captions), max_caption_length).long()

	for i, cap in enumerate(captions):
		end = min(cap_lengths[i], max_caption_length)

		tokens[i, :end] = cap[:end]

	cap_lengths = torch.Tensor(cap_lengths).long()

	latent_targets = torch.stack(latent_targets, 0) if latent_targets[0] != None else None

	return tokens, images, dists, caption_ids, list(image_ids), list(raw_captions), cap_lengths, idxs, latent_targets
