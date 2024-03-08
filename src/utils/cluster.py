import os


def update_config_with_username(config, user_name):
	"""

	:param config:
	:param user_name:
	:return:
	"""

	config.dataset.root = config.dataset.root.format(user_name)
	config.dataset.annotation_path = config.dataset.annotation_path.format(user_name)
	config.dataset.vocab_path = config.dataset.vocab_path.format(user_name)

	config.experiment.wandb_dir = config.experiment.wandb_dir.format(user_name)
	config.experiment.out_dir = config.experiment.out_dir.format(user_name)
	config.experiment.cache_dir = config.experiment.cache_dir.format(user_name)

	return config


def copy_data_to_mem(config, split='train'):
	"""

	:param config:
	:param split:
	:return:
	"""
	if config.dataset.name == "f30k":
		os.system(
			"tar -xf {} -C /dev/shm".format(
				os.path.join(config.dataset.root, config.dataset.name, "flickr30k-images.tar.gz"),
			)
		)
	elif config.dataset.name == "coco":
		os.system(
			"mkdir ./src/datasets/coco-images"
		)
		if split == 'train':
			os.system(
				"unzip -qq {} -d /dev/shm".format(
					os.path.join(config.dataset.root, config.dataset.name, "train2014.zip")
				)
			)
			os.system(
				"mv ./src/datasets/train2014/*.jpg ./src/datasets/coco-images"
			)
		os.system(
			"unzip -qq {} -d /dev/shm".format(
				os.path.join(config.dataset.root, config.dataset.name, "val2014.zip")
			)
		)
		os.system(
			"mv ./src/datasets/val2014/*.jpg ./src/datasets/coco-images"
		)
	else:

		raise NotImplementedError("Dataset {} not implemented".format(config.dataset.name))
