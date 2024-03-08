import torch.optim as optim


def get_optimizer(optimizer_name, parameters, config, logger=None):
	"""

	:param optimizer_name:
	:param parameters:
	:param config:
	:param logger:
	:return:
	"""

	if logger:
		logger.log('creating [{}] from Config({})'.format(optimizer_name, config))

	if optimizer_name == 'adam':
		optimizer = optim.Adam(
			parameters,
			lr=float(config.optimizer.learning_rate),
			weight_decay=float(config.optimizer.weight_decay),
		)
	elif optimizer_name == 'adamw':
		optimizer = optim.AdamW(
			parameters,
			lr=float(config.optimizer.learning_rate),
			weight_decay=float(config.optimizer.weight_decay),
		)
	else:
		raise NotImplementedError

	return optimizer
