"""
Reference code: https://github.com/vinid/neg_clip/blob/main/src/training/scheduler.py
https://github.com/MauritsBleeker/mm-feature-suppression/blob/05044cbdb5ef24a709fe993f75969b5284a32330/src/utils/optimizers.py#L41

"""
import numpy as np
import torch.optim as optim


def assign_learning_rate(optimizer, new_lr):
    """

    :param optimizer:
    :param new_lr:
    :return:
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    """

    :param base_lr:
    :param warmup_length:
    :param step:
    :return:
    """
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_steps, steps, eta_min=0):
    """

    :param optimizer:
    :param base_lr:
    :param warmup_steps:
    :param steps:
    :param eta_min:
    :return:
    """

    def _lr_adjuster(step):
        if step < warmup_steps:
            lr = _warmup_lr(base_lr, warmup_steps, step)
        else:
            e = step - warmup_steps
            es = steps - warmup_steps
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * (base_lr - eta_min) + eta_min
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def get_lr_scheduler(scheduler_name, optimizer, config, n_iterations, logger=None):
    """
    :param scheduler_name:
    :param optimizer:
    :param config:
    :param n_iterations:
    :param logger:
    :return:
    """

    if logger:
        logger.log('creating [{}] from Config({})'.format(scheduler_name, config))

    if scheduler_name == 'cosine_annealing':

        lr_scheduler = cosine_lr(
            optimizer=optimizer,
            base_lr=float(config.optimizer.learning_rate),
            warmup_steps=config.optimizer.warmup_steps,
            steps=config.training.n_epochs * n_iterations
        )

    elif scheduler_name == 'stepLR':

        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=config.lr_scheduler.step_size,
            gamma=config.lr_scheduler.gamma
        )

    else:
        raise ValueError(f'Invalid scheduler name: {scheduler_name}')

    return lr_scheduler
