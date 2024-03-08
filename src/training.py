import os
import fire
import logging
import torch
import wandb
import munch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from contextlib import nullcontext
from functools import partial
from torch.cuda import amp
from torch.utils.data import DataLoader
from munch import Munch
from evaluation import Evaluator
from models.model import Model
from models.target_generator import TargetGenerator
from criterion.info_nce import InfoNCE
from criterion.triplet import TripletLoss
from data.dataset import Dataset, collate_fn
from utils.optimizer import get_optimizer
from utils.scheduler import get_lr_scheduler
from utils.cluster import copy_data_to_mem, update_config_with_username
from utils.decoding_loss import DecodingLoss
from utils.utils import update_config, get_device, load_json_annotations
from utils.vocab import Vocabulary


class Trainer(object):
    """
    Training class
    """

    def __init__(self, config):


        self.config = config

        self.json_file = load_json_annotations(self.config)

        self.dataset = Dataset(
            config=self.config,
            split='train',
            json_file=self.json_file
        )

        self.model = Model(config=self.config)

        if not self.config.experiment.development:

            wandb.watch(
                self.model,
                log='all',
                log_freq=self.config.training.log_step,
                idx=0
            )

        self.device = get_device()

        self.model.to_device(device=self.device)

        self.parameters = self.model.return_parameters()

        self.evaluator = Evaluator(
            config=self.config,
            split='val',
            model=self.model,
            json_file=self.json_file,
            device=self.device
        )

        self.data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.config.dataloader.batch_size,
            shuffle=True,
            collate_fn=partial(
                collate_fn,
                sort_captions=self.config.model.image_caption_encoder.name == "VSE" or self.config.model.image_caption_encoder.name == "BASE"
            ),
            num_workers=self.config.dataloader.num_workers
        )

        self.contrastive_loss = self.get_criterion()

        self.optimizer = get_optimizer(
            optimizer_name=self.config.optimizer.name,
            parameters=self.parameters,
            config=self.config
        )

        self.epoch = self.step = 0

        self.scaler = amp.GradScaler()

        self.lr_scheduler = get_lr_scheduler(
            scheduler_name=self.config.lr_scheduler.name,
            optimizer=self.optimizer,
            config=self.config,
            n_iterations=len(self.data_loader)
        )

        if self.config.model.target_decoder.decode_target:
            self.decoding_loss = DecodingLoss(config=self.config)

        self.best_score = -float('inf')

        if 'target_decoder' in self.config.model and self.config.model.target_decoder.decode_target:

            self.target_generator = TargetGenerator(device=self.device)


    def train(self):
        """
        Training function
        :return:
        """

        logging.info('--- Start with zero-shot evaluation before training ---')

        self.evaluator.evaluate(step=self.step) if not self.config.experiment.development else None

        logging.info('--- Start training ---')

        for epoch in range(0, self.config.training.n_epochs):

            self.training_epoch()

        self.store_model(file_name=self.config.training.model_save_file) if not self.config.experiment.development \
            else None

    def training_epoch(self):
        """
        Training epoch step
        :return:
        """

        self.model.train()
        self.epoch += 1

        for i, batch in enumerate(self.data_loader):

            self.iteration(batch, iteration=i)

            self.step += 1

        if self.config.lr_scheduler.name == 'stepLR':

            self.lr_scheduler.step()

        rsum, _ = self.evaluator.evaluate(step=self.step)

        if rsum > self.best_score and self.config.experiment.store_best_validation_model:

            logging.info('Store new best model in epoch: {0} '.format(self.epoch))

            self.best_score = rsum

            self.store_model(file_name=self.config.training.best_model_file) if not self.config.experiment.development \
                else None

        if self.config.shortcuts.use_shortcuts and not self.config.shortcuts.bits.use_bits and self.config.shortcuts.random_number:
            self.dataset.shortcut_sampler.set_offset()

    def iteration(self, batch, iteration):
        """
        Training iteration

        :param batch: tuple with all input tensors from training
        :param iteration: int with current training step
        :return:
        """

        self.optimizer.zero_grad()

        if self.config.reconstruction_constraint.use_constraint:
            self.decoding_loss.constraint_opt.zero_grad()

        if self.config.lr_scheduler.name == 'cosine_annealing':

            self.lr_scheduler(step=self.step)

        tokens, images, dists, caption_ids, image_ids, raw_captions, cap_lengths, idxs, _ = batch

        with amp.autocast() if self.config.training.use_fp16 else nullcontext():

            z_images, z_captions, logit_scale, reconstructions = self.model(
                images.to(self.device), tokens.to(self.device), cap_lengths
            )

            latent_targets = None

            if self.config.model.target_decoder.decode_target:

                latent_targets = self.target_generator(raw_captions, device=self.device)


            loss, contrastive_loss, caption_cross_entropy, image_cross_entropy, reconstruction_loss = self.compute_loss(
                z_images, z_captions, logit_scale, latent_targets, reconstructions
            )

        self.optimizer_step(loss=loss)

        if iteration % self.config.training.log_step == 0:

            self.logging(
                i=iteration, loss=loss, contrastive_loss=contrastive_loss, reconstruction_loss=reconstruction_loss
            )

    def optimizer_step(self, loss):
        """
        Perform optimization step

        :param loss: tesnor with loss value
        :return:
        """

        if self.config.training.use_fp16:
            self.scaler.scale(loss).backward()

        else:
            loss.backward()

        if self.config.training.grad_clip > 0:
            if self.config.training.use_fp16:
                self.scaler.unscale_(self.optimizer)

            nn.utils.clip_grad.clip_grad_norm_(self.parameters, self.config.training.grad_clip)

        if self.config.training.use_fp16:
            self.scaler.step(self.optimizer)
        else:
            self.optimizer.step()

        if self.config.model.target_decoder.decode_target and self.config.reconstruction_constraint.use_constraint:
            self.decoding_loss.constraint_opt.step()

    def get_criterion(self):
        """
        Return optimization function

        Returns: optimization function
        """
        if self.config.criterion.name == "infonce":

            return InfoNCE(device=self.device,
                           use_ifm=self.config.criterion.ifm.use_ifm,
                           epsilon=self.config.criterion.ifm.epsilon)
        elif self.config.criterion.name == "triplet":

            return TripletLoss(device=self.device,
                               margin=self.config.criterion.alpha)
        else:
            raise NotImplementedError

    def compute_loss(self, z_images, z_captions, logit_scale, latent_targets, reconstructions):
        """
        Compute the losses for training

        :param z_images: tensor with latents of the images
        :param z_captions: tensor with latents of the captions
        :param logit_scale: scaler for logit values
        :param reconstructions: tensor with the target reconstructions
        :param latent_targets: latent targets to reconstruct
        :return:  loss values
        """

        loss = contrastive_loss = cap_token_reconstruction = img_token_reconstruction = ltd_loss_value = 0

        if self.config.model.image_caption_encoder.train:

            contrastive_loss = self.contrastive_loss(z_images, z_captions, logit_scale)
            loss += contrastive_loss

        if self.config.model.target_decoder.decode_target:

            ltd_loss, ltd_loss_value = self.decoding_loss(reconstructions, latent_targets.to(self.device))
            loss += ltd_loss

        return loss, contrastive_loss, cap_token_reconstruction, img_token_reconstruction, ltd_loss_value

    def logging(self, loss, i, contrastive_loss=None, reconstruction_loss=None):
        """
        Log to WandB and print

        :param loss: loss value
        :param i: iteration counter
        :param contrastive_loss: loss value of the contrastive loss
        :param reconstruction_loss: loss value of the reconstruction loss
        :return:
        """

        logging.info(
            'Epoch: [{0}][{1}/{2}]\t''Loss value: {3}\t'.format(self.epoch, i, len(self.data_loader), loss.data)
        )

        if not self.config.experiment.development:

            wandb.log({
                'epoch': self.epoch,
                'step': self.step,
                'loss': loss.data,
                'lr': self.optimizer.param_groups[0]['lr']
            }, step=self.step)

            if contrastive_loss:
                wandb.log(
                    {'contrastive loss': contrastive_loss.data.data},
                    step=self.step
                )

        if reconstruction_loss:

            if reconstruction_loss:

                wandb.log({
                    'reconstruction_loss': reconstruction_loss.data,
                }, step=self.step)

                if self.config.reconstruction_constraint.use_constraint:
                    wandb.log({
                        'multiplier': self.decoding_loss.reconstruction_constraint.multiplier.data
                    }, step=self.step)

    def store_model(self, file_name):
        """
        Store model checkpoint

        :param file_name: file name of the model to store
        :return:
        """

        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': munch.unmunchify(self.config),
            'step': self.step
        }

        directory = os.path.join(self.config.experiment.out_dir, self.config.experiment.experiment_name)
        os.makedirs(directory, exist_ok=True)

        logging.info(f'Store best model in {directory}')
        torch.save(state_dict, os.path.join(directory, file_name))


def main(yaml_file, **kwargs):
    """

    :param yaml_file: yaml config file
    :param kwargs: additional input flags from command line
    :return:
    """

    config = Munch.fromYAML(open(yaml_file, 'rb'))

    if 'wandb_user' in kwargs:
        config.experiment.entity = kwargs['wandb_user']

    if 'user_name' in kwargs:
        config = update_config_with_username(config=config, user_name=kwargs['user_name'])

    if kwargs:
        config = update_config(config, kwargs)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    cudnn.benchmark = True

    wandb.init(
        project=config.experiment.wandb_project,
        entity=config.experiment.entity,
        name=config.experiment.experiment_name,
        dir=config.experiment.wandb_dir,
        config=munch.unmunchify(config),
        tags=[config.dataset.name]
    )

    if torch.cuda.is_available():
        # copy data to mem of the GPU node to speed up training
        copy_data_to_mem(config, split='train')

    trainer = Trainer(config=config)

    trainer.train()


if __name__ == '__main__':

    fire.Fire(main)
