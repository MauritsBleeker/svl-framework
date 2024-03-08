"""
Reference code: https://github.com/MauritsBleeker/reducing-predictive-feature-suppression/blob/main/src/utils/decoding_loss.py
"""
import torch
import torch.nn as nn
from criterion.target_reconstruction import TargetReconstruction
from utils.constraint import Constraint
from utils.constraint import ConstraintOptimizer


class DecodingLoss(nn.Module):

    def __init__(self, config):
        """

        :param self:
        :param config: config class
        :return:
        """
        super(DecodingLoss, self).__init__()

        self.config = config

        self.reconstruction_criterion = TargetReconstruction(
            reconstruction_metric=self.config.criterion.reconstruction_metric
        )

        self.beta = 1

        if self.config.reconstruction_constraint.use_constraint:

            self.reconstruction_constraint = Constraint(
                self.config.reconstruction_constraint.bound,
                'le',
                start_val=float(self.config.reconstruction_constraint.start_val),
            )

            self.constraint_opt = ConstraintOptimizer(
                torch.optim.SGD,
                params=self.reconstruction_constraint.parameters(),
                lr=5e-3,
                momentum=self.config.reconstruction_constraint.alpha,
                dampening=self.config.reconstruction_constraint.alpha,
            )

    def forward(self, reconstructions, targets):
        """

        :param reconstructions: predicted reconstruction
        :param targets: targets, either latent or input
        :return:
        """

        reconstruction_loss = self.reconstruction_criterion(reconstructions, targets)

        if self.config.reconstruction_constraint.use_constraint:
            constraint_loss = self.reconstruction_constraint(reconstruction_loss)[0]
            return constraint_loss, reconstruction_loss
        else:
            return self.beta * reconstruction_loss, reconstruction_loss
