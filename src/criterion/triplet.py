"""
Refence code: https://github.com/fartashf/vsepp/blob/master/model.py
"""
import torch
import torch.nn as nn
from utils.utils import matmul


class TripletLoss(nn.Module):
    """
    Compute triplet loss
    """

    def __init__(self, device, max_violation=True, margin=0):

        super(TripletLoss, self).__init__()
        
        self.device = device
        self.margin = margin
        self.sim = matmul
        self.max_violation = max_violation

    def forward(self, z_images, z_captions, *argv):
        """

        :param z_images: tensor with image latents
        :param z_captions: tensor with caption latents
        :param argv: not used
        :return: loss value
        """

        scores = self.sim(z_images, z_captions)
        diagonal = scores.diag().view(z_images.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_caption_retrieval = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_image_retrieval = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = mask.to(self.device)
        cost_caption_retrieval = cost_caption_retrieval.masked_fill_(I, 0)
        cost_image_retrieval = cost_image_retrieval.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_caption_retrieval = cost_caption_retrieval.max(1)[0]
            cost_image_retrieval = cost_image_retrieval.max(0)[0]

        return cost_caption_retrieval.sum() + cost_image_retrieval.sum()
