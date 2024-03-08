"""
Reference code: https://github.com/MauritsBleeker/reducing-predictive-feature-suppression/blob/main/src/models/caption_encoder.py
"""
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.pooling import AttentionPool2d
from utils.utils import get_device


AVAILABLE_MODELS = (
"resnet50", "resnet101", "resnet152",
)


class ProjectionHead(nn.Module):

    def __init__(self, in_features, projection_dim):
        """

        :param in_features:
        :param projection_dim:
        """
        super(ProjectionHead, self).__init__()

        self.projector = nn.Sequential(
            nn.Linear(in_features, in_features, bias=False),
            nn.ReLU(),
            nn.Linear(in_features, projection_dim, bias=False),
        )

        self.init_weights()

    def forward(self, x):
        """

        :param x:
        :return:
        """
        return self.projector(x)

    def init_weights(self):
        """

        :return:
        """

        nn.init.xavier_uniform_(self.projector[0].weight)
        nn.init.xavier_uniform_(self.projector[2].weight)


class EncoderText(nn.Module):

    def __init__(self, config):
        """

        :param config:
        """
        super(EncoderText, self).__init__()

        self.config = config

        self.word_dim = self.config.model.image_caption_encoder.word_dim
        self.embed_dim = self.config.model.image_caption_encoder.embed_dim

        self.embed = nn.Embedding(self.config.dataset.vocab_size, self.word_dim)

        # Sentence embedding
        self.rnn = nn.GRU(self.word_dim, self.embed_dim // 2, bidirectional=True, batch_first=True)

        self.fc = ProjectionHead(in_features=self.embed_dim, projection_dim=self.embed_dim)

        self.init_weights()

        self.device = get_device()

    def init_weights(self):
        """

        :return:
        """

        nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, captions, lengths, device='cpu'):
        """

        :param captions:
        :param lengths:
        :return:
        """
        # Embed word ids to vectors
        wemb_out = self.embed(captions)

        # Forward propagate RNNs
        packed = pack_padded_sequence(wemb_out, lengths.cpu(), batch_first=True)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(packed)
        padded = pad_packed_sequence(rnn_out, batch_first=True)

        # Reshape *final* output to (batch_size, hidden_size)
        I = lengths.expand(self.embed_dim, 1, -1).permute(2, 1, 0) - 1
        if torch.cuda.is_available():
            # TODO: Fix this
            I = I.to('cuda')

        out = torch.gather(padded[0], 1, I).squeeze(1)

        out = self.fc(out)

        return out


class EncoderImage(nn.Module):

    def __init__(self, config):
        super(EncoderImage, self).__init__()

        self.config = config

        assert self.config.model.image_caption_encoder.model_name in AVAILABLE_MODELS

        self.cnn = getattr(models, self.config.model.image_caption_encoder.model_name)(pretrained=True)
        self.backbone_dim = self.cnn.fc.in_features

        self.cnn.avgpool = nn.Identity()

        self.cnn.fc = nn.Identity()

        self.pooling = AttentionPool2d(
            spacial_dim=7,
            embed_dim=self.backbone_dim,
            num_heads=1,
            output_dim=self.config.model.image_caption_encoder.embed_dim
        )

    def forward(self, images):
        """

        :param images:
        :return:
        """

        feature_map = self.cnn(images).view(-1, self.backbone_dim, 7, 7)
        pooled = self.pooling(feature_map)

        return pooled


class BaseModel(nn.Module):

    def __init__(self, config):
        """

        :param config:
        """
        super(BaseModel, self).__init__()

        self.config = config

        self.image_encoder = EncoderImage(config=self.config)

        self.text_encoder = EncoderText(config=self.config)

        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / self.config.criterion.temperature),
            requires_grad=self.config.criterion.tune_temperature
        )

    def forward(self, images, tokens, lengths):
        """

        :param images:
        :param tokens:
        :param lengths:
        :return:
        """

        z_images = self.image_encoder(images)
        z_captions = self.text_encoder(tokens, lengths)

        logit_scale = self.logit_scale.exp()

        return z_images, z_captions, logit_scale
