"""
Reference code: https://github.com/fartashf/vsepp/blob/master/model.py
"""
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from utils.vocab import load_vocab
from utils.utils import get_device
from utils.pooling import AttentionPool2d

class EncoderImage(nn.Module):

    def __init__(self, config):
        """
        
        :param config:
        """

        super(EncoderImage, self).__init__()

        self.config = config
        
        self.embed_size = self.config.model.image_caption_encoder.embed_dim

        self.cnn = self.get_cnn(backbone=self.config.model.image_caption_encoder.model_name)
        self.backbone_dim = self.cnn.fc.in_features

        self.attention_pooling = False

        if 'img_pooling' in self.config.model.image_caption_encoder and self.config.model.image_caption_encoder.img_pooling == 'attention':

            self.attention_pooling = True

            self.cnn.avgpool = nn.Identity()

            self.cnn.fc = nn.Sequential()

            self.attention_pooling = AttentionPool2d(
                spacial_dim=7,
                embed_dim=self.backbone_dim,
                num_heads=1,
                output_dim=self.embed_size
            )
        else:

            self.fc = nn.Linear(self.backbone_dim, self.embed_size)

            self.cnn.fc = nn.Sequential()

            self.init_weights()

    def get_cnn(self, backbone):
        """

        :param backbone:
        :return:
        """

        if backbone == 'resnet50':
            model = models.__dict__[backbone](weights='ResNet50_Weights.DEFAULT')
        elif backbone == 'resnet152':
            model = models.__dict__[backbone](weights='ResNet152_Weights.DEFAULT')
        else:
            raise NotImplementedError

        return model

    def init_weights(self):
        """
        Xavier initialization for the fully connected layer
        :return: 
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """
        :param images: 
        :return: 
        """
        z_images = self.cnn(images)

        if self.attention_pooling:
            z_images = z_images.view(-1, self.backbone_dim, 7, 7)
            z_images = self.attention_pooling(z_images)

            return z_images

        z_images = self.fc(z_images)

        return z_images


class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_gru_layers):
        """
        
        :param vocab_size: 
        :param word_dim: 
        :param embed_size: 
        :param num_gru_layers:
        """
        super(EncoderText, self).__init__()
        
        self.embed_size = embed_size

        self.embed = nn.Embedding(vocab_size, word_dim)

        self.rnn = nn.GRU(word_dim, embed_size, num_gru_layers, batch_first=True)

        self.device = get_device()

        self.init_weights()

        self.device = get_device()

    def init_weights(self):
        """
        Uniform initialization for the fully connected layer
        :return: 
        """
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, tokens, lengths):
        """
        
        :param tokens: 
        :param lengths: 
        :return: 
        """

        tokens = self.embed(tokens)
        packed = pack_padded_sequence(tokens, lengths.cpu(), batch_first=True)

        z_captions, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(z_captions, batch_first=True)

        I = torch.tensor(lengths, dtype=torch.int64, device=self.device).reshape(-1, 1, 1)
        I = I.expand(tokens.size(0), 1, self.embed_size) - 1
        z_captions = torch.gather(padded[0], 1, I).squeeze(1)

        return z_captions


class VSE(nn.Module):
    def __init__(self, config):
        """

        :param config:
        """
        super(VSE, self).__init__()

        self.config = config

        self.image_encoder = EncoderImage(
            config=self.config
        )

        self.text_encoder = EncoderText(
            embed_size=self.config.model.image_caption_encoder.embed_dim,
            vocab_size=self.config.dataset.vocab_size,
            word_dim=self.config.model.image_caption_encoder.word_dim,
            num_gru_layers=self.config.model.image_caption_encoder.num_gru_layers
        )

        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / self.config.criterion.temperature),
            requires_grad=self.config.criterion.tune_temperature
        )

    def forward(self, images, tokens, lengths):
        """
        Compute the image and captions embeddings
        :param images:
        :param tokens:
        :param lengths:
        :return:

        """
        # Forward
        z_images = self.image_encoder(images)
        z_captions = self.text_encoder(tokens, lengths)

        logit_scale = self.logit_scale.exp()

        return z_images, z_captions, logit_scale
