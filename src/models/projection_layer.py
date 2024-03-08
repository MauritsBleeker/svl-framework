import torch.nn as nn
import torch


class PredictionLayer(nn.Module):

    def __init__(self, in_features, d_hidden, embed_layer, dropout_p=0.2):
        """

        :param in_features:
        :param d_hidden:
        :param embed_layer:
        """

        super(PredictionLayer, self).__init__()

        self.projection = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(p=dropout_p),
            nn.Linear(d_hidden, embed_layer.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embed_layer.embedding_dim),
            nn.Linear(d_hidden, embed_layer.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embed_layer.embedding_dim),
        )

        self.output_layer = nn.Linear(embed_layer.embedding_dim, embed_layer.num_embeddings, bias=False)
        self.output_layer.weight = nn.Parameter(embed_layer.weight)

        self.init_weights()

    def forward(self, x):
        """
        :param x:
        :return: logits
        """

        x = self.projection(x)
        logits = self.output_layer(x)

        return logits

    def init_weights(self, tie_output_to_embedding=None):
        """

        :return:
        """

        nn.init.xavier_uniform_(self.projection[1].weight)
        nn.init.xavier_uniform_(self.projection[5].weight)
        nn.init.xavier_uniform_(self.projection[8].weight)
