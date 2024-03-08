import torch.nn as nn
from sentence_transformers import SentenceTransformer


class TargetGenerator(nn.Module):

    def __init__(self, device):

        super(TargetGenerator, self).__init__()

        self.target_generator = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)


    def forward(self, captions, device):
        """

        :param captions:
        :param device:
        :return:
        """

        return self.target_generator.encode(
            captions,
            show_progress_bar=False,
            convert_to_numpy=False,
            convert_to_tensor=True,
            device=device
        )
