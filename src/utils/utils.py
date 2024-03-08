import json
import os
import torch
from operator import attrgetter


def matmul(images, captions):
    """
    Matmul between all the image and captions pairs
    :param images: tensor with image representations
    :param captions:  tensor with captions
    :return: dot product scores
    """
    return images.mm(captions.t())


def l2norm(x):
    """
    L2-normalize columns of X
    :param x: tensor
    :return:
    """
    norm = torch.pow(x, 2).sum(dim=1, keepdim=True).sqrt()
    x = torch.div(x, norm)
    return x


def load_json(root, file):
    """
    :param root: root folder of json file
    :param file: json file name
    :return:
    """

    return json.load(open(os.path.join(root, file)))


def update_config(config, kwargs):
    """
    Update config with flags from the commandline
    :param config: config class
    :param kwargs: additional arugments/overwrites
    :return:
    """

    for key, value in kwargs.items():
        try:
            _ = attrgetter(key)(config)
            subconfig = config

            for sub_key in key.split('.'):
                if isinstance(subconfig[sub_key], dict):
                    subconfig = subconfig[sub_key]
                else:
                    if subconfig[sub_key] is None or isinstance(subconfig[sub_key], type(value)):
                        subconfig[sub_key] = value
                    else:
                        raise Exception("wrong value type")

        except AttributeError:
            Exception("{} not in config".format(key))

    return config


def count_params(model):
    """
    Count number of parameters in PyTorch model/module
    :param model: PyTorch module
    :return: int with the number of parameters
    """

    return sum([m.numel() for m in model.parameters()])


def model_norm(model):
    """
    Simple function to return norm of the model weights
    :param model: PyTorch module
    :return: sum of the norm of all modules/layers in a model
    """

    return sum([m.norm() for m in model.parameters()])


def load_json_annotations(config):
    """
    Load json annotations
    :param config: Config class
    :return:
    """

    file_path = os.path.join(
        config.dataset.annotation_path,
        config.dataset.name,
        config.dataset.annotation_file
    )
    json_file = json.load(open(file_path, 'rb'))

    return json_file


def get_device():
    """
    Return device to run on cuda/cpu
    :return:
    """

    return 'cuda' if torch.cuda.is_available() else 'cpu' if torch.backends.mps.is_built() else 'cpu'
