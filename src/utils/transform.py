"""
Sources:
https://github.com/openai/CLIP/blob/3702849800aa56e2223035bccd1c6ef91c704ca8/clip/clip.py#L79
https://github.com/fartashf/vsepp/blob/abe382fd9c751d1b92c95030df8fb804a7d5ef53/data.py#L311
"""
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from torchvision import transforms
from utils.vocab import Vocabulary

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def image_transform_clip(crop_size):
    """

    :param crop_size: int
    :return: transformations
    """

    normlize = Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])

    img_transform = Compose([
        Resize((crop_size), interpolation=BICUBIC),
        CenterCrop(crop_size),
        _convert_image_to_rgb,
    ])

    return normlize, img_transform


def image_transform_vse(crop_size, split):
    """

    :param crop_size:
    :param split:
    :return:
    """
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    if split == 'train':
        img_transform = transforms.Compose([transforms.RandomResizedCrop(crop_size),
                  transforms.RandomHorizontalFlip()])
    elif split == 'val' or split == 'test':
        img_transform = Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    else:
        raise NotImplementedError

    normlize = Compose([transforms.ToTensor(), normalizer])

    return normlize, img_transform


def get_image_transform(config, split):
    """

    :param config:
    :param split:
    :return:
    """
    if config.model.image_caption_encoder.name == "clip":
        return image_transform_clip(crop_size=config.dataloader.crop_size)
    elif config.model.image_caption_encoder.name == "VSE" or config.model.image_caption_encoder.name == "BASE":
        return image_transform_vse(crop_size=config.dataloader.crop_size, split=split)
    else:
        raise NotImplementedError
