import random
import torchvision.transforms as transforms
from collections import defaultdict
from torchvision import datasets


class ShortcutSampler(object):

    def __init__(self, config, eval=False, n_images=None):
        """

        :param config: config class
        :param eval: do we use the sample in evaluation mode?
        :param n_images: number of images in the dataset
        """

        super(ShortcutSampler, self).__init__()

        self.config = config

        self.eval = eval

        self.mnist = datasets.MNIST(
            self.config.dataset.root,
            train=True,
            download=True,
            transform=None
        )

        self.minst_transform = transforms.ToPILImage()

        self.label_to_idx = defaultdict(list)

        for i, label in enumerate(self.mnist.train_labels):
            self.label_to_idx[int(label)].append(i)

        self.on_images = self.config.shortcuts.training.on_image if not eval else self.config.shortcuts.evaluation.on_image
        self.on_caption = self.config.shortcuts.training.on_caption if not eval else self.config.shortcuts.evaluation.on_caption

        self.n_images = n_images

        self.offset = None

        if not self.eval:
            self.set_offset()

    def set_offset(self):

        self.offset = random.randint(0, self.n_images - 1)

    def sample_shortcut(self, image=None, caption=None, shortcut_id=None):
        """

        :param image:
        :param caption:
        :param shortcut_id:
        :return:
        """
        if self.on_caption or self.on_images:

            if self.config.shortcuts.bits.use_bits or not shortcut_id:
                if (self.config.shortcuts.bits.random or not shortcut_id) and not self.eval:
                    shortcut_id = str(random.randint(0, (2 ** self.config.shortcuts.bits.n_bits) - 1))
                else:
                    shortcut_id = str(shortcut_id % (2 ** self.config.shortcuts.bits.n_bits))

            elif not self.eval and not self.config.shortcuts.bits.use_bits and self.config.shortcuts.random_number:
                # we use unique shortcuts, but not the same id every epoch. I.e., model can't over fit
                shortcut_id = str((int(shortcut_id) + self.offset) % self.n_images)

            if shortcut_id and type(shortcut_id) != str:
                shortcut_id = str(shortcut_id)

            list_of_digits = [int(i) for i in shortcut_id]

            list_of_digits = [0] * (self.config.shortcuts.n_digits - len(list_of_digits)) + list_of_digits

            mnist_images = [self.mnist.data[random.choice(self.label_to_idx[digit])] for digit in list_of_digits]

            if image and self.on_images:
                image = self.add_shortcut_to_img(image, mnist_images)

            if caption and self.on_caption:
                caption = self.add_shortcut_to_caption(caption, list_of_digits)

        return image, caption, shortcut_id

    def add_shortcut_to_img(self, image, mnist_digits):
        """

        :param image:
        :param mnist_digits:
        :return:
        """
        stride = image.size[0] // self.config.shortcuts.n_digits

        for i, digit in enumerate(mnist_digits):
            image.paste(self.minst_transform(digit), (i * stride, 0))

        return image

    def add_shortcut_to_caption(self, caption, list_of_digits):
        """

        :param caption:
        :param list_of_digits:
        :return:
        """

        assert type(caption) == str

        caption = caption + ' ' + ' '.join([str(digit) for digit in list_of_digits])

        return caption
