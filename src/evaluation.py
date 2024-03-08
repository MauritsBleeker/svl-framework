"""
Evaluation class
Reference code:
https://github.com/KunpengLi1994/VSRN/blob/master/evaluation.py
"""
import torch
import logging
import fire
import wandb
import numpy as np
from functools import partial
from munch import Munch
from torch.utils.data import DataLoader
from data.dataset import Dataset, collate_fn
from models.model import Model
from utils.cluster import copy_data_to_mem
from utils.utils import load_json_annotations, get_device, update_config
from utils.vocab import Vocabulary


class Evaluator(object):
    """
    Evaluation class
    """

    def __init__(self, config, split, model, json_file, device):
        """
		:param config: config class
		:param split: str, either validation or test
		:param model: model class
		:param json_file: json file with val/test data
		:param device: CPU/GPU
		"""

        self.config = config

        assert split in set(['val', 'test'])

        logging.info("Loading the {} evaluation set".format(split))

        self.split = split

        self.model = model

        self.dataset = Dataset(
            config=self.config,
            split=split,
            json_file=json_file
        )

        self.dataloader = DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.config.dataloader.eval_batch_size,
            num_workers=self.config.dataloader.num_workers,
            collate_fn=partial(
                collate_fn,
                sort_captions=self.config.model.image_caption_encoder.name == "VSE" or self.config.model.image_caption_encoder.name == "BASE"),
            pin_memory=True
        )

        self.device = get_device()

    @torch.no_grad()
    def encode_data(self):
        """
		encode the test data
		:return:
		"""

        caption_representations = np.zeros(
            (len(self.dataset.caption_ids), self.config.model.image_caption_encoder.embed_dim))
        image_representations = np.zeros(
            (len(self.dataset.caption_ids), self.config.model.image_caption_encoder.embed_dim))

        img_ids = np.zeros((len(self.dataset.caption_ids), 1))
        cap_ids = np.zeros((len(self.dataset.caption_ids), 1))

        for i, (
                tokens, images, dists, caption_ids, image_ids, raw_captions, cap_lengths, idx,
                _) in enumerate(
            self.dataloader):
            z_images, z_captions, _, _ = self.model(images.to(self.device), tokens.to(self.device),
                                                          cap_lengths.to(self.device))

            caption_representations[idx, :] = z_captions.cpu().numpy().copy()
            image_representations[idx, :] = z_images.cpu().numpy().copy()
            img_ids[idx, :] = np.array(image_ids).reshape(-1, 1)
            cap_ids[idx, :] = np.array(caption_ids).reshape(-1, 1)

        return image_representations, caption_representations, img_ids, cap_ids

    def i2t(self, image_representations, caption_representations):
        """
		:param image_representations: tensor with image representations
		:param caption_representations: tensor with caption representations
		:return:
		"""

        n_images = len(self.dataset) // self.config.dataset.captions_per_image

        ranks = np.zeros(n_images)
        top1 = np.zeros(n_images)
        r_precision = []

        for index in range(0, n_images):
            z_img = image_representations[self.config.dataset.captions_per_image * index].reshape(
                1, image_representations.shape[1]
            )

            sim = np.dot(z_img, caption_representations.T).flatten()
            ranking = np.argsort(sim)[::-1]

            rank = 1e20

            for i in range(
                    self.config.dataset.captions_per_image * index,
                    self.config.dataset.captions_per_image * index + self.config.dataset.captions_per_image, 1
            ):

                tmp = np.where(ranking == i)[0][0]

                if tmp < rank:
                    rank = tmp

            r_precision.append(
                (
                        (ranking[
                         :self.config.dataset.captions_per_image] >= self.config.dataset.captions_per_image * index) &
                        (ranking[
                         :self.config.dataset.captions_per_image] < self.config.dataset.captions_per_image * index + self.config.dataset.captions_per_image)
                ).sum() / self.config.dataset.captions_per_image
            )

            ranks[index] = rank
            top1[index] = ranking[0]

        r1 = self.recall_at_k(ranks, k=1)
        r5 = self.recall_at_k(ranks, k=5)
        r10 = self.recall_at_k(ranks, k=10)

        medr = np.floor(np.median(ranks)) + 1
        meanr = ranks.mean() + 1
        r_precision = np.mean(r_precision)

        return r1, r5, r10, medr, meanr, r_precision

    def t2i(self, image_representations, caption_representations):
        """

		:param image_representations: tensor with image representations
		:param caption_representations: tensor with caption representations
		:return:
		"""

        npts = image_representations.shape[0] // self.config.dataset.captions_per_image

        ims = np.array(
            [
                image_representations[i] for i in
                range(0, len(image_representations), self.config.dataset.captions_per_image)
            ]
        )

        ranks = np.zeros(self.config.dataset.captions_per_image * npts)
        top1 = np.zeros(self.config.dataset.captions_per_image * npts)

        for index in range(npts):

            # Get query captions
            queries = caption_representations[
                      self.config.dataset.captions_per_image * index:self.config.dataset.captions_per_image * index + self.config.dataset.captions_per_image
                      ]

            d = np.dot(queries, ims.T)

            inds = np.zeros(d.shape)

            for i in range(len(inds)):
                inds[i] = np.argsort(d[i])[::-1]
                ranks[self.config.dataset.captions_per_image * index + i] = np.where(inds[i] == index)[0][0]
                top1[self.config.dataset.captions_per_image * index + i] = inds[i][0]

        # Compute metrics
        r1 = self.recall_at_k(ranks, k=1)
        r5 = self.recall_at_k(ranks, k=5)
        r10 = self.recall_at_k(ranks, k=10)

        medr = np.floor(np.median(ranks)) + 1
        meanr = ranks.mean() + 1

        return r1, r5, r10, medr, meanr

    def recall_at_k(self, rankings, k):
        """

		:param rankings: numpy tensor with rankings
		:param k: 1, 5, 10, recall at k
		:return:
		"""
        return 100.0 * len(np.where(rankings < k)[0]) / len(rankings)

    def evaluate(self, step=None):
        """
		:param step: current training step (if we evaluate during training)
		:return:
		"""

        self.model.eval()

        logging.info('--- Start evaluation ---')

        image_representations, caption_representations, img_ids, cap_ids = self.encode_data()

        i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr, r_precision = self.i2t(
            image_representations,
            caption_representations
        )

        t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr, = self.t2i(
            image_representations,
            caption_representations
        )

        rsum = i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10

        self.log_results(
            i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr,
            t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr, rsum, step, r_precision,
        )

        return rsum, [i2t_r1, i2t_r5, i2t_r10, t2i_r1, t2i_r5, t2i_r10]

    def log_results(
            self, i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr, t2i_r1, t2i_r5,
            t2i_r10, t2i_medr, t2i_meanr, rsum, step, r_precision
    ):

        if self.split == 'val':
            if not self.config.experiment.development:
                wandb.log({
                    'i2t_r1': i2t_r1,
                    'i2t_r5': i2t_r5,
                    'i2t_r10': i2t_r10,
                    'i2t_medr': i2t_medr,
                    'i2t_meanr': i2t_meanr,
                    't2i_r1': t2i_r1,
                    't2i_r5': t2i_r5,
                    't2i_r10': t2i_r10,
                    't2i_medr': t2i_medr,
                    't2i_meanr': t2i_meanr,
                    'rsum': rsum,
                    'r_precision': r_precision
                }, step=step)

        logging.info(
            'Image to text (r@1, r@5, r@10, medr, meanr): %.1f, %.1f, %.1f, %.1f, %.1f' % (
                i2t_r1, i2t_r5, i2t_r10, i2t_medr, i2t_meanr
            ))
        logging.info(
            'Text to image (r@1, r@5, r@10, medr, meanr): %.1f, %.1f, %.1f, %.1f, %.1f' % (
                t2i_r1, t2i_r5, t2i_r10, t2i_medr, t2i_meanr
            ))
        logging.info("Recall sum: %.2f" % rsum)
        logging.info("Image to text r-precision: %.2f" % r_precision)


def main(path_to_model, split='test', data_root=None, zero_shot=False, copy_data=True, **kwargs):
    """
	:param path_to_model: path to model checkpoint
	:param split: test/validate split
	:param data_root: root folder of the data
	:param zero_shot: eval model in zero shot mode (i.e., pre-trained CLIP)
	:param copy_data: copy data to RAM mem of the node
	:return:
	"""

    checkpoint = torch.load(path_to_model, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    config = Munch.fromDict(checkpoint['config'])

    if kwargs:
        config = update_config(config, kwargs)

    if data_root:
        config.dataset.root = data_root

    if torch.cuda.is_available() and copy_data:
        copy_data_to_mem(config, split='test')

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    device = get_device()

    model = Model(config=config)

    if not zero_shot:
        model.load_state_dict(checkpoint['model'], strict=False)

    json_file = load_json_annotations(config)

    model.to_device(device)

    evaluator = Evaluator(config=config, split=split, model=model, json_file=json_file, device=device)

    evaluator.evaluate()


if __name__ == '__main__':
    fire.Fire(main)
