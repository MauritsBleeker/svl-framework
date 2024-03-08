"""
Reference code: https://github.com/fartashf/vsepp/blob/master/vocab.py
This script creates a vocabulary wrapper
"""
import nltk
import pickle
import os
import json
import argparse
from collections import Counter
from pycocotools.coco import COCO
from clip.simple_tokenizer import SimpleTokenizer
from nltk.tokenize import word_tokenize

ANNOTATIONS = {
    'coco': ['captions_train2014.json',
             'captions_val2014.json'],
    'f30k': ['dataset_flickr30k.json'],
}


class Vocabulary(object):
    """
    Simple vocabulary wrapper.
    """

    def __init__(self):

        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """

        :param word: string
        :return:
        """

        if word not in self.word2idx:

            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        """

        :param word: string with word
        :return: token index
        """
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def from_coco_json(path):
    """

    :param path: path to json file
    :return: list with captions
    """
    coco = COCO(path)
    ids = coco.anns.keys()
    captions = []
    for i, idx in enumerate(ids):
        captions.append(str(coco.anns[idx]['caption']))

    return captions


def from_flickr_json(path):
    """

    :param path: path to json file
    :return: list with captions
    """

    dataset = json.load(open(path, 'r'))['images']
    captions = []

    for i, d in enumerate(dataset):
        captions += [str(x['raw']) for x in d['sentences']]

    return captions


def from_txt(txt_file):
    """

    :param txt_file: text file with captions
    :return:
    """

    captions = []
    with open(txt_file, 'rb') as f:
        for line in f:
            captions.append(line.strip())
    return captions


def build_vocab(data_path, data_name, json, threshold):
    """

    :param data_path: path to data annotations
    :param data_name: name of the dataset
    :param json:
    :param threshold: threshold value to include word or not
    :return:
    """

    counter = Counter()
    for path in json[data_name]:
        full_path = f"{data_path}/annotations/{data_name}/{path}"
        if data_name == 'coco':
            captions = from_coco_json(full_path)
        elif data_name == 'f30k':
            captions = from_flickr_json(full_path)
        else:
            captions = from_txt(full_path)
        for i, caption in enumerate(captions):
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if i % 1000 == 0:
                print("[%d/%d] tokenized the captions." % (i, len(captions)))

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # add these tokens for the shortcuts
    vocab.add_word('0')
    vocab.add_word('1')
    vocab.add_word('2')
    vocab.add_word('3')
    vocab.add_word('4')
    vocab.add_word('5')
    vocab.add_word('6')
    vocab.add_word('7')
    vocab.add_word('8')
    vocab.add_word('9')

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def get_vocab(config, tokenizer):
    """

    :param config: config class
    :param tokenizer: tokenizer class
    :return:
    """

    if config.model.image_caption_encoder.name == "clip":
        return tokenizer.encoder
    elif config.model.image_caption_encoder.name == "VSE" or config.model.image_caption_encoder.name == "BASE":
        return load_vocab(config)
    else:
        raise NotImplementedError


def load_vocab(config):
    """

    :param config: config class
    :return:
    """

    vocab_path = os.path.join(config.dataset.vocab_path, config.dataset.vocab_file)
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab


def get_tokenizer(config):
    """

    config:
    :return:
    """
    if config.model.image_caption_encoder.name == "clip":
        return SimpleTokenizer()
    elif config.model.image_caption_encoder.name == "VSE" or config.model.image_caption_encoder.name == "BASE":
        return word_tokenize
    else:
        raise NotImplementedError


def main(arguments):
    vocab = build_vocab(data_path=arguments.data_path, data_name=arguments.data_name,
                        json=ANNOTATIONS, threshold=arguments.threshold)
    os.makedirs(arguments.vocab_dir, exist_ok=True)
    vocab_path = f'./{arguments.vocab_dir}/{arguments.data_name}_vocab.pkl'
    with open(vocab_path, 'wb+') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
    print('Saved vocabulary file to ', vocab_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./src/datasets')
    parser.add_argument('--data_name', default='coco',
                        help='coco, f30k')
    parser.add_argument('--vocab_dir', type=str, default='./src/vocab',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)
