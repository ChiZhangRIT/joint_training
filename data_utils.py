"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import json
import itertools
import pdb

from six.moves import urllib

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    if isinstance(space_separated_fragment, str):
        try:
          word = str.encode(space_separated_fragment)
        except:
          word = ''
        #   pdb.set_trace()
    else:
        word = space_separated_fragment
    words.extend(re.split(_WORD_SPLIT, word))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      print('>> Full Vocabulary Size :',len(vocab_list))
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):

  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=True):

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):

  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_custom_data(working_dir, train_enc, train_dec, enc_vocab_size, dec_vocab_size, tokenizer=None):

    # Create vocabulary of the appropriate size
    enc_vocab_path = os.path.join(working_dir, "vocab%d.enc" % enc_vocab_size)
    dec_vocab_path = os.path.join(working_dir, "vocab%d.dec" % dec_vocab_size)
    create_vocabulary(enc_vocab_path, train_enc, enc_vocab_size, tokenizer)
    create_vocabulary(dec_vocab_path, train_dec, dec_vocab_size, tokenizer)

    # Create token ids for the training data.
    enc_train_ids_path = train_enc + (".ids%d" % enc_vocab_size)
    dec_train_ids_path = train_dec + (".ids%d" % dec_vocab_size)
    data_to_token_ids(train_enc, enc_train_ids_path, enc_vocab_path, tokenizer)
    data_to_token_ids(train_dec, dec_train_ids_path, dec_vocab_path, tokenizer)

    return (enc_train_ids_path, dec_train_ids_path, enc_vocab_path, dec_vocab_path)


def parse_MSCOCO(MSCOCO_path, working_dir, permute=False):
    """
    Args:
        MSCOCO_path: path to MSCOCO dataset
        working_dir: folder where temporary data will be stored
        permute: boolean. If set to 'True', 'ab' and 'ba' are different pairs.

    Returns:
        im: image
        cap_A: encoder input
        cap_B: decoder input
    """

    # load data
    with open(MSCOCO_path + 'annotations/captions_train2014.json', 'rb') as f:
        train_data = json.load(f)

    # find all images with its captions
    imageid_to_captions = {}
    for i in xrange(len(train_data['annotations'])):
        caption = train_data['annotations'][i]['caption']
        imageid = train_data['annotations'][i]['image_id']
        imageid_to_captions.setdefault(imageid,[]).append(caption)
    id_to_filename = {}
    for i in xrange(len(train_data['images'])):
        filename = train_data['images'][i]['file_name']
        id = train_data['images'][i]['id']
        id_to_filename[id] = filename
    filename_to_imageid = dict((v,k) for k,v in id_to_filename.iteritems())

    filename_to_captions = {}
    im, cap_A, cap_B = [], [], []
    for k,v in filename_to_imageid.iteritems():
        # filename_to_captions has the structure of {filename: [caption1, caption2, ...]}
        filename_to_captions[k] = imageid_to_captions[v]

        # find all 2-element combinations of captions under each image
        comb_cap = list(itertools.combinations(imageid_to_captions[v], 2))
        for i in comb_cap:
            im.append(MSCOCO_path + 'train2014/' + k)
            cap_A.append(i[0])
            cap_B.append(i[1])
            if permute:
                im.append(MSCOCO_path + 'train2014/' + k)
                cap_A.append(i[1])
                cap_B.append(i[0])

    with open(working_dir + 'im_filename.txt', 'w') as f:
        for item in im:
            f.write("%s\n" % item)
    with open(working_dir + 'caption_a.txt', 'w') as f:
        for item in cap_A:
            f.write("%s\n" % item)
    with open(working_dir + 'caption_b.txt', 'w') as f:
        for item in cap_B:
            f.write("%s\n" % item)

    # return (im, cap_A, cap_B)
