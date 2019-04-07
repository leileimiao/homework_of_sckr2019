#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings       
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from pattern.en import tokenize
from time import time

import gensim
import logging
import multiprocessing
import os
import re
import sys
import chardet  


from gensim.models.fasttext import FastText as FT_gensim
from gensim.test.utils import datapath

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    return cleantext


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for root, dirs, files in os.walk(self.dirname):
            for filename in files:
                file_path = root + '/' + filename
                for line in open(file_path,'rb'):
                    sline = line.strip()
                    if sline == "":
                        continue
                    rline = cleanhtml(sline.decode('utf-8'))
                    tokenized_line = ' '.join(tokenize(rline))
                    is_alpha_word_line = [word for word in
                                          tokenized_line.lower().split()
                                          if word.isalpha()]
                    yield is_alpha_word_line


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ("Please use python train_with_gensim.py data_path")
        exit()
    data_path = sys.argv[1]
    begin = time()

    sentences = MySentences(data_path)
    '''
    model = gensim.models.Word2Vec(sentences,
                                   size=200,
                                   window=10,
                                   min_count=10,
                                   workers=multiprocessing.cpu_count())
    model.save("data/model/word2vec_gensim")
    model.wv.save_word2vec_format("data/model/word2vec_org",
                                  "data/model/vocabulary",
                                  binary=False)

    # Set file names for train and test data
    corpus_file = datapath('lee_background.cor')
    '''
    model_gensim = FT_gensim(sentences, size=150, window=8, min_count=5, iter=10,min_n = 3 , max_n = 6)
    model_gensim.save("model/fasttext_gensim")
    # build the vocabulary
    #model_gensim.build_vocab(corpus_file=corpus_file)

    # train the model
    '''
    model_gensim.train(
    corpus_file=corpus_file, epochs=model_gensim.epochs,
    total_examples=model_gensim.corpus_count, total_words=model_gensim.corpus_total_words)

    print(model_gensim)
    '''
    end = time()
    print ("Total procesing time: %d seconds" % (end - begin))