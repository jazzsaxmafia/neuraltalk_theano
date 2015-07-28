#-*- coding: utf-8 -*-
import sys
from imagernn.data_provider import *
from driver import *
import argparse
import ipdb

import numpy as np
import pandas as pd

import theano
import theano.tensor as T

from keras import initializations

dataset = 'flickr30k'
word_count_threshold = 5
word_embedding_dim = 256
image_embedding_dim= 256
hidden_size = 256
batch_size = 100

parser = argparse.ArgumentParser()
dp = getDataProvider(dataset)
misc = {}

(
    misc['wordtoix'],
    misc['ixtoword'],
    bias_init_vector
) = preProBuildWordVocab(dp.iterSentences('train'), word_count_threshold)

num_vocab = len(misc['wordtoix'])
num_images = dp.features.shape[1]

'''
일단
image encoder ( 4096 -> embedding dim )와
text encoder ( vocab dim -> embedding dim)을 정의하자

'''
We = initializations.uniform((4096, image_embedding_dim))
be = initializations.zero((image_embedding_dim,))
Ws = initializations.uniform((num_vocab, word_embedding_dim))


'''
text decoder (hidden dim -> vocab dim)을 정의하자

'''
Wd = initializations.uniform((hidden_size, num_vocab))
bd = initializations.zero((num_vocab,))

'''
이미지(batch) -> image_embedding_dim
'''
image = T.vector()
embedded_image = T.dot(image, We) + be
embedded_image = embedded_image.dimshuffle('x',0)

'''
sentence

'''
sentence = T.matrix()
embedded_sentence = T.dot(sentence, Ws)

'''
이미지를 sentence의 맨 앞에 붙임
'''
X = T.concatenate([embedded_image, embedded_sentence], axis=0)

'''
LSTM weight ( i, f, c, o에 대한 weight들 )
을 정의하자
'''
Wi = initializations.orthogonal((word_embedding_dim, word_embedding_dim))
Wf = initializations.orthogonal((word_embedding_dim, word_embedding_dim))
Wc = initializations.orthogonal((word_embedding_dim, word_embedding_dim))
Wo = initializations.orthogonal((word_embedding_dim, word_embedding_dim))

bi = initializations.zero((word_embedding_dim))
bf = initializations.zero((word_embedding_dim))
bc = initializations.zero((word_embedding_dim))
bo = initializations.zero((word_embedding_dim))

xi = T.dot(X, Wi) + bi
xf = T.dot(X, Wf) + bf
xc = T.dot(X, Wc) + bc
xo = T.dot(X, Wo) + bo

def _step():
    it = T.nnet.sigmoid(x
    pass


'''
Test
'''
print_x = theano.function(inputs=[image, sentence], outputs=X, allow_input_downcast=True)
print_image = theano.function(inputs=[image], outputs=embedded_image, allow_input_downcast=True)
print_sentence=theano.function(inputs=[sentence], outputs=embedded_sentence, allow_input_downcast=True)

batch = [dp.sampleImageSentencePair() for i in xrange(5)]
x = batch[0]
image_feat = x['image']['feat'].astype(theano.config.floatX)
sentence_index = [misc['wordtoix'][w] for w in x['sentence']['tokens'] if w in misc['wordtoix']]

n_vocab = len(misc['wordtoix'])
sentence_vector = np.zeros((len(sentence_index), n_vocab))
for ind in range(len(sentence_index)):
    sentence_vector[ind, sentence_index[ind]] = 1
