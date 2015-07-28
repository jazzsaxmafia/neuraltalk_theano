import sys
from imagernn.data_provider import *
from driver import *
import argparse
import ipdb

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.regularizers import l2

dataset = 'flickr30k'
word_count_threshold = 5
word_embedding_dim = 256
image_embedding_dim= 256
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

model = Sequential()

model.add(Embedding(input_dim=num_vocab,
                    output_dim=word_embedding_dim,
                    init='uniform',
                    W_regularizer=l2(0.01)))

model.add(
        LSTM(
            input_dim=1+word_embedding_dim+image_embedding_dim,
            output_dim=word_embedding_dim,
            activation='sigmoid',
            return_sequences=True)
        )

model.add(Dropout(0.5))
model.add(Dense(word_embedding_dim, num_vocab))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

batch = [dp.sampleImageSentencePair() for i in xrange(batch_size)]
wordtoix = misc['wordtoix']

for bat in batch:
    tokens = bat['sentence']['tokens']
    img = bat['image']
    ix = [0] + [ wordtoix[w] for w in tokens if w in wordtoix]

    ipdb.set_trace()

