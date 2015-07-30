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
num_epochs=100

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
image = T.matrix()
embedded_image = T.dot(image, We) + be
embedded_image = embedded_image.dimshuffle(0,'x',1)

'''
sentence

'''
sentence = T.matrix(dtype='int32')
mask = T.matrix()
embedded_sentence = Ws[sentence] # (batch, 문장길이, embedding_dim)

'''
이미지를 sentence의 맨 앞에 붙임
'''
X = T.concatenate([embedded_image, embedded_sentence], axis=1)
X = X.dimshuffle(1,0,2)

'''
LSTM weight ( i, f, c, o에 대한 weight들 )
을 정의하자
'''
def numpy_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

WLSTM = initializations.uniform((1+word_embedding_dim*2, 4*hidden_size))

bias = T.alloc(numpy_floatX(1.), batch_size, 1)

def _step(b, x_t, h_t_1, m_, c_, weight):
    Hin = T.concatenate([b, x_t, h_t_1], axis=1)
    IFOG = T.dot(Hin, weight)

    ifo = T.nnet.sigmoid(IFOG[:, :3*hidden_size])
    g = T.tanh(IFOG[:, 3*hidden_size:])

    IFOGf = T.concatenate([ifo, g], axis=1)

    c = IFOGf[:, :hidden_size] * IFOGf[:, 3*hidden_size:] + c_ * IFOGf[:,hidden_size:2*hidden_size]
    c = c * m_[:,None] + c_ * (1. - m_)[:,None]

    Hout = IFOGf[:, 2*hidden_size:3*hidden_size] * c
    Hout = Hout * m_[:,None] + h_t_1*(1. - m_)[:,None]
    return Hout, c

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
     grads = T.grad(cost=cost, wrt=params)
     updates = []
     for p, g in zip(params, grads):
         acc = theano.shared(p.get_value() * 0.)
         acc_new = rho * acc + (1 - rho) * g ** 2
         gradient_scaling = T.sqrt(acc_new + epsilon)
         g = g / gradient_scaling
         updates.append((acc, acc_new))
         updates.append((p, p - lr * g))
     return updates


(Houts, cells), updates = theano.scan(fn = lambda x, m, h, c, b, weight: _step(b,x,h, m, c, weight),
                   sequences=[X, mask.T],
                   outputs_info=
                    [
                        T.alloc(numpy_floatX(0.),batch_size, hidden_size),
                        T.alloc(numpy_floatX(0.),batch_size, hidden_size)
                    ],
                   non_sequences=[bias, WLSTM])

Houts = Houts.dimshuffle(1,0,2)
Y, updates = theano.scan(fn=lambda hout, wd,dd: T.nnet.softmax(T.dot(hout, wd)+dd),
                         sequences=[Houts],
                         non_sequences=[Wd,bd])

Y = Y[:,1:-1,:]
n_timestep=Y.shape[1]

losses,_ = theano.scan(fn=lambda y, m, sen: -T.log(y[T.arange(n_timestep), sen[1:]][mask != 0.0]),
                       sequences=[Y, mask, sentence])
loss = T.mean(losses)
params = [We, be, Ws, WLSTM, Wd, bd]
updates = RMSprop(cost=loss, params=params)
train = theano.function(inputs=[image, sentence, mask], outputs=losses, updates=updates, allow_input_downcast=True)

'''
sentence/tag를 추출하고, minibatch에 맞게 padding해줌
'''
def prepare_sentence(batches):
    sentence_index = map(lambda x: [misc['wordtoix'][w] for w in x['sentence']['tokens'] if w in misc['wordtoix']], batches)
    max_len = np.max(map(lambda x: len(x), sentence_index))

    sentence_vec = np.zeros((batch_size, max_len+1))
    mask = np.zeros((batch_size, max_len+2))

    for ind, sen in enumerate(sentence_index):
        sentence_vec[ind, 1:len(sen)+1] = np.array(sen).astype(int) # START symbol 감안해서 1
        mask[ind, :len(sen)+2] = 1 # START symbol 감안해서 1, 맨 앞에 이미지 붙이는거 감안해서 1

    return sentence_vec, mask

'''
Train
'''
iteration=0
for epoch in range(num_epochs):
    for batch in dp.iterImageSentencePairBatch(max_batch_size=batch_size):

        image_feats = np.array(map(lambda x: x['image']['feat'].astype(theano.config.floatX), batch))
        sentence_vec, ma = prepare_sentence(batch)

        cost = train(image_feats, sentence_vec, ma)

        if np.mod(iteration, 1000) == 0:
            print "Saving model... iteration: ", iteration
            np.savez('./cv/checkpoint_'+str(iteration), We=We.get_value(),
                                                        be=be.get_value(),
                                                        Ws=Ws.get_value(),
                                                        WLSTM=WLSTM.get_value(),
                                                        Wd=Wd.get_value(),
                                                        bd=bd.get_value())

        iteration += 1

houts_test = f_hout(image_feats, sentence_vec, ma)
