#-*- coding: utf-8 -*-
from imagernn.data_provider import *
from driver import *
import ipdb
import cPickle

import numpy as np

import theano
import theano.tensor as T

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from util import *
from config import *

from keras import initializations

srng = RandomStreams()

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

def dropout(X):
    if dropout_prob > 0.0:
        X *= srng.binomial(p=dropout_prob, dtype=theano.config.floatX, size=X.shape)
        X /= dropout_prob

    return X

def build_model(dp, word_count_threshold, word_embedding_dim, image_embedding_dim, hidden_size, batch_size, num_vocab):

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
    X = dropout(X)

    '''
    LSTM weight ( i, f, c, o에 대한 weight들 )
    을 정의하자
    '''

    WLSTM = initializations.uniform((1+word_embedding_dim*2, 4*hidden_size))
    bias = T.alloc(numpy_floatX(1.), batch_size, 1)

    def _step(b, x_t, h_t_1, m_, c_, weight):

        Hin = T.concatenate([b, x_t, h_t_1], axis=1) # 1, x[t], h[t-1]을 concat

        IFOG = T.dot(Hin, weight)

        ifo = T.nnet.sigmoid(IFOG[:, :3*hidden_size])
        g = T.tanh(IFOG[:, 3*hidden_size:])

        IFOGf = T.concatenate([ifo, g], axis=1)

        c = IFOGf[:, :hidden_size] * IFOGf[:, 3*hidden_size:] + c_ * IFOGf[:,hidden_size:2*hidden_size]
        c = c * m_[:,None] + c_ * (1. - m_)[:,None]

        Hout = IFOGf[:, 2*hidden_size:3*hidden_size] * c
        Hout = Hout * m_[:,None] + h_t_1*(1. - m_)[:,None]
        return Hout, c



    (Houts, cells), updates = theano.scan(fn = lambda x, m, h, c, b, weight: _step(b,x,h, m, c, weight),
                       sequences=[X, mask.T],
                       outputs_info=
                        [
                            T.alloc(numpy_floatX(0.),batch_size, hidden_size),
                            T.alloc(numpy_floatX(0.),batch_size, hidden_size)
                        ],
                       non_sequences=[bias, WLSTM])

    Houts = Houts.dimshuffle(1,0,2)
    Y, updates = theano.scan(fn=lambda hout, wd,dd: T.dot(hout, wd) + dd, #T.nnet.softmax(T.dot(hout, wd)+dd),
                             sequences=[Houts],
                             non_sequences=[Wd,bd])


    Y = Y[:,1:,:]
    n_timestep=Y.shape[1]

    losses,_ = theano.scan(fn=lambda y, m, sen: -T.log(1e-20 + y[T.arange(n_timestep), sen[1:]][mask != 0.0]),
                           sequences=[Y, mask, sentence])

    loss = T.sum(losses) / Y.shape[0]
    loss += regularization_ratio * 0.5 * T.sum(WLSTM * WLSTM)
    loss += regularization_ratio * 0.5 * T.sum(Wd * Wd)

    params = [We, be, Ws, WLSTM, Wd, bd]
    updates = RMSprop(cost=loss, params=params)
    train_function = theano.function(inputs=[image, sentence, mask], outputs=loss, updates=updates, allow_input_downcast=True)
    Y_function = theano.function(inputs=[image, sentence, mask], outputs=Y, updates=updates, allow_input_downcast=True)
    Hout_function = theano.function(inputs=[image, sentence, mask], outputs=Houts, updates=updates, allow_input_downcast=True)


    return train_function, params, Y_function, Hout_function

def train():

    batch_size = 100
    num_epochs=100

    dp = getDataProvider(dataset)

    misc = {}

    (
        misc['wordtoix'],
        misc['ixtoword'],
        bias_init_vector
    ) = preProBuildWordVocab(dp.iterSentences('train'), word_count_threshold)

    wordtoix = misc['wordtoix']
    ixtoword = misc['ixtoword']

    with open(os.path.join(checkpoint_path,'wordtoix'), 'w') as f:
        cPickle.dump(wordtoix, f)

    with open(os.path.join(checkpoint_path,'ixtoword'), 'w') as f:
        cPickle.dump(ixtoword, f)

    num_vocab = len(misc['wordtoix'])

    train_function, params, Y_function, Hout_function = build_model(
            dp, word_count_threshold, word_embedding_dim, image_embedding_dim, hidden_size, batch_size, num_vocab)

    [We, be, Ws, WLSTM, Wd, bd] = params

    '''
    sentence/tag를 추출하고, minibatch에 맞게 padding해줌
    '''
    def prepare_sentence(batches):
        sentence_index = map(lambda x: [misc['wordtoix'][w] for w in x['sentence']['tokens'] if w in misc['wordtoix']], batches)
        max_len = np.max(map(lambda x: len(x), sentence_index))

        sentence_vec = np.zeros((batch_size, max_len+2))
        mask = np.zeros((batch_size, max_len+2))

        for ind, sen in enumerate(sentence_index):
            sentence_vec[ind, 1:len(sen)+1] = np.array(sen).astype(int) # START symbol 감안해서 1
            sentence_vec[ind, len(sen)+1] = 0
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

            cost = train_function(image_feats, sentence_vec, ma)
            Y = Y_function(image_feats, sentence_vec, ma)
            Houts = Hout_function(image_feats, sentence_vec, ma)

            print cost
            print Houts
            ipdb.set_trace()

            if np.mod(iteration, 1000) == 0:
                print "Saving model... iteration: ", iteration
                np.savez(os.path.join(checkpoint_path, 'checkpoint_'+str(iteration)), We=We.get_value(),
                                                            be=be.get_value(),
                                                            Ws=Ws.get_value(),
                                                            WLSTM=WLSTM.get_value(),
                                                            Wd=Wd.get_value(),
                                                            bd=bd.get_value())

            iteration += 1

