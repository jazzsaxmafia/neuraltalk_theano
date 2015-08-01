import numpy as np
import theano
import theano.tensor as T
from imagernn.data_provider import *
import cPickle
import ipdb
from util import *
from config import *

def load_params(npy_path='./cv/checkpoint_62000.npz'):

    params = np.load(npy_path)

    We = to_shared(params['We'])
    be = to_shared(params['be'])
    Ws = to_shared(params['Ws'])
    WLSTM = to_shared(params['WLSTM'])
    Wd = to_shared(params['Wd'])
    bd = to_shared(params['bd'])

    shared_params = [We, be, Ws, WLSTM, Wd, bd]

    return shared_params

def build_test_function(checkpoint_file, database, ixtoword_path, hidden_size):


    [We, be, Ws, WLSTM, Wd, bd] = load_params(checkpoint_file)


    image = T.vector()
    embedded_image = T.dot(image, We) + be

    def _step(b, x_t, h_t_1, c_, weight):
        Hin = T.concatenate([b, x_t, h_t_1])
        IFOG = T.dot(Hin, weight)

        ifo = T.nnet.sigmoid(IFOG[:3*hidden_size])
        g = T.tanh(IFOG[3*hidden_size:])

        IFOGf = T.concatenate([ifo, g])

        c = IFOGf[:hidden_size] * IFOG[3*hidden_size:] + c_ * IFOGf[hidden_size:2*hidden_size]
        Hout = IFOGf[2*hidden_size:3*hidden_size] * c

        Y = T.dot(Hout, Wd) + bd
        chosen_word = T.argmax(Y)

        x_next = Ws[chosen_word]

        return x_next, Hout, c, chosen_word

    bias = T.alloc(numpy_floatX(1.), 1)

    (
        X,
        Houts,
        Cs,
        chosen_words
    ), updates = theano.scan(fn=lambda x_t, h_t_1, c_, b, weight : _step(b, x_t, h_t_1, c_, weight),
                    outputs_info=
                        [
                            embedded_image, # X[t]
                            T.alloc(numpy_floatX(0.), hidden_size), # H[t-1]
                            T.alloc(numpy_floatX(0.), hidden_size), # c[t-1]
                            None # next word
                        ],
                    non_sequences=[bias, WLSTM],
                    n_steps=10)

    test_function = theano.function(inputs=[image], outputs=chosen_words, allow_input_downcast=True)
    return test_function

def test_main():

    dataset = 'flickr30k'
    checkpoint_file = os.path.join(checkpoint_path,'checkpoint_62000.npz')
    ixtoword_path = os.path.join(checkpoint_path, 'ixtoword')
    hidden_size = 256

    with open(ixtoword_path) as f: ixtoword = cPickle.load(f)
    test_function = build_test_function(checkpoint_file, dataset, ixtoword_path, hidden_size)

    dp = getDataProvider(dataset)
    sampled_data = dp.sampleImageSentencePair('test')['image']
    sampled_image = sampled_data['feat']
    result = test_function(sampled_image)
    result_sentence = ' '.join(map(lambda x: ixtoword[x], result))

    print result_sentence

    ipdb.set_trace()

