#-*- coding: utf-8 -*-
import ipdb
import numpy as np
import code

from imagernn.utils import initw
'''
WLSTM : encoding된 vector (image:256, image bias: 1, word:256) -> input/forget/cell/output (hidden*4)
Wd : hidden(256) -> output(dictionary크기)


'''

class LSTMGenerator:
  """
  A multimodal long short-term memory (LSTM) generator
  """

  @staticmethod
  def init(input_size, hidden_size, output_size):

    model = {}
    # Recurrent weights: take x_t, h_{t-1}, and bias unit
    # and produce the 3 gates and the input to cell signal
    model['WLSTM'] = initw(input_size + hidden_size + 1, 4 * hidden_size)
    # Decoder weights (e.g. mapping to vocabulary)
    model['Wd'] = initw(hidden_size, output_size) # decoder
    model['bd'] = np.zeros((1, output_size))

    update = ['WLSTM', 'Wd', 'bd']
    regularize = ['WLSTM', 'Wd']
    return { 'model' : model, 'update' : update, 'regularize' : regularize }

  @staticmethod
  def forward(Xi, Xs, model, params, **kwargs):
    """
    Xi is 1-d array of size D (containing the image representation)
    Xs is N x D (N time steps, rows are data containng word representations), and
    it is assumed that the first row is already filled in as the start token. So a
    sentence with 10 words will be of size 11xD in Xs.
    """
    predict_mode = kwargs.get('predict_mode', False)

    # Google paper concatenates the image to the word vectors as the first word vector
    X = np.row_stack([Xi, Xs]) #이미지를 마치 첫 번째 단어처럼 취급


    # options
    # use the version of LSTM with tanh? Otherwise dont use tanh (Google style)
    # following http://arxiv.org/abs/1409.3215
    tanhC_version = params.get('tanhC_version', 0)
    drop_prob_encoder = params.get('drop_prob_encoder', 0.0)
    drop_prob_decoder = params.get('drop_prob_decoder', 0.0)

    if drop_prob_encoder > 0: # if we want dropout on the encoder
      # inverted version of dropout here. Suppose the drop_prob is 0.5, then during training
      # we are going to drop half of the units. In this inverted version we also boost the activations
      # of the remaining 50% by 2.0 (scale). The nice property of this is that during prediction time
      # we don't have to do any scailing, since all 100% of units will be active, but at their base
      # firing rate, giving 100% of the "energy". So the neurons later in the pipeline dont't change
      # their expected firing rate magnitudes
      '''
      predict때는 dropout ㄴㄴ
      Training 때는 dropout
      dropout 비율이 50%라 치면, 나머지 50%만 학습시키는 꼴이니까 얘들은 영향력 2배로 늘림
      '''
      if not predict_mode: # and we are in training mode
        scale = 1.0 / (1.0 - drop_prob_encoder)
        U = (np.random.rand(*(X.shape)) < (1 - drop_prob_encoder)) * scale # generate scaled mask
        X *= U # drop!

    # follows http://arxiv.org/pdf/1409.2329.pdf
    WLSTM = model['WLSTM']
    n = X.shape[0]
    d = model['Wd'].shape[0] # size of hidden layer
    Hin = np.zeros((n, WLSTM.shape[0])) # xt, ht-1, bias
    Hout = np.zeros((n, d))
    IFOG = np.zeros((n, d * 4)) # input/forget/output/g 함께 취급
    IFOGf = np.zeros((n, d * 4)) # after nonlinearity
    C = np.zeros((n, d))
    for t in xrange(n): # 단어 하나하나에 대해 iteration
      # set input
      prev = np.zeros(d) if t == 0 else Hout[t-1]
      Hin[t,0] = 1
      Hin[t,1:1+d] = X[t]
      Hin[t,1+d:] = prev

      # compute all gate activations. dots:
      IFOG[t] = Hin[t].dot(WLSTM)

      # non-linearities
      IFOGf[t,:3*d] = 1.0/(1.0+np.exp(-IFOG[t,:3*d])) # sigmoids; these are the gates
      IFOGf[t,3*d:] = np.tanh(IFOG[t, 3*d:]) # tanh

      # compute the cell activation
      C[t] = IFOGf[t,:d] * IFOGf[t, 3*d:]
      if t > 0: C[t] += IFOGf[t,d:2*d] * C[t-1]
      if tanhC_version:
        Hout[t] = IFOGf[t,2*d:3*d] * np.tanh(C[t])
      else:
        Hout[t] = IFOGf[t,2*d:3*d] * C[t]

    if drop_prob_decoder > 0: # if we want dropout on the decoder
      if not predict_mode: # and we are in training mode
        scale2 = 1.0 / (1.0 - drop_prob_decoder)
        U2 = (np.random.rand(*(Hout.shape)) < (1 - drop_prob_decoder)) * scale2 # generate scaled mask
        Hout *= U2 # drop!

    # decoder at the end
    Wd = model['Wd'] # hidden -> Vocabulary
    bd = model['bd']
    # NOTE1: we are leaving out the first prediction, which was made for the image
    # and is meaningless.
    Y = Hout[1:, :].dot(Wd) + bd
    ''' LSTM 결과 ( 문장길이 x N ) 를 다시 decoder로 입력 '''

    cache = {}
    if not predict_mode:
      # we can expect to do a backward pass
      cache['WLSTM'] = WLSTM
      cache['Hout'] = Hout
      cache['Wd'] = Wd
      cache['IFOGf'] = IFOGf
      cache['IFOG'] = IFOG
      cache['C'] = C
      cache['X'] = X
      cache['Hin'] = Hin
      cache['tanhC_version'] = tanhC_version
      cache['drop_prob_encoder'] = drop_prob_encoder
      cache['drop_prob_decoder'] = drop_prob_decoder
      if drop_prob_encoder > 0: cache['U'] = U # keep the dropout masks around for backprop
      if drop_prob_decoder > 0: cache['U2'] = U2

    return Y, cache

  @staticmethod
  def backward(dY, cache):

    Wd = cache['Wd']
    Hout = cache['Hout']
    IFOG = cache['IFOG']
    IFOGf = cache['IFOGf']
    C = cache['C']
    Hin = cache['Hin']
    WLSTM = cache['WLSTM']
    X = cache['X']
    tanhC_version = cache['tanhC_version']
    drop_prob_encoder = cache['drop_prob_encoder']
    drop_prob_decoder = cache['drop_prob_decoder']
    n,d = Hout.shape

    # we have to add back a row of zeros, since in the forward pass
    # this information was not used. See NOTE1 above.
    '''
    dY: delta_L
    dHout: delta_L-1

    dWd = a * delta_L  (a: Hout)
    delta_L-1 = Wd * delta_L

    delta_L-1 기준으로 IFOG에 대한 gradient 차례로 구함
    '''
    dY = np.row_stack([np.zeros(dY.shape[1]), dY])

    # backprop the decoder
    dWd = Hout.transpose().dot(dY)
    dbd = np.sum(dY, axis=0, keepdims = True)
    dHout = dY.dot(Wd.transpose())

    # backprop dropout, if it was applied
    if drop_prob_decoder > 0:
      dHout *= cache['U2']

    # backprop the LSTM
    dIFOG = np.zeros(IFOG.shape) # 문장길이 x hidden_size
    dIFOGf = np.zeros(IFOGf.shape)
    dWLSTM = np.zeros(WLSTM.shape) # (image+hidden+bias) x (hidden*4)
    dHin = np.zeros(Hin.shape)
    dC = np.zeros(C.shape)
    dX = np.zeros(X.shape)
    for t in reversed(xrange(n)):
    # 문장 맨 뒤에서부터 시작

      '''
      h(t) = o(t) * C(t)
      dLoss/do = dLoss/dh * C(t) = dHout * C - (1)
      dLoss/dC = dLoss/dh * o(t) = dHout * o - (2)
      '''
      if tanhC_version:
        tanhCt = np.tanh(C[t]) # recompute this here
        dIFOGf[t,2*d:3*d] = tanhCt * dHout[t]
        # backprop tanh non-linearity first then continue backprop
        dC[t] += (1-tanhCt**2) * (IFOGf[t,2*d:3*d] * dHout[t])
      else:
        dIFOGf[t,2*d:3*d] = C[t] * dHout[t] # (1)
        dC[t] += IFOGf[t,2*d:3*d] * dHout[t]# (2)

      if t > 0:
        '''
        C(t) = f*C(t-1) + i*g
        dLoss/df = dLoss/dC(t) * C(t-1) - (3)
        dLoss/dC(t-1) = dLoss/dC(t) * f - (4)
        dLoss/di = dLoss/dC(t) * g - (5)
        dLoss/dg = dLoss/dC(t) * i - (6)

        i,f,o,g는 모두 이미 activation거친 상태.
        즉 (1)~(6)은 모두 dLoss/da 의 형태임
        dLoss/dz = sigmoid/tanh'(z) * dLoss/da

        참고로 tanh' = 1 - tanh^2
        '''
        dIFOGf[t,d:2*d] = C[t-1] * dC[t] # (3)
        dC[t-1] += IFOGf[t,d:2*d] * dC[t]# (4)

      dIFOGf[t,:d] = IFOGf[t, 3*d:] * dC[t] #(5)
      dIFOGf[t, 3*d:] = IFOGf[t,:d] * dC[t] #(6)

      # backprop activation functions
      dIFOG[t,3*d:] = (1 - IFOGf[t, 3*d:] ** 2) * dIFOGf[t,3*d:]
      y = IFOGf[t,:3*d]
      dIFOG[t,:3*d] = (y*(1.0-y)) * dIFOGf[t,:3*d]

      # backprop matrix multiply
      '''
      Hin[t] : [1 X(t) h(t-1) ]
      IFOG[t] = np.dot(Hin[t], WLSTM)

      dIFOG: delta
      dLoss/dWLSTM = Hin * delta

      dLoss/dHin = dLoss/dIFOG * WLSTM.T
      '''
      dWLSTM += np.outer(Hin[t], dIFOG[t])
      dHin[t] = dIFOG[t].dot(WLSTM.transpose())

      # backprop the identity transforms into Hin
      '''
      dHin은 bias, input, hidden의 조합 (1+256+256)
      '''
      dX[t] = dHin[t,1:1+d] # 4096 -> d 로 가는 image encoder update
      if t > 0:
        dHout[t-1] += dHin[t,1+d:] # dLoss/dHout(t-1) = dLoss/dHin(t)

    if drop_prob_encoder > 0: # backprop encoder dropout
      dX *= cache['U']

    return { 'WLSTM': dWLSTM, 'Wd': dWd, 'bd': dbd, 'dXi': dX[0,:], 'dXs': dX[1:,:] }

  @staticmethod
  def predict(Xi, model, Ws, params, **kwargs):
    """
    Run in prediction mode with beam search. The input is the vector Xi, which
    should be a 1-D array that contains the encoded image vector. We go from there.
    Ws should be NxD array where N is size of vocabulary + 1. So there should be exactly
    as many rows in Ws as there are outputs in the decoder Y. We are passing in Ws like
    this because we may not want it to be exactly model['Ws']. For example it could be
    fixed word vectors from somewhere else.
    """
    tanhC_version = params['tanhC_version']
    beam_size = kwargs.get('beam_size', 1)

    WLSTM = model['WLSTM']
    d = model['Wd'].shape[0] # size of hidden layer
    Wd = model['Wd']
    bd = model['bd']

    # lets define a helper function that does a single LSTM tick
    def LSTMtick(x, h_prev, c_prev):
      t = 0

      # setup the input vector
      Hin = np.zeros((1,WLSTM.shape[0])) # xt, ht-1, bias
      Hin[t,0] = 1
      Hin[t,1:1+d] = x
      Hin[t,1+d:] = h_prev

      # LSTM tick forward
      IFOG = np.zeros((1, d * 4))
      IFOGf = np.zeros((1, d * 4))
      C = np.zeros((1, d))
      Hout = np.zeros((1, d))
      IFOG[t] = Hin[t].dot(WLSTM)
      IFOGf[t,:3*d] = 1.0/(1.0+np.exp(-IFOG[t,:3*d]))
      IFOGf[t,3*d:] = np.tanh(IFOG[t, 3*d:])
      C[t] = IFOGf[t,:d] * IFOGf[t, 3*d:] + IFOGf[t,d:2*d] * c_prev
      if tanhC_version:
        Hout[t] = IFOGf[t,2*d:3*d] * np.tanh(C[t])
      else:
        Hout[t] = IFOGf[t,2*d:3*d] * C[t]
      Y = Hout.dot(Wd) + bd
      return (Y, Hout, C) # return output, new hidden, new cell

    # forward prop the image
    (y0, h, c) = LSTMtick(Xi, np.zeros(d), np.zeros(d))

    # perform BEAM search. NOTE: I am not very confident in this implementation since I don't have
    # a lot of experience with these models. This implements my current understanding but I'm not
    # sure how to handle beams that predict END tokens. TODO: research this more.
    if beam_size > 1:
      # log probability, indices of words predicted in this beam so far, and the hidden and cell states
      beams = [(0.0, [], h, c)]
      nsteps = 0
      while True:
        beam_candidates = []
        for b in beams:
          ixprev = b[1][-1] if b[1] else 0 # start off with the word where this beam left off
          if ixprev == 0 and b[1]:
            # this beam predicted end token. Keep in the candidates but don't expand it out any more
            beam_candidates.append(b)
            continue
          (y1, h1, c1) = LSTMtick(Ws[ixprev], b[2], b[3])
          y1 = y1.ravel() # make into 1D vector
          maxy1 = np.amax(y1)
          e1 = np.exp(y1 - maxy1) # for numerical stability shift into good numerical range
          p1 = e1 / np.sum(e1)
          y1 = np.log(1e-20 + p1) # and back to log domain
          top_indices = np.argsort(-y1)  # we do -y because we want decreasing order
          for i in xrange(beam_size):
            wordix = top_indices[i]
            beam_candidates.append((b[0] + y1[wordix], b[1] + [wordix], h1, c1))
        beam_candidates.sort(reverse = True) # decreasing order
        beams = beam_candidates[:beam_size] # truncate to get new beams
        nsteps += 1
        if nsteps >= 20: # bad things are probably happening, break out
          break
      # strip the intermediates
      predictions = [(b[0], b[1]) for b in beams]
    else:
      # greedy inference. lets write it up independently, should be bit faster and simpler
      ixprev = 0
      nsteps = 0
      predix = []
      predlogprob = 0.0
      while True:
        (y1, h, c) = LSTMtick(Ws[ixprev], h, c)
        ixprev, ixlogprob = ymax(y1)
        predix.append(ixprev)
        predlogprob += ixlogprob
        nsteps += 1
        if ixprev == 0 or nsteps >= 20:
          break
      predictions = [(predlogprob, predix)]

    return predictions

def ymax(y):
  """ simple helper function here that takes unnormalized logprobs """
  y1 = y.ravel() # make sure 1d
  maxy1 = np.amax(y1)
  e1 = np.exp(y1 - maxy1) # for numerical stability shift into good numerical range
  p1 = e1 / np.sum(e1)
  y1 = np.log(1e-20 + p1) # guard against zero probabilities just in case
  ix = np.argmax(y1)
  return (ix, y1[ix])
