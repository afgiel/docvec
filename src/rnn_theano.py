import os
import numpy as np
from theano import tensor as T
import cPickle as cpkl

from rnn_lang import get_data
from rnn_slu import RNNSLU
from glove_wrapper import GloveWrapper

data_root = '../data/wordinds/'
print 'starting L matrix construction'
gw = GloveWrapper(verbose=True)

train_files = os.listdir(data_root+ '/train/')

print 'getting train and test data'
train_x, train_y, train_D, train_class = get_data(gw, train_files, 'train/')

print 'initializing'
drnn = RNNSLU(train_D[-1]+1, gw)
print 'intialized'

num_epochs = 25
learn_rate = 0.01

for e in range(num_epochs):
    for i in range(len(train_x)):
        x = np.array(train_x[i]).astype('int32')
        y = np.array(train_y[i]).astype('int32')
        drnn.sentence_train(x, y, learn_rate, train_D[i])
        drnn.normalize()
        if i % 10 == 0:
            print 'epoch: ', e, ' sentence: ', i



print 'saving model'
with open('../data/drnnlm_theano_model.pkl', 'w') as model_file:
    cpkl.dump(drnn, model_file, protocol=cpkl.HIGHEST_PROTOCOL)
print 'model saved'
