import os 
import pickle as pkl
import pdb
import itertools
import numpy as np

from glove_wrapper import GloveWrapper 
from simple_drnnlm import SimpleDRNNLM

def get_data(gw, categories, subdir):
    train = []
    correct = []
    d = []
    doc_ind = 0
    delims = [gw.get_index('.'), gw.get_index('?'), gw.get_index('!')]
    for i in range(len(categories)):
        cat = categories[i]
        f = open(data_root + subdir + cat, 'rb')
        doc_list = pkl.load(f) # [doc index][timestep] = word index in glove matrix
        for doc in doc_list:

            docs = [[]]
            #group into sentences
            for x, y in itertools.groupby(doc, lambda z: z in delims):
                docs[-1].extend(y)
                if x: docs.append([])
            # make train and test
            for d in docs:
                train.append(d[:-1])
                correct.append(d[1:])
                d.append(doc_ind)
            doc_ind += 1
        f.close()
    return train, correct, d

data_root = '../data/wordinds/'
print 'starting L matrix construction'
gw = GloveWrapper(verbose=True)
L0 = gw.L

train_files = os.listdir(data_root+ '/train/')

train_x, train_y, train_D = get_data(gw, train_files, 'train/') 
test_x, test_y, test_D= get_data(gw, train_files, 'test/') 
D0 = np.random.randn(train_D[-1] + 1, 300)
print 'got train and test data'
# test parameters; you probably want to change these
model = SimpleDRNNLM(L0, D0, U0 = L0, alpha=0.1, rseed=10, bptt=1)
print 'constructed model, training...'
model.custom_train_sgd(train_x,train_y, train_D, apply_to=['W','U','L','D'], printevery=1, costevery=1)
print 'training done'

# generate sequences
def seq_to_words(seq):
        return [gw.get_word(s) for s in seq]
        

for i in range(10):
    seq, J = model.generate_sequence(gw.get_index("SSTART"), gw.get_index("EEND"), maxlen=100)
    print " ".join(seq_to_words(seq))
