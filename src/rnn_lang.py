import os 
import pickle as pkl
import pdb
import itertools

from glove_wrapper import GloveWrapper 
from rnnlm import RNNLM

def get_data(gw, categories, subdir):
    train = []
    correct = []
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
        f.close()
    return train, correct

data_root = '../data/wordinds/'
print 'starting L matrix construction'
gw = GloveWrapper(verbose=True)
L0 = gw.L

train_files = os.listdir(data_root+ '/train/')

train_x, train_y = get_data(gw, train_files, 'train/') 
test_x, test_y = get_data(gw, train_files, 'test/') 
print 'got train and test data'
# test parameters; you probably want to change these
model = RNNLM(L0, U0 = L0, alpha=0.1, rseed=10, bptt=1)
print 'constructed model, training...'
model.train_sgd(train_x,train_y, printevery=1, costevery=1)
print 'training done'

# generate sequences
def seq_to_words(seq):
        return [gw.get_word(s) for s in seq]
        

for i in range(10):
    seq, J = model.generate_sequence(gw.get_index("<s>"), gw.get_index("</s>"), maxlen=100)
    print " ".join(seq_to_words(seq))
