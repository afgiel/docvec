import os
import pickle as pkl
import pdb
import itertools
import numpy as np

from glove_wrapper import GloveWrapper
from simple_drnnlm import SimpleDRNNLM

def get_data(gw, categories, subdir):
    data_root = '../data/wordinds/'
    xs = []
    ys = []
    ds = []
    cs = [] # class
    doc_ind = 0
    docs_per = 10
    for i in range(len(categories)):
        cat = categories[i]
        f = open(data_root + subdir + cat, 'rb')
        doc_list = pkl.load(f) # [doc index][timestep] = word index in glove matrix

        for doc in doc_list[:docs_per]:

            # make train and test
            for d in doc:
                xs.append(d[:-1])
                ys.append(d[1:])
                ds.append(doc_ind)
            cs.append(i)
            doc_ind += 1
        f.close()
    return xs, ys, ds, cs

# generate sequences
def seq_to_words(seq):
    return [gw.get_word(s) for s in seq]

def main():
    data_root = '../data/wordinds/'
    print 'starting L matrix construction'
    gw = GloveWrapper(verbose=True)
    L0 = gw.L

    train_files = os.listdir(data_root+ '/train/')

    print 'getting train and test data'
    train_x, train_y, train_D, _ = get_data(gw, train_files, 'train/') 
    pdb.set_trace()
    test_x, test_y, test_D, _ = get_data(gw, train_files, 'test/') 
    D0 = np.random.randn(train_D[-1] + 1, 300)
    print 'got train and test data'

    n_epochs = 25
    train_x = train_x*n_epochs
    train_y = train_y*n_epochs
    train_D = train_D*n_epochs

    model = SimpleDRNNLM(L0, D0, U0 = L0, alpha=0.05, rseed=10, bptt=3)
    print 'constructed model, training...'
    model.custom_train_sgd(train_x,train_y, train_D, apply_to=['H','U','L','D'], printevery=5, costevery=25)
    print 'training done'

    print 'saving model'
    with open('../data/simple_drnnlm_model.pkl', 'w') as model_file:
        pkl.dump(model, model_file) 
    print 'model saved'

    for i in range(10):
        seq, J = model.generate_sequence(1, gw.get_index("SSTART"), gw.get_index("EEND"), maxlen=100)
        print " ".join(seq_to_words(seq))

if __name__ == "__main__":
    main()
