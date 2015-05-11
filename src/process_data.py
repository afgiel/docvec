import os
import pickle

from get_vecs import TextPreprocessor 

from sklearn.datasets import fetch_20newsgroups

data_root = '../data/wordinds/'

dataset_train = fetch_20newsgroups(subset='all', remove=('headers', 'footers'))
print len(dataset_train.data)

tp = TextPreprocessor()

print "start getting inds"
for cat in dataset_train.target_names:
    print 'processing ' + cat + '...'
    
    # training data
    documents = []
    for doc in fetch_20newsgroups(subset='train', remove=('headers', 'footers'), categories=[cat]).data:
        documents.append(tp.doc_to_inds(doc))
    cat_f = open(data_root + '/train/' + cat + '.pkl', 'wb')
    pickle.dump(documents, cat_f)
    cat_f.close()

    # test data
    documents = []
    for doc in fetch_20newsgroups(subset='test', remove=('headers', 'footers'), categories=[cat]).data:
        documents.append(tp.doc_to_inds(doc))
    cat_f = open(data_root + '/test/' + cat + '.pkl', 'wb')
    pickle.dump(documents, cat_f)
    cat_f.close()

print 'done!'
