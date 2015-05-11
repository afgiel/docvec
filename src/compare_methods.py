import os

import numpy as np
import pickle as pkl

from glove_wrapper import GloveWrapper

from sklearn.svm import LinearSVC

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

data_root = '../data/wordinds/'


def get_average_wordvectors(gw, categories, subdir):
    x = []
    y = []

    for i in range(len(categories)):
        cat = categories[i]
        f = open(data_root + subdir + cat + '.pkl', 'rb')
        doc_list = pkl.load(f) # [doc index][timestep] = word index in glove matrix
        for doc in doc_list:
            avg_vec = np.array([gw.get_vec(w) for w in doc]).sum(axis=0) / len(doc) 
            x.append(avg_vec)
            y.append(i)
        f.close()
    return x, y
        


dataset_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers'))
dataset_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers'))
categories = dataset_train.target_names
print 'start glove wrapper'
gw = GloveWrapper(verbose=True)

# average word vectors
print 'getting avg word vecs'
X_avg_vecs, y_avg_vecs  = get_average_wordvectors(gw, categories, 'train/')
X_avg_vecs_test, y_avg_vecs_test = get_average_wordvectors(gw, categories, 'test/')
print X_avg_vecs[0].shape
# word counts
print 'getting word counts'
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(dataset_train.data)
X_test_counts = count_vect.transform(dataset_test.data)

# tfidf
print 'getting tfidf'
tfidf_vect = TfidfVectorizer()
X_train_tfidf = tfidf_vect.fit_transform(dataset_train.data)
X_test_tfidf = tfidf_vect.transform(dataset_test.data)


#SVM classifier
clf_svm = LinearSVC()
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)

print 'results using an SVM classfier\n'

print 'average word vector'
clf_svm.fit(X_avg_vecs, y_avg_vecs)
y_pred = clf_svm.predict(X_avg_vecs_test)
print(classification_report(y_avg_vecs_test, y_pred, target_names=categories))


print 'word counts'
clf_svm = LinearSVC()
clf_svm.fit(X_train_counts, dataset_train.target)
y_pred = clf_svm.predict(X_test_counts)
print(classification_report(dataset_test.target, y_pred, target_names=categories))


print 'tfidf'
clf_svm = LinearSVC()
clf_svm.fit(X_train_tfidf, dataset_train.target)
y_pred = clf_svm.predict(X_test_tfidf)
print(classification_report(dataset_test.target, y_pred, target_names=categories))

