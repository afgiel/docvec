from get_vecs import TextPreprocessor 
import glove_wrapper

from sklearn.datasets import fetch_20newsgroups


dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers'))
tp = TextPreprocessor()

print "start getting inds"

inds = tp.doc_to_inds(dataset.data[0])

print dataset.data[0]
print inds
