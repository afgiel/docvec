import os
import pdb
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import decomposition

from rnn_lang import get_data
from glove_wrapper import GloveWrapper
from nearest_neighbors import DocNN


data_root = '../data/wordinds/'
test_files = os.listdir(data_root+ '/train/') # list of all 20 categories
num_classes = 20
num_docs = 200
colors = iter(cm.rainbow(np.linspace(0,1,num_classes)))#"bgrcmykw"

#gw = GloveWrapper(verbose=True)

#test_x, test_y, test_D, labels = get_data(gw, test_files, 'test/')


model = None
print 'loading model'
with open('../data/drnnlm_theano_model.pkl', 'r') as model_file:
    model = pkl.load(model_file)
print 'model loaded'

#D0 = np.random.randn(train_D[-1] + 1, 300)
#D = model.generate_docvecs(test_x, test_y, test_D, D0)

D = model[:num_docs]#model.sparams.D[:num_docs]
labels = labels[:num_docs]
pca = decomposition.PCA(n_components=2)
D_plot = pca.fit(D).transform(D)

for cat in range(num_classes):
    cat_inds = [i for i, y in enumerate(labels) if y == cat]
    plt.scatter(D_plot[cat_inds, 0], D_plot[cat_inds, 1], c=next(colors))

plt.title('PCA representation of test set')
#plt.show()
plt.savefig('Theano_PCA.png')
# plot here

knn = DocNN(D, k=3)
for label in range(len(labels)):
    vecs, indices = knn.doc_nearest(label)
    print 'label: %d' % labels[label]
    print [labels[ind] for ind in indices]

