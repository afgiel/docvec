import string
import nltk

import numpy as np

from glove_wrapper import GloveWrapper

class TextPreprocessor():
    def __init__(self):
        self.glove_vecs = GloveWrapper(verbose=True)

    def doc_to_inds(self, doc):

        doc = doc.lower()
        doc = doc.replace('\n', ' ')

        inds = []

        for sent in nltk.tokenize.sent_tokenize(doc):
            words = nltk.word_tokenize(sent)
            sent_of_inds = []
            sent_of_inds.append(self.glove_vecs.get_index('SSTART'))
            for word in words:
                sent_of_inds.append(self.glove_vecs.get_index(word))
            sent_of_inds.append(self.glove_vecs.get_index('EEND'))
            inds.append(sent_of_inds)

        return inds

