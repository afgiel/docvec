import string

import numpy as np

from glove_wrapper import GloveWrapper

class TextPreprocessor():
    def __init__(self):
        self.glove_vecs = GloveWrapper(verbose=True)

    def doc_to_inds(self, doc):
        
        doc = doc.lower()
        doc = doc.replace('\n', ' ')

        for punct in string.punctuation:
            doc = doc.replace(punct, " " + punct + " ")
        doc = doc.replace('  ', ' ').rstrip()

        tokens = doc.split(' ')
        
        inds = [0]* len(tokens)

        for i in range(len(tokens)):
            inds[i] = self.glove_vecs.get_index(tokens[i]) 

        return inds
            
