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
            #if punct in ['.', '?', '!']:
            #    doc = doc.replace(punct, " " + punct + " EEND SSTART ")
            #else: 
            doc = doc.replace(punct, " " + punct + " ")
        doc = doc.replace('  ', ' ').rstrip()

        tokens = doc.split(' ')
        
        inds = [0]* len(tokens)

        for i in range(len(tokens)):
            inds[i] = self.glove_vecs.get_index(tokens[i]) 

        start_ind = self.glove_vecs.get_index('SSTART')
        end_ind = self.glove_vecs.get_index('EEND')
        
        inds.insert(0, start_ind)
        #if inds[-1] == start_ind:
        #    inds.pop()
        #if inds[-1] != end_ind:
        inds.append(end_ind)
        return inds
            
