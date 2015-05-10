import numpy as np

PATH_TO_DATA = '../data/glove.6B.300d.txt'
NUM_TOKENS = 400000 + 1
NUM_DIM = 300

UNK = 'UUUNKKK'

class GloveWrapper():

    def __init__(self, verbose=False):

        self.L = np.zeros((NUM_TOKENS, NUM_DIM))
        self.mapping = {}

        if verbose: print 'INSTANTIATING GLOVE MATRIX'

        with open(PATH_TO_DATA, 'r') as glove_file:
            for index, line in enumerate(glove_file):
                if verbose and index % 50000 == 0:
                    print '\tGLOVE INDEX: %d' % index
                line_split = line.split(' ')
                word = line_split[0]
                if word.startswith('U'): print word
                vec = np.array(line_split[1:])
                assert len(vec) == NUM_DIM
                self.mapping[word] = index
                self.L[index] = vec

        self.mapping[UNK] = NUM_TOKENS - 1
        self.L[self.mapping[UNK]] = np.random.uniform(-1., 1., (NUM_DIM,))

    def get(self, word):
        if word in self.mapping:
            return self.L[self.mapping[word]]
        else:
            return self.L[self.mapping[UNK]]
