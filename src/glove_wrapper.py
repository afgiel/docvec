import numpy as np

PATH_TO_DATA = '../data/glove.6B.300d.txt'
NUM_TOKENS = 400000 + 3
NUM_DIM = 300

UNK = 'UUUNKKK'
START = '<START>'
END = '<END>'

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
                vec = np.array(line_split[1:])
                assert len(vec) == NUM_DIM
                self.mapping[word] = index
                self.L[index] = vec

        self.mapping[UNK] = NUM_TOKENS - 1
        self.L[self.mapping[UNK]] = np.random.uniform(-1., 1., (NUM_DIM,))

        self.mapping[START] = NUM_TOKENS - 2
        self.L[self.mapping[START]] = np.random.uniform(-1., 1., (NUM_DIM,))

        self.mapping[END] = NUM_TOKENS - 3
        self.L[self.mapping[END]] = np.random.uniform(-1., 1., (NUM_DIM,))

        self.word_mapping = dict(zip(self.mapping.values(),self.mapping.keys()))

    def get_index(self, word):
        if word in self.mapping:
            return self.mapping[word]
        else:
            return self.mapping[UNK]

   def get_word(self, index):
        if index in self.word_mapping:
            return self.word_mapping[index]
        else:
            return UNK

    def get_vec(self, index):
        return self.L[index]

    def set_vec(self, index, vec):
        assert len(vec) == NUM_DIM
        self.L[index] = vec
