import theano
import numpy
from theano import tensor as T
from glove_wrapper import GloveWrapper

from collections import OrderedDict

class RNNSLU(object):
    ''' elman neural net model '''
    def __init__(self, nd, gw):
        '''
        nh :: dimension of the hidden layer
        nc :: number of classes size of vocab for language model
        nd :: number of documents
        ne :: number of word embeddings in the vocabulary
        de :: dimension of the word embeddings

        '''
        # parameters of the model
        # need to change these embeddings to be the glove vectors
        ne = gw.L.shape[0]
        de = gw.L.shape[1]
        nh = de
        nc = ne

        self.emb = theano.shared(name='embeddings',
                                 value=gw.L.astype(theano.config.floatX))
        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (de, nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nc,
                               dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros((1, nh)).astype(theano.config.floatX))
        # not sure if this d is best, but need a way to pass it in the the recurrence function based of
        # the doc being viewed
        self.d = theano.shared(name='d',
                               value=numpy.zeros((nd,nh)).astype(theano.config.floatX))




        # matrix to mult doc vec by to create addition to softmax calc
        self.g = theano.shared(name='g',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
        # bundle
        self.params = [self.emb, self.wx, self.wh, self.w,
                       self.bh, self.b, self.h0, self.d, self.g]

        doc_num = T.iscalar('doc_num')
        idxs = T.ivector()
        x = self.emb[idxs]
        #print 'x ndim', x.ndim
        #x = theano.printing.Print('x')(x)
        doc_vec = self.d[doc_num]
        #doc_vec = theano.printing.Print('doc')(doc_vec)
        y_sentence = T.ivector('y_sentence')  # labels

        def recurrence(x_t, h_tm1, doc):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx)
                                 + T.dot(h_tm1, self.wh) + self.bh + doc)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b + T.dot(doc, self.g))
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                non_sequences=doc_vec,
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0])

        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        lr = T.scalar('lr')

        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))

        self.classify = theano.function(inputs=[idxs, doc_num], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr, doc_num],
                                              outputs=sentence_nll,
                                              updates=sentence_updates)

        self.normalize = theano.function(inputs=[],
                                         updates={self.emb:
                                                  self.emb /
                                                  T.sqrt((self.emb**2)
                                                  .sum(axis=1))
                                                  .dimshuffle(0, 'x')})

        def __get_state__(self):
            return self.d

       def __getitem__(self, index):
            return self.d[index]
