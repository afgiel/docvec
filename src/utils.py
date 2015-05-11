import numpy as np
import theano
import theano.tensor as T


def prepare_dataset(X_train, y_train,
       X_val, y_val,
       X_test, y_test):

    print '\tnumpying'
	X_train = np.array(X_train)
 	y_train = np.array(y_train)
 	X_valid = np.array(X_valid)
 	y_valid = np.array(y_valid)
 	X_test = np.array(X_test)
 	y_test = np.array(y_test)

	print '\tcreating theano shared vars'
	return dict(
		X_train=theano.shared(lasagne.utils.floatX(X_train)),
		y_train=T.cast(theano.shared(y_train), 'int32'),
		X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
		y_valid=T.cast(theano.shared(y_valid), 'int32'),
		X_test=theano.shared(lasagne.utils.floatX(X_test)),
		y_test=T.cast(theano.shared(y_test), 'int32'),
		num_examples_train=X_train.shape[0],
		num_examples_valid=X_valid.shape[0],
		num_examples_test=X_test.shape[0],
		input_dim=X_train.shape[1],
		output_dim=NUM_CLASSES,
		)

