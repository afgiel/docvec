# general imports
import itertools
import time
import os

# scientific imports
import numpy as np
import lasagne
import theano
import theano.tensor as T
from recurrent import LSTMLayer, ReshapeLayer
import utils

NUM_CLASSES = 20
NUM_EPOCHS = 5
BATCH_SIZE = 100
NUM_HIDDEN_UNITS = 256
LEARNING_RATE = 0.00001
MOMENTUM = 0.9
REG_STRENGTH = 0.00001

def run_lstm(X_train, y_train,
       X_val, y_val,
       X_test, y_test):

    print 'PREPARING DATASET'
    dataset = utils.prepare_dataset(X_train, y_train, X_val, y_val, X_test, y_test)
    print 'BUILDING MODEL'
    output_layer = build_model(
        input_dim = dataset['input_dim'],
        output_dim = dataset['output_dim'],
    )
    print 'CREATING ITER FUNCS'
    iter_funcs = create_iter_functions(dataset, output_layer)
    for epoch in train(iter_funcs, dataset):
        print("Epoch %d of %d" % (epoch['number'], num_epochs))
        print("\ttraining loss:\t\t%.6f" % epoch['train_loss'])
        print("\tvalidation loss:\t\t%.6f" % epoch['valid_loss'])
        print("\tvalidation accuracy:\t\t%.2f %%" %
                (epoch['valid_accuracy'] * 100))

        if epoch['number'] >= num_epochs:
          break

  return output_layer

def train(iter_funcs, dataset, batch_size=BATCH_SIZE):
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size
    num_batches_test = dataset['num_examples_test'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            print '\tbatch %d of %d' % (b, num_batches_train)
            tick = time.time()
            batch_train_loss = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)
            toc = time.time()
            print '\t\t loss: %f' % (batch_train_loss)
            print '\t\t took %f' % (toc - tick)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
            }


def create_iter_functions(dataset, output_layer,
                          X_tensor_type=T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE,
                          momentum=MOMENTUM,
                          reg_strength=REG_STRENGTH):
    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y')
    batch_slice = slice(
      batch_index * batch_size, (batch_index + 1) * batch_size)

    objective = lasagne.objectives.Objective(output_layer, loss_function=lasagne.objectives.multinomial_nll)

    reg = lasagne.regularization.l2(output_layer)
    loss_train = objective.get_loss(X_batch, target=y_batch) + REG_STRENGTH*reg
    loss_eval = objective.get_loss(X_batch, target=y_batch, deterministic=True)

    pred = T.argmax(output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch))

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.nesterov_momentum(
      loss_train, all_params, learning_rate, momentum)

    iter_train = theano.function(
      [batch_index], loss_train,
      updates=updates,
      givens={
        X_batch: dataset['X_train'][batch_slice],
        y_batch: dataset['y_train'][batch_slice],
        },
      )

    iter_valid = theano.function(
      [batch_index], [loss_eval, accuracy],
      givens={
        X_batch: dataset['X_valid'][batch_slice],
        y_batch: dataset['y_valid'][batch_slice],
        },
      )

    iter_test = theano.function(
      [batch_index], [loss_eval, accuracy],
      givens={
        X_batch: dataset['X_test'][batch_slice],
        y_batch: dataset['y_test'][batch_slice],
        },
      )

    return dict(
      train=iter_train,
      valid=iter_valid,
      test=iter_test,
      )



def build_model(input_dim, output_dim,
                batch_size=BATCH_SIZE, num_hidden_units=NUM_HIDDEN_UNITS):
    seq_length = 10 # WHAT DO I DO HERE
	l_in = lasagne.layers.InputLayer(
        shape=(batch_size, seq_length, input_dim),
    )
	l_rec = LSTMLayer(
        l_in,
        num_units=num_hidden_units
    )
    l_reshape = ReshapeLayer(
        l_rec,
        shape=(batch_size*seq_length, input_dim)
    )
    l_rec_out = lasagne.layers.DenseLayer(
        l_reshape,
        num_units=NUM_CLASSES,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    l_out = ReshapeLayer(
        l_rec_out,
        shape=(batch_size, seq_length, output_dim)
    )

	return l_out
