from numpy import *
import itertools
import time
import sys

# Import NN utils
from nn.base import NNBase
from nn.math import softmax, sigmoid, make_onehot
from nn.math import MultinomialSampler, multinomial_sample
from misc import random_weight_matrix


class SimpleDRNNLM(NNBase):

    def __init__(self, L0, D0, U0=None,
                 alpha=0.005, rseed=10, bptt=1):

        self.hdim = L0.shape[1] # word vector dimensions
        self.vdim = L0.shape[0] # vocab size
        self.ddim = D0.shape[0] # doc size
        param_dims = dict(H = (self.hdim, self.hdim),
                          U = L0.shape)
        # note that only L gets sparse updates
        param_dims_sparse = dict(L = L0.shape, D = D0.shape)
        NNBase.__init__(self, param_dims, param_dims_sparse)

        #### YOUR CODE HERE ####

        self.bptt = bptt
        self.alpha = alpha

        # Initialize word vectors
        self.sparams.L = L0.copy()
        self.sparams.D = D0.copy()
        self.params.U = random.randn(self.vdim, self.hdim)*0.1

        # Initialize H matrix, as with W and U in part 1
        self.params.H = random_weight_matrix(self.hdim, self.hdim)

        #### END YOUR CODE ####


    def _acc_grads(self, xs, ys, d):

        # Expect xs as list of indices
        ns = len(xs)

        # make matrix here of corresponding h(t)
        # hs[-1] = initial hidden state (zeros)
        hs = zeros((ns+1, self.hdim))

        # predicted probas
        ps = zeros((ns, self.vdim))

        zs = zeros((ns+1, self.hdim))

        ##
        # Forward propagation
        for t in xrange(ns):
            x_t = xs[t]
            zs[t] = self.params.H.dot(hs[t-1]) + self.sparams.L[x_t] + self.sparams.D[d]
            hs[t] = sigmoid(zs[t])
            ps[t] = softmax(self.params.U.dot(hs[t]))

        ##
        # Backward propagation through time

        d_grad = zeros_like(self.sparams.D[0])
        for t in reversed(xrange(ns)):
            delta = zeros((ns, self.hdim))
            p_t = ps[t]
            eps_t = p_t - make_onehot(ys[t], len(p_t))
            self.grads.U += outer(eps_t, hs[t])
            sig_prime_t = sigmoid(zs[t])*(1.-sigmoid(zs[t]))
            delta[t] = sig_prime_t * self.params.U.T.dot(eps_t)
            self.sgrads.L[xs[t]] = delta[t].copy()
            d_grad += delta[t].copy()
            self.grads.H += outer(delta[t], hs[t-1])
            for i in xrange(1, self.bptt):
                j = t-i
                if j < 0: continue
                sig_prime_j = sigmoid(zs[j])*(1.-sigmoid(zs[j]))
                delta[j] = sig_prime_j * self.params.H.T.dot(delta[j+1])
                self.sgrads.L[xs[j]] = delta[j].copy()
                d_grad += delta[j].copy()
                self.grads.H += outer(delta[j], hs[j-1])

        self.sgrads.D[d] = d_grad

    def grad_check(self, x, y, d, outfd=sys.stderr, **kwargs):
        """
        Wrapper for gradient check on RNNs;
        ensures that backprop-through-time is run to completion,
        computing the full gradient for the loss as summed over
        the input sequence and predictions.

        Do not modify this function!
        """
        bptt_old = self.bptt
        self.bptt = len(y)
        print >> outfd, "NOTE: temporarily setting self.bptt = len(y) = %d to compute true gradient." % self.bptt
        self.custom_grad_check(x, y, d, outfd=outfd, **kwargs)
        self.bptt = bptt_old
        print >> outfd, "Reset self.bptt = %d" % self.bptt


    def custom_grad_check(self, x, y, d, eps=1e-4, tol=1e-6,
                   outfd=sys.stderr, verbose=False,
                   skiplist=[]):

        # Accumulate gradients in self.grads
        self._reset_grad_acc()
        self._acc_grads(x, y, d)
        self.sgrads.coalesce() # combine sparse updates

        ##
        # Loop over dense parameters
        for name in self.params.names():
            if name in skiplist: continue
            theta = self.params[name]
            grad_computed = self.grads[name]
            grad_approx = zeros(theta.shape)
            for ij, v in ndenumerate(theta):
                tij = theta[ij]
                theta[ij] = tij + eps
                Jplus  = self.compute_loss(x, y, d)
                theta[ij] = tij - eps
                Jminus = self.compute_loss(x, y, d)
                theta[ij] = tij # reset
                grad_approx[ij] = (Jplus - Jminus)/(2*eps)
            # Compute Frobenius norm
            grad_delta = linalg.norm(grad_approx - grad_computed)
            print >> outfd, "grad_check: dJ/d%s error norm = %.04g" % (name, grad_delta),
            print >> outfd, ("[ok]" if grad_delta < tol else "**ERROR**")
            print >> outfd, "    %s dims: %s = %d elem" % (name, str(list(theta.shape)), prod(theta.shape))
            if verbose and (grad_delta > tol): # DEBUG
                print >> outfd, "Numerical: \n" + str(grad_approx)
                print >> outfd, "Computed:  \n" + str(grad_computed)

        ##
        # Loop over sparse parameters
        for name in self.sparams.names():
            if name in skiplist: continue
            theta_full = self.sparams[name]
            # Loop over all sparse updates for this parameter
            for idx, grad_computed in self.sgrads[name]:
                theta = theta_full[idx] # view of update block
                grad_approx = zeros(theta.shape)
                # Loop over all indices within update block
                for ij, v in ndenumerate(theta):
                    tij = theta[ij]
                    theta[ij] = tij + eps
                    Jplus  = self.compute_loss(x, y, d)
                    theta[ij] = tij - eps
                    Jminus = self.compute_loss(x, y, d)
                    theta[ij] = tij # reset
                    grad_approx[ij] = (Jplus - Jminus)/(2*eps)
                # Compute Frobenius norm
                grad_delta = linalg.norm(grad_approx - grad_computed)
                print >> outfd, "grad_check: dJ/d%s[%s] error norm = %.04g" % (name, idx, grad_delta),
                print >> outfd, ("[ok]" if grad_delta < tol else "**ERROR**")
                print >> outfd, "    %s[%s] dims: %s = %d elem" % (name, idx, str(list(theta.shape)), prod(theta.shape))
                if verbose and (grad_delta > tol): # DEBUG
                    print >> outfd, "Numerical: \n" + str(grad_approx)
                    print >> outfd, "Computed:  \n" + str(grad_computed)

        self._reset_grad_acc()



    def compute_seq_loss(self, xs, ys, d):
        """
        Compute the total cross-entropy loss
        for an input sequence xs and output
        sequence (labels) ys.

        You should run the RNN forward,
        compute cross-entropy loss at each timestep,
        and return the sum of the point losses.
        """

        J = 0
        ns = len(xs)
        hs = zeros((ns+1, self.hdim))
        for t in xrange(ns):
            x_t = xs[t]
            zs_t = self.params.H.dot(hs[t-1]) + self.sparams.L[x_t] + self.sparams.D[d]
            hs[t] = sigmoid(zs_t)
            ps_t = softmax(self.params.U.dot(hs[t]))
            J += -1*log(ps_t[ys[t]])
        return J


    def compute_loss(self, X, Y, D):
        """
        Compute total loss over a dataset.
        (wrapper for compute_seq_loss)

        Do not modify this function!
        """
        if not isinstance(X[0], ndarray): # single example
            return self.compute_seq_loss(X, Y, D)
        else: # multiple examples
            losses = []
            for i in range(len(X)):
                xs = X[i]
                ys = Y[i]
                d = D[i]
                losses.append(self.compute_seq_loss(xs,ys,d))
            return sum(losses)

    def compute_mean_loss(self, X, Y, D):
        """
        Normalize loss by total number of points.

        Do not modify this function!
        """
        J = self.compute_loss(X, Y, D)
        ntot = sum(map(len,Y))
        return J / float(ntot)

    def generate_docvecs(self, X, y, ds, D0):
        self.sparams.D = D0.copy() 
        self.custom_train_sgd(X, y, ds, apply_to=['D']) 
        return self.sparams.D

    def custom_train_sgd(self, X, y, ds, apply_to=[],
                  idxiter=None, alphaiter=None,
                  printevery=10000, costevery=10000,
                  devidx=None):
        if idxiter == None: # default training schedule
            idxiter = xrange(len(y))
        if alphaiter == None: # default training schedule
            alphaiter = itertools.repeat(self.alpha)

        costs = []
        counter = 0
        t0 = time.time()

        try:
            print "Begin SGD..."
            for idx, alpha in itertools.izip(idxiter, alphaiter):
                if counter % printevery == 0:
                    print "  Seen %d in %.02f s" % (counter, time.time() - t0)
                if False and counter % costevery == 0:
                    if devidx != None:
                        cost = self.compute_mean_loss(X[devidx], y[devidx], ds[devidx])
                    else: cost = self.compute_mean_loss(X, y, ds)
                    costs.append((counter, cost))
                    print "  [%d]: mean loss %g" % (counter, cost)
                #pdb.set_trace()
                if hasattr(idx, "__iter__") and len(idx) > 1: # if iterable
                    self.custom_train_minibatch_sgd(X[idx], y[idx], ds[idx], alpha, apply_to)
                elif hasattr(idx, "__iter__") and len(idx) == 1: # single point
                    idx = idx[0]
                    self.custom_train_point_sgd(X[idx], y[idx], ds[idx], alpha, apply_to)
                else:
                    self.custom_train_point_sgd(X[idx], y[idx], ds[idx], alpha, apply_to)

                counter += 1
        except KeyboardInterrupt as ke:
            """
            Allow manual early termination.
            """
            print "SGD Interrupted: saw %d examples in %.02f seconds." % (counter, time.time() - t0)
            return costs

        # Wrap-up
        if devidx != None:
            cost = self.compute_mean_loss(X[devidx], y[devidx], ds[devidx])
        else: cost = self.compute_mean_loss(X, y, ds)
        costs.append((counter, cost))
        print "  [%d]: mean loss %g" % (counter, cost)
        print "SGD complete: %d examples in %.02f seconds." % (counter, time.time() - t0)

        return costs


    def custom_train_point_sgd(self, x, y, d, alpha, apply_to):
        """Generic single-point SGD"""
        self._reset_grad_acc()
        self._acc_grads(x, y, d)
        self._custom_apply_grad_acc(apply_to, alpha)

    def custom_train_minibatch_sgd(self, X, y, ds, alpha, apply_to):
        """
        Generic minibatch SGD
        """
        self._reset_grad_acc()
        for i in range(len(y)):
            self._acc_grads(X[i], y[i], ds[i])
        self._custom_apply_grad_acc(apply_to, alpha)

    def _custom_apply_grad_acc(self, apply_to, alpha=1.0):
        """
        Update parameters with accumulated gradients.

        alpha can be a scalar (as in SGD), or a vector
        of the same length as the full concatenated
        parameter vector (as in e.g. AdaGrad)
        """
        for param in apply_to:
            if param in self.params._name_to_idx:
                param -= alpha * self.grads[param]
            elif param in self.sparams._name_to_idx:
                self.custom_apply_to(param, alpha=-1*alpha)
            else:
                print 'PARAM NAME TO BE UPDATED NOT FOUND'
# DELETE BELOW THIS LINE ------------
        self.sgrads.apply_to(self.sparams, alpha=-1*alpha)


    def custom_apply_to(self, param, alpha=-1.0):
        """Apply sparse updates to parameter store."""
        ud = self.sgrads[param]
        for idx, v in ud: # idx, vec pairs
            self.sparams[param][idx] += alpha*v # in-place update


    def generate_sequence(self, d, init, end, maxlen=100):
        """
        Generate a sequence from the language model,
        by running the RNN forward and selecting,
        at each timestep, a random word from the
        a word from the emitted probability distribution.

        The MultinomialSampler class (in nn.math) may be helpful
        here for sampling a word. Use as:

            y = multinomial_sample(p)

        to sample an index y from the vector of probabilities p.


        Arguments:
            init = index of start word (word_to_num['<s>'])
            end = index of end word (word_to_num['</s>'])
            maxlen = maximum length to generate

        Returns:
            ys = sequence of indices
            J = total cross-entropy loss of generated sequence
        """

        J = 0 # total loss
        ys = [init] # emitted sequence

        hs = zeros((maxlen+1, self.hdim))

        curr = init
        t = 0
        while curr != end and len(ys) < maxlen:
            x_t = curr
            zs_t = self.params.H.dot(hs[t-1]) + self.sparams.L[x_t] + self.sparams.D[d]
            hs[t] = sigmoid(zs_t)
            ps_t = softmax(self.params.U.dot(hs[t]))
            y = multinomial_sample(ps_t)
            ys.append(y)
            curr = y
            J += -1*log(ps_t[y])
            t += 1

        return ys, J

