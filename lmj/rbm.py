# Copyright (c) 2011 Leif Johnson <leif@leifjohnson.net>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''An implementation of several types of Restricted Boltzmann Machines.

This code is largely based on the Matlab generously provided by Taylor, Hinton
and Roweis, and described in their 2006 NIPS paper, "Modeling Human Motion Using
Binary Hidden Variables". Their code and results are available online at
http://www.cs.nyu.edu/~gwtaylor/publications/nips2006mhmublv/.

There are more RBM implementations in this module than just the Taylor
"conditional RBM," though. The basic (non-Temporal) RBM is based on the Taylor,
Hinton, and Roweis code, but stripped of the dynamic bias terms and refactored
into an object-oriented style. The convolutional RBM code is based on the 2009
ICML paper by Lee, Grosse, Ranganath and Ng, "Convolutional Deep Belief Networks
for Scalable Unsupervised Learning of Hierarchical Representations".

The mean-covariance RBM is based on XXX.

All implementations incorporate an option to train hidden unit biases using a
sparsity criterion, as described in the 2008 NIPS paper by Lee, Ekanadham and
Ng, "Sparse Deep Belief Net Model for Visual Area V2".

Most RBM implementations  provide an option to treat visible units as either
binary or gaussian. Training networks with gaussian visible units is a tricky
dance of parameter-twiddling, but binary units seem quite stable in their
learning and convergence properties.

Finally, although I have tried to ensure that the code is correct, there are
probably many bugs, all of which are my own doing. I wrote this code to get a
better intuitive understanding for the RBM family of machine learning
algorithms, but I do not claim that the code is useful for a particular purpose
or produces state-of-the-art results. Mostly I hope that this code is readable
so that others can use it to better understand how this whole RBM thing works.
'''

import numpy as np
import logging
import numpy.random as rng


def sigmoid(eta):
    '''Return the logistic sigmoid function of the argument.'''
    return 1. / (1. + np.exp(-eta))


def identity(eta):
    '''Return the identity function of the argument.'''
    return eta


def bernoulli(p):
    '''Return an array of boolean samples from Bernoulli(p).

    Parameters
    ----------
    p : ndarray
        This array should contain values in [0, 1].

    Returns
    -------
    An array of the same shape as p. Each value in the result will be a boolean
    indicating whether a single Bernoulli trial succeeded for the corresponding
    element of `p`.
    '''
    return rng.rand(*p.shape) < p


class RBM(object):
    '''A Restricted Boltzmann Machine (RBM) is a probabilistic model of data.

    RBMs have two layers of variables (here called "units," in keeping with
    neural network terminology) -- a "visible" layer that models data in the
    world, and a "hidden" layer that is imagined to generate the data in the
    visible layer. RBMs are inspired by the (unrestricted) Boltzmann Machine, a
    model from statistical physics in which each unit is connected to all other
    units, and the states of unobserved variables can be inferred by using a
    sampling procedure.

    The full connectivity of an unrestricted Boltzmann Machine makes inference
    difficult, requiring sampling or some other approximate technique. RBMs
    restrict this more general model by requiring that the visible and hidden
    units form a fully connected, undirected, bipartite graph. In this way, each
    of the visible units is independent from the other visible units when
    conditioned on the state of the hidden layer, and each of the hidden units
    is independent of the others when conditioned on the state of the visible
    layer. This conditional independence makes inference tractable for the units
    in a single RBM.

    To "encode" a signal by determining the state of the hidden units given some
    visible data ("signal"),

    1. the signal is presented to the visible units, and
    2. the states of the hidden units are sampled from the conditional
       distribution given the visible data.

    To "decode" an encoding in the hidden units,

    3. the states of the visible units are sampled from the conditional
       distribution given the states of the hidden units.

    Once a signal has been encoded and then decoded,

    4. the sampled visible units can be compared directly with the original
       visible data.

    Training takes place by presenting a number of data points to the network,
    encoding the data, reconstructing it from the hidden states, and encoding
    the reconstruction in the hidden units again. Then, using contrastive
    divergence (Hinton 2002; Hinton & Salakhutdinov 2006), the gradient is
    approximated using the correlations between visible and hidden units in the
    first encoding and the same correlations in the second encoding.
    '''

    def __init__(self, num_visible, num_hidden, binary=True, scale=0.001):
        '''Initialize a restricted boltzmann machine.

        Parameters
        ----------
        num_visible : int
            The number of visible units.

        num_hidden : int
            The number of hidden units.

        binary : bool
            True if the visible units are binary, False if the visible units are
            normally distributed.

        scale : float
            Sample initial weights from N(0, scale).
        '''
        self.weights = scale * rng.randn(num_hidden, num_visible)
        self.hid_bias = scale * rng.randn(num_hidden, 1)
        self.vis_bias = scale * rng.randn(num_visible, 1)

        self._visible = binary and sigmoid or identity

    @property
    def num_hidden(self):
        return len(self.hid_bias)

    @property
    def num_visible(self):
        return len(self.vis_bias)

    def hidden_expectation(self, visible, bias=0.):
        '''Given visible data, return the expected hidden unit values.'''
        return sigmoid(np.dot(self.weights, visible.T) + self.hid_bias + bias)

    def visible_expectation(self, hidden, bias=0.):
        '''Given hidden states, return the expected visible unit values.'''
        return self._visible(np.dot(hidden.T, self.weights) + self.vis_bias.T + bias)

    def iter_passes(self, visible):
        '''Repeatedly pass the given visible layer up and then back down.

        Parameters
        ----------
        visible : ndarray
            The initial state of the visible layer.

        Returns
        -------
        Generates a sequence of (visible, hidden) states. The first pair will be
        the (original visible, resulting hidden) states, followed by pairs
        containing the values from (visible down-pass, hidden up-pass).
        '''
        while True:
            hidden = self.hidden_expectation(visible)
            yield visible, hidden
            visible = self.visible_expectation(bernoulli(hidden))

    def reconstruct(self, visible, passes=1):
        '''Reconstruct a given visible layer through the hidden layer.

        Parameters
        ----------
        visible : ndarray
            The initial state of the visible layer.

        passes : int
            The number of up- and down-passes.

        Returns
        -------
        An array containing the reconstructed visible layer after the specified
        number of up- and down- passes.
        '''
        for i, (visible, _) in enumerate(self.iter_passes(visible)):
            if i + 1 == passes:
                return visible


class Trainer(object):
    '''
    '''

    def __init__(self, rbm, momentum=0., l2=0., target_sparsity=None):
        '''
        '''
        self.rbm = rbm
        self.momentum = momentum
        self.l2 = l2
        self.target_sparsity = target_sparsity

        self.grad_weights = np.zeros(rbm.weights.shape, float)
        self.grad_vis = np.zeros(rbm.vis_bias.shape, float)
        self.grad_hid = np.zeros(rbm.hid_bias.shape, float)

    def learn(self, visible, learning_rate=0.2):
        '''
        '''
        gradients = self.calculate_gradients(visible)
        self.apply_gradients(*gradients, learning_rate=learning_rate)

    def calculate_gradients(self, visible_batch):
        '''Calculate gradients for a batch of visible data.

        Returns a 3-tuple of gradients: weights, visible bias, hidden bias.

        visible_batch: A (batch size, visible units) array of visible data. Each
          row represents one visible data sample.
        '''
        passes = self.rbm.iter_passes(visible_batch)
        v0, h0 = passes.next()
        v1, h1 = passes.next()
        gw = (np.dot(h0, v0) - np.dot(h1, v1)) / len(visible_batch)
        gv = (v0 - v1).mean(axis=0)
        gh = (h0 - h1).mean(axis=1)
        if self.target_sparsity is not None:
            gh = self.target_sparsity - h0.mean(axis=1)

        logging.debug('displacement: %.3g, hidden std: %.3g',
                      np.linalg.norm(gv), h0.std(axis=1).mean())
        # make sure we pass ndarrays
        gv = gv.reshape(gv.shape[0],1)
        gh = gh.reshape(gh.shape[0],1)
        return gw, gv, gh

    def apply_gradients(self, weights, visible, hidden, learning_rate=0.2):
        '''
        '''
        def update(name, g, _g, l2=0):
            target = getattr(self.rbm, name)
            g *= 1 - self.momentum
            g += self.momentum * (g - l2 * target)
            target += learning_rate * g
            _g[:] = g

        update('vis_bias', visible, self.grad_vis)
        update('hid_bias', hidden, self.grad_hid)
        update('weights', weights, self.grad_weights, self.l2)


class Temporal(RBM):
    '''An RBM that incorporates temporal (dynamic) visible and hidden biases.

    This RBM is based on work and code by Taylor, Hinton, and Roweis (2006).
    '''

    def __init__(self, num_visible, num_hidden, order, binary=True, scale=0.001):
        '''
        '''
        super(TemporalRBM, self).__init__(
            num_visible, num_hidden, binary=binary, scale=scale)

        self.order = order

        self.vis_dyn_bias = scale * rng.randn(order, num_visible, num_visible)
        self.hid_dyn_bias = scale * rng.randn(order, num_hidden, num_visible) - 1.

    def iter_passes(self, frames):
        '''Repeatedly pass the given visible layer up and then back down.

        Generates the resulting sequence of (visible, hidden) states.

        visible: An (order, visible units) array containing frames of visible
          data to "prime" the network. The temporal order of the frames is
          assumed to be reversed, so frames[0] will be the current visible
          frame, frames[1] is the previous frame, etc.
        '''
        vdb = self.vis_dyn_bias[0]
        vis_dyn_bias = collections.deque(
            [np.dot(self.vis_dyn_bias[i], f).T for i, f in enumerate(frames)],
            maxlen=self.order)

        hdb = self.hid_dyn_bias[0]
        hid_dyn_bias = collections.deque(
            [np.dot(self.hid_dyn_bias[i], f).T for i, f in enumerate(frames)],
            maxlen=self.order)

        visible = frames[0]
        while True:
            hidden = self.hidden_expectation(visible, sum(hid_dyn_bias))
            yield visible, hidden
            visible = self.visible_expectation(bernoulli(hidden), sum(vis_dyn_bias))
            vis_dyn_bias.append(np.dot(vdb, visible))
            hid_dyn_bias.append(np.dot(hdb, visible))


class TemporalTrainer(Trainer):
    '''
    '''

    def __init__(self, rbm, momentum=0.2, l2=0.1, target_sparsity=None):
        '''
        '''
        super(TemporalTrainer, self).__init__(rbm, momentum, l2, target_sparsity)
        self.grad_dyn_vis = np.zeros(rbm.hid_dyn_bias.shape, float)
        self.grad_dyn_hid = np.zeros(rbm.hid_dyn_bias.shape, float)

    def calculate_gradients(self, frames_batch):
        '''Calculate gradients using contrastive divergence.

        Returns a 5-tuple of gradients: weights, visible bias, hidden bias,
        dynamic visible bias, and dynamic hidden bias.

        frames_batch: An (order, visible units, batch size) array containing a
          batch of frames of visible data.

          Frames are assumed to be reversed temporally, across the order
          dimension, i.e., frames_batch[0] is the current visible frame in each
          batch element, frames_batch[1] is the previous visible frame,
          frames_batch[2] is the one before that, etc.
        '''
        order, _, batch_size = frames_batch.shape
        assert order == self.rbm.order

        vis_bias = sum(np.dot(self.rbm.vis_dyn_bias[i], f).T for i, f in enumerate(frames_batch))
        hid_bias = sum(np.dot(self.rbm.hid_dyn_bias[i], f).T for i, f in enumerate(frames_batch))

        v0 = frames_batch[0].T
        h0 = self.rbm.hidden_expectation(v0, hid_bias)
        v1 = self.rbm.visible_expectation(bernoulli(h0), vis_bias)
        h1 = self.rbm.hidden_expectation(v1, hid_bias)

        gw = (np.dot(h0.T, v0) - np.dot(h1.T, v1)) / batch_size
        gv = (v0 - v1).mean(axis=0)
        gh = (h0 - h1).mean(axis=0)

        gvd = np.zeros(self.rbm.vis_dyn_bias.shape, float)
        ghd = np.zeros(self.rbm.hid_dyn_bias.shape, float)
        v = v0 - self.rbm.vis_bias - vis_bias
        for i, f in enumerate(frames_batch):
            gvd[i] += np.dot(v.T, f)
            ghd[i] += np.dot(h0.T, f)
        v = v1 - self.rbm.vis_bias - vis_bias
        for i, f in enumerate(frames_batch):
            gvd[i] -= np.dot(v.T, f)
            ghd[i] -= np.dot(h1.T, f)

        return gw, gv, gh, gvd, ghd

    def apply_gradients(self, weights, visible, hidden, visible_dyn, hidden_dyn,
                        learning_rate=0.2):
        '''
        '''
        def update(name, g, _g, l2=0):
            target = getattr(self.rbm, name)
            g *= 1 - self.momentum
            g += self.momentum * (g - l2 * target)
            target += learning_rate * g
            _g[:] = g

        update('vis_bias', visible, self.grad_vis)
        update('hid_bias', hidden, self.grad_hid)
        update('weights', weights, self.grad_weights, self.l2)
        update('vis_dyn_bias', visible_dyn, self.grad_vis_dyn, self.l2)
        update('hid_dyn_bias', hidden_dyn, self.grad_hid_dyn, self.l2)


import scipy.signal
convolve = scipy.signal.convolve


class Convolutional(RBM):
    '''
    '''

    def __init__(self, num_filters, filter_shape, pool_shape, binary=True, scale=0.001):
        '''Initialize a convolutional restricted boltzmann machine.

        num_filters: The number of convolution filters.
        filter_shape: An ordered pair describing the shape of the filters.
        pool_shape: An ordered pair describing the shape of the pooling groups.
        binary: True if the visible units are binary, False if the visible units
          are normally distributed.
        scale: Scale initial values by this parameter.
        '''
        self.num_filters = num_filters

        self.weights = scale * rng.randn(num_filters, *filter_shape)
        self.vis_bias = scale * rng.randn()
        self.hid_bias = scale * rng.randn(num_filters)

        self._visible = binary and sigmoid or identity
        self._pool_shape = pool_shape

    def _pool(self, hidden):
        '''Given activity in the hidden units, pool it into groups.'''
        _, r, c = hidden.shape
        rows, cols = self._pool_shape
        active = np.exp(hidden.T)
        pool = np.zeros(active.shape, float)
        for j in range(int(np.ceil(float(c) / cols))):
            cslice = slice(j * cols, (j + 1) * cols)
            for i in range(int(np.ceil(float(r) / rows))):
                mask = (cslice, slice(i * rows, (i + 1) * rows))
                pool[mask] = active[mask].sum(axis=0).sum(axis=0)
        return pool.T

    def pooled_expectation(self, visible, bias=0.):
        '''Given visible data, return the expected pooling unit values.'''
        activation = np.exp(np.array([
            convolve(visible, self.weights[k, ::-1, ::-1], 'valid')
            for k in range(self.num_filters)]).T + self.hid_bias + bias).T
        return 1. - 1. / (1. + self._pool(activation))

    def hidden_expectation(self, visible, bias=0.):
        '''Given visible data, return the expected hidden unit values.'''
        activation = np.exp(np.array([
            convolve(visible, self.weights[k, ::-1, ::-1], 'valid')
            for k in range(self.num_filters)]).T + self.hid_bias + bias).T
        return activation / (1. + self._pool(activation))

    def visible_expectation(self, hidden, bias=0.):
        '''Given hidden states, return the expected visible unit values.'''
        activation = sum(
            convolve(hidden[k], self.weights[k], 'full')
            for k in range(self.num_filters)) + self.vis_bias + bias
        return self._visible(activation)


class ConvolutionalTrainer(Trainer):
    '''
    '''

    def calculate_gradients(self, visible):
        '''Calculate gradients for an instance of visible data.

        Returns a 3-tuple of gradients: weights, visible bias, hidden bias.

        visible: A single array of visible data.
        '''
        v0 = visible
        h0 = self.rbm.hidden_expectation(v0)
        v1 = self.rbm.visible_expectation(bernoulli(h0))
        h1 = self.rbm.hidden_expectation(v1)

        # h0.shape == h1.shape == (num_filters, visible_rows - filter_rows + 1, visible_columns - filter_columns + 1)
        # v0.shape == v1.shape == (visible_rows, visible_columns)

        gw = np.array([
            convolve(v0, h0[k, ::-1, ::-1], 'valid') -
            convolve(v1, h1[k, ::-1, ::-1], 'valid')
            for k in range(self.rbm.num_filters)])
        gv = (v0 - v1).sum()
        gh = (h0 - h1).sum(axis=-1).sum(axis=-1)
        if self.target_sparsity is not None:
            h = self.target_sparsity - self.rbm.hidden_expectation(visible).mean(axis=-1).mean(axis=-1)

        logging.debug('displacement: %.3g, hidden activations: %.3g',
                      np.linalg.norm(gv), h0.mean(axis=-1).mean(axis=-1).std())

        return gw, gv, gh


class MeanCovariance(RBM):
    '''
    '''

    def __init__(self, num_visible, num_mean, num_precision, scale=0.001):
        '''Initialize a mean-covariance restricted boltzmann machine.

        num_visible: The number of visible units.
        num_mean: The number of units in the hidden mean vector.
        num_precision: The number of units in the hidden precision vector.
        '''
        super(MeanCovariance, self).__init__(
            num_visible, num_mean, binary=False, scale=scale)

        # replace the hidden bias to reflect the precision units.
        self.hid_bias = scale * rng.randn(num_precision, 1)

        self.hid_mean = scale * rng.randn(num_mean, 1)

        self.hid_factor_u = scale * -abs(rng.randn(num_precision - 1))
        self.hid_factor_c = scale * -abs(rng.randn(num_precision))
        self.hid_factor_l = scale * -abs(rng.randn(num_precision - 1))

        self.vis_factor = scale * rng.randn(num_visible, num_precision)

    @property
    def hid_factor(self):
        return (numpy.diag(self.hid_factor_u, 1) +
                numpy.diag(self.hid_factor_c, 0) +
                numpy.diag(self.hid_factor_l, -1))

    def hidden_expectation(self, visible):
        '''Given visible data, return the expected hidden unit values.'''
        z = numpy.dot(visible.T, self.vis_factor)
        return sigmoid(numpy.dot(z * z, self.hid_factor).T + self.hid_bias)

    def visible_expectation(self, hidden):
        '''Given hidden states, return the expected visible unit values.'''
        z = numpy.diag(numpy.dot(-self.hid_factor.T, hidden))
        Sinv = numpy.dot(self.vis_factor, numpy.dot(z, self.vis_factor.T))
        return numpy.dot(numpy.dot(numpy.pinv(Sinv), self.weights), self.hid_mean)


class MeanCovarianceTrainer(Trainer):
    '''
    '''
