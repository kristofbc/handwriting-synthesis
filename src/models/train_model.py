#!/usr/bin/env python # -*- coding: utf-8 -*-

# Generate handwriting from a sequence of characters
# ==================================================

import time
import os
import click
import math
import inspect
import sys
import logging
import random
import copy
from logging.handlers import RotatingFileHandler
from logging import handlers

import numpy as np
import scipy.stats
import cPickle as pickle

try:
    import cupy
except:
    cupy = np
    pass

import chainer
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import variable
from chainer import link
from chainer import function
from chainer import initializers
from chainer import initializer
from chainer.utils import type_check

from batch_generator import BatchGenerator

INPUT_SIZE = 3 # (x, y, end_of_stroke)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ==============
# Helpers (hlps)
# ==============

# ================
# Functions (fcts)
# ================

# ============
# Links (lnks)
# ============
class SoftWindow(chainer.Chain):
    """
        Attention mechanism: SoftWindow
        Args:
            num_mixtures (int): Number of softwindow mixtures
    """
    def __init__(self, num_mixtures):
        super(SoftWindow, self).__init__()
        self._num_mixtures = num_mixtures
        
        with self.init_scope():
            self.layer_linear_alpha = L.Linear(self._num_mixtures, initialW=TruncatedNormal(std=0.075))
            self.layer_linear_beta = L.Linear(self._num_mixtures, initialW=TruncatedNormal(std=0.075))
            self.layer_linear_kappa = L.Linear(self._num_mixtures, initialW=TruncatedNormal(std=0.075))

    def __call__(self, x, cs, k_prev):
        """
            Execute the attention mechanism
            Args:

        """
        batch_size, seq_len, num_letters = cs.shape

        alpha = self.layer_linear_alpha(x)
        alpha = F.exp(alpha)
        beta = self.layer_linear_beta(x)
        beta = F.exp(beta)
        kappa = self.layer_linear_kappa(x)
        kappa = F.exp(kappa)

        a = F.expand_dims(alpha, axis=2)
        b = F.expand_dims(beta, axis=2)
        k = F.expand_dims(k_prev + kappa, axis=2)

        u = -F.expand_dims(F.expand_dims(self.xp.arange(0, seq_len).astype(self.xp.float32), axis=0), axis=0)
        u = F.broadcast_to(u, (k.shape[0], k.shape[1], seq_len))
        phi = F.exp(-F.square(F.broadcast_to(k, (k.shape[0], k.shape[1], seq_len)) + u) * F.broadcast_to(b, (b.shape[0], b.shape[1], seq_len)))
        phi = phi * F.broadcast_to(a, (a.shape[0], a.shape[1], seq_len)) # (batch_size, mixtures, length)
        phi = F.sum(phi, axis=1, keepdims=True) # (batch_size, 1, length)

        # If the network is done generating the sequence
        # (64, 10, 65) > (64,) = (64,)
        # finish = F.where(phi[:, 0, -1] > F.max(phi[:, 0, :-1], axis=1), 1., 0.) # not implemented
        finish = variable.Variable(phi.data[:, 0, -1] > self.xp.max(phi.data[:, 0, :-1], axis=1)) # Break the chain for the finish variable (it's not differentiable anyways)

        k = F.squeeze(k, axis=2)
        window = F.squeeze(F.batch_matmul(phi, cs), axis=1)
        phi = F.squeeze(phi, axis=1)
        finish = F.expand_dims(finish, axis=1)
        return window, k, phi, finish

class MixtureDensity(chainer.Chain):
    """
        MixtureDensity layer
        Args:
            num_mixtures (int): number of mixture components
    """
    def __init__(self, num_mixtures, bias=0.):
        super(MixtureDensity, self).__init__()
        self._num_mixtures = num_mixtures
        self._bias = bias

        with self.init_scope():
            self.layer_linear_e = L.Linear(1, initialW=TruncatedNormal(std=0.075))
            self.layer_linear_pi = L.Linear(self._num_mixtures, initialW=TruncatedNormal(std=0.075))
            self.layer_linear_mu1 = L.Linear(self._num_mixtures, initialW=TruncatedNormal(std=0.075))
            self.layer_linear_mu2 = L.Linear(self._num_mixtures, initialW=TruncatedNormal(std=0.075))
            self.layer_linear_s1 = L.Linear(self._num_mixtures, initialW=TruncatedNormal(std=0.075))
            self.layer_linear_s2 = L.Linear(self._num_mixtures, initialW=TruncatedNormal(std=0.075))
            self.layer_linear_rho = L.Linear(self._num_mixtures, initialW=TruncatedNormal(std=0.075))

    def __call__(self, x):
        """
            Execute the MixtureDensity layer
            Args:
                x (float[][]): the input to extract the MD components
        """
        e = self.layer_linear_e(x)
        pi = self.layer_linear_pi(x)
        mu1 = self.layer_linear_mu1(x)
        mu2 = self.layer_linear_mu2(x)
        s1 = self.layer_linear_s1(x)
        s2 = self.layer_linear_s2(x)
        rho = self.layer_linear_rho(x)

        e = F.sigmoid(e)
        pi = F.softmax(pi * (1. + self._bias))
        s1 = F.exp(s1 - self._bias)
        s2 = F.exp(s2 - self._bias)
        rho = F.tanh(rho)

        return e, pi, mu1, mu2, s1, s2, rho


# ==================
# Initializer (init)
# ==================
class TruncatedNormal(initializer.Initializer):

    """Initialize an array with a truncated normal distribution

    All values with more than 'magnitude' standard deviation from the mean are dropped and re-picked.
    Similar to TensorFlow:truncated_normal and Keras:truncated_normal

    Args:
        mean (float): mean of the normal distribution
        std (float): standard deviation of the normal distribution
        magnitude (int): magnitude of the value from the mean before re-sampling
        dtype: Data type specifier
    """

    def __init__(self, mean=0.0, std=1.0, magnitude=2., dtype=None):
        self.mean = mean
        self.std = std
        self.magnitude = magnitude
        super(TruncatedNormal, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
            dtype = self.dtype
        else:
            dtype = array.dtype

        xp = cuda.get_array_module(array)
        a, b = self.mean - self.magnitude * self.std, self.mean + self.magnitude * self.std
        # Use scipy truncated normal function
        dist = scipy.stats.truncnorm.rvs(a, b, loc=self.mean, scale=self.std, size=array.shape)
        array[...] = xp.asarray(dist).astype(dtype)


class LayerNormalization(chainer.Link):
    """
        Implementation of LayerNormalization from Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization.
        Args:
            hidden_size (int|shape): shape of the hidden size
            bias_init (float): optional bias parameter
            gain_init (float): optional gain parameter
            epsilon (float): computation stability parameters
    """
    def __init__(self, hidden_size, bias_init = 0., gain_init = 1., epsilon = 1e-6):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size
        self.epsilon = epsilon

        with self.init_scope():
            self.bias = variable.Parameter(bias_init)
            self.gain = variable.Parameter(gain_init)

            if hidden_size is not None:
                self._initialize_params(hidden_size)

    def _initialize_params(self, size):
        self.bias.initialize(size)
        self.gain.initialize(size)
        self.hidden_size = size

    def __call__(self, x):
        """
            Apply the LayerNormalization on in the input "x"
            Args:
                x (float[][]): input tensor to re-center and re-scale (layer normalize)
            Returns:
                float[][]
        """
        if self.hidden_size is None:
            self._initialize_params(x.shape)

        # Layer Normalization parameters
        size = x.shape[1]
        mu = F.sum(x, axis=1, keepdims=True) / size
        mu = F.broadcast_to(mu, x.shape)
        sigma = F.sqrt(
            (F.sum(F.square(x - mu), axis=1, keepdims=True) / size) + self.epsilon
        )
        sigma = F.broadcast_to(sigma, x.shape)

        # Transformation
        outputs = (x - mu) / sigma
        # Affine transformation
        outputs = (outputs * self.gain) + self.bias
        
        return outputs

class LayerNormalizationStatelessLSTM(chainer.Chain):
    """
    Implementation of an layer normalized StatelessLSTM: see "Layer Normalization" by Ba. J. et al.

    Args:
        n_units (int): Number of units inside this LSTM
        forget_bias_init (float): bias added to the forget gate before sigmoid activation
        norm_bias_init (float): optional bias parameter
        norm_gain_init (float): optional gain parameter
    """
    def __init__(self, n_units, forget_bias_init = 0., norm_bias_init = 1., norm_gain_init = 0.):
        super(LayerNormalizationStatelessLSTM, self).__init__()

        self.n_units = n_units
        self.forget_bias = forget_bias_init

        with self.init_scope():
            self.h_x = L.Linear(4*n_units)
            self.h_h = L.Linear(4*n_units)
            self.norm_c = LayerNormalization(None, norm_bias_init, norm_gain_init)
            self.norm_x = LayerNormalization(None, norm_bias_init, norm_gain_init)
            self.norm_h = LayerNormalization(None, norm_bias_init, norm_gain_init)

    def __call__(self, c, h, x, b = None):
        """
            Perform the LSTM op
            Args:
                inputs (float[][]): input tensor containing "x" to transform
                c (float[][]): previous LSTM cell state
                h (float[][]): previous LSTM output
                x (float[][]): current input to transform
                b (float[][]): optional bias added to the gates
        """
        f_i_o_g = self.norm_x(self.h_x(x)) + self.norm_h(self.h_h(h))
        if b is not None:
            f_i_o_g = F.bias(f_i_o_g, b)

        # Compute the LSTM using Chainer's function to be able to use LayerNormalization
        def extract_gates(x):
            r = F.reshape(x, (x.shape[0], x.shape[1] // 4, 4) + x.shape[2:])
            return F.split_axis(r, 4, axis=2)

        f, i, o, g = extract_gates(f_i_o_g)
        # Remove unused dimension and apply transformation
        f = F.sigmoid(F.squeeze(f, axis=2) + self.forget_bias)
        i = F.sigmoid(F.squeeze(i, axis=2))
        o = F.sigmoid(F.squeeze(o, axis=2))
        g = F.tanh(F.squeeze(g, axis=2))

        # Transform
        ct = g * i + f * c
        # Apply LayerNormalization
        ht = o * F.tanh(self.norm_c(ct))

        return ct, ht

# =============
# Models (mdls)
# ============= 
class Model(chainer.Chain):
    """
        Alex Graves handwriting model
        Args:
            num_units (int): Number of units for a rnn layer
            rnn_layers (int): Number of rnn layers
            window_mixtures (int): Number of softwindow mixtures
            output_mixtures (int): Number of output mixtures by the MDN
    """
    def __init__(self, num_units, rnn_layers, output_mixtures, window_mixtures, bias=0.):
        super(Model, self).__init__()
        self._num_units = num_units
        self._rnn_layers = rnn_layers
        self._window_mixtures = window_mixtures
        self._output_mixtures = output_mixtures
        self._states = None
        self._e, self._pi, self._mu1, self._mu2, self._s1, self._s2, self._rho = None, None, None, None, None, None, None
        self._window, self._phi, self._kappa, self._finish = None, None, None, None
        self.loss = 0.

        with self.init_scope():
            self.layer_window = SoftWindow(self._window_mixtures)
            #self.layer_lstms = [L.StatelessLSTM(self._num_units) for _ in xrange(self._rnn_layers)]
            self.layer_lstms = [LayerNormalizationStatelessLSTM(self._num_units) for _ in xrange(self._rnn_layers)]
            self.layer_mixture_density = MixtureDensity(self._output_mixtures, bias)

    def to_gpu(self, device=None):
        super(Model, self).to_gpu(device)
        for i in xrange(len(self.layer_lstms)):
            self.layer_lstms[i].to_gpu(device)

    def reset_state(self, state=None):
        """
            Reset the model to the inital state
        """
        if state is None:
            self._states = None
        else:
            for s in xrange(len(self._states)):
                if state.shape != self._states[s].shape:
                    self._states[s] *= F.broadcast_to(state, self._states[s].shape)
                else:
                    self._states[s] *= state

        self._e, self._pi, self._mu1, self._mu2, self._s1, self._s2, self._rho = None, None, None, None, None, None, None
        self._window, self._phi, self._kappa, self._finish = None, None, None, None
        self.loss = 0.

    def get_mdn(self):
        return self._e, self._pi, self._mu1, self._mu2, self._s1, self._s2, self._rho

    def get_window(self):
        # self._states[-3] = window
        # self._states[-2] = k
        # self._states[-1] = finish
        return self._states[-3], self._states[-2], self._states[-1], self._phi

    def __call__(self, inputs):
        """
            Execute the handwriting model
            Args:
                inputs (float[][]): x, cs
        """
        data, cs = inputs
        seq_len, num_letters = cs.shape[1:]
        batch_size, t_max, x_dim = data.shape

        if self._states is None:
            sizes = [self._num_units] * self._rnn_layers * 2 + [num_letters, self._window_mixtures, 1]
            # 9: (64, 400) * 6 [LSTM:cs, LSTM:hs], (64, 80) [SoftWindow:window], (64, 10) [SoftWindow:k], (64, 1) [SoftWindow:finish]
            self._states = [self.xp.zeros((batch_size, s), dtype=self.xp.float32) for s in sizes]
            #for s in xrange(len(self._states)):
            #    print(self._states[s].shape)
            #exit()

        in_coords = data[:, :-1, :] # (batch_size, time step, coord)
        out_coords = data[:, 1:, :]

        # Unroll the RNN for each time steps
        outs = None
        for t in xrange(in_coords.shape[1]):
            #print("{0}/{1}".format(t+1, in_coords.shape[1]))
            x_now = in_coords[:, t, :]

            # self._states[-3] = window
            # self._states[-2] = k
            # self._states[-1] = finish
            # self._states[2n] = c
            # self._states[2n+1] = h
            window, k, finish = self._states[-3:]
            phi = None
            state_output = []
            output_prev = []
            for layer in xrange(len(self.layer_lstms)):
                # LSTM
                x = F.concat([x_now, window] + output_prev, axis=1)
                c_now, h_now = self._states[2*layer], self._states[2*layer+1]
                c_new, h_new = self.layer_lstms[layer](c_now, h_now, x)
                output_prev = [h_new]
                state_output += [c_new, h_new]

                if layer == 0:
                    # Attention Mechanism
                    window, k, phi, finish = self.layer_window(h_new, cs, k)

            self._states = state_output + [window, k, finish]
            self._phi = phi
            outs = F.expand_dims(output_prev[0], axis=0) if outs is None else F.concat((outs, F.expand_dims(output_prev[0], axis=0)), axis=0)

        # MDN
        outs = F.reshape(outs, (-1, self._num_units))
        e, pi, mu1, mu2, s1, s2, rho = self.layer_mixture_density(outs)
        self._e, self._pi, self._mu1, self._mu2, self._s1, self._s2, self._rho = e, pi, mu1, mu2, s1, s2, rho

        # Loss 
        coords = F.reshape(out_coords, (-1, 3))
        coords = F.expand_dims(coords, axis=2)
        x1, x2, eos = F.separate(coords, axis=1)

        epsilon = 1e-8
        z_rho = 1. - F.square(rho)
        z_x1_mu1 = (F.broadcast_to(x1, mu1.shape) - mu1) / s1
        z_x2_mu2 = (F.broadcast_to(x2, mu2.shape) - mu2) / s2
        z = F.square(z_x1_mu1) + F.square(z_x2_mu2) - 2. * rho * z_x1_mu1 * z_x2_mu2
        n = 1. / (2. * np.pi * s1 * s2 * F.sqrt(z_rho)) * F.exp(-z / (2. * z_rho))

        loss_right = eos * e + (1. - eos) * (1. - e)
        loss_right = F.squeeze(loss_right, axis=1)
        loss_right = F.log(loss_right + epsilon)
        loss_left = F.sum(pi * n, axis=1)
        loss_left = F.log(loss_left + epsilon)
        loss = F.mean(-loss_left - loss_right)

        del in_coords, out_coords, data, cs, outs

        self.loss = loss
        return self.loss

# ===============================================
# Main entry point of the training process (main)
# ===============================================

def main(data_dir, output_dir, batch_size, min_sequence_length, validation_split, epochs, grad_clip, resume_dir, resume_model, resume_optimizer, resume_stats, gpu, save_interval, validation_interval, truncated_backprop_interval, rnn_layers_number, rnn_cells_number, win_unit_number, mix_comp_number, random_seed, learning_rate, debug):
    """ Save the args for this run """
    arguments = {}
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for i in args:
        arguments[i] = values[i]

    # Snapshot directory
    model_suffix_dir = "{0}-{1}".format(time.strftime("%Y%m%d-%H%M%S"), batch_size)
    training_suffix = "{0}".format("training")
    state_suffix = "{0}".format("state")

    save_dir = output_dir + '/' + model_suffix_dir
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)

    """ Setup the global logger """
    logger = logging.getLogger()
    logFormatter = logging.Formatter(log_fmt)

    # File logger
    #fh = logging.FileHandler("{0}/logs.txt".format(save_dir))
    #fh.setFormatter(logFormatter)
    #logger.addHandler(fh)

    # stdout logger
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logFormatter)
    logger.addHandler(ch)

    """ Train the model based on the data saved in ../processed """
    logger.info("Run arguments")
    logger.info(arguments)
    logger.info('Training the model')
    logger.info('GPU: {}'.format(gpu))
    logger.info('Mini-batch size: {}'.format(batch_size))
    logger.info('# epochs: {}'.format(epochs))

    """ Chainer's debug mode """
    if debug == 1:
        logger.info("Enabling Chainer's debug mode")
        chainer.config.debug = True

    """ Fetching the model and the inputs """
    logger.info("Fetching the model and the inputs")
    dataset_size = BatchGenerator.dataset_size(data_dir)
    validation_size = int(math.floor(dataset_size*validation_split))
    train_size = int(dataset_size-validation_size)
    batch_generator_train = BatchGenerator(data_dir, batch_size, min_sequence_length, 0, train_size)
    batch_generator_validation = BatchGenerator(data_dir, batch_size, min_sequence_length, train_size)

    history_network_train = []
    history_network_valid = []
    offset_epoch = 0

    """ Create the model """
    def op_models(models, op, *args):
        rets = []
        for i in xrange(len(models)):
            rets.append(op(i, models[i], *args))

        return rets

    models = []
    if gpu == -1:
        logger.info("Creating the model")
        m = Model(rnn_cells_number, rnn_layers_number, mix_comp_number, win_unit_number)
        models.append(m)
    else:
        gpu = gpu.split(',')
        logger.info("Creating {0} model(s)".format(len(gpu)))

        for i in xrange(len(gpu)):
            if i == 0:
                m = Model(rnn_cells_number, rnn_layers_number, mix_comp_number, win_unit_number)    
            else:
                m = copy.deepcopy(models[0])

            models.append(m)

    """ Enable cupy, if available """
    if gpu > -1:
        logger.info("Enabling CUpy")
        chainer.cuda.get_device_from_id(gpu).use()
        xp = cupy
        model.to_gpu()
    else:
        xp = np

    """ Setup the model """
    logger.info("Setuping the model")
    #optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.90, beta2=0.999, eps=1e-08)
    optimizer = chainer.optimizers.Adam(alpha=learning_rate)
    optimizer.setup(models[0])

    if grad_clip is not 0:
        optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

    if resume_dir:
        # Resume model and optimizer
        logger.info("Loading state from {}".format(output_dir + '/' + resume_dir))
        if resume_model != "":
            chainer.serializers.load_npz(output_dir + "/" + resume_dir + "/" + resume_model, models[0])
            for i in xrange(len(models[1:])):
                models[i] = copy.deepcopy(models[0])

        if resume_optimizer != "":
            chainer.serializers.load_npz(output_dir + "/" + resume_dir + "/" + resume_optimizer, optimizer)
        # Resume statistics
        if resume_stats == 1:
            try:
                history_network_train = list(np.load(output_dir + "/" + resume_dir + "/history-network-train.npy"))
                history_network_valid = list(np.load(output_dir + "/" + resume_dir + "/history-network-valid.npy"))
            except:
                logger.info("Unable to find history network train or valid")

            offset_epoch = len(history_network_train)

    """ Start training """
    xp.random.seed(random_seed)
    np.random.seed(random_seed)
    time_epoch_start = time.time()
    history_train = []
    history_valid = []
    n_batches = None

    # Exponential Decay on learning rate helper
    def exponential_decay(base_learning_rate, decay_steps, decay_rate, staircase=False, optimizer_lr_attr="lr"):
        """
        Lower the learning rate of the optimizer via an exponential decay
        Args:
            base_learning_rate (float): initial learning rate
            decay_steps (int): divided by the current steps
            decay_rate (float): multiplied to (current_steps/decay_steps)
            staircase (bool): if true, the learning rate takes a discrete interval
        Returns:
            (Function) function computing the decayed learning rate at each iteration

        Example:
            lr_decay = exponential_decay(0.001, 10000, 0.5, staircase=True)
            ...
            current_iteration = 0
            for batch in mini_batch:
                ...
                optimizer.lr = lr_decay(optimizer.t)
        """
        def cb(current_step):
            """
            Called every iteration, return the decayed learning rate
            Args:
                current_step (int): current iteration number
            Returns:
                (float) the decayed learning rate
            """
            steps = current_step / decay_steps
            if staircase:
                steps = math.floor(steps)
            return base_learning_rate * (decay_rate ** steps)

        return cb

    batches_per_epoch = 1000
    batches_per_epoch_valid = int(len(batch_generator_validation.dataset)/batch_size) # Compute all the batches inside one epoch
    itr = 0
    accum_loss = 0
    best_valid_loss = 0
    if truncated_backprop_interval > 0:
        lr_decay = exponential_decay(learning_rate, int(math.floor(10000/truncated_backprop_interval)), 0.5, staircase=True)
    else:
        lr_decay = exponential_decay(learning_rate, 10000, 0.5, staircase=True) # If no truncated bpp

    logger.info("Starting training with {0} mini-batches for {1} epochs".format(batches_per_epoch, epochs))
    for e in xrange(offset_epoch, epochs):
        logger.info("Epoch #{0}/{1}".format(e+1, epochs))
        for b in xrange(1, batches_per_epoch+1):
            time_iteration_start = time.time()
            coords, seq, reset, needed = batch_generator_train.next_batch()
            if needed:
                #print("Reset state")
                #model.reset_state(xp.asarray(reset))
                def reset_state(i, m, reset):
                    if m.xp != np:
                        m.reset_state(chainer.cuda.to_gpu(reset.copy(), m._device_id))
                    else:
                        m.reset_state(reset.copy())

                op_models(models, reset_state, reset)

            """ Train the model """
            def make_inputs_models(i, m, coords, seq):
                if m.xp != np:
                    return [chainer.cuda.to_gpu(coords.copy(), m._device_id), chainer.cuda.to_gpu(seq.copy(), m._device_id)]
                else:
                    return [coords, seq]

            inputs = op_models(models, make_inputs_models, coords, seq)
            losses = op_models(models, lambda i, m, x: m(x[0]), inputs)
            #loss_t = model([xp.asarray(coords), xp.asarray(seq)])
            #accum_loss += loss_t

            # Truncated back-propagation
            if truncated_backprop_interval == 0 or (b+1)%truncated_backprop_interval == 0 or b == batches_per_epoch+1:
                op_models(models, lambda i, model: model.cleargrads())
                #model.cleargrads()
                #accum_loss.backward()
                op_models(losses, lambda i, loss: loss.backward())

                if truncated_backprop_interval > 0:
                    #accum_loss.unchain_backward()
                    op_models(losses, lambda i, loss: loss.unchain_backward())

                if len(models) > 1:
                    op_models(models[0:1], 
                              lambda i, model, models: [model.addgrads(models[j]) for j in xrange(len(models))], 
                              models[1:])

                optimizer.update()

                if len(models) > 1:
                    op_models(models[1:], 
                              lambda i, model, master: model.copyparams(master), 
                              models[0])
                
                # Exponential decay on the learning rate
                optimizer.alpha = lr_decay(optimizer.t)
                losses = []
                #accum_loss = 0

            loss = cuda.to_cpu(models[0].loss.data)
            #del loss_t

            time_iteration_end = time.time()-time_iteration_start
            history_train.append([loss, time_iteration_end])
            logger.info("[TRAIN] Epoch #{0} ({1}/{2}): loss = {3}, time = {4}".format(e+1, b, batches_per_epoch, loss, time_iteration_end))
            itr += 1

        # Compile global stats
        history_train = np.asarray(history_train)
        train_mean = history_train[:, 0].mean()
        train_std = history_train[:, 0].std()
        train_min = history_train[:, 0].min()
        train_max = history_train[:, 0].max()
        train_med = np.median(history_train[:, 0])
        train_time_sum = history_train[:, 1].sum()
        history_network_train.append([
            train_mean, train_std, train_min, train_max, train_med, train_time_sum
        ])
        logger.info("[TRAIN] Epoch #{0} (COMPLETED IN {1}): mean = {2}, std = {3}, min = {4}, max = {5}, med = {6}".format(e+1, train_time_sum, train_mean, train_std, train_min, train_max, train_med))

        # Check if we should validate the data
        if (e+1) % validation_interval == 0:
            with chainer.using_config('train', False):
                with function.no_backprop_mode():
                    # Reset completely the state before the validation
                    models[0].reset_state()
                    for b in xrange(batches_per_epoch_valid):
                        time_iteration_start = time.time()
                        coords, seq, reset, needed = batch_generator_validation.next_batch()
                        if needed:
                            #print("Reset state")
                            models[0].reset_state(xp.asarray(reset))

                        loss_t = models[0]([xp.asarray(coords), xp.asarray(seq)])
                        loss = cuda.to_cpu(loss_t.data)
                        del loss_t

                        time_iteration_end = time.time()-time_iteration_start
                        history_valid.append([loss, time_iteration_end])
                        logger.info("[VALID] Epoch #{0} ({1}/{2}): loss = {3}, time = {4}".format(e+1, b, batches_per_epoch_valid, loss, time_iteration_end))

                    # All the validation mini-batches are processed
                    history_valid = np.asarray(history_valid)
                    valid_mean = history_valid[:, 0].mean()
                    valid_std = history_valid[:, 0].std()
                    valid_min = history_valid[:, 0].min()
                    valid_max = history_valid[:, 0].max()
                    valid_med = np.median(history_valid)
                    valid_time_sum = history_valid[:, 1].sum()
                    history_network_valid.append([valid_mean, valid_std, valid_min, valid_max, valid_med, valid_time_sum])
                    logger.info("[VALID] Epoch #{0} (COMPLETED IN {1}): mean = {2}, std = {3}, min = {4}, max = {5}, med = {6}"
                                .format(e+1, valid_time_sum, valid_mean, valid_std, valid_min, valid_max, valid_med))

                    # Completely reset the state for the next training phase
                    model.reset_state()
                    batch_generator_validation.reset()
                    batch_generator_train.reset()


            # If a "best" model is found, save it
            if valid_mean < best_valid_loss:
                best_valid_loss = valid_mean

                logger.info("Best fit found for epoch {0} with loss {1}".format(e+1, best_valid_loss))

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Save the model and optimizer
                chainer.serializers.save_npz(save_dir + '/model-best', model)
                chainer.serializers.save_npz(save_dir + '/state-best', optimizer)

        # Check if we should save the data
        if (e+1) % save_interval == 0:
            logger.info("Saving the model, optimizer and history")

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Save the model and optimizer
            chainer.serializers.save_npz(save_dir + '/model-{}'.format(e+1), model)
            chainer.serializers.save_npz(save_dir + '/state-{}'.format(e+1), optimizer)

            # Save the stats
            np.save(save_dir + '/history-network-train', np.asarray(history_network_train))
            np.save(save_dir + '/history-network-valid', np.asarray(history_network_valid))

        # Reset global epochs vars
        history_train = []
        history_valid = []


# ===============
# CLI Entry point
# ===============


@click.command()
@click.option('--data_dir', type=click.Path(exists=True), default='data/processed/OnlineHandWriting/tf', help='Directory containing the data.')
@click.option('--output_dir', type=click.Path(exists=False), default='models', help='Directory for model checkpoints.')
@click.option('--batch_size', type=click.INT, default=64, help='Size of the mini-batches.')
@click.option('--min_sequence_length', type=click.INT, default=256, help='Minimum size of a data sequence.')
@click.option('--validation_split', type=click.FLOAT, default=0.15, help='Use this split size of the validation set.')
@click.option('--epochs', type=click.INT, default=30, help='Number of epoch for training.')
@click.option('--grad_clip', type=click.INT, default=3, help='Gradient clip value.')
@click.option('--resume_dir', type=click.STRING, default='', help='Directory name for resuming the optimization from a snapshot.')
@click.option('--resume_model', type=click.STRING, default='', help='Name of the model snapshot.')
@click.option('--resume_optimizer', type=click.STRING, default='', help='Name of the optimizer snapshot.')
@click.option('--resume_stats', type=click.INT, default=0, help='Resume the statistics (new stats will be combined with old statistics).')
@click.option('--gpu', type=click.INT, default=-1, help='GPU ID (negative value is CPU).')
@click.option('--save_interval', type=click.INT, default=10, help='How often the model should be saved.')
@click.option('--validation_interval', type=click.INT, default=1, help='How often the model should be validated.')
@click.option('--truncated_backprop_interval', type=click.INT, default=10, help='Run the truncated backpropagation algorithm at each n iteration.')
@click.option('--rnn_layers_number', type=click.INT, default=3, help='Number of layers for the RNN.')
@click.option('--rnn_cells_number', type=click.INT, default=400, help='Number of LSTM cells per layer.')
@click.option('--win_unit_number', type=click.INT, default=10, help='Number of soft-window components.')
@click.option('--mix_comp_number', type=click.INT, default=20, help='Numver of Gaussian components for mixture density output.')
@click.option('--random_seed', type=click.INT, default=None, help='Number of Gaussian components for mixture density output.')
@click.option('--learning_rate', type=click.FLOAT, default=0.001, help='Learning rate of the optimizer.')
@click.option('--debug', type=click.INT, default=0, help='Chainer debugging mode.')
def cli(**kwargs):
    main(**kwargs)


if __name__ == '__main__':
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)

    cli()
