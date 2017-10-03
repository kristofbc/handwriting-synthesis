#!/usr/bin/env python # -*- coding: utf-8 -*-

# Generate handwriting from a sequence of characters
# ==================================================

import time
import os
import click
import math
import inspect
import sys
import random
import logging
from logging.handlers import RotatingFileHandler
from logging import handlers

import numpy as np
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

#from net.lstm import LSTM
#from net.adaptive_weight_noise import AdaptiveWeightNoise
#from net.soft_window import SoftWindow
#from net.mixture_density_outputs import MixtureDensityOutputs

from net.functions.adaptive_weight_noise import adaptive_weight_noise
from net.functions.soft_window import soft_window
#from net.functions.mixture_density_outputs import mixture_density_outputs
from functions.connection.mixture_density_network import mixture_density_network
from links.connection.lstm import LSTM
from utils import mean_squared_error

INPUT_SIZE = 5 # (x, y, end_of_stroke)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ==============
# Helpers (hlps)
# ==============
def get_max_sequence_length(sequences):
    length = []
    for i in xrange(len(sequences)):
        length.extend([sequences[i].size])
    return int(np.asarray(length).max())

def group_data(data, characters):
    if len(data) != len(characters):
        raise ValueError("data should have the same amount of characters")

    xp = cuda.get_array_module(*data)
    grouped = []
    for i in xrange(len(data)):
        grouped.append(xp.asarray([data[i], characters[i]]))

    return xp.vstack(grouped)

def pad_data(data, characters):
    max_length_data = 0
    max_length_characters = 0
    # Get the maximum length of each arrays
    tmp1 = []
    tmp2 = []
    for i in xrange(len(data)):
        if len(data[i]) > max_length_data:
            max_length_data = len(data[i])

    # Pad each arrays to be the same length
    for i in xrange(len(data)):
        if len(data[i]) != max_length_data:
            pad_length = max_length_data-len(data[i])
            pad = np.full((pad_length, 5), 0)
            pad[:, 2:5] = 2.
            data[i] = np.vstack([data[i], pad])
        tmp1.append(np.asarray(data[i]))
        tmp2.append(np.asarray(characters[i]))
    
    return np.asarray(tmp1), np.asarray(tmp2) 

def one_hot(data, characters, n_chars, n_max_seq_length):
    xp = cuda.get_array_module(*data)
    cs = xp.zeros((len(data), n_chars, n_max_seq_length), dtype=xp.float32)

    for i in xrange(len(data)):
        for j in xrange(len(characters[i])):
            k = characters[i][j]
            cs[i, k, j] = 1.0

    return cs

def get_expanded_stroke_position(positions):
    # Each position time step is composed of
    # deltaX, deltaY, p1, p2, p3
    # deltaX, deltaY is the pen position's offsets
    # p1 = pen is touching paper
    # p2 = pen will be lifted from the paper (don't draw next stroke)
    # p3 = handwriting is completed
    new_positions = []
    for stroke in positions:
        parts = np.zeros((len(stroke), 5), dtype=np.float32)
        idx_mask = np.where(stroke[:, 2] == 2.)[0]
        parts[:, 0:2] = stroke[:, 0:2]
        parts[:, 3] = stroke[:, 2]
        parts[:, 2] = 1 - stroke[:, 2]

        if len(idx_mask) > 0:
            parts[idx_mask, 2:5] = 2.
            parts[idx_mask[0]-1, 4] = 1
        else:
            parts[-1][4] = 1

        new_positions.append(parts)

    return np.asarray(new_positions)

# ================
# Functions (fcts)
# ================
class SoftWindowFunction(function.Function):
    def check_type_forward(self, in_types):
        pass

    def forward(self, inputs):
        a_h, b_h, k_h, cs, k_prev = inputs

        if not type_check.same_types(*inputs):
            raise ValueError("numpy and cupy must not be used together\n"
                             "type(a_h): {0}, type(b_h): {1}, type(k_h): {2}, type(cs): {3}, type(k_prev): {4}"
                             .format(type(a_h), type(b_h), type(k_h), type(cs), type(k_prev)))

        batch_size, W, u = cs.shape
        K = a_h.shape[1]
        xp = cuda.get_array_module(*inputs)

        a_h = xp.exp(a_h).reshape((batch_size, K, 1))
        b_h = xp.exp(b_h).reshape((batch_size, K, 1))
        k_h = k_prev + xp.exp(k_h)
        k_h = xp.reshape(k_h, (batch_size, K, 1))

        self.a_h = a_h
        self.b_h = b_h
        self.k_h = k_h

        # Compute phi's parameters
        #us = xp.linspace(0, u-1, u)
        us = xp.arange(u, dtype=xp.float32).reshape((1, 1, u))
        phi = a_h * xp.exp(-b_h * xp.square(k_h - us))

        self.phi = phi
        
        phi = xp.sum(phi, axis=1)

        # Finalize the soft window computation
        w = xp.matmul(cs, phi.reshape((batch_size, u, 1)))

        return w.reshape((batch_size, W)), k_h.reshape((batch_size, K)), phi

    def backward(self, inputs, grad_outputs):
        a_h, b_h, k_h, cs, k_prev = inputs

        batch_size, W, u = cs.shape
        K = a_h.shape[1]
        xp = cuda.get_array_module(*inputs)
        us = xp.arange(u, dtype=xp.float32).reshape((1, 1, u))

        # Get the gradient output w.r.t the loss fow "w" and "k"
        gw, gk = grad_outputs[0:2]

        if gw is None:
            gw = 0.
        if gk is None:
            gk = 0.

        # Compute e using forward and gradient values. Eq. 56
        g_e = self.phi * xp.matmul(gw.reshape(batch_size, 1, W), cs)

        # Gradients of the original parameters. Eq. 57 to 60
        g_a_h = xp.sum(g_e, axis=2)
        b = self.b_h.reshape((batch_size, K))
        g_b_h = -b * xp.sum(g_e * xp.square(self.k_h - us), axis=2)
        g_k_p = gk + 2. * b * xp.sum(g_e * (us - self.k_h), axis=2) 
        g_k_h = xp.exp(k_h) * g_k_p

        return g_a_h, g_b_h, g_k_h, None, g_k_p,

        

# ============
# Links (lnks)
# ============
class SoftWindow(chainer.Link):
    """
        The SoftWindow act as an attention mechanism which controls the alignment between the text and the pen position

        Args:
            mixture_size (int): size of the mixture "k"
            unit_size (int): size of the mixture hidden units
    """
    def __init__(self, mixture_size, unit_size):
        super(SoftWindow, self).__init__()

        self.mixture_size = mixture_size
        self.unit_size = unit_size
        self.k_prev = None
        self.w = None
        self.phi = None

    def reset_state(self):
        """
            Reset Variables
        """
        self.k_prev = None
        self.w = None
        self.phi = None
    
    def __call__(self, inputs):
        """
            Perform the SoftWindow text-pen alignment prediction

            Args:
                inputs (float[][]): input tensor containing "x" and the character sequence "cs"
            Returns:
                window (float)
        """
        x, cs = inputs
        batch_size, W, u = cs.shape

        # Extract the soft window's parameters
        a_h, b_h, k_h = F.split_axis(x, [self.mixture_size, 2 * self.mixture_size], axis=1)
        K = a_h.shape[1]

        if self.k_prev is None:
            self.k_prev = variable.Variable(self.xp.zeros_like(k_h, dtype=self.xp.float32))

        self.w, self.k_prev, self.phi = SoftWindowFunction()(a_h, b_h, k_h, cs, self.k_prev)

        return self.w

class MixtureDensityNetwork(chainer.Link):
    """
        The Mixture-Density-Network outputs a parametrised mixture distribution.
        
        Args:
            n_mdn_comp (int): number of MDN components
            n_units (int): number of cells units
            prob_bias (float): bias added to the pi's probability
    """
    def __init__(self, n_mdn_comp, n_units, prob_bias = 0.):
        super(MixtureDensityNetwork, self).__init__()

        self.n_mdn_comp = n_mdn_comp
        self.n_unit = n_units
        self.p_bias = prob_bias

        self.q, self.pi, self.mu_x1, self.mu_x2, self.s_x1, self.s_x2, self.rho = None, None, None, None, None, None, None
        self.gamma = None
        self.loss = None

    def reset_state(self):
        """
            Reset the Variables
        """
        self.q, self.pi, self.mu_x1, self.mu_x2, self.s_x1, self.s_x2, self.rho = None, None, None, None, None, None, None
        self.gamma = None
        self.loss = None
 
    def __call__(self, inputs):
        """
            Perform the MDN prediction

            Args:
                inputs (float[][]): input tensor 
            Returns:
                loss (float)
        """
        x, y = inputs
        xp = cuda.get_array_module(*x)

        # Extract the MDN's parameters
        q_h, pi_h, mu_x1_h, mu_x2_h, s_x1_h, s_x2_h, rho_h = F.split_axis(
            y, [
                3, 3+self.n_mdn_comp, 3+2*self.n_mdn_comp, 3+3*self.n_mdn_comp, 3+4*self.n_mdn_comp, 3+5*self.n_mdn_comp
            ], axis=1
        )

        # Add the bias to the parameter to change the shape of the prediction
        p_bias = 1. if self.p_bias == 0 else self.p_bias
        q_h /= p_bias
        pi_h /= p_bias
        s_x1_h *= p_bias
        s_x2_h *= p_bias

        self.loss, _, _, _, self.q, self.pi, self.mu_x1, self.mu_x2, self.s_x1, self.s_x2, self.rho = mixture_density_network(
            x, q_h, pi_h, mu_x1_h, mu_x2_h, s_x1_h, s_x2_h, rho_h
        )

        return self.loss 

class Linear(chainer.Link):
    """
        Linear link with custom weight and bias initialization
        
        Args:
            in_size (int): Dimension of input vector
            out_size (tuple): Dimension of output vector
        Returns:
            float[][]
    """
    def __init__(self, in_size, out_size):
        super(Linear, self).__init__()

        self.out_size = out_size
        self.in_size = in_size

        with self.init_scope():
            self.W = chainer.Parameter(chainer.initializers.Normal(0.075))
            self.b = chainer.Parameter(chainer.initializers.Normal(0.075))

            # Initialization is on call (don't initialize unecessary params if they're given each time
            #if in_size is not None:
                #self._initialize_params(in_size)

    def _initialize_params(self, in_size):
        self.in_size = in_size
        self.W.initialize((self.out_size, in_size))
        self.b.initialize((self.out_size))

    def __call__(self, x, W=None, b=None):
        """
            Perform the Linear operation with custom weights and bias

            Args:
                x (float[][]): input tensor "x" to transform
                W (float[][]): input weights
                b (float[]): input bias
            Returns
                float[][]
        """
        if W is None and b is None and self.W.data is None:
            if self.in_size is None:
                self._initialize_params(x.size // x.shape[0])
            else:
                self._initialize_params(self.in_size)

        if W is None:
            W = self.W
        if b is None:
            b = self.b

        return F.linear(x, W, b)

class LinearLSTM(chainer.Chain):
    """
        Alex Graves' LSTM implementation
        Args:
            n_units (int): Number of units inside this LSTM
    """
    def __init__(self, n_units):
        super(LinearLSTM, self).__init__()

        self.n_units = n_units
        self.h = None
        self.c = None

        with self.init_scope():
            self.h_x = Linear(None, n_units)

    def reset_state(self):
        """
            Reset the internal state of the LSTM
        """
        self.h = None
        self.c = None

    def __call__(self, inputs, W, b):
        """
            Perform the LSTM op
            Args:
                inputs (float[][]): input tensor containing "x" to transform
        """
        x = inputs
        if self.h is not None:
            x += self.h_x(self.h, W, b)

        if self.c is None:
            self.c = variable.Variable(self.xp.zeros((len(inputs), self.n_units), dtype=self.xp.float32))

        self.c, self.h = F.lstm(self.c, x)
        return self.h

class AdaptiveWeightNoise(chainer.Link):
    """
        Alex Grave's Adaptive Weight Noise
        From: Practical Variational Inference for Neural Networks.
    """
    def __init__(self, in_size, out_size, normal_scale=0.1, nobias=False, initial_bias=0):
        super(AdaptiveWeightNoise, self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        self.normal_scale = normal_scale
        self.nobias = nobias
        self.initial_bias = initial_bias

        with self.init_scope():
            if nobias:
                self.mu = chainer.Parameter(chainer.initializers.Normal(1))
            else:
                self.mu = chainer.Parameter(NormalBias(self.out_size, self.normal_scale, self.initial_bias))

            self.sigma = chainer.Parameter(initializers._get_initializer(self.xp.log(1e-8)))

            if in_size is not None:
                self._initialize_params(in_size)

    def _initialize_params(self, in_size):
        self.mu.initialize((self.out_size*in_size))
        self.sigma.initialize((self.out_size*in_size))
        self.in_size = in_size

    def get_test_weight(self, in_size=None):
        """
            When testing, do not generate AWN weights and biases, use the current configuration
            Args:
                in_size (int): Input size of the variable to transform
            Returns:
                weights(float[][])
        """
        if self.in_size is None:
            if in_size is None:
                raise ValueError("in_size should not be none for test weights")

            self.in_size = in_size

        if self.nobias:
            return F.reshape(self.mu, (self.out_size, self.in_size)), None
        else:
            w, b = F.split_axis(self.mu, [self.out_size*(self.in_size-1)], axis=0)
            return F.reshape(w, (self.out_size, self.in_size-1)), b

    def __call__(self, batch_size, in_size=None):
        """
            Update the weigths
            Args:
                batch_size (Variable): Size of the current batch
                in_size (int): Input size of the variable to transform
            Returns:
                weight (float[][]), loss (float)
        """
        if self.mu.data is None or self.sigma.data is None:
            if in_size is None:
                raise ValueError("AdaptiveWeightNoise requires a in_size to intialize it's Parameters")

            self._initialize_params(in_size)

        # Base parameters
        mu_h = F.broadcast_to(F.mean(self.mu), self.mu.shape)
        diff_mu_h = F.square(self.mu - mu_h)
        sigma_h = F.broadcast_to(F.mean(F.exp(self.sigma) + diff_mu_h), self.sigma.shape)

        # Weight and bias
        eps = variable.Variable(self.xp.random.randn(self.mu.size).astype(self.xp.float32))
        W = self.mu + eps  * F.sqrt(F.exp(self.sigma))

        # Loss
        loss_x = (F.log(sigma_h) - self.sigma) / 2.
        loss_y = (diff_mu_h + F.exp(self.sigma) - sigma_h) / ((2. * sigma_h) + 1e-8)
        loss = F.reshape(F.sum(loss_x + loss_y), (1,)) / batch_size

        # Extract the bias if required
        if self.nobias:
            return F.reshape(W, (self.out_size, self.in_size)), None, loss
        else:
            w, b = F.split_axis(W, [self.out_size*(self.in_size-1)], axis=0)
            return F.reshape(w, (self.out_size, self.in_size-1)), b, loss

# ==================
# Initializer (init)
# ==================
class NormalBias(initializer.Initializer):
    """
        Initialize array with a normal distribution and bias.
        Mean is zero.
        Standard deviation is "scale".
        
        Args:
            out_size(int): Output size
            scale(float): Standard deviation of Gaussian distribution.
            bias(float): Inital bias value
            dtype: Data type specifier.
    """
    def __init__(self, out_size, scale=1., bias=0., dtype=None):
        self.out_size = out_size
        self.scale = scale
        self.bias = bias
        super(NormalBias, self).__init__(dtype)

    def __call__(self, array):
        # @NOTE We assume that array is 1D
        xp = cuda.get_array_module(array)
        in_size = array.shape[0]/self.out_size
        args = {'loc': 0.0, 'scale': self.scale, 'size': self.out_size*(in_size-1)}
        if xp is not np:
            # Only CuPy supports dtype option
            if self.dtype == np.float32 or self.dtype == np.float16:
                # float16 is not supported in cuRAND
                args['dtype'] = np.float32

        normal = xp.random.normal(**args).astype(self.dtype).reshape((1, args['size']))
        bias = xp.ones((1, self.out_size)).astype(self.dtype) * self.bias
        array[...] = xp.concatenate((normal, bias), axis=1).reshape(array.shape)
        

# =============
# Models (mdls)
# ============= 
class Model(chainer.Chain):
    """
        Synthesis model is defined as:
            - Batch (pen positions) -> LSTM1
            - Batch + one-hot (character sequences + LSTM1 -> SoftWindow
            - Batch + SoftWindow + LSTM 1 -> LSTM2
            - LSTM2 -> LSTM 3
            - LSTM3 -> MDM
            - MDN -> output

        Args:
            n_layers (int): number of hidden rnn layers
            n_units (int): number of hidden units in LSTM cells
            n_mixture_components (int): number of MDN components
            n_window_unit (int): number of parameters in the softwindow
            prob_bias (float): bias sampling value
        Returns
            loss (float)
    """

    def __init__(self, n_layers, n_units, n_mixture_components, n_window_unit, prob_bias = 0.):
        super(Model, self).__init__()

        with self.init_scope():
            # LSTMs layers
            self.lstm1 = LinearLSTM(n_units)
            self.lstm2 = LinearLSTM(n_units)
            self.lstm3 = LinearLSTM(n_units)

            # Attention mechanism
            self.sw = SoftWindow(n_window_unit, n_units)

            # Mixture Density Network
            self.mdn = MixtureDensityNetwork(n_mixture_components, n_units, prob_bias)

            # Linear connections for some layers
            self.x_lstm1 = Linear(n_layers + 2, n_units*4)
            self.x_lstm2 = Linear(n_layers + 2, n_units*4)
            self.x_lstm3 = Linear(n_layers + 2, n_units*4)
            self.lstm1_lstm2 = Linear(n_units, n_units * 4)
            self.lstm2_lstm3 = Linear(n_units, n_units * 4)
            self.lstm1_sw = Linear(n_units, n_window_unit * 3)
            # Input shape (here None) is the length of the vocabulary: it's dynamic
            self.sw_lstm1 = Linear(None, n_units * 4)
            self.sw_lstm2 = Linear(None, n_units * 4)
            self.sw_lstm3 = Linear(None, n_units * 4)
            self.h1_mdn = Linear(n_units, n_mixture_components * 5 + n_mixture_components + 3)
            self.h2_mdn = Linear(n_units, n_mixture_components * 5 + n_mixture_components + 3)
            self.h3_mdn = Linear(n_units, n_mixture_components * 5 + n_mixture_components + 3)

            # Noise for the linear connections
            self.awn_x_lstm1 = AdaptiveWeightNoise(n_layers + 3, n_units*4)
            self.awn_x_lstm2 = AdaptiveWeightNoise(n_layers + 3, n_units*4)
            self.awn_x_lstm3 = AdaptiveWeightNoise(n_layers + 3, n_units*4)
            self.awn_lstm1 = AdaptiveWeightNoise(n_units+1, n_units*4)
            self.awn_lstm2 = AdaptiveWeightNoise(n_units+1, n_units*4)
            self.awn_lstm3 = AdaptiveWeightNoise(n_units+1, n_units*4)
            self.awn_lstm1_lstm2 = AdaptiveWeightNoise(n_units+1, n_units * 4)
            self.awn_lstm2_lstm3 = AdaptiveWeightNoise(n_units+1, n_units * 4)
            self.awn_lstm1_sw = AdaptiveWeightNoise(n_units+1, n_window_unit * 3)
            # Input shape (here None) is the length of the vocabulary: it's dynamic
            self.awn_sw_lstm1 = AdaptiveWeightNoise(None, n_units * 4)
            self.awn_sw_lstm2 = AdaptiveWeightNoise(None, n_units * 4)
            self.awn_sw_lstm3 = AdaptiveWeightNoise(None, n_units * 4)
            self.awn_h1_mdn = AdaptiveWeightNoise(n_units+1, n_mixture_components * 5 + n_mixture_components + 3)
            self.awn_h2_mdn = AdaptiveWeightNoise(n_units+1, n_mixture_components * 5 + n_mixture_components + 3)
            self.awn_h3_mdn = AdaptiveWeightNoise(n_units+1, n_mixture_components * 5 + n_mixture_components + 3)

        self.n_units = n_units
        self.n_mixture_components = n_mixture_components
        self.n_window_unit = n_window_unit
        self.p_bias = prob_bias
        self.mdn_components = None
        self._awn_weights = {}
        self._awn_biases = {}

        self.loss = 0

    def reset_state(self): 
        """
            Reset own Variables and Links Variables
        """
        self.sw.reset_state()
        self.mdn.reset_state()
        self.lstm1.reset_state()
        self.lstm2.reset_state()
        self.lstm3.reset_state()
        self.loss = 0
        self.mdn_components = None

    def reset_awn(self):
        """
            Reset the weights and biases
        """
        self._awn_weights = {}
        self._awn_biases = {}

    def get_awn_weight_name(self, awn_link_name):
        """
            Get the link name containing the weight
            Args:
                awn_link_name (string): name of the awn link corresponding to the weight
            Returns:
                (string)
        """
        name = awn_link_name[4:] if awn_link_name[:4] == "awn_" else awn_link_name
        return "awn_" + name + "_weight"

    def get_awn_bias_name(self, awn_link_name):
        """
            Get the link name containing the bias
            Args:
                awn_link_name (string): name of the awn link corresponding to the bias
            Returns:
                (string)
        """
        name = awn_link_name[4:] if awn_link_name[:4] == "awn_" else awn_link_name
        return "awn_" + name + "_bias"

    def get_awn_weight(self, awn_link_name):
        """
            Get the generated weight
            Args:
                awn_link_name (string): name of the awn link corresponding to the weight
            Returns:
                (link.Link)
        """
        name = self.get_awn_weight_name(awn_link_name)
        return self._awn_weights[name]

    def get_awn_bias(self, awn_link_name):
        """
            Get the generated bias
            Args:
                awn_link_name (string): name of the awn link corresponding to the weight
            Returns:
                (link.Link)
        """
        name = self.get_awn_bias_name(awn_link_name)
        return self._awn_biases[name]

    def __call__(self, data, cs_data, n_batches):
        """
            Perform the handwriting prediction

            Args:
                inputs (float[][]): a tensor containing the positions (X), char sequence (cs)
            Returns:
                loss (float)
        """
        batch_size, t_max, x_dim = data.shape

        # Create the one-hot encoding
        cs = variable.Variable(self.xp.asarray(cs_data))

        # Helper to call Linear with AdaptiveWeightNoise
        def awn_op(name, x):
            weight_name = self.get_awn_weight_name(name)
            if weight_name not in self._awn_weights:
                bias_name = self.get_awn_bias_name(name)

                if chainer.config.train:
                    W, b, _ = self["awn_" + name](n_batches, (x.size // x.shape[0])+1)
                else:
                    W, b = self["awn_" + name].get_test_weight((x.size // x.shape[0])+1)

                self._awn_weights[weight_name] = W
                self._awn_biases[bias_name] = b

            W = self.get_awn_weight(name)
            b = self.get_awn_bias(name)

            return self[name](x, W, b)

        # Train all samples in batch
        loss = 0
        sw_lstm1 = None
        for t in xrange(t_max-1):
            x_now = variable.Variable(self.xp.asarray(data[:, t, :], dtype=self.xp.float32))
            x_next = variable.Variable(self.xp.asarray(data[:, t+1, :], dtype=self.xp.float32))

            # LSTM1
            #x_lstm1 = self.x_lstm1(x_now)
            x_lstm1 = awn_op("x_lstm1", x_now)
            if sw_lstm1 is not None:
                x_lstm1 += sw_lstm1
            #h1 = self.lstm1(x_lstm1)
            h1 = awn_op("lstm1", x_lstm1)
            
            # Attention Mechanism
            #h1_sw = self.lstm1_sw(h1)
            h1_sw = awn_op("lstm1_sw", h1)
            sw = self.sw([h1_sw, cs])
            #sw_lstm1 = self.sw_lstm1(sw)
            sw_lstm1 = awn_op("sw_lstm1", sw)

            # LSTM2
            #x_lstm2 = self.x_lstm2(x_now)
            #x_lstm2 += self.lstm1_lstm2(h1)
            #x_lstm2 += self.sw_lstm2(sw)
            #h2 = self.lstm2(x_lstm2)
            x_lstm2 = awn_op("x_lstm2", x_now)
            x_lstm2 += awn_op("lstm1_lstm2", h1)
            x_lstm2 += awn_op("sw_lstm2", sw)
            h2 = awn_op("lstm2", x_lstm2)

            # LSTM3
            #x_lstm3 = self.x_lstm3(x_now)
            #x_lstm3 += self.lstm2_lstm3(h2)
            #x_lstm3 += self.sw_lstm3(sw)
            #h3 = self.lstm3(x_lstm3)
            x_lstm3 = awn_op("x_lstm3", x_now)
            x_lstm3 += awn_op("lstm2_lstm3", h2)
            x_lstm3 += awn_op("sw_lstm3", sw)
            h3 = awn_op("lstm3", x_lstm3)

            # MDN
            #y = self.h1_mdn(h1)
            #y += self.h2_mdn(h2)
            #y += self.h3_mdn(h3)
            y = awn_op("h1_mdn", h1)
            y += awn_op("h2_mdn", h2)
            y += awn_op("h3_mdn", h3)
            loss += F.average(self.mdn([x_next, y]))

            # Store the mdn components 
            if self.mdn_components is None:
                self.mdn_components = self.xp.zeros((batch_size, t_max-1, self.n_mixture_components * 5 + self.n_mixture_components + 3))

            self.mdn_components[0:batch_size, t, 0:3] = self.mdn.q.data
            self.mdn_components[0:batch_size, t, 3:(self.n_mixture_components+3)] = self.mdn.pi.data
            self.mdn_components[0:batch_size, t, (1*self.n_mixture_components+3):(2*self.n_mixture_components+3)] = self.mdn.mu_x1.data
            self.mdn_components[0:batch_size, t, (2*self.n_mixture_components+3):(3*self.n_mixture_components+3)] = self.mdn.mu_x2.data
            self.mdn_components[0:batch_size, t, (3*self.n_mixture_components+3):(4*self.n_mixture_components+3)] = self.mdn.s_x1.data
            self.mdn_components[0:batch_size, t, (4*self.n_mixture_components+3):(5*self.n_mixture_components+3)] = self.mdn.s_x2.data
            self.mdn_components[0:batch_size, t, (5*self.n_mixture_components+3):(6*self.n_mixture_components+3)] = self.mdn.rho.data

        self.loss = loss

        return self.loss

    def sample(self, data, cs_data, n_batches, t_prime_len=0):
        """
            Sample from the trained model

            Args:
                inputs (float[][]): a tensor containing the positions (X), char sequence (cs)
            Returns:
                loss (float)
        """
        batch_size, t_max, x_dim = data.shape
        # @TODO: Support more than 1 batch_size

        # Create the strokes array
        def gen_strokes(p_s, n_s):
            arr = self.xp.zeros((1, 2, INPUT_SIZE)).astype(self.xp.float32)
            arr[0][0] = p_s
            arr[0][1] = n_s
            return arr

        # Get best position from pdf
        def get_pdf_idx(x, pdf):
            acc = 0
            for i in xrange(0, pdf.size):
                acc += pdf[i]
                if acc >= x:
                    return i
            
            raise ValueError("Unable to sample from the pdf")

        # Start the drawing at (0, 0) or the priming data
        if t_prime_len != 0:
            x_data = gen_strokes(data[0][0], data[0][1])
        else:
            x_data = gen_strokes(
                self.xp.zeros((1, INPUT_SIZE)).astype(self.xp.float32),
                self.xp.asarray([[0, 0, 1, 0, 0]]).astype(self.xp.float32)
            )

        # @TODO: should dynamically stop the drawing
        loss_network = self.xp.zeros((t_max-1, 1))
        all_mdn_components = self.xp.zeros((t_max-1, self.n_mixture_components * 5 + self.n_mixture_components + 3))
        strokes = self.xp.zeros((t_max-1, INPUT_SIZE))
        print(t_max-1, t_prime_len-1)
        for t in xrange(t_max-1):
            # Prediction
            loss = self(x_data, cs_data, n_batches)
            loss_network[t] = loss.data

            if t < t_prime_len-2:
                stroke = x_data[0][1] # "Current Next"
                x_data = gen_strokes(data[0][t+1], data[0][t+2])
            else:
                # Generate the next potential prediction
                q = self.mdn_components[0:1, 0, 0:3][0]
                pi = self.mdn_components[0:1, 0, 3:(self.n_mixture_components+3)][0]
                mu_x1 = self.mdn_components[0:1, 0, (1*self.n_mixture_components+3):(2*self.n_mixture_components+3)][0]
                mu_x2 = self.mdn_components[0:1, 0, (2*self.n_mixture_components+3):(3*self.n_mixture_components+3)][0]
                s_x1 = self.mdn_components[0:1, 0, (3*self.n_mixture_components+3):(4*self.n_mixture_components+3)][0]
                s_x2 = self.mdn_components[0:1, 0, (4*self.n_mixture_components+3):(5*self.n_mixture_components+3)][0]
                rho = self.mdn_components[0:1, 0, (5*self.n_mixture_components+3):(6*self.n_mixture_components+3)][0]

                #idx_pos = pi.argmax()
                idx_pos = get_pdf_idx(random.random(), pi)
                eos_pos = get_pdf_idx(random.random(), q)
                mean = self.xp.asarray([mu_x1[idx_pos], mu_x2[idx_pos]])
                cov = self.xp.asarray([
                    [s_x1[idx_pos]*s_x1[idx_pos], rho[idx_pos]*s_x1[idx_pos]*s_x2[idx_pos]],
                    [rho[idx_pos]*s_x1[idx_pos]*s_x2[idx_pos], s_x2[idx_pos]*s_x2[idx_pos]]
                ])
                x = self.xp.random.multivariate_normal(mean, cov, 1)
                x1_pred, x2_pred = x[0][0], x[0][1]

                eos_pred = self.xp.asarray([0,0,0]).astype(self.xp.float32)
                eos_pred[eos_pos] = 1.

                stroke = self.xp.concatenate((self.xp.asarray([x1_pred, x2_pred]), eos_pred)).astype(self.xp.float32)
                x_data = gen_strokes(
                    stroke, # Current
                    self.xp.asarray([[0, 0, 0, 0, 0]]).astype(self.xp.float32) # Next
                )

            # Store the information
            strokes[t] = stroke
            all_mdn_components[t:(t+1), :] = self.mdn_components[0:1]
            self.mdn_components = None

        return self.xp.sum(loss_network) / (batch_size * t_max-1), strokes, all_mdn_components

# ===============================================
# Main entry point of the training process (main)
# ===============================================


def main(data_dir, output_dir, batch_size, peephole, epochs, grad_clip, resume_dir, resume_model, resume_optimizer, resume_stats, gpu, adaptive_noise, update_weight, use_weight_noise, save_interval, validation_interval, truncated_back_prop_len, truncated_data_samples, rnn_layers_number, rnn_cells_number, win_unit_number, mix_comp_number, random_seed, learning_rate, debug):
    """ Save the args for this run """
    arguments = {}
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for i in args:
        arguments[i] = values[i]

    # Snapshot directory
    model_suffix_dir = "{0}-{1}-{2}".format(time.strftime("%Y%m%d-%H%M%S"), 'with_peephole' if peephole == 1 else 'no_peephole', batch_size)
    training_suffix = "{0}".format("training")
    state_suffix = "{0}".format("state")

    save_dir = output_dir + '/' + model_suffix_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    """ Setup the global logger """
    logger = logging.getLogger()
    logFormatter = logging.Formatter(log_fmt)

    # File logger
    fh = logging.FileHandler("{0}/logs.txt".format(save_dir))
    fh.setFormatter(logFormatter)
    logger.addHandler(fh)

    # stdout logger
    #ch = logging.StreamHandler(sys.stdout)
    #ch.setFormatter(logFormatter)
    #logger.addHandler(ch)

    """ Train the model based on the data saved in ../processed """
    logger.info("Run arguments")
    logger.info(arguments)
    logger.info('Training the model')
    logger.info('Model: {}'.format('with peephole' if peephole == 1 else 'no peephole'))
    logger.info('GPU: {}'.format(gpu))
    logger.info('Mini-batch size: {}'.format(batch_size))
    logger.info('# epochs: {}'.format(epochs))

    """ Chainer's debug mode """
    if debug == 1:
        logger.info("Enabling Chainer's debug mode")
        chainer.config.debug = True

    """ Fetching the model and the inputs """
    def load_data(path):
        f = open(path, "rb")
        data = pickle.load(f)
        f.close()
        return data

    logger.info("Fetching the model and the inputs")
    train_data = load_data(data_dir + "/train/train_data")
    train_characters = load_data(data_dir + "/train/train_characters")
    valid_data = load_data(data_dir + "/valid/valid_data")
    valid_characters = load_data(data_dir + "/valid/valid_characters")
    vocab = load_data(data_dir + "/vocabulary")

    # Each position time step is composed of
    # deltaX, deltaY, p1, p2, p3
    # deltaX, deltaY is the pen position's offsets
    # p1 = pen is touching paper
    # p2 = pen will be lifted from the paper (don't draw next stroke)
    # p3 = handwriting is completed
    train_data = get_expanded_stroke_position(train_data)
    valid_data = get_expanded_stroke_position(valid_data)

    np.save("three_to_five.npy", train_data)
    exit()

    n_max_seq_length = max(get_max_sequence_length(train_characters), get_max_sequence_length(valid_characters))
    n_chars = len(vocab)
    history_network_train = []
    history_network_valid = []
    offset_epoch = 0

    """ Create the model """
    logger.info("Creating the model")
    if peephole == 0:
        model = Model(rnn_layers_number, rnn_cells_number, mix_comp_number, win_unit_number)    
    #else:
        #model = ModelPeephole(n_chars, INPUT_SIZE, win_unit_number, rnn_cells_number, mix_comp_number, rnn_layers_number)

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
    #optimizer = chainer.optimizers.Adam(alpha=learning_rate)
    optimizer = chainer.optimizers.RMSpropGraves()
    optimizer.setup(model)

    if grad_clip is not 0:
        optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

    if resume_dir:
        # Resume model and optimizer
        logger.info("Loading state from {}".format(output_dir + '/' + resume_dir))
        if resume_model != "":
            chainer.serializers.load_npz(output_dir + "/" + resume_dir + "/" + resume_model, model)
        if resume_optimizer != "":
            chainer.serializers.load_npz(output_dir + "/" + resume_dir + "/" + resume_optimizer, optimizer)
        # Resume statistics
        if resume_stats == 1:
            history_network_train = list(np.load(output_dir + "/" + resume_dir + "/history-network-train.npy"))
            history_network_valid = list(np.load(output_dir + "/" + resume_dir + "/history-network-valid.npy"))
            offset_epoch = len(history_network_train)

    """ Prepare data """
    sets = []
    for i in [[train_data, train_characters], [valid_data, valid_characters]]:
        data, characters = i
        grouped = group_data(data, characters)
        sets.append(list(grouped))

    train_set, valid_set = sets

    """ Create the data iterators """
    train_iter = chainer.iterators.SerialIterator(train_set, batch_size, repeat=True, shuffle=True)
    valid_iter = chainer.iterators.SerialIterator(valid_set, batch_size, repeat=True, shuffle=True)

    """ Start training """
    xp.random.seed(random_seed)
    np.random.seed(random_seed)
    time_epoch_start = time.time()
    history_train = []
    history_valid = []
    n_batches = None

    epoch_batch = math.ceil(len(train_set)/batch_size)
    logger.info("Starting training with {0} mini-batches for {1} epochs".format(epoch_batch, epochs))
    while (train_iter.epoch+offset_epoch) < epochs:
        epoch = train_iter.epoch + offset_epoch
        batch = np.asarray(train_iter.next())
        time_iteration_start = time.time()

        # For AdaptiveWeightNoise, keep track of the number of batches
        if n_batches is None:
            n_batches = variable.Variable(xp.asarray(xp.zeros(1)+epoch_batch).astype(xp.float32))
        
        # Unpack the training data
        train_data_batch, train_characters_batch = pad_data(batch[:, 0], batch[:, 1])
        cs_data = one_hot(train_data_batch, train_characters, n_chars, n_max_seq_length)

        # Train the batch
        model.cleargrads()
        model.reset_awn()
        loss_t = model(train_data_batch, cs_data, n_batches)

        # Truncated back-prop at each time-step
        #model.cleargrads()
        loss_t.backward()
        loss_t.unchain_backward()
        optimizer.update()

        time_iteration_end = time.time()-time_iteration_start
        #loss = cuda.to_cpu(model.loss.data)
        loss = cuda.to_cpu(loss_t.data)
        history_train.append([loss, time_iteration_end])
        model.reset_state()
        logger.info("[TRAIN] Epoch #{0} ({1}/{2}): loss = {3}, time = {4}".format(epoch+1, len(history_train), epoch_batch, loss, time_iteration_end))

        # Check of epoch is completed: all the mini-batches are completed
        if train_iter.is_new_epoch:
            history_train = np.asarray(history_train)
            train_mean = history_train[:, 0].mean()
            train_std = history_train[:, 0].std()
            train_min = history_train[:, 0].min()
            train_max = history_train[:, 0].max()
            train_med = np.median(history_train)
            train_time_sum = history_train[:, 1].sum()
            history_network_train.append([train_mean, train_std, train_min, train_max, train_med, train_time_sum])
            logger.info("[TRAIN] Epoch #{0} (COMPLETED IN {1}): mean = {2}, std = {3}, min = {4}, max = {5}, med = {6}"
                        .format(epoch+1, train_time_sum, train_mean, train_std, train_min, train_max, train_med))
 
            # Check if we should validate the data
            if epoch % validation_interval == 0:
                with chainer.using_config('train', False) and chainer.no_backprop_mode():
                    while True:
                    #for valid_batch in valid_iter:
                        time_iteration_start = time.time()

                        # Unpack the validation data
                        valid_batch = np.asarray(valid_iter.next())
                        valid_data_batch, valid_characters_batch = pad_data(valid_batch[:, 0], valid_batch[:, 1])
                        valid_cs_data = one_hot(valid_data_batch, valid_characters, n_chars, n_max_seq_length)

                        # Train the batch
                        model(valid_data_batch, valid_cs_data, n_batches)
                        time_iteration_end = time.time()-time_iteration_start
                        loss = cuda.to_cpu(model.loss.data)
                        history_valid.append([loss, time_iteration_end])
                        logger.info("[VALID] Epoch #{0} ({1}/{2}): loss = {3}, time = {4}".format(epoch+1, len(history_valid), math.ceil(len(valid_set)/batch_size), loss, time_iteration_end))

                        # MSE between the prediction and the ground truth
                        # MSE for (deltaX, deltaY) and eos components
                        #mdn_outputs = model.mdn_outputs
                        #pi_o = np.vstack(mdn_outputs[:, 1])
                        #mu_x_o = np.vstack(mdn_outputs[:, 2])
                        #mu_y_o = np.vstack(mdn_outputs[:, 3])
                        #eos_o = np.vstack(mdn_outputs[:, 0])

                        #max_pi = [x.argmax() for x in pi_o]
                        #mu_x = [mu_x_o[i][max_pi[i]] for i in xrange(len(mu_x_o))]
                        #mu_y = [mu_y_o[i][max_pi[i]] for i in xrange(len(mu_y_o))]
                        #mse_pos = mean_squared_error(valid_data_batch[:, 0:2], np.vstack([[mu_x[i], mu_y[i]] for i in xrange(len(mu_x))]))
                        #mse_eos_p1 = mean_squared_error(valid_data[:, 2], eos_o[:, 0])
                        #mse_eos_p2 = mean_squared_error(valid_data[:, 3], eos_o[:, 1])
                        #mse_eos_p3 = mean_squared_error(valid_data[:, 4], eos_o[:, 2])

                        model.reset_state()
                        if valid_iter.is_new_epoch:
                            break

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
                                .format(epoch+1, valid_time_sum, valid_mean, valid_std, valid_min, valid_max, valid_med))

                    valid_iter.reset()

            # Check if we should save the data
            if epoch % save_interval == 0:
                logger.info("Saving the model, optimizer and history")

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Save the model and optimizer
                chainer.serializers.save_npz(save_dir + '/model-{}'.format(epoch+1), model)
                chainer.serializers.save_npz(save_dir + '/state-{}'.format(epoch+1), optimizer)

                # Save the stats
                np.save(save_dir + '/history-network-train', np.asarray(history_network_train))
                np.save(save_dir + '/history-network-valid', np.asarray(history_network_valid))

            # Reset global epochs vars
            history_train = []
            history_valid = []
            n_batches = None



# ===============
# CLI Entry point
# ===============


@click.command()
@click.option('--data_dir', type=click.Path(exists=True), default='data/processed/OnlineHandWriting', help='Directory containing the data.')
@click.option('--output_dir', type=click.Path(exists=False), default='models', help='Directory for model checkpoints.')
@click.option('--batch_size', type=click.INT, default=64, help='Size of the mini-batches.')
@click.option('--peephole', type=click.INT, default=0, help='LSTM with Peephole.')
@click.option('--epochs', type=click.INT, default=500, help='Number of epoch for training.')
@click.option('--grad_clip', type=click.INT, default=0, help='Threshold for the gradient clipping.')
@click.option('--resume_dir', type=click.STRING, default='', help='Directory name for resuming the optimization from a snapshot.')
@click.option('--resume_model', type=click.STRING, default='', help='Name of the model snapshot.')
@click.option('--resume_optimizer', type=click.STRING, default='', help='Name of the optimizer snapshot.')
@click.option('--resume_stats', type=click.INT, default=0, help='Resume the statistics (new stats will be combined with old statistics).')
@click.option('--gpu', type=click.INT, default=-1, help='GPU ID (negative value is CPU).')
@click.option('--adaptive_noise', type=click.INT, default=1, help='Use Adaptive Weight Noise in the training process.')
@click.option('--update_weight', type=click.INT, default=1, help='Update weights in the training process.')
@click.option('--use_weight_noise', type=click.INT, default=1, help='Use weight noise in the training process.')
@click.option('--save_interval', type=click.INT, default=1, help='How often the model should be saved.')
@click.option('--validation_interval', type=click.INT, default=1, help='How often the model should be validated.')
@click.option('--truncated_back_prop_len', type=click.INT, default=50, help='Number of backpropagation before stopping.')
@click.option('--truncated_data_samples', type=click.INT, default=500, help='Number of samples to use inside a data.')
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
