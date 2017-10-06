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

from links.connection.lstm import LSTM
from links.connection.linear import Linear
from links.connection.linear_lstm import LinearLSTM
from links.connection.linear_layer_normalization_lstm import LinearLayerNormalizationLSTM
from links.connection.adaptive_weight_noise import AdaptiveWeightNoise
from links.connection.mixture_density_network import MixtureDensityNetwork
from links.connection.soft_window import SoftWindow
from links.normalization.layer_normalization import LayerNormalization

from utils import get_max_sequence_length
from utils import group_data
from utils import pad_data
from utils import one_hot

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

# ==================
# Initializer (init)
# ==================

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
            #self.lstm1 = LinearLSTM(n_units)
            #self.lstm2 = LinearLSTM(n_units)
            #self.lstm3 = LinearLSTM(n_units)
            self.lstm1 = LinearLayerNormalizationLSTM(n_units, forget_bias_init = 1., norm_bias_init = 1., norm_gain_init = 0.)
            self.lstm2 = LinearLayerNormalizationLSTM(n_units, forget_bias_init = 1., norm_bias_init = 1., norm_gain_init = 0.)
            self.lstm3 = LinearLayerNormalizationLSTM(n_units, forget_bias_init = 1., norm_bias_init = 1., norm_gain_init = 0.)
            
            # Attention mechanism
            self.sw = SoftWindow(n_window_unit, n_units)

            # Mixture Density Network
            self.mdn = MixtureDensityNetwork(n_mixture_components, n_units, prob_bias)

            # Linear connections for some layers
            self.x_lstm1 = Linear(n_layers, n_units*4, no_bias=True)
            self.x_lstm2 = Linear(n_layers, n_units*4, no_bias=True)
            self.x_lstm3 = Linear(n_layers, n_units*4, no_bias=True)
            self.lstm1_lstm2 = Linear(n_units, n_units * 4)
            self.lstm2_lstm3 = Linear(n_units, n_units * 4)
            self.lstm1_sw = Linear(n_units, n_window_unit * 3)
            # Input shape (here None) is the length of the vocabulary: it's dynamic
            self.sw_lstm1 = Linear(None, n_units * 4)
            self.sw_lstm2 = Linear(None, n_units * 4)
            self.sw_lstm3 = Linear(None, n_units * 4)
            self.h1_mdn = Linear(n_units, 1 + n_mixture_components * 6)
            self.h2_mdn = Linear(n_units, 1 + n_mixture_components * 6)
            self.h3_mdn = Linear(n_units, 1 + n_mixture_components * 6)

            # Noise for the linear connections
            self.awn_x_lstm1 = AdaptiveWeightNoise(n_layers, n_units*4, no_bias=True)
            self.awn_x_lstm2 = AdaptiveWeightNoise(n_layers, n_units*4, no_bias=True)
            self.awn_x_lstm3 = AdaptiveWeightNoise(n_layers, n_units*4, no_bias=True)
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
            self.awn_h1_mdn = AdaptiveWeightNoise(n_units+1, 1 + n_mixture_components * 6)
            self.awn_h2_mdn = AdaptiveWeightNoise(n_units+1, 1 + n_mixture_components * 6)
            self.awn_h3_mdn = AdaptiveWeightNoise(n_units+1, 1 + n_mixture_components * 6)

        self.n_units = n_units
        self.n_mixture_components = n_mixture_components
        self.n_window_unit = n_window_unit
        self.p_bias = prob_bias
        self.mdn_components = None
        self._awn_weights = {}
        self._awn_biases = {}

        self.loss = 0
        self.loss_complex = None

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
        self.loss_complex = None
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

                loss_complex = None
                if chainer.config.train:
                    W, b, loss_complex = self["awn_" + name](n_batches, (x.size // x.shape[0])+1)
                else:
                    W, b = self["awn_" + name].get_test_weight((x.size // x.shape[0])+1)

                if loss_complex is not None:
                    loss_complex = F.reshape(loss_complex, (1,1))
                    self.loss_complex = loss_complex if self.loss_complex is None else F.concat((self.loss_complex, loss_complex))

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
            loss += F.sum(self.mdn([x_next, y])) / batch_size

            # Store the mdn components 
            if self.mdn_components is None:
                self.mdn_components = self.xp.zeros((batch_size, t_max-1, 1 + 6 * self.n_mixture_components))

            self.mdn_components[0:batch_size, t, 0:1] = self.mdn.eos.data
            self.mdn_components[0:batch_size, t, 1:(self.n_mixture_components+1)] = self.mdn.pi.data
            self.mdn_components[0:batch_size, t, (1*self.n_mixture_components+1):(2*self.n_mixture_components+1)] = self.mdn.mu_x1.data
            self.mdn_components[0:batch_size, t, (2*self.n_mixture_components+1):(3*self.n_mixture_components+1)] = self.mdn.mu_x2.data
            self.mdn_components[0:batch_size, t, (3*self.n_mixture_components+1):(4*self.n_mixture_components+1)] = self.mdn.s_x1.data
            self.mdn_components[0:batch_size, t, (4*self.n_mixture_components+1):(5*self.n_mixture_components+1)] = self.mdn.s_x2.data
            self.mdn_components[0:batch_size, t, (5*self.n_mixture_components+1):(6*self.n_mixture_components+1)] = self.mdn.rho.data


        #loss = (batch_size * t_max)
        #self.loss = loss = F.sum(loss) / (batch_size * t_max)
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
            arr = self.xp.zeros((1, 2, 3)).astype(self.xp.float32)
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
                self.xp.asarray([[0, 0, 1]]).astype(self.xp.float32)
            )

        # @TODO: should dynamically stop the drawing
        loss_network = self.xp.zeros((t_max, 1))
        all_mdn_components = self.xp.zeros((t_max, 1 + 6 * self.n_mixture_components))
        strokes = self.xp.zeros((t_max, INPUT_SIZE))
        for t in xrange(t_max-1):
            # Prediction
            loss = self(x_data, cs_data, n_batches)
            loss_network[t] = loss.data

            if t < t_prime_len-2:
                print("Prime: {0}/{1}".format(t, t_prime_len-2))
                stroke = x_data[0][1] # "Current Next"
                x_data = gen_strokes(data[0][t+1], data[0][t+2])
            else:
                # Generate the next potential prediction
                eos = self.mdn_components[0:1, 0, 0:1][0]
                pi = self.mdn_components[0:1, 0, 1:(self.n_mixture_components+1)][0]
                mu_x1 = self.mdn_components[0:1, 0, (1*self.n_mixture_components+1):(2*self.n_mixture_components+1)][0]
                mu_x2 = self.mdn_components[0:1, 0, (2*self.n_mixture_components+1):(3*self.n_mixture_components+1)][0]
                s_x1 = self.mdn_components[0:1, 0, (3*self.n_mixture_components+1):(4*self.n_mixture_components+1)][0]
                s_x2 = self.mdn_components[0:1, 0, (4*self.n_mixture_components+1):(5*self.n_mixture_components+1)][0]
                rho = self.mdn_components[0:1, 0, (5*self.n_mixture_components+1):(6*self.n_mixture_components+1)][0]

                #idx_pos = pi.argmax()
                idx_pos = get_pdf_idx(random.random(), pi)
                mean = self.xp.asarray([mu_x1[idx_pos], mu_x2[idx_pos]])
                cov = self.xp.asarray([
                    [s_x1[idx_pos]*s_x1[idx_pos], rho[idx_pos]*s_x1[idx_pos]*s_x2[idx_pos]],
                    [rho[idx_pos]*s_x1[idx_pos]*s_x2[idx_pos], s_x2[idx_pos]*s_x2[idx_pos]]
                ])
                x = self.xp.random.multivariate_normal(mean, cov, 1)
                x1_pred, x2_pred = x[0][0], x[0][1]
                eos_pred = 1.0 if eos[0] > 0.10 else 0.0

                stroke = self.xp.asarray([[x1_pred, x2_pred, eos_pred]]).astype(self.xp.float32)
                x_data = gen_strokes(
                    stroke, # Current
                    self.xp.asarray([[0, 0, 0]]).astype(self.xp.float32) # Next
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
    optimizer = chainer.optimizers.Adam(alpha=learning_rate)
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
        loss_complex = cuda.to_cpu(model.loss_complex.data)
        history_train.append([loss, time_iteration_end] + list(loss_complex.flatten()))
        model.reset_state()
        logger.info("[TRAIN] Epoch #{0} ({1}/{2}): loss = {3}, time = {4}".format(epoch+1, len(history_train), epoch_batch, loss, time_iteration_end))

        # Check of epoch is completed: all the mini-batches are completed
        if train_iter.is_new_epoch:
            history_train = np.asarray(history_train)
            train_mean = history_train[:, 0].mean()
            train_std = history_train[:, 0].std()
            train_min = history_train[:, 0].min()
            train_max = history_train[:, 0].max()
            train_med = np.median(history_train[:, 0])
            train_time_sum = history_train[:, 1].sum()
            train_complex_mean = history_train[:, 2:].mean()
            train_complex_std = history_train[:, 2:].std()
            train_complex_max = history_train[:, 2:].max()
            train_complex_min = history_train[:, 2:].min()
            train_complex_med = np.median(history_train[:, 2:])
            history_network_train.append([
                train_mean, train_std, train_min, train_max, train_med, train_time_sum,
                train_complex_mean, train_complex_std, train_complex_min, train_complex_max, train_complex_med
            ])
            logger.info("[TRAIN] Epoch #{0} (COMPLETED IN {1}): mean = {2}, std = {3}, min = {4}, max = {5}, med = {6}, complex(mean) = {7}, complex(std) = {8}, complex(min) = {9}, complex(max) = {10}, complex(med) = {11}".format(epoch+1, train_time_sum, train_mean, train_std, train_min, train_max, train_med, train_complex_mean, train_complex_std, train_complex_min, train_complex_max, train_complex_med))
 
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
@click.option('--save_interval', type=click.INT, default=10, help='How often the model should be saved.')
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
