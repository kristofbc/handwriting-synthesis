#!/usr/bin/env python # -*- coding: utf-8 -*-

# Generate handwriting from a sequence of characters
# ==================================================

import time
import os
import click
import logging
import math

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

#from net.lstm import LSTM
#from net.adaptive_weight_noise import AdaptiveWeightNoise
#from net.soft_window import SoftWindow
#from net.mixture_density_outputs import MixtureDensityOutputs

from net.functions.adaptive_weight_noise import adaptive_weight_noise
from net.functions.soft_window import soft_window
from net.functions.mixture_density_outputs import mixture_density_outputs

INPUT_SIZE = 3 # (x, y, end_of_stroke)

# ==============
# Helpers (hlps)
# ==============
def load_data(path):
    f = open(path, "rb")
    data = pickle.load(f)
    f.close()
    return data

def get_max_sequence_length(sequences):
    length = []
    for i in xrange(len(sequences)):
        length.extend([sequences[i].size])
    return int(np.asarray(length).max())

def get_padded_data(train_data, train_characters, padding=True):
    max_length_data = 0
    max_length_characters = 0
    # Get the maximum length of each arrays
    tmp1 = []
    tmp2 = []
    for i in xrange(len(train_data)):
        if len(train_data[i]) > max_length_data:
            max_length_data = len(train_data[i])
        #if len(train_characters[i]) > max_length_characters:
        #    max_length_characters = len(train_characters[i])

    # Pad each arrays to be the same length
    for i in xrange(len(train_data)):
        if padding and len(train_data[i]) != max_length_data:
            pad_length = max_length_data-len(train_data[i])
            pad = np.full((pad_length, 3), 0)
            pad[:,2] = 2.0
            train_data[i] = np.vstack([train_data[i], pad])
        tmp1.append(np.asarray(train_data[i]))
        tmp2.append(np.asarray(train_characters[i]))
    
    return np.asarray(tmp1), np.asarray(tmp2)

# ================
# Functions (fcts)
# ================

# ============
# Links (lnks)
# ============
class AdaptiveWeightNoise(link.Link):
    def __init__(self, in_size, out_size, weight_scale=1, no_bias=False, use_weight_noise=True):
        super(AdaptiveWeightNoise, self).__init__()

        with self.init_scope():
            if no_bias:
                M = np.random.normal(0, weight_scale, (out_size*in_size)).astype(np.float32)
            else:
                M = np.random.normal(0, weight_scale, (out_size*(in_size-1))).astype(np.float32)
                M = M.reshape((1, out_size*(in_size-1)))
                M = np.c_[M, np.zeros((1, out_size)).astype(np.float32)]
                M = M.reshape(out_size*in_size)

            self.M = chainer.Parameter(M)
            self.logS2 = chainer.Parameter(np.log((np.ones((out_size*in_size))*1e-8).astype(np.float32)))

        self.in_size  = in_size
        self.out_size = out_size
        self.no_bias = no_bias
        self.use_weight_noise = use_weight_noise
    
    def __call__(self, batch_size):
        """
            Called the Alex Grave Adaptive Weight Noise

            Args:
                batch_size  (~chainer.Variable): 
                    (batch size) * (number of truncated backward gradient calculation for a training dataset)

            Returns:
                ~chainer.Variable: Output of the linear layer.

        """
        
        self.fWb, loss = adaptive_weight_noise(batch_size, self.M, self.logS2, self.use_weight_noise) 
        
        if self.no_bias:
            return F.reshape(self.fWb, (self.out_size, self.in_size)), loss
        else:
            self.fW, self.fb = F.split_axis(self.fWb, np.asarray([(self.in_size -1)*self.out_size]), axis=0)
            return F.reshape(self.fW, (self.out_size, self.in_size -1)), self.fb, loss

class SoftWindow(link.Link):
    """
        SoftWindow layer.
        This is a SoftWindow layer as a chain. 
        This is a link that wraps the :func:`~chainer.functions.mixturedensityoutputs` function,
        and holds a weight matrix ``W`` and optionally a bias vector ``b`` as parameters.

        Args:
            mix_size (int): number of mixture components.
    """
    
    def __init__(self, mix_size):
        super(SoftWindow, self).__init__()

        self.mix_size = mix_size
        self.ws = None
        self.eow = None
        self.reset_state()
        
    def reset_state(self):
        self.k_prev = None
            
    def __call__(self, cs, ls, h, W, b):
        """
            cs   :   one-hot-encoding of a length U character sequence 
            h    :   input vector (summation of affine transformation of outputs from hidden layers)
        """
        y = F.linear(h, W, b)
        
        a_hat, b_hat, k_hat =  F.split_axis(y, np.asarray([self.mix_size, 2*self.mix_size]), axis=1) 
        
        if self.k_prev is None:
            self.k_prev = variable.Variable(self.xp.zeros_like(k_hat.data))
            
        self.ws, self.k_prev, self.eow = soft_window(cs, ls, a_hat, b_hat, k_hat, self.k_prev)
        return self.ws, self.eow

class MixtureDensityOutputs(link.Link):
    """
        Mixture-Density-Outputs layer.

        This is a Mixture-Density-Outputs layer as a chain. 
        This is a link that wraps the :func:`~chainer.functions.mixturedensityoutputs` function,
        and holds a weight matrix ``W`` and optionally a bias vector ``b`` as parameters.
    
        Args:
            mix_size (int): number of mixture components.
    """
    
    def __init__(self, mix_size):
        super(MixtureDensityOutputs, self).__init__()

        self.loss = None
        self.xpred = None
        self.eos = None
        self.pi = None
        self.mux = None
        self.muy = None
        self.sgx = None
        self.sgy = None
        self.rho = None
        self.mix_size = mix_size
    
    def __call__(self, xnext, eow, h1, h2, h3, W1, W2, W3, b1, b2, b3, prob_bias):
        """
            xnext   :   next state of a pen. ndim=(batchsize,3)
            h       :   input vector 
            W1, W2, W3: (h.shape[1], 1 + mix_size * 6)
            b1, b2, b3: (1, 1 + mix_size * 6)
            prob_bias:  probability bias
        """
        
        y  = F.linear(h1, W1, b1)
        y += F.linear(h2, W2, b2)
        y += F.linear(h3, W3, b3) 
        
        eos_hat, pi_hat, mu1_hat, mu2_hat, sg1_hat, sg2_hat, rho_hat = F.split_axis(
                y,
                np.asarray([1, 1+self.mix_size, 1+2*self.mix_size, 
                            1+3*self.mix_size, 1+4*self.mix_size, 1+5*self.mix_size]), axis=1)
        
        self.loss, self.xpred, self.eos, self.pi, self.mux, self.muy, self.sgx, self.sgy, self.rho = mixture_density_outputs(
            xnext, eow, eos_hat, pi_hat * (1. + prob_bias), mu1_hat, mu2_hat, sg1_hat - prob_bias, sg2_hat - prob_bias, rho_hat
        )
        
        return self.loss, self.xpred, self.eos, self.pi, self.mux, self.muy, self.sgx, self.sgy, self.rho
        

class LSTM(link.Link):
    """
        LSTM
        A wrapper around the cell states

        Args:
            out_size (int): number of cells inside the LSTM
    """
    def __init__(self, out_size):
        super(LSTM, self).__init__()
        self.state_size = out_size
        self.reset_state()

    def reset_state(self):
        self.c = self.h = None

    def __call__(self, x, W_lateral, b_lateral):
        lstm_in = x
        if self.h is not None:
            lstm_in += F.linear(self.h, W_lateral, b_lateral)
        if self.c is None:
            self.c = variable.Variable(self.xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype))

        self.c, self.h = F.lstm(self.c, lstm_in)
        return self.h
        

# =============
# Models (mdls)
# =============

class Model(chainer.Chain):
    """
        Handwriting Synthesis Model
    """
    def __init__(self, n_chars, input_size, win_unit_number, rnn_cells_number, mix_comp_number, rnn_layers_number, weight_scale=0.1, use_adaptive_noise=True):
        super(Model, self).__init__()

        with self.init_scope():
           self.awn_x_l1 = AdaptiveWeightNoise((input_size+1), 4*rnn_cells_number, weight_scale, no_bias=False, use_weight_noise=use_adaptive_noise)
           self.awn_x_l2 = AdaptiveWeightNoise((input_size+1), 4*rnn_cells_number, weight_scale, no_bias=False, use_weight_noise=use_adaptive_noise)
           self.awn_x_l3 = AdaptiveWeightNoise((input_size+1), 4*rnn_cells_number, weight_scale, no_bias=False, use_weight_noise=use_adaptive_noise)

           self.awn_l1_ws = AdaptiveWeightNoise((rnn_cells_number+1), (3*win_unit_number), weight_scale, no_bias=False, use_weight_noise=use_adaptive_noise)

           self.awn_ws_l1 = AdaptiveWeightNoise((n_chars+1), (4*rnn_cells_number), weight_scale, no_bias=False, use_weight_noise=use_adaptive_noise)
           self.awn_ws_l2 = AdaptiveWeightNoise((n_chars+1), (4*rnn_cells_number), weight_scale, no_bias=False, use_weight_noise=use_adaptive_noise)
           self.awn_ws_l3 = AdaptiveWeightNoise((n_chars+1), (4*rnn_cells_number), weight_scale, no_bias=False, use_weight_noise=use_adaptive_noise)

           self.awn_l1_l1 = AdaptiveWeightNoise((rnn_cells_number+1), (4*rnn_cells_number), weight_scale, no_bias=False, use_weight_noise=use_adaptive_noise)
           self.awn_l2_l2 = AdaptiveWeightNoise((rnn_cells_number+1), (4*rnn_cells_number), weight_scale, no_bias=False, use_weight_noise=use_adaptive_noise)
           self.awn_l3_l3 = AdaptiveWeightNoise((rnn_cells_number+1), (4*rnn_cells_number), weight_scale, no_bias=False, use_weight_noise=use_adaptive_noise)
           self.awn_l1_l2 = AdaptiveWeightNoise((rnn_cells_number+1), (4*rnn_cells_number), weight_scale, no_bias=False, use_weight_noise=use_adaptive_noise)
           self.awn_l2_l3 = AdaptiveWeightNoise((rnn_cells_number+1), (4*rnn_cells_number), weight_scale, no_bias=False, use_weight_noise=use_adaptive_noise)

           self.awn_l1_ms = AdaptiveWeightNoise((rnn_cells_number+1), (1+mix_comp_number*6), weight_scale, no_bias=False, use_weight_noise=use_adaptive_noise)
           self.awn_l2_ms = AdaptiveWeightNoise((rnn_cells_number+1), (1+mix_comp_number*6), weight_scale, no_bias=False, use_weight_noise=use_adaptive_noise)
           self.awn_l3_ms = AdaptiveWeightNoise((rnn_cells_number+1), (1+mix_comp_number*6), weight_scale, no_bias=False, use_weight_noise=use_adaptive_noise)

           self.ws = SoftWindow(win_unit_number)

           self.l1 = LSTM(rnn_cells_number)
           self.l2 = LSTM(rnn_cells_number)
           self.l3 = LSTM(rnn_cells_number)

           self.ms = MixtureDensityOutputs(mix_comp_number)

        self.win_unit_number = win_unit_number
        self.rnn_cells_number = rnn_layers_number
        self.mix_comp_number = mix_comp_number
        self.n_chars = n_chars
        self.use_adaptive_noise = use_adaptive_noise
        self._awn_weights = {}
        self._awn_biases = {}
        self.eow = None

        self.reset_state()

    def reset_state(self):
        self.ws.reset_state()
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()

        self.ws_output = None
        self.ws_to_l1 = None
        self.loss_complex = None

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

    def get_awn_weight_link(self, awn_link_name):
        """
            Get the link containing the weight

            Args:
                awn_link_name (string): name of the awn link corresponding to the weight
            Returns:
                (link.Link)
        """
        name = self.get_awn_weight_name(awn_link_name)
        return self._awn_weights[name]

    def get_awn_bias_link(self, awn_link_name):
        """
            Get the link containing the bias

            Args:
                awn_link_name (string): name of the awn link corresponding to the weight
            Returns:
                (link.Link)
        """
        name = self.get_awn_bias_name(awn_link_name)
        return self._awn_biases[name]


    def __call__(self, x, renew_weights=True):
        """ Unpack and configure the network """
        x_now, x_next, cs, ls, prob_bias, n_batches = x
        testing = not chainer.config.train

        # Use CuPY if available
        xp = cuda.get_array_module(*x_now.data)

        # Generate the weight and bias for adaptive weight noise
        if renew_weights:
            for child_link in self.children():
                if child_link.name[:3] != "awn":
                    continue
                # The weight and bias adds _weight and _bias to the child_link name.
                # E.g: awn_x_l1 => awn_x_l1_weight, awn_x_l1_bias
                weight_name = self.get_awn_weight_name(child_link.name)
                bias_name = self.get_awn_bias_name(child_link.name)
                if testing:
                    # Generate weight without weight noise
                    fW, b = F.split_axis(self.M, np.array([(self.in_size -1)*self.out_size]), axis=0)
                    W = F.reshape(fW, (self.out_size, self.in_size-1))
                    self._awn_weights[weight_name] = W
                    self._awn_biases[bias_name] = b
                else:
                    # Generate weight with weight noise
                    W, b, loss = child_link(n_batches)
                    self._awn_weights[weight_name] = W
                    self._awn_biases[bias_name] = b
                    if self.loss_complex is None:
                        self.loss_complex = F.reshape(loss, (1,1))
                    else:
                        self.loss_complex = F.concat((self.loss_complex, F.reshape(loss, (1,1))))

        """ Begins the transformations """
        # Linear
        l1_in = F.linear(x_now, self.get_awn_weight_link("x_l1"), self.get_awn_bias_link("x_l1"))
        l2_in = F.linear(x_now, self.get_awn_weight_link("x_l2"), self.get_awn_bias_link("x_l2"))
        l3_in = F.linear(x_now, self.get_awn_weight_link("x_l3"), self.get_awn_bias_link("x_l3"))

        # LSTM1
        if self.ws_output is not None:
            l1_in += self.ws_to_l1
        l1_h = self.l1(l1_in, self.get_awn_weight_link("l1_l1"), self.get_awn_bias_link("l1_l1"))

        # SoftWindow
        self.ws_output, self.eow = self.ws(cs, ls, l1_h, self.get_awn_weight_link("l1_ws"), self.get_awn_bias_link("l1_ws"))
        self.ws_to_l1 = F.linear(self.ws_output, self.get_awn_weight_link("ws_l1"), self.get_awn_bias_link("ws_l1"))
        self.ws_to_l2 = F.linear(self.ws_output, self.get_awn_weight_link("ws_l2"), self.get_awn_bias_link("ws_l2"))
        self.ws_to_l3 = F.linear(self.ws_output, self.get_awn_weight_link("ws_l3"), self.get_awn_bias_link("ws_l3"))

        # LSTM2
        l2_in += F.linear(l1_h, self.get_awn_weight_link("l1_l2"), self.get_awn_bias_link("l1_l2"))
        l2_in += self.ws_to_l2
        l2_h = self.l2(l2_in, self.get_awn_weight_link("l2_l2"), self.get_awn_bias_link("l2_l2"))

        # LSTM3
        l3_in += F.linear(l2_h, self.get_awn_weight_link("l2_l3"), self.get_awn_bias_link("l2_l3"))
        l3_in += self.ws_to_l3
        l3_h = self.l3(l3_in, self.get_awn_weight_link("l3_l3"), self.get_awn_bias_link("l3_l3"))

        # Mixture Density Network
        loss_network, x_pred, eos, pi, mux, muy, sgx, sgy, rho = self.ms(
            x_next, self.eow, l1_h, l2_h, l3_h,
            self.get_awn_weight_link("l1_ms"),
            self.get_awn_weight_link("l2_ms"),
            self.get_awn_weight_link("l3_ms"),
            self.get_awn_bias_link("l1_ms"),
            self.get_awn_bias_link("l2_ms"),
            self.get_awn_bias_link("l3_ms"),
            prob_bias
        )

        return loss_network, x_pred, eos, pi, mux, muy, sgx, sgy, rho, self.loss_complex
        

# ===============================================
# Main entry point of the training process (main)
# ===============================================


def main(data_dir, output_dir, batch_size, peephole, epochs, grad_clip, resume_dir, resume_model, resume_optimizer, gpu, adaptive_noise, update_weight, use_weight_noise, save_interval, truncated_back_prop_len, truncated_data_samples, rnn_layers_number, rnn_cells_number, win_unit_number, mix_comp_number, random_seed):
    """ Train the model based on the data saved in ../processed """
    logger = logging.getLogger(__name__)
    logger.info('Training the model')

    logger.info('Model: {}'.format('with peephole' if peephole == 1 else 'no peephole'))
    logger.info('GPU: {}'.format(gpu))
    logger.info('Mini-batch size: {}'.format(batch_size))
    logger.info('# epochs: {}'.format(epochs))

    model_suffix_dir = "{0}-{1}-{2}".format(time.strftime("%Y%m%d-%H%M%S"), 'with_peephole' if peephole == 1 else 'no_peephole', batch_size)
    training_suffix = "{0}".format("training")
    state_suffix = "{0}".format("state")

    """ Fetching the model and the inputs """
    logger.info("Fetching the model and the inputs")
    train_data = load_data(data_dir + "/train/train_data")
    train_characters = load_data(data_dir + "/train/train_characters")
    valid_data = load_data(data_dir + "/valid/valid_data")
    valid_characters = load_data(data_dir + "/valid/valid_characters")
    vocab = load_data(data_dir + "/vocabulary")


    """ Create the model """
    n_max_seq_length = max(get_max_sequence_length(train_characters), get_max_sequence_length(valid_characters))
    n_chars = len(vocab)
    wscale = 0.1

    logger.info("Creating the model")
    if peephole == 0:
        model = Model(n_chars, INPUT_SIZE, win_unit_number, rnn_cells_number, mix_comp_number, rnn_layers_number, use_adaptive_noise=True if use_weight_noise == 1 else False)
    else:
        model = ModelPeephole(n_chars, INPUT_SIZE, win_unit_number, rnn_cells_number, mix_comp_number, rnn_layers_number)

    """ Setup the model """
    logger.info("Setuping the model")
    optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.90, beta2=0.999, eps=1e-08)
    optimizer.setup(model)

    if grad_clip is not 0:
        optimizer.add_hook(chainer.optimizers.GradientClipping(grad_clip))

    if resume_dir:
        logger.info("Loading state from {}".format(output_dir + '/' + resume_dir))
        if resume_model != "":
            chainer.serializers.load_npz(output_dir + "/" + resume_dir + "/" + resume_model, model)
        if resume_optimizer != "":
            chainer.serializers.load_npz(output_dir + "/" + resume_dir + "/" + resume_optimizer, optimizer)

    """ Enable cupy, if available """
    if gpu > -1:
        logger.info("Enabling CUpy")
        chainer.cuda.get_device_from_id(gpu).use()
        xp = cupy
        model.to_gpu()
    else:
        xp = np

    # Create an array of [train_data, train_characters]
    group_set_training = []
    group_set_validation = []
    for i in xrange(len(train_data)):
        tmp = []
        tmp.append(train_data[i])
        tmp.append(train_characters[i])
        group_set_training.append(tmp)
    for i in xrange(len(valid_data)):
        tmp = []
        tmp.append(valid_data[i])
        tmp.append(valid_characters[i])
        group_set_validation.append(tmp)

    # Chainer Iteration class for the mini-batches
    #train_iter = chainer.iterators.SerialIterator(group_set_training, batch_size, repeat=True, shuffle=True)
    train_iter = chainer.iterators.SerialIterator(group_set_training, batch_size, repeat=True, shuffle=True)
    valid_iter = chainer.iterators.SerialIterator(group_set_validation, 256, repeat=False, shuffle=True)


    """ Begin training """
    accum_loss_network_train = np.zeros((epochs, 5))
    accum_loss_network_valid = np.zeros((epochs, 5))
    accum_loss_complex_train = np.zeros((epochs, 2, 15))

    min_valid_loss = np.zeros(1)
    thr_valid_loss = -1100.0

    xp.random.seed(random_seed)
    np.random.seed(random_seed)
    n_batches_counter, n_batches = None, None

    t_epoch_start = time.time()
    elapsed_forward = 0
    elapsed_backward = 0
    losses_network = None
    losses_complex = None
    while train_iter.epoch < epochs:
        epoch = train_iter.epoch
        logger.info("Beginning training for epoch {}".format(epoch+1))

        batch = np.array(train_iter.next())

        # The mini-batch should be padded to be the same length
        train_data_batch, train_characters_batch = get_padded_data(batch[:, 0], batch[:, 1])

        # count number of backward() (computaion of gradients). this value is important for # complex_loss! See Graves 2011 p5 equation 18 for detail !
        if not isinstance(n_batches, variable.Variable):
            n_batches_counter = xp.zeros(1)
            for i in xrange(int(math.floor(len(group_set_training)/batch_size))):
                n_batches_counter += 1

            n_batches = chainer.Variable(xp.asarray(n_batches_counter).astype(xp.int32))

        """ Run the training for all the data in the mini-batch """
        prob_bias = 0.0
        if len(train_data_batch) > 1:
            now = time.time()
            offset_batch_size, t_max, x_dim = train_data_batch.shape
            if truncated_data_samples is not -1:
                t_max = truncated_data_samples if t_max > truncated_data_samples else t_max

            # One-hot encoding of character for all the sequence
            cs_data = xp.zeros((offset_batch_size, n_chars, n_max_seq_length))
            ls_data = xp.zeros((offset_batch_size, 1))
            for j in xrange(offset_batch_size):
                for k in xrange(len(train_characters_batch[j])):
                    length = train_characters_batch[j][k]
                    cs_data[j, length, k] = 1.0

            cs = chainer.Variable(xp.asarray(cs_data).astype(xp.float32))
            ls = chainer.Variable(xp.asarray(ls_data).astype(xp.float32))

            # Training parameters
            loss_network = xp.zeros((offset_batch_size, 1))
            loss_complex = xp.zeros(1)
            accum_loss = 0
            model.cleargrads()

            # @TODO: Make the backpropagation length dynamic
            back_prop_len = t_max + 1 if t_max < 1600 else t_max / 2
            if truncated_back_prop_len != 0:
                back_prop_len = truncated_back_prop_len

            # For each data in the batchsize run the training
            for t in xrange(t_max-1):
                t_forward = time.time()
                x_now = chainer.Variable(xp.asarray(train_data_batch[0:offset_batch_size, t, 0:x_dim]).astype(xp.float32))
                x_next = chainer.Variable(xp.asarray(train_data_batch[0:offset_batch_size, t+1, 0:x_dim]).astype(xp.float32))
                logger.info("Training data {0}/{1} for epoch {2}".format(t+1, t_max, epoch+1))

                # The weight are updated only in the first iteration
                local_update_weight = update_weight if t == 0 else False

                #loss_i, x_pred, eos_i, pi_i, mux_i, muy_i, sgx_i, sgy_i, rho_i, loss_complex_i = model(x_now, x_next, cs, ls, prob_bias, n_batches, local_update_weight)
                x = [x_now, x_next, cs, ls, prob_bias, n_batches]
                loss_i, x_pred, eos_i, pi_i, mux_i, muy_i, sgx_i, sgy_i, rho_i, loss_complex_i = model(x, local_update_weight)

                # Get stats
                accum_loss += F.sum(loss_i)/offset_batch_size
                loss_network += loss_i.data
                elapsed_forward += time.time() - t_forward
                loss_complex_i_cpu = cuda.to_cpu(loss_complex_i.data)

                # Backprop the error
                if t > 0:
                    # Complete backprop
                    if back_prop_len == -1:
                        t_backward = time.time()
                        accum_loss.backward()
                        accum_loss.unchain_backward()
                        accum_loss = 0
                        elapsed_backward += time.time() - t_backward
                    # Truncated backprop
                    elif (t+1) % back_prop_len == back_prop_len-1 or t == t_max-3:
                        t_backward = time.time()
                        accum_loss.backward()
                        accum_loss.unchain_backward()
                        accum_loss = 0
                        elapsed_backward += time.time() - t_backward

            # Update parameters at the end of all the processed data from a mini-batch
            optimizer.update()
            cur_at = time.time()
            throuput = t_max/(cur_at - now)

            # Update global statistics
            losses_network = xp.copy(loss_network) if losses_network is None else xp.concatenate((losses_network, loss_network), axis=0)
            losses_complex = xp.copy(loss_complex_i.data) if losses_complex is None else xp.concatenate((losses_complex, loss_complex_i.data), axis=0)

            model.reset_state()
                
        """ All the training mini-batches have been processed """
        if train_iter.is_new_epoch:
            # Global results for one epoch
            losses_network_cpu = losses_network if xp == np else cuda.to_cpu(losses_network)
            losses_complex_cpu = losses_complex if xp == np else cuda.to_cpu(losses_complex)

            accum_loss_network_train[epoch, 0] = losses_network_cpu.mean()
            accum_loss_network_train[epoch, 1] = losses_network_cpu.std()
            accum_loss_network_train[epoch, 2] = losses_network_cpu.min()
            accum_loss_network_train[epoch, 3] = losses_network_cpu.max()
            accum_loss_network_train[epoch, 4] = np.median(losses_network_cpu)

            accum_loss_complex_train[epoch, 0, 0:] = losses_complex_cpu.mean(axis=0)
            accum_loss_complex_train[epoch, 1, 0:] = losses_complex_cpu.std(axis=0)

            elapsed_time = time.time() - t_epoch_start

            logger.info("Epoch: {0}/{1} completed for training".format(epoch+1, epochs))
            logger.info("Training loss network: {0:>9.4f} +- {1:<9.4f}  {2:>9.4f} < {3:>9.4f} < {4:>9.4f}"
                        .format(accum_loss_network_train[epoch, 0],
                                accum_loss_network_train[epoch, 1],
                                accum_loss_network_train[epoch, 2],
                                accum_loss_network_train[epoch, 4],
                                accum_loss_network_train[epoch, 3]))
            logger.info("Total elapsed time: {0:>9.2f}".format(elapsed_time))

            # Loss for each layers
            logger.info("Training loss complex (mean): {0:>9.4f}  {1:>9.4f}  {2:>9.4f}  {3:>9.4f} {4:>9.4f}  {5:>9.4f}  {6:>9.4f} {7:>9.4f}  {8:>9.4f}  {9:>9.4f}  {10:>9.4f} {11:>9.4f}  {12:>9.4f}  {13:>9.4f} {14:>9.4f}"
                        .format(accum_loss_complex_train[epoch, 0][0],
                                accum_loss_complex_train[epoch, 0][1],
                                accum_loss_complex_train[epoch, 0][2],
                                accum_loss_complex_train[epoch, 0][3],
                                accum_loss_complex_train[epoch, 0][4],
                                accum_loss_complex_train[epoch, 0][5],
                                accum_loss_complex_train[epoch, 0][6],
                                accum_loss_complex_train[epoch, 0][7],
                                accum_loss_complex_train[epoch, 0][8],
                                accum_loss_complex_train[epoch, 0][9],
                                accum_loss_complex_train[epoch, 0][10],
                                accum_loss_complex_train[epoch, 0][11],
                                accum_loss_complex_train[epoch, 0][12],
                                accum_loss_complex_train[epoch, 0][13],
                                accum_loss_complex_train[epoch, 0][14]))
            logger.info("Training loss complex (std): {0:>9.4f}  {1:>9.4f}  {2:>9.4f}  {3:>9.4f} {4:>9.4f}  {5:>9.4f}  {6:>9.4f}{7:>9.4f}  {8:>9.4f}  {9:>9.4f}  {10:>9.4f} {11:>9.4f}  {12:>9.4f}  {13:>9.4f}  {14:>9.4f}"
                        .format(epoch,
                                accum_loss_complex_train[epoch, 1][0],
                                accum_loss_complex_train[epoch, 1][1],
                                accum_loss_complex_train[epoch, 1][2],
                                accum_loss_complex_train[epoch, 1][3],
                                accum_loss_complex_train[epoch, 1][4],
                                accum_loss_complex_train[epoch, 1][5],
                                accum_loss_complex_train[epoch, 1][6],
                                accum_loss_complex_train[epoch, 1][7],
                                accum_loss_complex_train[epoch, 1][8],
                                accum_loss_complex_train[epoch, 1][9],
                                accum_loss_complex_train[epoch, 1][10],
                                accum_loss_complex_train[epoch, 1][11],
                                accum_loss_complex_train[epoch, 1][12],
                                accum_loss_complex_train[epoch, 1][13],
                                accum_loss_complex_train[epoch, 1][14]))

            """ Save the model """
            if epoch % save_interval == 0:
                accum_loss_network_train_cpu = accum_loss_network_train if xp == np else cuda.to_cpu(accum_loss_network_train)
                accum_loss_complex_train_cpu = accum_loss_complex_train if xp == np else cuda.to_cpu(accum_loss_complex_train)

                save_dir = output_dir + '/' + model_suffix_dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                logger.info("Saving the results for epoch {}".format(epoch+1))
                np.save(save_dir + '/loss_network_train', accum_loss_network_train_cpu)
                np.save(save_dir + '/loss_complex_train', accum_loss_complex_train_cpu)

                logger.info("Saving the model and the optimizer for epoch {}".format(epoch+1))
                chainer.serializers.save_npz(save_dir + '/model-' + str(epoch), model)
                chainer.serializers.save_npz(save_dir + '/state-' + str(epoch), optimizer)

            """ Validation step """
            prob_bias = 0.0
            losses_valid = None
            for valid_batch in valid_iter:
                valid_batch = np.array(valid_batch)
                valid_data_batch, valid_characters_batch = get_padded_data(valid_batch[:, 0], valid_batch[:, 1], False)
                offset_valid_batch_size, t_max_valid, x_dim_valid = valid_data_batch.shape
                if truncated_data_samples is not -1:
                    t_max_valid = truncated_data_samples if t_max_valid > truncated_data_samples else t_max_valid

                # Training parameters
                loss_network = xp.zeros((offset_valid_batch_size, 1))
                loss_complex = xp.zeros((offset_valid_batch_size, 1))

                # One-hot encoding of character for all the sequence
                cs_data = xp.zeros((offset_valid_batch_size, n_chars, n_max_seq_length))
                ls_data = xp.zeros((offset_valid_batch_size, 1))
                for j in xrange(len(valid_characters_batch)):
                    for k in xrange(len(valid_characters_batch[j])):
                        length = valid_characters_batch[j][k]
                        cs_data[j, length, k] = 1.0

                
                with chainer.no_backprop_mode():
                    cs = chainer.Variable(xp.asarray(cs_data).astype(xp.float32))
                    ls = chainer.Variable(xp.asarray(ls_data).astype(xp.float32))
 
                # For each data in the batchsize run the validation
                for t in xrange(t_max-1):
                    with chainer.no_backprop_mode():
                        x_now = chainer.Variable(xp.asarray(valid_data_batch[0:offset_valid_batch_size, t, 0:x_dim_valid]).astype(xp.float32))
                        x_next = chainer.Variable(xp.asarray(valid_data_batch[0:offset_valid_batch_size, t+1, 0:x_dim_valid]).astype(xp.float32))
                        logger.info("Validating data {0}/{1} for epoch {2}".format(t+1, t_max, epoch+1))
                        #loss_i, x_pred, eos_i, pi_i, mux_i, muy_i, sgx_i, sgy_i, rho_i, loss_complex_i = model(x_now, x_next, cs, ls, prob_bias, n_batches, renew_weights=False, testing=True)
                        x = [x_now, x_next, cs, ls, prob_bias, n_batches]
                        loss_i, x_pred, eos_i, pi_i, mux_i, muy_i, sgx_i, sgy_i, rho_i, loss_complex_i = model(x, renew_weights=False)

                    loss_network += loss_i.data

                losses_valid = xp.copy(loss_network) if losses_valid is None else xp.concatenate((losses_valid, loss_network), axis=0)

                # Reset the model after each batch of data for validation
                model.reset_state()

            # Invalidate the iteration count for the next epoch
            valid_iter.reset()

            """ Statistics for validation """
            losses_network_cpu = losses_valid if xp == np else cuda.to_cpu(losses_valid)

            accum_loss_network_valid[epoch, 0] = losses_network_cpu.mean()
            accum_loss_network_valid[epoch, 1] = losses_network_cpu.std()
            accum_loss_network_valid[epoch, 2] = losses_network_cpu.min()
            accum_loss_network_valid[epoch, 3] = losses_network_cpu.max()
            accum_loss_network_valid[epoch, 4] = np.median(losses_network_cpu)

            logger.info("Epoch: {0}/{1} completed for validation".format(epoch+1, epochs))
            logger.info("Validation loss network: {0:>9.4f} +- {1:<9.4f}  {2:>9.4f} < {3:>9.4f} < {4:>9.4f}"
                        .format(accum_loss_network_valid[epoch, 0],
                                accum_loss_network_valid[epoch, 1],
                                accum_loss_network_valid[epoch, 2],
                                accum_loss_network_valid[epoch, 4],
                                accum_loss_network_valid[epoch, 3]))

            """ Create the best fit model and optimizer """
            if accum_loss_network_valid[epoch, 0] < min_valid_loss[0] and accum_loss_network_valid[epoch, 0] < thr_valid_loss:
                logger.info("Best fit for model and optmizer was found, saving them")
                chainer.serializers.save_npz(save_dir + '/model-best', model)
                chainer.serializers.save_npz(save_dir + '/state-best', optimizer)
                min_valid_loss[0] = accum_loss_network_valid[epoch, 0]

            if epoch % save_interval == 0:
                accum_loss_network_valid_cpu = accum_loss_network_valid if xp == np else cuda.to_cpu(accum_loss_network_valid)
                np.save(save_dir + '/loss_network_valid', accum_loss_network_valid_cpu)

            """ Reset the variables for the next iteration """
            n_batches = None
            t_epoch_start = time.time()
            elapsed_forward = 0
            elapsed_backward = 0
            losses_network = None
            losses_complex = None

    # Subsequent scripts can use the results of this network without having to open the npy files
    return accum_loss_network_train, accum_loss_network_valid


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
@click.option('--gpu', type=click.INT, default=-1, help='GPU ID (negative value is CPU).')
@click.option('--adaptive_noise', type=click.INT, default=1, help='Use Adaptive Weight Noise in the training process.')
@click.option('--update_weight', type=click.INT, default=1, help='Update weights in the training process.')
@click.option('--use_weight_noise', type=click.INT, default=1, help='Use weight noise in the training process.')
@click.option('--save_interval', type=click.INT, default=1, help='How often the model should be saved.')
@click.option('--truncated_back_prop_len', type=click.INT, default=50, help='Number of backpropagation before stopping.')
@click.option('--truncated_data_samples', type=click.INT, default=500, help='Number of samples to use inside a data.')
@click.option('--rnn_layers_number', type=click.INT, default=3, help='Number of layers for the RNN.')
@click.option('--rnn_cells_number', type=click.INT, default=400, help='Number of LSTM cells per layer.')
@click.option('--win_unit_number', type=click.INT, default=10, help='Number of soft-window components.')
@click.option('--mix_comp_number', type=click.INT, default=20, help='Numver of Gaussian components for mixture density output.')
@click.option('--random_seed', type=click.INT, default=None, help='Number of Gaussian components for mixture density output.')
def cli(**kwargs):
    main(**kwargs)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    cli()
