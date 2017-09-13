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
#from net.functions.mixture_density_outputs import mixture_density_outputs
from functions.connection.mixture_density_network import mixture_density_network

INPUT_SIZE = 3 # (x, y, end_of_stroke)

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
        The SoftWindow act as an attention mechanism which controls the alignment between the text and the pen position

        Args:
            mixture_size (int): size of the mixture "k"
    """
    def __init__(self, mixture_size):
        super(SoftWindow, self).__init__()

        with self.init_scope():
            self.input_linear = L.linear(None)

        self.mixture_size = mixture_size
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
        batch_size, W, u = cs

        # Extract the soft window's parameters
        x_h = self.input_linear(x)
        a_h, b_h, k_h = F.split_axis(x_h, [self.mix_comp_number, 2 * self.mix_comp_number], axis=1)

        if self.k_prev:
            self.k_prev = variable.Variables(self.xp.zeros_like(k_h, dtype=xp.float32))

        a_h = F.exp(a_h)
        b_h = F.exp(b_h)
        k_h = self.k_prev + F.exp(k_h)
        self.k_prev = k_h

        # Compute phi's parameters
        us = xp.linspace(0, u-1, u)
        p_k = F.square(k_h - us)
        p_b = F.scale(-b, p_k)
        p_b = F.exp(p_b)
        p_a = F.scale(a_h, p_b)
        phi = F.sum(p_a, axis=1, keepdims=True)
        self.phi = phi

        # Finalize the soft window computation
        w = F.matmul(phi, cs)
        w = F.reshape(w, (batch_size, W))
        self.w = w

        return w

class MixtureDensityNetwork(chainer.Chain):
    """
        The Mixture-Density-Network outputs a parametrised mixture distribution.
        
        Args:
            n_mdn_comp (int): number of MDN components
            prob_bias (int): bias added to the pi's probability
    """
    def __init__(self, n_mdn_comp, prob_bias = 0):
        super(MixtureDensityNetwork, self).__init__()

        with self.init_scope():
            self.input_linear = L.linear(None)

        self.n_mdn_comp = n_mdn_comp
        self.p_bias = prob_bias
    
    def __call__(self, inputs):
        """
            Perform the MDN prediction

            Args:
                inputs (float[][]): input tensor 
            Returns:
                loss (float)
        """
        x = inputs

        # Extract the MDN's parameters
        eos_hat, pi_hat, mu_x1_hat, mu_x2_hat, s_x1_hat, s_x2_hat, rho_hat = F.split_axis(
            x, np.asarray([1 + i*self.n_mdn_comp for i in xrange(5+1)]), axis=1
        )


        # Add the bias to the parameter to change the shape of the prediction
            

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
            n_units (int): number of hidden units in LSTM cells
            n_mixture_components (int): number of MDN components
            n_window_unit (int): number of parameters in the softwindow
        Returns
            loss (float)
    """

    def __init__(self, n_units, n_mixture_components, n_window_unit):
        super(Model, self).__init__()

        with self.init_scope():
            # LSTMs layers
            self.lstm1 = L.LSTM(n_units)
            self.lstm2 = L.LSTM(n_units)
            self.lstm3 = L.LSTM(n_units)
            
            # Attention mechanism
            self.soft_window = SoftWindow(n_window_unit)

            # Mixture Density Network

            # Linear connections for some layers
            self.sw_input = L.linear(None)
            #self.lstm3_output = L.linear(None)


# ===============================================
# Main entry point of the training process (main)
# ===============================================


def main(data_dir, output_dir, batch_size, peephole, epochs, grad_clip, resume_dir, resume_model, resume_optimizer, gpu, adaptive_noise, update_weight, use_weight_noise, save_interval, truncated_back_prop_len, truncated_data_samples, rnn_layers_number, rnn_cells_number, win_unit_number, mix_comp_number, random_seed, debug):
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

    """ Chainer's debug mode """
    if debug == 1:
        logger.info("Enabling Chainer's debug mode")
        chainer.config.debug = True

    """ Fetching the model and the inputs """
    def load_data(path):
        f = open(path, "rb")
        data = pickle.load(f)
        f.close()
        return f

    logger.info("Fetching the model and the inputs")
    train_data = load_data(data_dir + "/train/train_data")
    train_characters = load_data(data_dir + "/train/train_characters")
    valid_data = load_data(data_dir + "/valid/valid_data")
    valid_characters = load_data(data_dir + "/valid/valid_characters")
    vocab = load_data(data_dir + "/vocabulary")

    """ Create the model """
    logger.info("Creating the model")
    #if peephole == 0:
        
    #else:
        #model = ModelPeephole(n_chars, INPUT_SIZE, win_unit_number, rnn_cells_number, mix_comp_number, rnn_layers_number)

    """ Setup the model """
    logger.info("Setuping the model")
    optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.90, beta2=0.999, eps=1e-08)
    optimizer.setup(model)

    if grad_clip is not 0:
        optimizer.add_hook(chainer.optimizers.GradientClipping(grad_clip))

    """ Enable cupy, if available """
    if gpu > -1:
        logger.info("Enabling CUpy")
        chainer.cuda.get_device_from_id(gpu).use()
        xp = cupy
        model.to_gpu()
    else:
        xp = np

    if resume_dir:
        logger.info("Loading state from {}".format(output_dir + '/' + resume_dir))
        if resume_model != "":
            chainer.serializers.load_npz(output_dir + "/" + resume_dir + "/" + resume_model, model)
        if resume_optimizer != "":
            chainer.serializers.load_npz(output_dir + "/" + resume_dir + "/" + resume_optimizer, optimizer)


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
@click.option('--debug', type=click.INT, default=0, help='Chainer debugging mode.')
def cli(**kwargs):
    main(**kwargs)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    cli()
