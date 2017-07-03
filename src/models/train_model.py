#!/usr/bin/env python # -*- coding: utf-8 -*-

# Generate handwriting from a sequence of characters
# ==================================================

import time
import os
import click
import logging

import numpy as np
import cPickle as pickle

import chainer
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import variable

from net.lstm import LSTM
from net.adaptive_weight_noise import AdaptiveWeightNoise
from net.soft_window import SoftWindow
from net.mixture_density_outputs import MixtureDensityOutputs

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

class Model(chainer.Chain):
    """
    Handwriting Synthesis Network Model. No peephole

    Args:
        n_vocab:
    """
    def __init__(self, batch_size, n_vocab, n_win_units, n_rnn_cells, n_mix_comp, wscale=0.1, use_weight_noise=True):
        super(Model, self).__init__(
            awn_x_l1  = AdaptiveWeightNoise((3+1), 4*n_rnn_cells, wscale, nobias=False, use_weight_noise=use_weight_noise),
            awn_x_l2  = AdaptiveWeightNoise((3+1), 4*n_rnn_cells, wscale, nobias=False, use_weight_noise=use_weight_noise), 
            awn_x_l3  = AdaptiveWeightNoise((3+1), 4*n_rnn_cells, wscale, nobias=False, use_weight_noise=use_weight_noise),
            awn_l1_ws = AdaptiveWeightNoise((n_rnn_cells+1),(3*n_win_units), wscale, nobias=False, use_weight_noise=use_weight_noise),
            awn_ws_l1 = AdaptiveWeightNoise((n_vocab + 1), (4*n_rnn_cells), wscale, nobias=False, use_weight_noise=use_weight_noise),
            awn_ws_l2 = AdaptiveWeightNoise((n_vocab + 1), (4*n_rnn_cells), wscale, nobias=False, use_weight_noise=use_weight_noise),
            awn_ws_l3 = AdaptiveWeightNoise((n_vocab + 1), (4*n_rnn_cells), wscale, nobias=False, use_weight_noise=use_weight_noise),
            awn_l1_l1 = AdaptiveWeightNoise((n_rnn_cells + 1), (4*n_rnn_cells), wscale, nobias=False, use_weight_noise=use_weight_noise),
            awn_l2_l2 = AdaptiveWeightNoise((n_rnn_cells + 1), (4*n_rnn_cells), wscale, nobias=False, use_weight_noise=use_weight_noise),
            awn_l3_l3 = AdaptiveWeightNoise((n_rnn_cells + 1), (4*n_rnn_cells), wscale, nobias=False, use_weight_noise=use_weight_noise),
            awn_l1_l2 = AdaptiveWeightNoise((n_rnn_cells + 1), (4*n_rnn_cells), wscale, nobias=False, use_weight_noise=use_weight_noise),
            awn_l2_l3 = AdaptiveWeightNoise((n_rnn_cells + 1), (4*n_rnn_cells), wscale, nobias=False, use_weight_noise=use_weight_noise),
            awn_l1_ms = AdaptiveWeightNoise((n_rnn_cells + 1), (1 + n_mix_comp*6), wscale, nobias=False, use_weight_noise=use_weight_noise),
            awn_l2_ms = AdaptiveWeightNoise((n_rnn_cells + 1), (1 + n_mix_comp*6), wscale, nobias=False, use_weight_noise=use_weight_noise),
            awn_l3_ms = AdaptiveWeightNoise((n_rnn_cells + 1), (1 + n_mix_comp*6), wscale, nobias=False, use_weight_noise=use_weight_noise),

            ws=SoftWindow(n_win_units),
            l1=LSTM(n_rnn_cells),
            l2=LSTM(n_rnn_cells),
            l3=LSTM(n_rnn_cells),
            ms=MixtureDensityOutputs(n_mix_comp) 
        )
        #self.train = train
        self.n_rnn_cells = n_rnn_cells
        self.n_win_units = n_win_units
        self.n_mix_comp  = n_mix_comp
        self.n_vocab = n_vocab
        self.reset_state()

    def reset_state(self):
        self.ws.reset_state()
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.wo = None          # output from 'soft window', ws
        self.ws_to_l1 = None
        self.loss_complex = None

    def __call__(self, x_now, xnext, cs, ls, prob_bias, renew_weights, testing=False):
        xp = cuda.get_array_module(*x_now.data)

        if renew_weights:

            if testing:
                # Generate weights w/o Adaptive Noises
                def get_weights_for_testing(self):
                    fW, b = split_axis.split_axis(self.M, numpy.asarray([(self.in_size -1)*self.out_size]), axis=0)
                    W = reshape.reshape(fW, (self.out_size, self.in_size -1))
                    return W, b

                self.w_x_l1, self.b_x_l1    = get_weights_for_testing(self.awn_x_l1)
                self.w_x_l2, self.b_x_l2    = get_weights_for_testing(self.awn_x_l2)
                self.w_x_l3, self.b_x_l3    = get_weights_for_testing(self.awn_x_l3)
                self.w_l1_ws, self.b_l1_ws  = get_weights_for_testing(self.awn_l1_ws)
                self.w_ws_l1, self.b_ws_l1  = get_weights_for_testing(self.awn_ws_l1)
                self.w_ws_l2, self.b_ws_l2  = get_weights_for_testing(self.awn_ws_l2)
                self.w_ws_l3, self.b_ws_l3  = get_weights_for_testing(self.awn_ws_l3)
                self.w_l1_l1, self.b_l1_l1  = get_weights_for_testing(self.awn_l1_l1)
                self.w_l2_l2, self.b_l2_l2  = get_weights_for_testing(self.awn_l2_l2)
                self.w_l3_l3, self.b_l3_l3  = get_weights_for_testing(self.awn_l3_l3)
                self.w_l1_l2, self.b_l1_l2  = get_weights_for_testing(self.awn_l1_l2)
                self.w_l2_l3, self.b_l2_l3  = get_weights_for_testing(self.awn_l2_l3)
                self.w_l1_ms, self.b_l1_ms  = get_weights_for_testing(self.awn_l1_ms)
                self.w_l2_ms, self.b_l2_ms  = get_weights_for_testing(self.awn_l2_ms)
                self.w_l3_ms, self.b_l3_ms  = get_weights_for_testing(self.awn_l3_ms)

            else:
                # Generate weights w/ Adaptive Noises
                self.w_x_l1, self.b_x_l1, loss_ = self.awn_x_l1()
                self.loss_complex = reshape.reshape(loss_, (1,1))

                self.w_x_l2, self.b_x_l2, loss_ = self.awn_x_l2()
                self.loss_complex = concat.concat((self.loss_complex, reshape.reshape(loss_, (1,1))))

                self.w_x_l3, self.b_x_l3, loss_ = self.awn_x_l3()
                self.loss_complex = concat.concat((self.loss_complex, reshape.reshape(loss_, (1,1))))

                self.w_l1_ws, self.b_l1_ws, loss_ = self.awn_l1_ws()
                self.loss_complex = concat.concat((self.loss_complex, reshape.reshape(loss_, (1,1))))

                self.w_ws_l1, self.b_ws_l1, loss_ = self.awn_ws_l1()
                self.loss_complex = concat.concat((self.loss_complex, reshape.reshape(loss_, (1,1))))

                self.w_ws_l2, self.b_ws_l2, loss_ = self.awn_ws_l2()
                self.loss_complex = concat.concat((self.loss_complex, reshape.reshape(loss_, (1,1))))

                self.w_ws_l3, self.b_ws_l3, loss_ = self.awn_ws_l3()
                self.loss_complex = concat.concat((self.loss_complex, reshape.reshape(loss_, (1,1))))

                self.w_l1_l1, self.b_l1_l1, loss_  = self.awn_l1_l1()
                self.loss_complex = concat.concat((self.loss_complex, reshape.reshape(loss_, (1,1))))

                self.w_l2_l2, self.b_l2_l2, loss_ = self.awn_l2_l2()
                self.loss_complex = concat.concat((self.loss_complex, reshape.reshape(loss_, (1,1))))

                self.w_l3_l3, self.b_l3_l3, loss_ = self.awn_l3_l3()
                self.loss_complex = concat.concat((self.loss_complex, reshape.reshape(loss_, (1,1))))

                self.w_l1_l2, self.b_l1_l2, loss_ = self.awn_l1_l2()
                self.loss_complex = concat.concat((self.loss_complex, reshape.reshape(loss_, (1,1))))

                self.w_l2_l3, self.b_l2_l3, loss_ = self.awn_l2_l3()
                self.loss_complex = concat.concat((self.loss_complex, reshape.reshape(loss_, (1,1))))

                self.w_l1_ms, self.b_l1_ms, loss_  = self.awn_l1_ms()
                self.loss_complex = concat.concat((self.loss_complex, reshape.reshape(loss_, (1,1))))

                self.w_l2_ms, self.b_l2_ms, loss_  = self.awn_l2_ms()
                self.loss_complex = concat.concat((self.loss_complex, reshape.reshape(loss_, (1,1))))

                self.w_l3_ms, self.b_l3_ms, loss_  = self.awn_l3_ms()
                self.loss_complex = concat.concat((self.loss_complex, reshape.reshape(loss_, (1,1))))

        # xs(t)   --> l1(t)
        l1_in = F.linear(x_now, self.w_x_l1, self.b_x_l1)
        # xs(t)   --> l2(t)
        l2_in = F.linear(x_now, self.w_x_l2, self.b_x_l2)
        # xs(t)   --> l3(t)
        l3_in = F.linear(x_now, self.w_x_l3, self.b_x_l3)


        # LSTM1
        if self.wo is not None:
            # ws(t-1) --> l1(t)
            l1_in +=  self.ws_to_l1
            # h1 <-- LSTM(l1_in)
            h1 = self.l1(l1_in, self.w_l1_l1, self.b_l1_l1)  

        # SoftWindow       cs, h1(t) --> ws(t), eow(t)
        self.wo, self.eow  = self.ws(cs, ls, h1, self.w_l1_ws, self.b_l1_ws)
        self.ws_to_l1 = F.linear(self.wo, self.w_ws_l1, self.b_ws_l1)
        self.ws_to_l2 = F.linear(self.wo, self.w_ws_l2, self.b_ws_l2)
        self.ws_to_l3 = F.linear(self.wo, self.w_ws_l3, self.b_ws_l3)

        # LSTM2
        l2_in   += F.linear(h1, self.w_l1_l2, self.b_l1_l2)
        l2_in   += self.ws_to_l2
        h2       = self.l2(l2_in, self.w_l2_l2, self.b_l2_l2)

        # LSTM3
        l3_in   += F.linear(h2, self.w_l2_l3, self.b_l2_l3)
        l3_in   += self.ws_to_l3
        h3       = self.l3(l3_in, self.w_l3_l3, self.b_l3_l3)

        #self.eow = chainer.Variable(xp.zeros_like(xnext.data).astype(xp.float32), volatile='auto')
        # Mixture Density Network
        loss_network, xpred, eos_, pi_, mux_, muy_, sgx_, sgy_, rho_  = self.ms(
            xnext, 
            self.eow, 
            h1, h2, h3, 
            self.w_l1_ms, 
            self.w_l2_ms,
            self.w_l3_ms,
            self.b_l1_ms, 
            self.b_l2_ms,
            self.b_l3_ms,
            prob_bias
        )

        return loss_network, xpred, eos_, pi_, mux_, muy_, sgx_, sgy_, rho_ , self.loss_complex

class ModelPeephole(chainer.Chain):
    """
    Handwriting Synthesis Network Model. No peephole

    xs: (Variable) state of a pen position
    cs: (Variable) one-hot-encoded vector of a given set of character sequences.
    ndim=(batchsize, n_chars, max_sequence_length)
    lstm1, lstm2, lstm3: (Chain) lstm+peephole layer
    ws: (Chain) soft window for alignment
    ms: (Chain) mixture density outputs
    loss: (Variable)
    """
    def __init__(self, batch_size, n_vocab, input_size, n_win_units, n_rnn_cells, n_mix_comp):
        super(ModelPeephole, self).__init__(
            awn=AdaptiveWeightNoise( 
                (3 + 1)            *(3*4*n_rnn_cells) + \
                (n_rnn_cells+1)          *n_win_units + \
                (n_vocab + 1)      *(3*4*n_rnn_cells) + \
                (3*n_rnn_cells)      *(4*n_rnn_cells) + \
                3                    *(3*n_rnn_cells) + \
                (2*n_rnn_cells + 2)  *(4*n_rnn_cells) + \
                3*(n_rnn_cells + 1)*(1 + n_mix_comp*6)     # (l1, l2, l3) --> ms
            ),
            ws=SoftWindow(n_win_units),
            l1=LSTMpeephole(n_rnn_cells),       # LSTM as l1
            l2=LSTMpeephole(n_rnn_cells),       # LSTM as l2
            l3=LSTMpeephole(n_rnn_cells),       # LSTM as l3
            ms=MixtureDensityOutputs3(n_mix_comp) 
        )
        #self.train = train
        self.batch_size = batch_size
        self.n_rnn_cells = n_rnn_cells
        self.reset_state()
        self.split_sections = numpy.cumsum(
            numpy.asarray(
                [3            *3*4*n_rnn_cells,       # fw_x_ls
                 1            *3*4*n_rnn_cells,       # fb_x_ls
                 n_rnn_cells      *n_win_units,       # fw_l1_ws
                 1      *n_win_units,       # fb_l1_ws
                 n_vocab      *3*4*n_rnn_cells,       # fw_ws_ls
                 1      *3*4*n_rnn_cells,       # fb_ws_ls
                 n_rnn_cells    *4*n_rnn_cells,       # fw_l1_l1
                 n_rnn_cells    *4*n_rnn_cells,       # fw_l2_l2
                 n_rnn_cells    *4*n_rnn_cells,       # fw_l3_l3
                 n_rnn_cells    *4*n_rnn_cells,       # fw_l1_l2
                 1    *4*n_rnn_cells,       # fb_l1_l2
                 n_rnn_cells    *4*n_rnn_cells,       # fw_l2_l3
                 1    *4*n_rnn_cells,       # fb_l2_l3
                 (n_rnn_cells)  *(1 + n_mix_comp*6),  # fw_l1_ms
                 1  *(1 + n_mix_comp*6),  # fb_l1_ms
                 (n_rnn_cells)  *(1 + n_mix_comp*6),  # fw_l2_ms
                 1  *(1 + n_mix_comp*6),  # fb_l2_ms
                 (n_rnn_cells)  *(1 + n_mix_comp*6),  # fw_l3_ms
                 1  *(1 + n_mix_comp*6),  # fb_l3_ms
                 1  *3*n_rnn_cells,       # fw_l1_peephole
                 1  *3*n_rnn_cells,       # fw_l2_peephole
                 1  *3*n_rnn_cells         # fw_l3_peephole
                ]
            )
        )

    def reset_state(self):
        self.ws.reset_state()
        self.l1.reset_state()
        self.l2.reset_state()
        self.l3.reset_state()
        self.wo = None          # output from 'soft window', ws
        self.ws_to_l1 = None

    def __call__(self, x_now, xnext, cs, ls, prob_bias, enoise):

        n_batches = variable.Variable(xp.asarray([self.batch_size], dtype=x_now.data.dtype), volatile='auto')
        n_rnn_cells = self.n_rnn_cells

        # Adaptive weight noise
        fW, loss_complex = self.awn(enoise, n_batches)

        fw_x_ls, fb_x_ls,   \
                fw_l1_ws, fb_l1_ws, \
                fw_ws_ls, fb_ws_ls, \
                fw_l1_l1, fw_l2_l2, fw_l3_l3, \
                fw_l1_l2, fb_l1_l2, \
                fw_l2_l3, fb_l2_l3, \
                fw_l1_ms, fb_l1_ms, \
                fw_l2_ms, fb_l2_ms, \
                fw_l2_ms, fb_l2_ms, \
                fw_l1_peephole,     \
                fw_l2_peephole,     \
                fw_l3_peephole      \
                = F.split_axis(fW, self.split_sections[0:-1], axis=0)

        # xs(t)   --> l1(t), l2(t), l3(t)
        y_x = F.linear(x_now, F.reshape(fw_x_ls, (3, 3*4*n_rnn_cells)), F.reshape(fb_x_ls, (1, 3*4*n_rnn_cells)))
        l1_in, l2_in, l3_in = F.split_axis(y_x, numpy.asarray([4*n_rnn_cells, 8*n_rnn_cells]), axis=1)

        # LSTM1
        if self.wo is not None:
            l1_in +=  self.ws_to_l1                     # ws(t-1) --> l1(t)
            h1 = self.l1(l1_in, F.reshape(fw_l1_l1, (n_rnn_cells, 4*n_rnn_cells)), F.reshape(fw_l1_peephole, (1, 3*n_rnn_cells)))

        # SoftWindow       cs, h1(t) --> ws(t), eow(t)
        self.wo, self.eow  = self.ws(cs, ls, h1, F.reshape(fw_l1_ws, (n_vocab, 3*4*n_rnn_cells)), F.reshape(fb_ws_ls, (1, 3*4*n_rnn_cells)))
        y_w = F.linear(self.wo, F.reshape(fw_ws_ls, (n_vocab, 3*4*n_rnn_cells)), F.reshape(fb_ws_ls, (1, 3*4*n_rnn_cells)))

        self.ws_to_l1, ws_to_l2, ws_to_l3 = F.split_axis(y_w, numpy.asarray([4*n_rnn_cells, 8*n_rnn_cells]), axis=1)

        # LSTM2
        l2_in   += F.linear(h1, F.reshape(fw_l1_l2, (n_rnn_cells, 4*n_rnn_cells)), F.reshape(fb_l1_l2, (1, 4*n_rnn_cells)))
        l2_in   += ws_to_l2
        h2       = self.l2(l2_in, F.reshape(fw_l2_l2, (n_rnn_cells, 4*n_rnn_cells)), F.reshape(fw_l2_peephole, (1, 3*n_rnn_cells)))

        # LSTM3
        l3_in   += F.linear(h2, F.reshape(fw_l2_l3, (n_rnn_cells, 4*n_rnn_cells)), F.reshape(fb_l2_l3, (1, 4*n_rnn_cells)))
        l3_in   += ws_to_l3
        h3       = self.l3(l3_in, F.reshape(fw_l3_l3, (n_rnn_cells, 4*n_rnn_cells)), F.reshape(fw_l3_peephole, (1, 3*n_rnn_cells)))

        # Mixture Density Network
        loss_network, xpred, eos_, pi_, mux_, muy_, sgx_, sgy_, rho_  = self.ms(
            xnext, 
            self.eow, 
            h1, h2, h3, 
            F.reshape(fw_l1_ms, (n_rnn_cells, 1 + n_mix_comp*6)), 
            F.reshape(fw_l2_ms, (n_rnn_cells, 1 + n_mix_comp*6)),
            F.reshape(fw_l3_ms, (n_rnn_cells, 1 + n_mix_comp*6)),
            F.reshape(fb_l1_ms, (1,           1 + n_mix_comp*6)), 
            F.reshape(fb_l2_ms, (1,           1 + n_mix_comp*6)),
            F.reshape(fb_l3_ms, (1,           1 + n_mix_comp*6)),
            prob_bias
        )

        return loss_network, xpred, eos_, pi_, mux_, muy_, sgx_, sgy_, rho_, loss_complex


# ===============================================
# Main entry point of the training process (main)
# ===============================================


@click.command()
@click.option('--data_dir', type=click.Path(exists=True), default='data/processed/OnlineHandWriting', help='Directory containing the data.')
@click.option('--output_dir', type=click.Path(exists=False), default='models', help='Directory for model checkpoints.')
@click.option('--batch_size', type=click.INT, default=64, help='Size of the mini-batches.')
@click.option('--peephole', type=click.INT, default=0, help='LSTM with Peephole.')
@click.option('--epochs', type=click.INT, default=500, help='Number of epoch for training.')
@click.option('--grad_clip', type=click.INT, default=0, help='Threshold for the gradient clipping.')
@click.option('--resume', type=click.STRING, default='', help='Resume the optimization from a snapshot.')
@click.option('--gpu', type=click.INT, default=-1, help='GPU ID (negative value is CPU).')
@click.option('--rnn_layers_number', type=click.INT, default=3, help='Number of layers for the RNN.')
@click.option('--rnn_cells_number', type=click.INT, default=400, help='Number of LSTM cells per layer.')
@click.option('--win_unit_number', type=click.INT, default=10, help='Number of soft-window components.')
@click.option('--mix_comp_number', type=click.INT, default=20, help='Numver of Gaussian components for mixture density output.')
@click.option('--random_seed', type=click.INT, default=None, help='Numver of Gaussian components for mixture density output.')
def main(data_dir, output_dir, batch_size, peephole, epochs, grad_clip, resume, gpu, rnn_layers_number, rnn_cells_number, win_unit_number, mix_comp_number, random_seed):
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

    """ Enable cupy, if available """
    if gpu > -1:
        logger.info("Enabling CUpy")
        xp = cuda.cupy
    else:
        xp = np

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
        model = Model(batch_size, n_chars, INPUT_SIZE, win_unit_number, rnn_cells_number, mix_comp_number, rnn_layers_number)
    else:
        model = ModelPeephole(batch_size, n_chars, INPUT_SIZE, win_unit_number, rnn_cells_number, mix_comp_number, rnn_layers_number)

    """ Setup the model """
    logger.info("Setuping the model")
    optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.90, beta2=0.999, eps=1e-08)
    optimizer.setup(model)

    if grad_clip is not 0:
        optimizers.add_hook(chainer.optimizers.GradientClipping(grad_clip))

    if resume:
        logger.info("Loading state from {}".format(output_dir + '/' + resume))
        chainer.serializers.load_npz(output_dir + "/" + resume, optimizer)

    """ Begin training """
    accum_loss_network_train = np.zeros((epochs, 5))
    accum_loss_network_valid = np.zeros((epochs, 5))
    accum_loss_complex_train = np.zeros((epochs, 2, 15))

    min_valid_loss = np.zeros(1)
    thr_valid_loss = -1100.0

    xp.random.seed(random_seed)
    np.random.seed(random_seed)
    for epoch in xrange(epochs):
        logger.info("Beginning trainin for epoch {}".format(epoch))



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
