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

class Model(chainer.Chain):
    """
    Handwriting Synthesis Network Model. No peephole

    Args:
        n_vocab:
    """
    def __init__(self, n_vocab, input_size, n_win_units, n_rnn_cells, n_mix_comp, n_rnn_layers, wscale=0.1, use_weight_noise=True):
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
    """    
    def get_weights_for_testing(self):
        fW, b = split_axis.split_axis(self.M, numpy.asarray([(self.in_size -1)*self.out_size]), axis=0)
        W = reshape.reshape(fW, (self.out_size, self.in_size -1))
        return W, b
    """
    def __call__(self, x_now, xnext, cs, ls, prob_bias, n_batches, renew_weights, testing=False):
        xp = cuda.get_array_module(*x_now.data)
        
        if renew_weights:
            
            if testing:
                # Generate weights w/o Adaptive Noises
                def get_weights_for_testing(self):
                    fW, b = F.split_axis(self.M, np.asarray([(self.in_size -1)*self.out_size]), axis=0)
                    W = F.reshape(fW, (self.out_size, self.in_size -1))
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
                self.w_x_l1, self.b_x_l1, loss_ = self.awn_x_l1(n_batches)
                self.loss_complex = F.reshape(loss_, (1,1))
                
                self.w_x_l2, self.b_x_l2, loss_ = self.awn_x_l2(n_batches)
                self.loss_complex = F.concat((self.loss_complex, F.reshape(loss_, (1,1))))
                
                self.w_x_l3, self.b_x_l3, loss_ = self.awn_x_l3(n_batches)
                self.loss_complex = F.concat((self.loss_complex, F.reshape(loss_, (1,1))))
                
                self.w_l1_ws, self.b_l1_ws, loss_ = self.awn_l1_ws(n_batches)
                self.loss_complex = F.concat((self.loss_complex, F.reshape(loss_, (1,1))))
                
                self.w_ws_l1, self.b_ws_l1, loss_ = self.awn_ws_l1(n_batches)
                self.loss_complex = F.concat((self.loss_complex, F.reshape(loss_, (1,1))))
                
                self.w_ws_l2, self.b_ws_l2, loss_ = self.awn_ws_l2(n_batches)
                self.loss_complex = F.concat((self.loss_complex, F.reshape(loss_, (1,1))))
                
                self.w_ws_l3, self.b_ws_l3, loss_ = self.awn_ws_l3(n_batches)
                self.loss_complex = F.concat((self.loss_complex, F.reshape(loss_, (1,1))))
                
                self.w_l1_l1, self.b_l1_l1, loss_  = self.awn_l1_l1(n_batches)
                self.loss_complex = F.concat((self.loss_complex, F.reshape(loss_, (1,1))))
                
                self.w_l2_l2, self.b_l2_l2, loss_ = self.awn_l2_l2(n_batches)
                self.loss_complex = F.concat((self.loss_complex, F.reshape(loss_, (1,1))))
                
                self.w_l3_l3, self.b_l3_l3, loss_ = self.awn_l3_l3(n_batches)
                self.loss_complex = F.concat((self.loss_complex, F.reshape(loss_, (1,1))))
                
                self.w_l1_l2, self.b_l1_l2, loss_ = self.awn_l1_l2(n_batches)
                self.loss_complex = F.concat((self.loss_complex, F.reshape(loss_, (1,1))))
                
                self.w_l2_l3, self.b_l2_l3, loss_ = self.awn_l2_l3(n_batches)
                self.loss_complex = F.concat((self.loss_complex, F.reshape(loss_, (1,1))))
                
                self.w_l1_ms, self.b_l1_ms, loss_  = self.awn_l1_ms(n_batches)
                self.loss_complex = F.concat((self.loss_complex, F.reshape(loss_, (1,1))))
                
                self.w_l2_ms, self.b_l2_ms, loss_  = self.awn_l2_ms(n_batches)
                self.loss_complex = F.concat((self.loss_complex, F.reshape(loss_, (1,1))))
                
                self.w_l3_ms, self.b_l3_ms, loss_  = self.awn_l3_ms(n_batches)
                self.loss_complex = F.concat((self.loss_complex, F.reshape(loss_, (1,1))))
            
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
    def __init__(self, n_vocab, input_size, n_win_units, n_rnn_cells, n_mix_comp):
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

    def __call__(self, x_now, xnext, cs, ls, prob_bias, batch_size, enoise):

        n_batches = variable.Variable(xp.asarray([batch_size], dtype=x_now.data.dtype), volatile='auto')
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


def main(data_dir, output_dir, batch_size, peephole, epochs, grad_clip, resume, gpu, adaptive_noise, update_weight, use_weight_noise, save_interval, truncated_back_prop_len, truncated_data_samples, rnn_layers_number, rnn_cells_number, win_unit_number, mix_comp_number, random_seed):
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
        model = Model(n_chars, INPUT_SIZE, win_unit_number, rnn_cells_number, mix_comp_number, rnn_layers_number, use_weight_noise=True if use_weight_noise == 1 else False)
    else:
        model = ModelPeephole(n_chars, INPUT_SIZE, win_unit_number, rnn_cells_number, mix_comp_number, rnn_layers_number)

    """ Setup the model """
    logger.info("Setuping the model")
    optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.90, beta2=0.999, eps=1e-08)
    optimizer.setup(model)

    if grad_clip is not 0:
        optimizers.add_hook(chainer.optimizers.GradientClipping(grad_clip))

    if resume:
        logger.info("Loading state from {}".format(output_dir + '/' + resume))
        chainer.serializers.load_npz(output_dir + "/" + resume, optimizer)

    """ Enable cupy, if available """
    if gpu > -1:
        logger.info("Enabling CUpy")
        cuda.get_device(gpu).use()
        xp = cuda.cupy
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

            n_batches = chainer.Variable(xp.asarray(n_batches_counter).astype(xp.int32), volatile='auto')

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
            model.zerograds()

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

                loss_i, x_pred, eos_i, pi_i, mux_i, muy_i, sgx_i, sgy_i, rho_i, loss_complex_i = model(x_now, x_next, cs, ls, prob_bias, n_batches, local_update_weight)

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

                # One-hot encoding of character for all the sequence
                cs_data = xp.zeros((offset_valid_batch_size, n_chars, n_max_seq_length))
                ls_data = xp.zeros((offset_valid_batch_size, 1))
                for j in xrange(len(valid_characters_batch)):
                    for k in xrange(len(valid_characters_batch[j])):
                        length = valid_characters_batch[j][k]
                        cs_data[j, length, k] = 1.0

                cs = chainer.Variable(xp.asarray(cs_data).astype(xp.float32), volatile='on')
                ls = chainer.Variable(xp.asarray(ls_data).astype(xp.float32), volatile='on')

                # Training parameters
                loss_network = xp.zeros((offset_valid_batch_size, 1))
                loss_complex = xp.zeros((offset_valid_batch_size, 1))
                
                # For each data in the batchsize run the validation
                for t in xrange(t_max-1):
                    x_now = chainer.Variable(xp.asarray(valid_data_batch[0:offset_valid_batch_size, t, 0:x_dim_valid]).astype(xp.float32), volatile='on')
                    x_next = chainer.Variable(xp.asarray(valid_data_batch[0:offset_valid_batch_size, t+1, 0:x_dim_valid]).astype(xp.float32), volatile='on')
                    logger.info("Validating data {0}/{1} for epoch {2}".format(t+1, t_max, epoch+1))
                    loss_i, x_pred, eos_i, pi_i, mux_i, muy_i, sgx_i, sgy_i, rho_i, loss_complex_i = model(x_now, x_next, cs, ls, prob_bias, n_batches, renew_weights=False, testing=True)
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

            if epoch % save_interval:
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
@click.option('--resume', type=click.STRING, default='', help='Resume the optimization from a snapshot.')
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
