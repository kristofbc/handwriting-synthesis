#!/usr/bin/env python # -*- coding: utf-8 -*-

# Generate handwriting from a sequence of characters
# ==================================================

import time
import os
import click
import logging
import math
import copy

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
from chainer.utils import type_check

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
            pad = np.full((pad_length, 3), 0)
            pad[:,2] = 2.0
            data[i] = np.vstack([data[i], pad])
        tmp1.append(np.asarray(data[i]))
        tmp2.append(np.asarray(characters[i]))
    
    return np.asarray(tmp1).astype(np.float32), np.asarray(tmp2) 

def one_hot(data, characters, n_chars, n_max_seq_length):
    cs = np.zeros((len(data), n_chars, n_max_seq_length), dtype=np.float32)

    for i in xrange(len(data)):
        for j in xrange(len(characters[i])):
            k = characters[i][j]
            cs[i, k, j] = 1.0

    return cs

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

        with self.init_scope():
            #self.input_linear = L.Linear(3*mixture_size)
            self.mixture_W = chainer.Parameter(chainer.initializers.Normal(0.075), (3*mixture_size, unit_size))
            self.mixture_b = chainer.Parameter(chainer.initializers.Normal(0.25), (3*mixture_size))

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
        #x_h = self.input_linear(x)
        x_h = F.linear(x, self.mixture_W, self.mixture_b)
        a_h, b_h, k_h = F.split_axis(x_h, [self.mixture_size, 2 * self.mixture_size], axis=1)
        K = a_h.shape[1]

        if self.k_prev is None:
            with cuda.get_device_from_id(self._device_id):
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

        with self.init_scope():
            #self.input_linear = L.Linear(1 + n_mdn_comp * 6) # end_of_stroke + 6 parameters per Gaussian
            self.mdn_W = chainer.Parameter(chainer.initializers.Normal(0.075), (1 + n_mdn_comp * 6, n_units))
            self.mdn_b = chainer.Parameter(chainer.initializers.Normal(0.075), (1 + n_mdn_comp * 6))

        self.n_mdn_comp = n_mdn_comp
        self.n_unit = n_units
        self.p_bias = prob_bias

        self.eos, self.pi, self.mu_x1, self.mu_x2, self.s_x1, self.s_x2, self.rho = None, None, None, None, None, None, None
        self.gamma = None
        self.loss = None

    def reset_state(self):
        """
            Reset the Variables
        """
        self.eos, self.pi, self.mu_x1, self.mu_x2, self.s_x1, self.s_x2, self.rho = None, None, None, None, None, None, None
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
        x, x_next = inputs
        xp = cuda.get_array_module(*x)

        # Extract the MDN's parameters
        #x = self.input_linear(x)
        x = F.linear(x, self.mdn_W, self.mdn_b)

        eos_h, pi_h, mu_x1_h, mu_x2_h, s_x1_h, s_x2_h, rho_h = F.split_axis(
            x, np.asarray([1 + i*self.n_mdn_comp for i in xrange(5+1)]), axis=1
        )

        # Add the bias to the parameter to change the shape of the prediction
        pi_h *= (1. + self.p_bias)
        s_x1_h -= self.p_bias
        s_x2_h -= self.p_bias

        self.loss, _, self.eos, self.pi, self.mu_x1, self.mu_x2, self.s_x1, self.s_x2, self.rho = mixture_density_network(
            x, eos_h, pi_h, mu_x1_h, mu_x2_h, s_x1_h, s_x2_h, rho_h
        )

        self.loss = F.sum(self.loss)
        return self.loss

        # Compute the parameters used in the MDN. Eq 18 to 22
        z_eos = 1. / (1. + F.exp(eos_h))
        z_pi = F.softmax(pi_h)
        z_mu_x1 = mu_x2_h
        z_mu_x2 = mu_x2_h
        z_s_x1 = F.exp(s_x1_h)
        z_s_x2 = F.exp(s_x2_h)
        z_rho = F.tanh(rho_h)

        self.eos = z_eos
        self.pi = z_pi
        self.mu_x1 = z_mu_x1
        self.mu_x2 = z_mu_x2
        self.s_x1 = z_s_x1
        self.s_x2 = z_s_x2
        self.rho = z_rho

        x1 = x_next[:, 0:1]
        x2 = x_next[:, 1:2]
        x3 = x_next[:, 2:3]

        # Compute "Z". Eq 25
        x1_b = F.broadcast_to(x1, z_mu_x1.shape)
        x2_b = F.broadcast_to(x2, z_mu_x2.shape)
        z = (F.square(x1_b - z_mu_x1)/F.square(z_s_x1)) + \
            (F.square(x2_b - z_mu_x2)/F.square(z_s_x2)) - \
            ((2. * z_rho * (x1_b - z_mu_x1) * (x2_b - z_mu_x2))/(z_s_x1*z_s_x2))

        # Compute "N". Eq 24
        n = (1. / ((2. * np.pi * z_s_x1 * z_s_x2 * F.sqrt(1. - F.square(z_rho))) + 1e-20)) * F.exp(-z / (2. * (1. - F.square(z_rho))))

        # Compute the loss. Eq 26
        gamma = z_pi * n
        gamma_sum = F.sum(gamma, 1, keepdims=True) + 1e-20
        self.gamma = gamma / F.broadcast_to(gamma_sum, gamma.shape)
        loss_left = -F.log(gamma_sum)

        #z_sum = (z_eos * x3) + ((1. - z_eos) * (1. - x3)) + 1e-20
        #loss_right = -F.log(z_sum)
        loss_right = -x3 * F.log(z_eos) - (1. - x3) * F.log(1. - z_eos)
        
        # @TODO probably mask the "2" values!
        loss = loss_left + loss_right
        loss = F.sum(loss)

        return loss
        

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
            prob_bias (float): bias sampling value
        Returns
            loss (float)
    """

    def __init__(self, n_units, n_mixture_components, n_window_unit, prob_bias = 0.):
        super(Model, self).__init__()

        with self.init_scope():
            # LSTMs layers
            self.lstm1 = L.LSTM(n_units)
            self.lstm2 = L.LSTM(n_units)
            self.lstm3 = L.LSTM(n_units)
            
            # Attention mechanism
            self.sw = SoftWindow(n_window_unit, n_units)

            # Mixture Density Network
            self.mdn = MixtureDensityNetwork(n_mixture_components, n_units, prob_bias)
            # Linear connections for some layers

        self.n_units = n_units
        self.n_mixture_components = n_mixture_components
        self.n_window_unit = n_window_unit
        self.p_bias = prob_bias

        self.loss = None

    def reset_state(self):
        """
            Reset own Variables and Links Variables
        """
        self.sw.reset_state()
        self.mdn.reset_state()
        self.loss = None

    def __call__(self, inputs):
        """
            Perform the handwriting prediction

            Args:
                inputs (float[][]): a tensor containing current position (X), next position (Xt+1), char sequence (cs)
            Returns:
                loss (float)
        """
        data, cs_data = inputs
        batch_size, t_max, x_dim = data.shape

        with cuda.get_device_from_id(self._device_id):
            cs = variable.Variable(cs_data)

        # Train all samples in batch
        loss = 0
        for t in xrange(t_max-1):
            with cuda.get_device_from_id(self._device_id):
                x_now = variable.Variable(data[:, t, :])
                x_next = variable.Variable(data[:, t+1, :])

            # LSTM1
            x = self.lstm1(x_now)
            
            # Attention Mechanism
            sw = self.sw([x, cs])

            # LSTM2
            x = F.concat((x, sw), axis=1)
            x = F.concat((x, x_now), axis=1)
            x = self.lstm2(x)

            # LSTM3
            x = self.lstm3(x)

            # MDN
            loss += self.mdn([x, x_next])
            #print(loss)

        loss /= (batch_size * t_max)
        self.loss = loss

        return self.loss


# ===============================================
# Main entry point of the training process (main)
# ===============================================


def main(data_dir, output_dir, batch_size, peephole, epochs, grad_clip, resume_dir, resume_model, resume_optimizer, gpu, adaptive_noise, update_weight, use_weight_noise, save_interval, validation_interval, truncated_back_prop_len, truncated_data_samples, rnn_layers_number, rnn_cells_number, win_unit_number, mix_comp_number, random_seed, debug):
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
        return data

    logger.info("Fetching the model and the inputs")
    train_data = load_data(data_dir + "/train/train_data")
    train_characters = load_data(data_dir + "/train/train_characters")
    valid_data = load_data(data_dir + "/valid/valid_data")
    valid_characters = load_data(data_dir + "/valid/valid_characters")
    vocab = load_data(data_dir + "/vocabulary")

    n_max_seq_length = max(get_max_sequence_length(train_characters), get_max_sequence_length(valid_characters))
    n_chars = len(vocab)

    """ Create the model """
    def op_models(models, op, *args):
        rets = []
        for i in xrange(len(models)):
            rets.append(op(i, models[i], *args))

        return rets

    gpu = gpu.split(',')
    logger.info("Creating {0} model(s)".format(len(gpu)))
    models = []
    for i in xrange(len(gpu)):
        if i == 0:
            if peephole == 0:
                models.append(Model(rnn_cells_number, mix_comp_number, win_unit_number))
            else:
                raise ValueError("Peephole is not yet supported")
                #model = ModelPeephole(n_chars, INPUT_SIZE, win_unit_number, rnn_cells_number, mix_comp_number, rnn_layers_number)
        else:
            models.append(copy.deepcopy(models[0]))

    """ Setup the model """
    logger.info("Setuping {0} model(s)".format(len(gpu)))
    # Optimizer is only on the "master" model
    #optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.90, beta2=0.999, eps=1e-08)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(models[0])

    if grad_clip is not 0:
        optimizer.add_hook(chainer.optimizers.GradientClipping(grad_clip))

    if resume_dir:
        logger.info("Loading state from {}".format(output_dir + '/' + resume_dir))
        if resume_model != "":
            chainer.serializers.load_npz(output_dir + "/" + resume_dir + "/" + resume_model, model)

            if len(models) > 1:
                for i in models[1:]:
                    models[i] = copy.deepcopy(models[0])

        if resume_optimizer != "":
            chainer.serializers.load_npz(output_dir + "/" + resume_dir + "/" + resume_optimizer, optimizer)

    """ Enable cupy, if available """
    for i in xrange(len(gpu)):
        if int(gpu[i]) > -1:
            logger.info("Enabling CUpy for model #{0}".format(i))
            #chainer.cuda.get_device_from_id(gpu).use()
            #xp = cupy
            models[i].to_gpu(int(gpu[i]))
            xp = np
        else:
            xp = np

    """ Prepare data """
    sets = []
    for i in [[train_data, train_characters], [valid_data, valid_characters]]:
        data, characters = i
        grouped = group_data(data, characters)
        sets.append(grouped)

    train_set, valid_set = sets

    """ Create the data iterators """
    train_iter = chainer.iterators.SerialIterator(train_set, batch_size, repeat=True, shuffle=True)
    valid_iter = chainer.iterators.SerialIterator(valid_set, batch_size, repeat=True, shuffle=True)

    """ Start training """
    xp.random.seed(random_seed)
    np.random.seed(random_seed)
    time_epoch_start = time.time()
    history_network_train = []
    history_network_valid = []
    history_train = []
    history_valid = []

    logger.info("Starting training with {0} mini-batches for {1} epochs".format(math.ceil(len(train_set)/batch_size), epochs))
    while train_iter.epoch < epochs:
        epoch = train_iter.epoch
        batch = np.asarray(train_iter.next())
        time_iteration_start = time.time()
        
        # Unpack the training data
        train_data_batch, train_characters_batch = pad_data(batch[:, 0], batch[:, 1])
        cs_data = one_hot(train_data_batch, train_characters_batch, n_chars, n_max_seq_length)

        # Train the batch
        #optimizer.update(model, [train_data_batch, train_characters_batch, cs_data])
        def make_inputs_models(i, model, b_size):
            t_d_batch = train_data_batch[i*b_size:(i+1)*b_size]
            cs_d_batch = cs_data[i*b_size:(i+1)*b_size]

            if model.xp != np:
                t_d_batch = cuda.to_gpu(t_d_batch.copy(), model._device_id)
                cs_d_batch = cuda.to_gpu(cs_d_batch.copy(), model._device_id)

            return [t_d_batch, cs_d_batch]

        inputs_models = op_models(models, make_inputs_models, batch_size/len(models))
        losses = op_models(models, lambda i, model, inputs: model(inputs[i]), inputs_models) 

        # Truncated back-prop at each time-step
        op_models(models, lambda i, model: model.cleargrads())
        op_models(losses, lambda i, loss: loss.backward())
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

        time_iteration_end = time.time()-time_iteration_start
        loss = cuda.to_cpu(models[0].loss.data)
        #loss = op_models(models, lambda i, model: cuda.to_cpu(model.loss.data))
        #loss = np.asarray(loss).sum()
        history_train.append([loss, time_iteration_end])

        op_models(models, lambda i, model: model.reset_state())

        logger.info("[TRAIN] Epoch #{0} ({1}/{2}): loss = {3}, time = {4}".format(epoch+1, len(history_train), math.ceil(len(train_set)/batch_size), loss, time_iteration_end))

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
                        models[0]([valid_data_batch, valid_characters_batch, valid_cs_data])
                        time_iteration_end = time.time()-time_iteration_start
                        loss = cuda.to_cpu(models[0].loss.data)
                        history_valid.append([loss, time_iteration_end])
                        logger.info("[VALID] Epoch #{0} ({1}/{2}): loss = {3}, time = {4}".format(epoch+1, len(history_valid), math.ceil(len(valid_set)/batch_size), loss, time_iteration_end))

                        models[0].reset_state()
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
                save_dir = output_dir + '/' + model_suffix_dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # Save the model and optimizer
                np.save(save_dir + '/model-{}'.format(epoch+1), models[0])
                np.save(save_dir + '/state-{}'.format(epoch+1), optimizer)

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
@click.option('--data_dir', type=click.Path(exists=True), default='data/processed/OnlineHandWriting', help='Directory containing the data.')
@click.option('--output_dir', type=click.Path(exists=False), default='models', help='Directory for model checkpoints.')
@click.option('--batch_size', type=click.INT, default=64, help='Size of the mini-batches.')
@click.option('--peephole', type=click.INT, default=0, help='LSTM with Peephole.')
@click.option('--epochs', type=click.INT, default=500, help='Number of epoch for training.')
@click.option('--grad_clip', type=click.INT, default=0, help='Threshold for the gradient clipping.')
@click.option('--resume_dir', type=click.STRING, default='', help='Directory name for resuming the optimization from a snapshot.')
@click.option('--resume_model', type=click.STRING, default='', help='Name of the model snapshot.')
@click.option('--resume_optimizer', type=click.STRING, default='', help='Name of the optimizer snapshot.')
@click.option('--gpu', type=click.STRING, default="-1", help='GPU ID (negative value is CPU).')
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
@click.option('--debug', type=click.INT, default=0, help='Chainer debugging mode.')
def cli(**kwargs):
    main(**kwargs)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    cli()
