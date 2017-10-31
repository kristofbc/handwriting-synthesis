#!/usr/bin/env python # -*- coding: utf-8 -*-

# Generate a prediction based on a trained model
# ==============================================

import numpy as np
import chainer
import chainer.functions as F
from chainer import variable

try:
    import cupy
except:
    cupy = np
    pass

from train_model import Model

import click
import os
import time
import logging
import random
import math

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

# =================
# Helpers functions
# =================

# ===============================================
# Main entry point of the training process (main)
# ===============================================

def main(model_dir, model_name, text, models_dir, data_dir, prime_index, batchsize, gpu, peephole, grad_clip, truncated_back_prop_len, truncated_data_samples, rnn_layers_number, rnn_cells_number, win_unit_number, mix_comp_number, random_seed):
    """ Generate a handwriting sequence of ASCII characters """
    logger = logging.getLogger(__name__)
    path = models_dir + '/' + model_dir
    if not os.path.exists(path + '/' + model_name):
        raise ValueError("Directory {} does not exists".format(path))

    """ Import required data """
    translation = np.load(os.path.join(data_dir, 'translation'))
    rev_translation = {v: k for k, v in translation.items()}
    charset = [rev_translation[i] for i in range(len(rev_translation))]
    charset[0] = ''

    """ Load the model """
    logger.info("Importing the model {}".format(model_name))
    model = Model(rnn_cells_number, rnn_layers_number, mix_comp_number, win_unit_number)

    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()
        model.to_gpu()
        xp = cupy
    else:
        xp = np
    
    chainer.serializers.load_npz(path + "/" + model_name, model)
    logger.info("Model imported successfully")

    """ Proceed to sampling """
    text_chars = np.asarray([translation.get(c, 0) for c in text])
    coords = np.asarray([0., 0., 1.])
    coords = [coords]

    # Priming
    style = None
    prime_len = 0
    if prime_index > -1:
        # Priming consist of first training the network with the character sequence + strokes then
        #   generating the requested sequence of character and finally clipping the output to the requested character
        styles = np.load(os.path.join(data_dir, 'styles'))
        if prime_index > len(styles[0]):
            raise ValueError("Prime index does not exists")
        
        style_coords = styles[0][prime_index]
        style_text = styles[1][prime_index]
        prime_len = len(style_coords)
        coords = list(style_coords)
        coord = coords[0] # Set the first pen stroke as the first element to process
        text_chars = np.r_[style_text, text_chars] # Concatenate on axis 1 the prime text + synthesis text ascii characters
        sequence_prime = np.eye(len(translation), dtype=np.float32)[style_text]
        sequence_prime = np.expand_dims(np.concatenate([sequence_prime, np.zeros((1, len(translation)))]), axis=0)

    sequence = np.eye(len(translation), dtype=np.float32)[text_chars]
    sequence = np.expand_dims(np.concatenate([sequence, np.zeros((1, len(translation)))]), axis=0)

    phi_data, window_data, kappa_data, stroke_data, lstms_cs_data, lstms_hs_data = [], [], [], [], [], []
    model.reset_state()
    losses = []
    for s in xrange(1, 60 * len(text_chars) + 1):
        is_priming = False
        if s < prime_len: # [0, prime_len] is priming, [prime_len+1, inf] is synthesis
            is_priming = True

        logger.info("[{:5d}] Sampling ... {}".format(s, "priming" if is_priming else "synthesis"))

        coord = coord[None, None, ...]
        coord = np.concatenate((coord, coord), axis=1)
        loss_t = model([
            coord.astype(np.float32),
            sequence_prime.astype(np.float32) if is_priming else sequence.astype(np.float32)
        ])
        

        if is_priming:
            # Use the real coordinate when priming
            coord = coords[s]
        else:
            # Synthesis mode
            def sample(e, mu1, mu2, s1, s2, rho):
                cov = np.asarray([[s1 * s1, s1 * s2 * rho], [s1 * s2 * rho, s2 * s2]])
                mean = np.asarray([mu1, mu2])
                
                x1, x2 = np.random.multivariate_normal(mean, cov)
                end = np.random.binomial(1, e)
                return np.asarray([x1, x2, end])

            e, pi, mu1, mu2, s1, s2, rho = model.get_mdn()
            e = chainer.cuda.to_cpu(e.data)
            pi = chainer.cuda.to_cpu(pi.data)
            mu1 = chainer.cuda.to_cpu(mu1.data)
            mu2 = chainer.cuda.to_cpu(mu2.data)
            s1 = chainer.cuda.to_cpu(s1.data)
            s2 = chainer.cuda.to_cpu(s2.data)
            rho = chainer.cuda.to_cpu(rho.data)

            g = np.random.choice(np.arange(pi.shape[1]), p=pi[0])
            coord = sample(e[0, 0], mu1[0, g], mu2[0, g], s1[0, g], s2[0, g], rho[0, g])
            coords += [coord]
            stroke_data += [[mu1[0, g], mu2[0, g], s1[0, g], s2[0, g], rho[0, g], coord[2]]]
            
            # Extract LSTM data here
            # ...

            window, kappa, finish, phi = model.get_window()
            window = chainer.cuda.to_cpu(window.data)
            kappa = chainer.cuda.to_cpu(kappa.data)
            finish = chainer.cuda.to_cpu(finish.data)
            phi = chainer.cuda.to_cpu(phi.data)

            phi_data += [phi[0, :]]
            window_data += [window[0, :]]
            kappa_data += [kappa[0, :]]

            if finish[0, 0] > 0.8:
                break

    logger.info("Finished sampling")
    coords = np.asarray(coords[prime_len:])
    coords[-1, 2] = 1.

    """ Save the synthesis data """
    # Save the data ...
    #np.save(os.path.join(path, model_name, '{}-lstm-cs.npy'.format(now)), lstm_cs)
    #np.save(os.path.join(path, model_name, '{}-lstm-hs.npy'.format(now)), lstm_hs)
    #np.save(os.path.join(path, model_name, '{}-window.npy'.format(now)), window_data)
    #np.save(os.path.join(path, model_name, '{}-phi.npy'.format(now)), phi_data)
    #np.save(os.path.join(path, model_name, '{}-stroke.npy'.format(now)), stroke_data)
    #np.save(os.path.join(path, model_name, '{}-coords.npy'.format(now)), coords)

    """ Plot the data """
    epsilon = 1e-8
    strokes = np.array(stroke_data)
    strokes[:, :2] = np.cumsum(strokes[:, :2], axis=0)
    minx1, maxx1 = np.min(strokes[:, 0]), np.max(strokes[:, 0])
    minx2, maxx2 = np.min(strokes[:, 1]), np.max(strokes[:, 1])

    def split_strokes(points):
        points = np.asarray(points)
        strokes = []
        b = 0
        for i in range(len(points)):
            if points[i, 2] == 1.:
                strokes += [points[b: i + 1, :2].copy()]
                b = i + 1
        return strokes

    def cumsum(points):
        sums = np.cumsum(points[:, :2], axis=0)
        return np.concatenate([sums, points[:, 2:]], axis=1)

    plt.figure(1)
    plt.title(text)
    for stroke in split_strokes(cumsum(np.asarray(coords))):
        plt.plot(stroke[:, 0], -stroke[:, 1])
    plt.axes().set_aspect('equal')
    plt.show()

# ===============
# CLI Entry point
# ===============

@click.command()
@click.argument('model_dir', type=click.STRING)
@click.argument('model_name', type=click.STRING)
@click.argument('text', type=click.STRING)
@click.option('--models_dir', type=click.Path(exists=True), default='models', help='Directory containing the models.')
@click.option('--data_dir', type=click.Path(exists=True), default='data/processed/OnlineHandWriting/tf', help='Directory containing data.')
@click.option('--prime_index', type=click.INT, default=-1, help='Priming index.')
@click.option('--batchsize', type=click.INT, default=1, help='Control the number of MDN outputs.')
@click.option('--gpu', type=click.INT, default=-1, help='ID of the gpu to use')
@click.option('--peephole', type=click.INT, default=0, help='LSTM with Peephole.')
@click.option('--grad_clip', type=click.INT, default=0, help='Threshold for the gradient clipping.')
@click.option('--truncated_back_prop_len', type=click.INT, default=50, help='Number of backpropagation before stopping.')
@click.option('--truncated_data_samples', type=click.INT, default=500, help='Number of samples to use inside a data.')
@click.option('--rnn_layers_number', type=click.INT, default=3, help='Number of layers for the RNN.')
@click.option('--rnn_cells_number', type=click.INT, default=400, help='Number of LSTM cells per layer.')
@click.option('--win_unit_number', type=click.INT, default=10, help='Number of soft-window components.')
@click.option('--mix_comp_number', type=click.INT, default=20, help='Number of Gaussian components for mixture density output.')
@click.option('--random_seed', type=click.INT, default=None, help='Numver of Gaussian components for mixture density output.')
def cli(**kwargs):
    main(**kwargs)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    cli()
