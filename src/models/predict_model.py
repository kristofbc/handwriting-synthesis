#!/usr/bin/env python # -*- coding: utf-8 -*-

# Generate a prediction based on a trained model
# ==============================================

import numpy as np
import chainer
import chainer.functions as F

from train_model import Model
from train_model import get_padded_data
from train_model import get_max_sequence_length

import click
import os
import time
import logging

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

# =================
# Helpers functions
# =================


def get_char_to_index(text):
    """
        Get the index corresponding to the character

        Args:
            text (str): the text to convert
        Returns:
            (int[])
    """
    vocab = {}
    cha_index = []
    for l in text:
        tmp = []
        for s in l:
            dataset = np.ndarray(len(s[0]), dtype=np.int32)
            for i, cha in enumerate(s[0]):
                if cha not in vocab:
                    vocab[cha] = len(vocab)
                dataset[i] = vocab[cha]
            tmp.extend(dataset)
        cha_index.append(np.asarray(tmp))
    return cha_index, vocab

def draw_from_strokes(strokes, plt_inst=None):
    """
        Draw the image based on the strokes array

        Args:
            strokes (int[][]): An array containing the position of the strokes
            plt_inst (pyplot): An instance of matplotlib\\pyplot
        Returns:
            (pyplot)
    """
    if plt_inst is None:
        plt_inst = plt

    # Remove the last position
    last = [0,0,0]

    positions = []
    pos_list = [[], []]
    for point in strokes:
        # New letter
        if point[2] == 1:
            positions = positions + pos_list
            pos_list = [[], []]

        sum_x = last[0]+point[0]
        sum_y = last[1]+point[1]
        last = [sum_x, sum_y]

        # Continue the current letter
        if point[2] != 1:
            pos_list[0].append(sum_x)
            pos_list[1].append(sum_y)

    plt_inst.gca().invert_yaxis()
    plt_inst.plot(*positions)
    return plt_inst


def visualize_dataset(data_index, data_dir, train_set=True):
    if not os.path.exists(data_dir):
        raise ValueError("Directory {} does not exists".format(data_dir))

    data_index = int(data_index)
    suffix = "train" if train_set is True else "valid"
    data = np.load(data_dir + "/{0}/{1}_data".format(suffix, suffix))
    texts = np.load(data_dir + "/{0}/{1}_text".format(suffix, suffix))
    stats = np.load(data_dir + "/statistics")

    if data_index > len(data) or data_index > len(texts):
        raise ValueError("Data index {} does not exists in dataset".format(data_index))
    
    strokes = data[data_index]
    text = texts[data_index]

    # Must "un-normalize" the data
    strokes[:, 1] *= stats[3]
    strokes[:, 0] *= stats[2]
    strokes[:, 1] += stats[1]
    strokes[:, 0] += stats[0]

    # Draw the generate handwritten text
    plt.figure(1)
    plt.subplot("{0}{1}{2}".format(1, 1, 1))
    plt.title(text)
    draw_from_strokes(strokes, plt)
    plt.show()

# ===============================================
# Main entry point of the training process (main)
# ===============================================

def main(model_dir, model_name, text, models_dir, data_dir, downscale_factor, gpu, peephole, grad_clip, truncated_back_prop_len, truncated_data_samples, rnn_layers_number, rnn_cells_number, win_unit_number, mix_comp_number, random_seed):
    """ Generate a handwriting sequence of ASCII characters """
    logger = logging.getLogger(__name__)
    path = models_dir + '/' + model_dir
    if not os.path.exists(path + '/' + model_name):
        raise ValueError("Directory {} does not exists".format(path))

    """ Parse the input character sequences """
    # Create the vocabulary array
    logger.info("Parsing the input character sequence")
    if not isinstance(text, list):
        # Special case when user wants to visualize data set
        if "dataset:" in text:
            logger.info("Visualizing dataset")
            data_index = text[len("dataset:"):]
            visualize_dataset(data_index[:data_index.index(":")], data_dir, True if data_index[data_index.index(":")+1:] == "train" else False)
            return
        text = [text]

    cha_index, vocabulary = get_char_to_index(text)
    n_chars = len(vocabulary)
    train_characters = cha_index[0:len(text)]
    n_max_seq_length = get_max_sequence_length(train_characters)
    
    """ Import the trained model """
    logger.info("Importing the model {}".format(model_name))
    input_size = 3 # dimensions of x (x, y, end-of-stroke)

    # @TODO: should check if peephole is requested
    # @TODO: character size should be dynamic (74 is the length of the current data)
    original_vocabulary = np.load(data_dir + "/vocabulary")
    stats = np.load(data_dir + "/statistics")
    n_chars_training = len(original_vocabulary)
    model = Model(n_chars_training, input_size, win_unit_number, rnn_cells_number, mix_comp_number, rnn_layers_number)
    
    # Load the model
    chainer.serializers.load_npz(path + "/" + model_name, model)
    logger.info("Model imported successfully")

    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()
        xp = chainer.cupy
    else:
        xp = np

    """ Create the one-hot vocabulary sequence """
    batchsize = len(text)
    cs_data = xp.zeros((batchsize, n_chars, n_max_seq_length))
    ls_data = xp.zeros((batchsize, 1))
    for j in xrange(batchsize):
        for k in xrange(len(train_characters[j])):
            length = train_characters[j][k]
            cs_data[j, length, k] = 1.0

    # @TODO: Make sure the length of the data match the input of the model
    pad = xp.zeros((1, min(n_chars_training, abs(n_chars_training-n_chars)), n_max_seq_length)).astype(xp.float32)
    #pad.fill(2.0)
    cs_data = xp.concatenate((cs_data, pad), axis=1)

    cs = chainer.Variable(xp.asarray(cs_data).astype(xp.float32))
    ls = chainer.Variable(xp.asarray(ls_data).astype(xp.float32))

    # Start the prediction at postion 0, 0
    x_now = chainer.Variable(xp.asarray([[0,0,1]], dtype=np.float32))
    x_next = chainer.Variable(xp.asarray([[-1.0, -1.0, -1.0]]).astype(xp.float32))
    prob_bias = 0.0
    n_batches = chainer.Variable(xp.asarray(1+xp.zeros(1)).astype(xp.int32), volatile='auto')
    loss_network = xp.zeros((1, 1))

    # The loop is defined by the backprop length
    strokes = []
    for i in xrange(truncated_back_prop_len):
        # The weight are updated only in the first iteration
        local_update_weight = True if i == 0 else False

        # Predict the next stroke
        loss_i, x_pred, eos_i, pi_i, mux_i, muy_i, sgx_i, sgy_i, rho_i, loss_complex_i = model(x_now, x_next, cs, ls, prob_bias, n_batches, local_update_weight, testing=False)
        loss_network += loss_i.data

        # Transform the output
        # ...

        # Store the results of the stoke
        strokes = x_pred.data if len(strokes) == 0 else xp.concatenate((strokes, x_pred.data))

        # New positions
        x_now = x_pred.data.astype(xp.float32)

    # Compile the results
    losses_network_cpu = chainer.cuda.to_cpu(xp.copy(loss_network))

    # @TODO: must fetch the stats dynamically
    # Must "un-normalize" the generated strokes
    strokes[:, 1] *= stats[3]
    strokes[:, 0] *= stats[2]
    strokes[:, 1] += stats[1]
    strokes[:, 0] += stats[0]

    # Draw the generate handwritten text
    plt.figure(1)
    strokes = xp.concatenate((strokes, xp.asarray([[0.0, 0.0, 1.0]])))
    for i in xrange(len(text)):
        plt.subplot("{0}{1}{2}".format(len(text), 1, i+1))
        plt.title(text)
        draw_from_strokes(strokes, plt)
        plt.show()

    print(cs_data)
    exit()


# ===============
# CLI Entry point
# ===============

@click.command()
@click.argument('model_dir', type=click.STRING)
@click.argument('model_name', type=click.STRING)
@click.argument('text', type=click.STRING)
@click.option('--models_dir', type=click.Path(exists=True), default='models', help='Directory containing the models.')
@click.option('--data_dir', type=click.Path(exists=True), default='data/processed/OnlineHandWriting', help='Directory containing data.')
@click.option('--downscale_factor', type=click.FLOAT, default=0.5, help='Downscale the image by this factor.')
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
