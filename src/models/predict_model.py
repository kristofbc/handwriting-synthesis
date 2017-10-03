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
from train_model import get_max_sequence_length
from train_model import get_expanded_stroke_position

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


def get_char_to_index(text, vocabulary=None):
    """
        Get the index corresponding to the character

        Args:
            text (str): the text to convert
            vocabulary (dict): available vocabulary
        Returns:
            (int[])
    """
    vocab = vocabulary if vocabulary is not None else {}
    cha_index = []
    for l in text:
        tmp = []
        for s in l:
            dataset = np.ndarray(len(s[0]), dtype=np.int32)
            for i, cha in enumerate(s[0]):
                if cha not in vocabulary:
                    if vocabulary is not None:
                        raise ValueError("Character '{}' is not in vocabulary.".format(cha))
                    else:
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

def main(model_dir, model_name, text, models_dir, data_dir, prime_index, batchsize, gpu, peephole, grad_clip, truncated_back_prop_len, truncated_data_samples, rnn_layers_number, rnn_cells_number, win_unit_number, mix_comp_number, random_seed):
    """ Generate a handwriting sequence of ASCII characters """
    logger = logging.getLogger(__name__)
    path = models_dir + '/' + model_dir
    if not os.path.exists(path + '/' + model_name):
        raise ValueError("Directory {} does not exists".format(path))

    """ Import required data """
    logger.info("Importing dataset")
    original_vocabulary = np.load(data_dir + "/vocabulary")
    stats = np.load(data_dir + "/statistics")
    train_data = np.load(data_dir + "/train/train_data")
    train_characters = np.load(data_dir + "/train/train_characters")
    vocabulary = np.load(data_dir + "/vocabulary")

    # Each position time step is composed of
    # deltaX, deltaY, p1, p2, p3
    # deltaX, deltaY is the pen position's offsets
    # p1 = pen is touching paper
    # p2 = pen will be lifted from the paper (don't draw next stroke)
    # p3 = handwriting is completed
    train_data = get_expanded_stroke_position(train_data)
    #valid_data = get_expanded_stroke_position(valid_data)

    """ Parse the input character sequences """
    # Create the vocabulary array
    logger.info("Parsing the input character sequence")
    data_index = []
    if not isinstance(text, list):
        # Special case when user wants to visualize data set
        if "dataset:" in text:
            logger.info("Visualizing dataset")
            data_index = text[len("dataset:"):]
            visualize_dataset(data_index[:data_index.index(":")], data_dir, True if data_index[data_index.index(":")+1:] == "train" else False)
            return
        # Special case when the user wants to compare the generation against ground true
        elif "gt:" in text:
            logger.info("Importing requested ground-truth")
            data_index = text[len("gt:"):]
            
            if data_index == "*":
                data_index = range(len(train_data))

            if isinstance(data_index, basestring):
                data_index = [int(data_index)]

            text = []
            inv_map = {v: k for k, v in original_vocabulary.iteritems()}
            for idx in data_index:
                if idx > len(train_data):
                    raise ValueError("Index {} is not in training set".format(idx))
                
                tmp = ''.join([inv_map[char_idx] for char_idx in train_characters[idx]])
                text.append(tmp)
        # Default text parameter
        else:
            text = [text]

    #n_chars = len(vocabulary)
    # @TODO: character size should be dynamic (83 is the length of the current data)
    n_chars = 81#len(vocab)
    
    """ Import the trained model """
    logger.info("Importing the model {}".format(model_name))
    input_size = 5 # dimensions of x (x, y, end-of-stroke)

    # @TODO: should check if peephole is requested
    #n_chars_training = len(original_vocabulary)
    model = Model(rnn_layers_number, rnn_cells_number, mix_comp_number, win_unit_number)
    
    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()
        model.to_gpu()
        xp = cupy
    else:
        xp = np

    # Load the model
    logger.info("Model imported successfully")
    chainer.serializers.load_npz(path + "/" + model_name, model)

    """ Priming """
    # Priming consist of using the original author's sentence, then appending the new sentence to it
    if prime_index == -2:
        # display available authors styles
        logger.info("Available styles between: {0} and {1}".format(0, len(train_data)-1))
        logger.info('To visualize a style: TEXT="dataset:{styleid}:train')
        exit()

    if prime_index != -1:
        if prime_index > len(train_data):
            raise ValueError("Prime index is not in train_data")

        text_train_character, _ = get_char_to_index(text, vocabulary)
        batchsize = len(text_train_character)
        n_max_seq_length = get_max_sequence_length(text_train_character)

        t_max_prime, x_dim_prime = train_data[prime_index].shape
        t_data_prime = np.where(train_data[prime_index][:, 2] == 2)[0].min()

        prime_data_original = train_data[prime_index][0:t_data_prime, :]
        prime_data_original = np.expand_dims(prime_data_original, axis=0)
        prime_data = prime_data_original.copy()
        for i in xrange(batchsize-1):
            prime_data = np.r_[prime_data, prime_data_original]

        prime_train_characters_data = train_characters[prime_index]
        prime_len = len(prime_train_characters_data)
        prime_train_characters = []
        for i in xrange(batchsize):
            prime_train_characters.append(np.r_[prime_train_characters_data, text_train_character[i]])

    else:
        batchsize = 1
        n_max_seq_length = get_max_sequence_length(train_characters)
        t_data_prime = 0
        prime_len = 0
        prime_train_characters = train_characters

    """ Create the one-hot vocabulary sequence """
    #batchsize = len(text)
    #batchsize = len(train_characters)
    cs_data = xp.zeros((batchsize, n_chars, n_max_seq_length + prime_len)).astype(xp.float32)
    ls_data = xp.zeros((batchsize, 1))
    for j in xrange(batchsize):
        for k in xrange(len(prime_train_characters[j])):
            length = prime_train_characters[j][k]
            cs_data[j, length, k] = 1.0
        ls_data[j, 0] = k

    # @TODO: Make sure the length of the data match the input of the model
    #pad = xp.zeros((1, min(n_chars_training, abs(n_chars_training-n_chars)), n_max_seq_length)).astype(xp.float32)
    #pad.fill(2.0)
    #cs_data = xp.concatenate((cs_data, pad), axis=1)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        for j in xrange(len(text)):
            #x_data = xp.zeros((batchsize, 3)).astype(xp.float32)
            #x_next_data = xp.ones((batchsize, 3)).astype(xp.float32) * (-1.0)

            prob_bias = 0.0
            loss_network = xp.zeros((batchsize, 1))

            # The loop is defined by the backprop length
            strokes = []
            mse = 0
            cursor = truncated_back_prop_len if len(data_index) == 0 else len(train_data[data_index[j]])
            cursor += t_data_prime
            mdn_states = np.zeros((batchsize, cursor, mix_comp_number * 5 + mix_comp_number + 3))
            n_batches = variable.Variable(xp.ones(1).astype(xp.float32)*len(train_data))

            if prime_len > 0:
                x_data = prime_data
                x_data = xp.concatenate((x_data, xp.zeros((1, cursor-t_data_prime, input_size)).astype(xp.float32)), axis=1)
                #x_data = xp.zeros((1, cursor, input_size)).astype(xp.float32)
            else:
                x_data = xp.zeros((1, cursor, input_size)).astype(xp.float32)

            loss_network, strokes, mdn_components = model.sample(x_data, cs_data, n_batches, t_data_prime)
            states = xp.concatenate((mdn_components, xp.full((mdn_components.shape[0], 1), loss_network), strokes), axis=1)

            # Concatenate the ground-truth
            if len(data_index) == 0:
                states = xp.concatenate((states, xp.zeros_like(strokes)), axis=1)
            else:
                if t_data_prime > 0:
                    fill_train = xp.concatenate((xp.zeros((t_data_prime-1, input_size)), train_data[data_index[j]]), axis=0)
                else:
                    fill_train = train_data[data_index[j]]
                states = xp.concatenate((states, fill_train), axis=1)

            # Save the results
            np.save(path + "/{0}-mdn.npy".format(text[j].replace(" ", "-")), np.asarray([states]))
            model.reset_state()

# ===============
# CLI Entry point
# ===============

@click.command()
@click.argument('model_dir', type=click.STRING)
@click.argument('model_name', type=click.STRING)
@click.argument('text', type=click.STRING)
@click.option('--models_dir', type=click.Path(exists=True), default='models', help='Directory containing the models.')
@click.option('--data_dir', type=click.Path(exists=True), default='data/processed/OnlineHandWriting', help='Directory containing data.')
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
