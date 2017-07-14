#!/usr/bin/env python # -*- coding: utf-8 -*-

# Generate a prediction based on a trained model
# ==============================================

import numpy as np
import chainer
import chainer.functions as F

from train_model import Model
from train_model import get_max_sequence_length

import click
import os
import time
import logging


# =================
# Helpers functions
# =================


def get_char_to_index(text):
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
        text = [text]
    cha_index, vocabulary = get_char_to_index(text)
    n_chars = len(vocabulary)
    train_characters = cha_index[0:len(text)]
    n_max_seq_length = get_max_sequence_length(train_characters)
    
    """ Import the trained model """
    logger.info("Importing the model {}".format(model_name))
    input_size = 3 # dimensions of x (x, y, end-of-stroke)

    # @TODO: should check if peephole is requested
    # @TODO: character size should be dynamic (74 is the length of the current data
    n_chars_training = len(np.load(data_dir + "/vocabulary" ))
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
