#!/usr/bin/env python # -*- coding: utf-8 -*-

# Generate a prediction based on a trained model
# ==============================================

import numpy as np
import chainer
import chainer.functions as F

try:
    import cupy
except:
    cupy = np
    pass

from train_model import Model
from train_model import get_padded_data
from train_model import get_max_sequence_length

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

def main(model_dir, model_name, text, models_dir, data_dir, batchsize, gpu, peephole, grad_clip, truncated_back_prop_len, truncated_data_samples, rnn_layers_number, rnn_cells_number, win_unit_number, mix_comp_number, random_seed):
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
    n_chars = 83
    n_max_seq_length = get_max_sequence_length(train_characters)
    
    """ Import the trained model """
    logger.info("Importing the model {}".format(model_name))
    input_size = 3 # dimensions of x (x, y, end-of-stroke)

    # @TODO: should check if peephole is requested
    #n_chars_training = len(original_vocabulary)
    model = Model(n_chars, input_size, win_unit_number, rnn_cells_number, mix_comp_number, rnn_layers_number, use_adaptive_noise=False)
    
    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()
        model.to_gpu()
        xp = cupy
    else:
        xp = np

    # Load the model
    logger.info("Model imported successfully")
    chainer.serializers.load_npz(path + "/" + model_name, model)

    """ Create the one-hot vocabulary sequence """
    #batchsize = len(text)
    #batchsize = len(train_characters)
    cs_data = xp.zeros((batchsize, n_chars, n_max_seq_length))
    ls_data = xp.zeros((batchsize, 1))
    for j in xrange(batchsize):
        for k in xrange(len(train_characters[j])):
            length = train_characters[j][k]
            cs_data[j, length, k] = 1.0
        ls_data[j, 0] = k

    # @TODO: Make sure the length of the data match the input of the model
    #pad = xp.zeros((1, min(n_chars_training, abs(n_chars_training-n_chars)), n_max_seq_length)).astype(xp.float32)
    #pad.fill(2.0)
    #cs_data = xp.concatenate((cs_data, pad), axis=1)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        cs = chainer.Variable(xp.asarray(cs_data).astype(xp.float32))
        ls = chainer.Variable(xp.asarray(ls_data).astype(xp.float32))
        n_batches = chainer.Variable(xp.asarray(batchsize+xp.zeros(1)).astype(xp.int32))

        for j in xrange(len(text)):
            x_data = xp.zeros((batchsize, 3)).astype(xp.float32)
            x_next_data = xp.ones((batchsize, 3)).astype(xp.float32) * (-1.0)

            prob_bias = 0.0
            loss_network = xp.zeros((batchsize, 1))

            # The loop is defined by the backprop length
            strokes = []
            mse = 0
            cursor = truncated_back_prop_len if len(data_index) == 0 else len(train_data[data_index[j]])-1
            for i in xrange(cursor):
                logger.info("Iteration {}".format(i))
                # Start the prediction at postion 0, 0
                x_now = chainer.Variable(xp.asarray(x_data[0:batchsize, 0:3]).astype(xp.float32))
                x_next = chainer.Variable(xp.asarray(x_next_data[0:batchsize, 0:3].astype(xp.float32)))
                
                # The weight are updated only in the first iteration
                local_update_weight = True if i == 0 else False

                # Predict the next stroke
                x = [x_now, x_next, cs, ls, prob_bias, n_batches]
                loss_i, x_pred, eos_i, pi_i, mux_i, muy_i, sgx_i, sgy_i, rho_i, loss_complex_i = model(x, renew_weights=local_update_weight)
                loss_network += loss_i.data

                # From the distribution fetch a potential pen point
                pi_i_cpu = chainer.cuda.to_cpu(pi_i.data)
                x_pred_cpu = chainer.cuda.to_cpu(x_pred.data)
                eos_i_cpu = chainer.cuda.to_cpu(eos_i.data)
                mux_i_cpu = chainer.cuda.to_cpu(mux_i.data)
                muy_i_cpu = chainer.cuda.to_cpu(muy_i.data)
                sgx_i_cpu = chainer.cuda.to_cpu(sgx_i.data)
                sgy_i_cpu = chainer.cuda.to_cpu(sgy_i.data)
                rho_i_cpu = chainer.cuda.to_cpu(rho_i.data)
                for k in xrange(pi_i_cpu.shape[0]):
                    def get_point_index(x, pi):
                        summ = 0
                        for i in xrange(len(pi)):
                            #summ += pi[i]
                            summ = pi[i]
                            if summ.data >= x:
                                return i
                        raise ValueError("Unable to sample index from distribution")

                    #idx_pos = get_point_index(random.random(), pi_i[k])
                    idx_pos = np.random.choice(range(len(pi_i_cpu[k])), p=pi_i_cpu[k])

                    # From the index, perform a simple gaussian 2d to get the next positions
                    def get_next_position_gaussian_2d(mux, muy, sgx, sgy, rho):
                        mean = np.asarray([mux, muy])
                        covar = np.asarray([[sgx*sgx, rho*sgx*sgy], [rho*sgx*sgy, sgy*sgy]])

                        x = np.random.multivariate_normal(mean, covar, 1)
                        #plt.plot(x[0][0], x[0][1], 'x')
                        #plt.axis('equal')
                        #plt.show()

                        return x[0][0], x[0][1]

                    x1_pred, x2_pred = get_next_position_gaussian_2d(mux_i_cpu[k][idx], muy_i_cpu[k][idx], sgx_i_cpu[k][idx], sgy_i_cpu[k][idx], rho_i_cpu[k][idx])
                    #x1_pred = x_pred_cpu[k][0]
                    #x2_pred = x_pred_cpu[k][1]

                    #eos_pred = 1 if random.random() < eos_i[k].data else 0
                    eos_pred = 1.0 if eos_i_cpu[k] > 0.10 else 0.0
                    
                    x_data[k, 0] = x1_pred
                    x_data[k, 1] = x2_pred
                    x_data[k, 2] = eos_pred

                # Transform the output
                #x_data[:, 0:2] = x_pred.data[:, 0:2]
                #x_data[:, 2:] = xp.where(x_pred.data[:, 2:] > 0.10, 1.0, x_pred.data[:, 2:])
                #x_data[:, 2:] = xp.where(x_pred.data[:, 2:] > 1.0, 2.0, x_data[:, 2:])

                # Compare ground truth if requested
                x_data_cpu = chainer.cuda.to_cpu(x_data)
                if len(data_index) > 0:
                    x_gt = train_data[data_index[j]][i+1]
                    # @TODO: x_data[0] is harcoded
                    mse += np.sum(np.square(x_gt[0:2] - x_data_cpu[0, 0:2]))/cursor

                #mse += F.sum(F.square(x_next[:, 0:2] - x_pred[:, 0:2]))/len(x_pred)

                # Store the results of the stoke
                # @TODO: x_data[0] is harcoded
                strokes = np.asarray([x_data_cpu[0]]) if len(strokes) == 0 else np.concatenate((strokes, np.asarray([x_data_cpu[0]])))

                # New positions
                #x_now = x_pred.data.astype(xp.float32)

            # Compile the results
            losses_network_cpu = chainer.cuda.to_cpu(xp.copy(loss_network))
            mse_cpu = mse
            model.reset_state()

            """ Generate the handwriting sequence """
            # @TODO: must fetch the stats dynamically
            # Must "un-normalize" the generated strokes
            def denormalize(data):
                data[:, 1] *= stats[3]
                data[:, 0] *= stats[2]
                data[:, 1] += stats[1]
                data[:, 0] += stats[0]

                return data

            # Draw the generate handwritten text
            plt.cla(); plt.clf()
            plt.figure(1)
            #strokes = xp.concatenate((strokes, xp.asarray([[0.0, 0.0, 1.0]])))

            logger.info("Generating the handwriting sequence for '{}'".format(text[j]))
            
            # Draw the ground-truth and the ground-truth-prediction overlay if requested
            if len(data_index) > 0:
                plt.subplot("{0}{1}{2}".format(2, 1, 1))
                plt.title(text[j] + " (MSE: {})".format(mse_cpu))
                gt_stroke = denormalize(train_data[data_index[j]])
                draw_from_strokes(gt_stroke, plt)
                
                plt.subplot("{0}{1}{2}".format(2, 1, 2))
                pred_stroke = denormalize(strokes)
                draw_from_strokes(pred_stroke, plt)
            else:
                plt.subplot("{0}{1}{2}".format(1, 1, 1))
                plt.title(text[j])
                pred_stroke = denormalize(strokes)
                draw_from_strokes(pred_stroke, plt)

            plt.show()
            #plt.savefig(path + "/" + text[i].replace(" ", "-") + ".png")


# ===============
# CLI Entry point
# ===============

@click.command()
@click.argument('model_dir', type=click.STRING)
@click.argument('model_name', type=click.STRING)
@click.argument('text', type=click.STRING)
@click.option('--models_dir', type=click.Path(exists=True), default='models', help='Directory containing the models.')
@click.option('--data_dir', type=click.Path(exists=True), default='data/processed/OnlineHandWriting', help='Directory containing data.')
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
