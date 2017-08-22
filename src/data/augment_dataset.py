#!/usr/bin/env python # -*- coding: utf-8 -*-

# Data Augmentation for the current dataset
## Rotate and scale the data
# ==================================================

import os
import click
import logging
import math
import random

import scipy.misc
import numpy as np
import cPickle as pickle

from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation


# ==============
# Helpers (hlps)
# ==============
def load_data(path):
    f = open(path, "rb")
    data = pickle.load(f)
    f.close()
    return data

def transformation_resize(matrix, factorW=None, factorH=None):
    """
        Resize the matrix by factor
        
        Args:
            matrix (float[][]): the matrix representing the pen positions
            factorW (int): transformation width factor
            factorH (height): transformed height factor
        Returns
            (float[][])
    """
    if factorW is None:
        factorW = random.choice(range(5, 15))/10.0
    if factorH is None:
        factorH = random.choice(range(5, 15))/10.0

    res = np.copy(matrix)
    res[:, 0:1] *= factorW
    res[:, 1:2] *= factorH
    
    return res

def transformation_rotate(matrix, angle=None):
    """
        Rotate a matrix by a angle

        Args:
            matrix (float[][]): the matrix representing the pen positions
            angle (float): angle of the rotation
        Returns
            (float[][])
    """
    if angle is None:
        angle = random.choice(range(-45, 45)) * 1.0

    
    res = np.copy(matrix)
    res[:, 0:2] = scipy.ndimage.interpolation.rotate(res[:, 0:2], angle, reshape=False)
    
    return res

# ===============================================
# Main entry point of the training process (main)
# ===============================================


def main(data_dir, out_dir, training_multiplier, validation_multiplier):
    """ Augment data saved in ../processed """
    logger = logging.getLogger(__name__)
    logger.info('Augmenting data')

    """ Fetching the model and the inputs """
    logger.info("Fetching the model and the inputs")
    stats = np.load(data_dir + "/statistics")
    train_data, train_characters = [], []
    train_augment_length = 0
    if training_multiplier > 1:
        train_data = load_data(data_dir + "/train/train_data")
        train_characters = load_data(data_dir + "/train/train_characters")
        train_augment_length = math.ceil(len(train_data)*training_multiplier)
        logger.info("Augmenting training data from {0} to {1}".format(len(train_data), train_augment_length))

    #valid_data, valid_characters = [], []
    #valid_augment_length = 0
    #if validation_multiplier > 1:
    #    valid_data = load_data(data_dir + "/valid/valid_data")
    #    valid_characters = load_data(data_dir + "/valid/valid_characters")
    #    valid_augment_length = math.ceil(len(valid_data)*validation_multiplier)
    #    logger.info("Augmenting validation data from {0} to {1}".format(len(valid_data), valid_augment_length))
    
    """ Perform the augmentation on the already normalized data """
    def perform_augmentation(data, characters, length):
        ops = [transformation_resize, transformation_rotate]
        transformations = []

        i = 0
        while length > 0:
            # Randomly select a transformation to apply, n times
            op = random.choice(ops)
            multiplier_length = range(1,min(5,int(length)))
            
            if len(multiplier_length) == 0:
                break

            multiplier = random.choice(multiplier_length)
            length -= multiplier
            
            while multiplier > 0:
                res = op(data[i])
                transformations.append([i, res, characters[i]])
                multiplier -= 1

            i += 1

            if i > len(data):
                i = 0

        return transformations

    if train_augment_length > 0:
        augmented = perform_augmentation(train_data, train_characters, train_augment_length-len(train_data))

        # Visualize the result VS the ground truth
        group = []
        previous = None
        for i in xrange(len(augmented)):
            strokes_gen = augmented[i][1]
            strokes = [strokes_gen]

            # New group of strokes is begining, remember the index
            if previous is None:
                previous = augmented[i][0]
                strokes = [train_data[previous], strokes_gen]

            # All augmentation of the same idx are processed, plot them
            if previous != augmented[i][0] and len(group) > 0:
                plt.cla(); plt.clf()
                plt.figure(1)
                for k in xrange(len(group)):
                    plt.subplot("{0}{1}{2}".format(len(group), 1, k+1))
                    plt.gca().invert_yaxis()
                    plt.plot(*group[k])
                plt.show()
                previous = None
                group = []
            
            # Normalize and compute positions
            for j in xrange(len(strokes)):
                # Must "un-normalize" the data
                strokes[j][:, 1] *= stats[3]
                strokes[j][:, 0] *= stats[2]
                strokes[j][:, 1] += stats[1]
                strokes[j][:, 0] += stats[0]

                # Remove the last position
                last = [0,0,0]

                positions = []
                pos_list = [[], []]
                for point in strokes[j]:
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
                
                group.append(positions)


# ===============
# CLI Entry point
# ===============


@click.command()
@click.option('--data_dir', type=click.Path(exists=True), default='data/processed/OnlineHandWriting', help='Directory containing data.')
@click.option('--out_dir', type=click.Path(exists=False), default='data/interim/OnlineHandWriting', help='Output directory of the augmented data')
@click.option('--training_multiplier', type=click.FLOAT, default=2.0, help='Increase the training data set by this multiplier.')
@click.option('--validation_multiplier', type=click.FLOAT, default=1.1, help='Increase the training data set by this multiplier.')
def cli(**kwargs):
    main(**kwargs)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    cli()
