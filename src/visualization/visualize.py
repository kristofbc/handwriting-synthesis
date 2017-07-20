#!/usr/bin/env python # -*- coding: utf-8 -*-

# Visualize the generated model
# =============================

import time
import os
import click
import logging
import math

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

import chainer
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import variable


# ========================
# Helpers functions (hlpr)
# ========================

def get_coordinates(data, std=[]):
    """
        Extract the coordinate used for plotting for a network

        Args:
            data (float[]): 1D array containing the data to plot
            std (float[]): 1D array to create the "box" arround the curve
        Returns:
            (float[]), (float[]), (float[])
    """
    coord = []
    box = []
    y_min = np.min(data, axis=0)
    y_max = np.max(data, axis=0)

    # Scale the data between range [-1.0, 1.0]
    #data = scale_data(data, mins=y_min, maxs=y_max)

    for i in xrange(len(data)):
        # Create the "box" around the curve
        if len(std) == len(data):
            box.append([data[i] - 1.0 * std[i], data[i] + 1.0 * std[i]])

        coord.append([i, data[i]])

    return np.array(coord, dtype=np.float32), np.array(box, dtype=np.float32), [0, len(coord), y_min, y_max]

def scale_data(data, high=1.0, low=-1.0, maxs=None, mins=None):
    """
        Scale data between [low, high]

        Args:
            data (float[]): 1D array of values to scale
            high (float): upperbound of the scale
            low (float): lowerbound of the scale
            maxs (float): max value in data
            mins (float): min value in data
        Returns:
            (float[])
    """
    if mins is None:
        mins = np.min(data, axis=0)
    if maxs is None:
        maxs = np.max(data, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - data)) / rng)

def plot_data(coordinate, box=[], plt_inst=None, **kwargs):
    """
        Plot the coordinate with the "std box" around the curve

        Args:
            coordinate (float[]): 1D array of the coordinate to plot
            box (float[]): 1D array of the box around the curve
            plt_inst (pyplot): pyplot instance
        Returns:
            (plt_inst)
    """
    if plt_inst is None:
        plt_inst = plt
    
    if len(box) == len(coordinate):
        plt_inst.fill_between(np.arange(len(box)), box[:, 0:1].squeeze(), box[:, 1:].squeeze(), zorder=1, alpha=0.2)

    plt_inst.plot(coordinate[:, 0:1].squeeze(), coordinate[:, 1:].squeeze(), **kwargs)

    return plt_inst

def plot_loss_networks(train_network, valid_network, x_label="Epoch", y_label="Loss", title="Network loss"):
    """
        Plot multiple loss networks on the same graph

        Args:
            train_network (float[][]): the train network loss array
            valid_network (float[][]): the valid network loss array
        Returns:
            (pyplot)
    """
    # Extract the coordinate of the losses
    coord_network_train, box_network_train, stats_network_train = [], [], []
    coord_network_valid, box_network_valid, stats_network_valid = [], [], []
    if len(train_network) > 0:
        coord_network_train, box_network_train, stats_network_train = get_coordinates(train_network[:, 0], train_network[:, 1])
    if len(valid_network) > 0:
        coord_network_valid, box_network_valid, stats_network_valid = get_coordinates(valid_network[:, 0], valid_network[:, 1])

    plt.figure(1)
    plt.subplot("{0}{1}{2}".format(1, 1, 1))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title + " (iteration #{})".format(len(coord_network_train) if len(coord_network_train) > 0 else len(coord_network_valid)))
    plt.ylim(
        min(stats_network_train[2] if len(stats_network_train) > 0 else 0, stats_network_valid[2] if len(stats_network_valid) > 0 else 0)-100, 
        max(stats_network_train[3] if len(stats_network_train) > 0 else 0, stats_network_valid[3] if len(stats_network_valid) > 0 else 0)+100
    )

    if len(coord_network_train) > 0:
        plot_data(coord_network_train, box_network_train, plt, label="Train")
    if len(coord_network_valid) > 0:
        plot_data(coord_network_valid, box_network_valid, plt, label="Test")

    plt.legend(ncol=2 if len(coord_network_train) > 0 and len(coord_network_valid) > 0 else 1, loc="upper right", fontsize=10)

    return plt


# ===================================================
# Main entry point of the visualization process (main)
# ===================================================


def main(model_name, models_dir, output_path):
    """ Train the model based on the data saved in ../processed """
    logger = logging.getLogger(__name__)
    logger.info('Visualization of the model')

    model_path = models_dir + "/" + model_name
    if not os.path.exists(model_path):
        raise ValueError("The model {0} does not exists in {1}".format(model_name, models_dir))

    if not os.path.exists(output_path):
        raise ValueError("Output path {} does not exists.".format(output_path))

    """ Load the result stats files """
    # Training
    loss_network_train = None
    if os.path.exists(model_path + "/loss_network_train.npy"):
        loss_network_train = np.load(model_path + "/loss_network_train.npy")
    loss_complex_train = None
    if os.path.exists(model_path + "/loss_complex_train.npy"):
        loss_complex_train = np.load(model_path + "/loss_complex_train.npy")
    # Validation
    loss_network_valid = None
    if os.path.exists(model_path + "/loss_network_valid.npy"):
        loss_network_valid = np.load(model_path + "/loss_network_valid.npy")

    """ Generate the loss curves """
    if loss_network_train is None and loss_complex_train is None and loss_network_valid is None:
        raise RuntimeError("Unable to generate the figures: no values found")

    # Extract the coordinate of the losses
    plt_inst = plot_loss_networks(loss_network_train if loss_network_train is not None else [], loss_network_valid if loss_network_valid is not None else [])
    #plt_inst.show()
    iteration_number = len(loss_network_train) if len(loss_network_train) > 0 else len(loss_network_valid)
    plt_inst.savefig(output_path + "/" + model_name + "-iteration-{}".format(iteration_number) + ".png")

# ======================
# CLI entry point (clie)
# ======================


@click.command()
@click.argument('model_name', type=click.STRING)
@click.option('--models_dir', type=click.Path(exists=True), default='models', help='Directory containing the models.')
@click.option('--output_path', type=click.Path(exists=True), default='reports', help='Directory containing the output visualization.')
def cli(**kwargs):
    main(**kwargs)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    cli()
