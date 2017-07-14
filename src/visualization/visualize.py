#!/usr/bin/env python # -*- coding: utf-8 -*-

# Visualize the generated model
# =============================

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


# ===================================================
# Main entry point of the visualization process (main)
# ===================================================


@click.command()
@click.argument('model_dir', type=click.STRING)
@click.argument('model_name', type=click.STRING)
@click.option('--models_dir', type=click.Path(exists=True), default='models', help='Directory containing the models.')
@click.option('--data_dir', type=click.Path(exists=True), default='data/processed/OnlineHandWriting', help='Directory containing the data.')
def main(model_dir, model_name, models_dir, data_dir):
    """ Train the model based on the data saved in ../processed """
    logger = logging.getLogger(__name__)
    logger.info('Visualization of the model')

    path = models_dir + '/' + model_dir
    if not os.path.exists(paths):
        raise ValueError("The path {} does not exits".format(path))

    model_path = path + "/" + model_name
    if not os.path.exists(model_path):
        raise ValueError("The model {0} does not exists in {1}".format(model_name, path))

    """ Load the result stats files """
    # Training
    loss_network_train = np.load(model_path + "/loss_network_train")
    loss_complex_train = np.load(model_path + "/loss_complex_train")
    # Validation
    loss_network_valid = np.load(model_path + "/loss_network_valid")

    """ Generate the loss curves """


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #logging.basicConfig(level=logging.INFO, format=log_fmt, stream=sys.stdout)
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
