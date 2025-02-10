import math

import numpy as np 
import tensorflow as tf
from tensorflow import keras

import model
import utils
import evaluation


def main(dataname):
    """
    Main function to configure hyperparameters and train the model.

    Args:
        dataname (str): Name of the dataset.

    Returns:
        None
    """
    configs = utils.config_pare_SITE(
        iterations=2000,
        lr_rate=0.001,
        lr_weight_decay=0.001,
        flag_early_stop=True,
        use_batch=1024,
        rep_alpha=0.1,
        out_dropout=0.4,
        GNN_dropout=0.4,
        rep_dropout=0.4,
        inp_dropout=0.0,
        rep_hidden_layer=3,
        rep_hidden_shape=[100, 100, 100],
        GNN_hidden_layer=3,
        GNN_hidden_shape=[100, 100, 100],
        out_T_layer=3,
        out_C_layer=3,
        out_hidden_shape=[100, 100, 100],
        reg_lambda=0.01,
        k=2
    )

    utils.implement(
        config=configs,
        data_name=dataname,
        Model_name=model.SITE,
        activation=tf.nn.relu
    )


if __name__ == '__main__':
    main('flickr')

