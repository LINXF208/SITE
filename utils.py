import math
import os
import random

import numpy as np 
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import scipy.io as scio
import matplotlib.pyplot as plt

import evaluation
import model


def mmd2_lin(X, t, p):
    """ Compute the linear MMD. 
    Args:
        X (tf.Tensor): Representation matrix.
        t (tf.Tensor): Treatment assignment vector (binary: 0 or 1).
        p (float): Probability of treatment.

    Returns:
        tf.Tensor: computed linear MMD value.
    """
    it = tf.where(t > 0)[:, 0]
    ic = tf.where(t < 1)[:, 0]

    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)
    
    mean_control = tf.reduce_mean(Xc, axis=0)
    mean_treated = tf.reduce_mean(Xt, axis=0)

    mmd = tf.reduce_sum(tf.square(2.0 * p * mean_treated - 2.0 * (1.0 - p) * mean_control))

    return mmd 


def divide_t_c(concated_data, input_t):
    """Divide units into Treated and Control groups.

    Args:
        concated_data (tf.Tensor): The dataset containing all units.
        input_t (tf.Tensor): Binary tensor (0: Control, 1: Treated).

    Returns:
        tuple: (group_T, group_C, i0, i1)
            - group_T (tf.Tensor): Treated group data.
            - group_C (tf.Tensor): Control group data.
            - i0 (tf.Tensor): Indices of the Control group.
            - i1 (tf.Tensor): Indices of the Treated group.
    """
    i0 = tf.cast((tf.where(input_t < 1)[:, 0]), tf.int32)
    i1 = tf.cast((tf.where(input_t > 0)[:, 0]), tf.int32)

    group_T = tf.gather(concated_data, i1)
    group_C = tf.gather(concated_data, i0)

    return group_T, group_C, i0, i1


def config_pare_SITE(
        iterations,
        lr_rate,
        lr_weight_decay,
        flag_early_stop,
        use_batch,
        rep_alpha,
        out_dropout,
        GNN_dropout,
        rep_dropout,
        inp_dropout,
        rep_hidden_layer,
        rep_hidden_shape,
        GNN_hidden_layer,
        GNN_hidden_shape,
        out_T_layer,
        out_C_layer,
        out_hidden_shape,
        reg_lambda,
        k
    ):
    """
    Configure hyperparameters for the SITE model.

    Args:
        iterations (int): Number of training iterations.
        lr_rate (float): Learning rate for Adam optimizer.
        lr_weight_decay (float): Weight decay for learning rate.
        flag_early_stop (bool): Whether to enable early stopping.
        use_batch (int): Batch size for training.
        rep_alpha (float): Regularization parameter for representation learning.
        out_dropout (float): Dropout rate for output layers.
        GNN_dropout (float): Dropout rate for GNN layers.
        rep_dropout (float): Dropout rate for representation layers.
        inp_dropout (float): Dropout rate for input
        rep_hidden_layer (int): Number of hidden layers in representation network.
        rep_hidden_shape (list): Shape of hidden layers in representation network.
        GNN_hidden_layer (int): Number of hidden layers in GNN.
        GNN_hidden_shape (list): Shape of hidden layers in GNN.
        out_T_layer (int): Number of layers in treatment outcome predictor.
        out_C_layer (int): Number of layers in control outcome predictor.
        out_hidden_shape (list): Shape of hidden layers in outcome predictors.
        reg_lambda (float): Regularization parameter for the model.
        k (int): Number of neighbors for GNN aggregation.

    Returns:
        dict: A dictionary containing all hyperparameters, i.e., config.
    """

    config = {
        # Parameters for Adam
        "lr_rate": lr_rate,
        "lr_dc": lr_weight_decay, 

        # Parameters for training
        "iterations": iterations,
        "flag_early_stop": flag_early_stop,
        "use_batch": use_batch,

        # Parameters for dropout rate
        "out_dropout": out_dropout,
        "GNN_dropout": GNN_dropout,
        "rep_dropout": rep_dropout,
        "inp_dropout": inp_dropout,

        # Parameters for model
        "rep_hidden_layer": rep_hidden_layer,
        "rep_hidden_shape": rep_hidden_shape,
        "GNN_hidden_layer": GNN_hidden_layer,
        "GNN_hidden_shape": GNN_hidden_shape,
        "out_T_layer": out_T_layer,
        "out_C_layer": out_C_layer,
        "out_hidden_shape": out_hidden_shape,

        # Other parameters 
        "reg_lambda": reg_lambda,
        "rep_alpha": rep_alpha,
        "k": k
    }

    return config


def load_data(data_name='flickr'):
    """
    Load the dataset, here is an example for the Flickr dataset.

    Args:
        data_name (str, optional): Name of the dataset. Defaults to 'flickr'.

    Returns:
        list: [features, adjacency matrix, treatments, factual outcomes, potential outcomes (y1, y0)]
    """
    if data_name == 'flickr':
        all_x = np.load("data/flk/flk_x.npy")
        adj = np.load("data/flk/flk_A.npy")
        all_t = np.load("data/flk/flk_t.npy")
        all_yf = np.load("data/flk/flk_yf.npy")
        all_y1 = np.load("data/flk/flk_y1.npy")
        all_y0 = np.load("data/flk/flk_y0.npy")
        print("Flickr dataset was Loaded.")

    data = [all_x, adj, all_t, all_yf, all_y1, all_y0]

    return data


def split_train_val_test(data, train_ratio, val_ratio, test_ratio,seed=42):
    """
    Split data indices into training, validation, and test sets.

    Args:
        data (array-like): The dataset to split.
        train_ratio (float): Proportion of the dataset to use for training.
        val_ratio (float): Proportion of the dataset to use for validation.
        test_ratio (float): Proportion of the dataset to use for testing.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (train_indices, val_indices, test_indices)
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")

    if len(data) == 0:
        raise ValueError("Data cannot be empty.")

    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(data))

    train_set_size = int(len(data) * train_ratio)
    val_set_size = int(len(data) * val_ratio)

    train_indices = shuffled_indices[:train_set_size]
    val_indices = shuffled_indices[train_set_size:train_set_size+val_set_size]
    test_indices = shuffled_indices[train_set_size+val_set_size:]
    
    return train_indices,val_indices,test_indices


def normalize_adj(mx):
    """
    Compute the symmetric normalized adjacency matrix.

    Args:
        mx (tf.Tensor): Adjacency matrix (square, [N, N]).

    Returns:
        tf.Tensor: Normalized adjacency matrix (L).
    """
    rowsum = tf.reduce_sum(mx, axis = -1)
    msq_rowsum = tf.pow(rowsum, -0.5)

    msq_D = tf.raw_ops.MatrixDiag(diagonal = msq_rowsum)

    L = tf.matmul(tf.matmul(msq_D, mx), msq_D)

    return L


def train(
        model_name,
        train_input,
        agg_features_train,
        train_yf,
        val_input,
        agg_features_val,
        val_yf,
        config_hyperparameters, 
        max_iterations,
        train_indices, 
        flag_early_stop=False, 
        activation=tf.nn.relu
    ):
    """
    Train process.

    Args:
        model_name (tf.keras.Model): The model class to be trained.
        train_input (np.array): Input data of training units.
        agg_features_train (np.array): Aggregated features of training units.
        train_yf (np.array): Factual outcomes of training units.
        val_input (np.array): Input data of units in the validation set.
        agg_features_val (np.array): Aggregated validation features.
        val_yf (np.array): Factual outcomes of validation units.
        config_hyperparameters (dict): Model hyperparameters.
        max_iterations (int): Maximum number of training iterations.
        train_indices (list or np.array): Indices for training data.
        flag_early_stop (bool): Whether to enable early stopping.
        activation (function): Activation function used in the model.

    Returns:
        tf.keras.Model: The trained model.
    """
    cur_model = model_name(config_hyperparameters,activation=activation) 

    losslist = []
    loss_list_val = []
    sum_loss = 0
    sum_val_loss = 0
    count = 0

    for i in range(max_iterations):
        print("iter",i)
        batch_indices = random.sample(range(0, len(train_indices)), config_hyperparameters['use_batch'])

        batch_input = tf.cast(np.array(train_input)[batch_indices], tf.float32)
        batch_y = tf.cast(np.array(train_yf)[batch_indices], tf.float32)
        batch_agg = tf.cast(np.array(agg_features_train)[batch_indices], tf.float32)

        total_loss = cur_model.network_learn(batch_input, batch_agg, batch_y)

        train_loss = cur_model.val_y(train_input, agg_features_train, train_yf)
        val_loss = cur_model.val_y(val_input, agg_features_val, val_yf)
        print("train loss",train_loss)
        print("val loss",val_loss)

        sum_loss += train_loss
        sum_val_loss += val_loss
        if (i+1) % 20 == 0:
            if len(loss_list_val) > 0 and sum_val_loss/20 >= loss_list_val[-1]:
                count += 1
            else:
                count = 0

            if flag_early_stop:
                if i > 400 and count >= 1:
                    print("Early stopping triggered.")
                    break

            losslist.append(sum_loss/20)
            loss_list_val.append(sum_val_loss/20)

            sum_loss = 0
            sum_val_loss = 0

    return cur_model

 
def save_mymodel(save_path, save_name, need_save_model):
    """
    Save the model weights to a specified path.

    Args:
        save_path (str): Directory to save the model weights.
        save_name (str): Filename for the saved weights.
        need_save_model (tf.keras.Model): Model instance to be saved.

    Returns:
        None
    """
    cur_path = save_path + '/' + save_name

    need_save_model.save_weights(cur_path)
    print("Already saved the model's weights in file" + cur_path)


def load_mymodel(load_path, load_name, need_load_model, config_hyperparameters, activation):
    """
    Load a saved model from a specified path.

    Args:
        load_path (str): Directory where the model is saved.
        load_name (str): Filename of the saved model.
        need_load_model (tf.keras.Model): Model class to instantiate.
        config_hyperparameters (dict): Model configuration parameters.
        activation (tf activation function): Activation function.

    Returns:
        tf.keras.Model: Loaded model instance.
    """
    cur_model = need_load_model(config_hyperparameters, activation)

    cur_path = load_path + '/' + load_name

    cur_model.load_weights(cur_path)
    print("Model successfully loaded.")

    return cur_model


def implement(config, data_name, model_name, activation):
    """ 
    Train, evaluate, and save model with results.

    Args:
        config (dict): Hyperparameters.
        data_name (str): Dataset name.
        model_name (class): Model class.
        activation (function): Activation function.
    """
    # Load data.
    data = load_data(data_name)
    x, adj, all_t, all_yf, y1, y0 = data

    # Preprocess.
    all_t = all_t.reshape(len(all_t), 1)
    y1 = y1.reshape(len(y1), 1)
    y0 = y0.reshape(len(y0), 1)
    all_yf = all_yf.reshape(len(all_yf), 1)
    all_input_self = np.concatenate([x, all_t], axis=1)
    all_ite_true = y1 - y0

    # Spilt train/val/test sets.
    train_indices,val_indices,test_indices = split_train_val_test(x, 0.7, 0.15, 0.15)

    # Compture L
    init_adj_plus_I = ((adj > 0) + 0.0).T + np.eye(adj.shape[0])
    L = normalize_adj(init_adj_plus_I)

    # Aggregate interference-related information before training.
    final_A = L
    for i in range(config['k'] - 1):
        final_A =  np.matmul(final_A, L) 
    agg_features = np.matmul(final_A, np.array(all_input_self))

    # ndarray -> tf.Tensor
    all_input_self = tf.cast(all_input_self, tf.float32)
    cur_yf = tf.cast(all_yf, tf.float32)

    train_input = tf.gather(all_input_self, train_indices)
    val_input = tf.gather(all_input_self, val_indices)
    test_input = tf.gather(all_input_self, test_indices)

    train_yf = tf.gather(cur_yf, train_indices)
    val_yf = tf.gather(cur_yf, val_indices)
    test_yf = tf.gather(cur_yf, test_indices)

    agg_features_train = tf.gather(agg_features, train_indices)
    agg_features_val = tf.gather(agg_features, val_indices)
    agg_features_test = tf.gather(agg_features, test_indices)

    train_ite_true = all_ite_true[train_indices]
    val_ite_true = all_ite_true[val_indices]
    test_ite_true = all_ite_true[test_indices]

    os.makedirs(cur_save_path, exist_ok=True)

    for cur_i in range(10):
        # Train and revalute model with ten runs.
        cur_model = train(
                        model_name,
                        train_input,
                        agg_features_train,
                        train_yf,
                        val_input,
                        agg_features_val,
                        val_yf,
                        config,config["iterations"],
                        train_indices,
                        config["flag_early_stop"],
                        activation=activation
                        )

        cur_save_model_name = "model"
        cur_save_path = './save_Models/data_' + data_name + "_" + str(model_name)[8:-2] + "_repeat_" + str(cur_i)
        save_mymodel(cur_save_path, cur_save_model_name, cur_model)

        val_pehe, val_msey= evaluation.evaluate_msey_pehe(cur_model, val_input, agg_features_val, val_yf, val_ite_true)
        cur_val_results = [val_pehe, val_msey]
        cur_val_results_name = './results/val_results_'+ data_name + str(model_name)[8:-2]+'_'+"reapted_" + str(cur_i)
        save_results(cur_val_results, cur_val_results_name)

        test_pehe, test_msey= evaluation.evaluate_msey_pehe(cur_model, test_input, agg_features_test, test_yf, test_ite_true)
        cur_test_results = [test_pehe, test_msey]
        cur_test_results_name = './results/test_results_'+ data_name + str(model_name)[8:-2]+'_'+"reapted_" + str(cur_i)
        save_results(cur_test_results, cur_test_results_name)


def save_results(save_result, save_name):
    """
    Save results as a .npy file in the specified directory.

    Args:
        save_result (numpy array): Data to save.
        save_path (str): Directory where the results should be saved.
        save_name (str): Filename for saving the results.

    Returns:
        None
    """
    np.save(save_name, save_result) 
    print("saved all results ")



