import utils
import evaluation
import numpy as np 
import tensorflow as tf
import pandas as pd
import math
import statsmodels.api as sm
from tensorflow import keras
from sklearn.model_selection import train_test_split
import seaborn as sns
import model

def main(dataname):
    

    configs,activation = utils.config_pare_SITE(iterations=2000,lr_rate=0.001,lr_weigh_decay=0.001,flag_early_stop=True,use_batch=1024,
    GNN_alpha= 0.1,rep_alpha=0.1,out_dropout=0.4,GNN_dropout=0.4,rep_dropout=0.4,inp_dropout = 0.0,rep_hidden_layer=3,rep_hidden_shape=[100,100,100],
    GNN_hidden_layer=3,GNN_hidden_shape=[100,100,100],out_T_layer=3,out_C_layer=3,out_hidden_shape=[100,100,100],activation=tf.nn.relu,
    reg_lambda=0.01,k=2)

    utils.implement(config=configs,data_name=dataname,
                              Model_name=model.SITE,activation=activation)

if __name__ == '__main__':

    main('flickr')

