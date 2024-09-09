import numpy as np 
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.utils import shuffle
import pandas as pd
import math
import os
import evaluation
from tensorflow import keras
import matplotlib.pyplot as plt
import scipy.io as scio
import model
import random

def mmd2_lin( X,t,p):
    ''' Linear MMD '''

    it = tf.where(t>0)[:,0]
    ic = tf.where(t<1)[:,0]

    Xc = tf.gather(X,ic)
    Xt = tf.gather(X,it)

    mean_control = tf.reduce_mean(Xc,axis=0)
    mean_treated = tf.reduce_mean(Xt,axis=0)

    mmd = tf.reduce_sum(tf.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))

    return mmd 


    

def divide_TC( concated_data,input_t):
    #temp = tf.concat([hidden,GNN,weighted_G],1)
    #print("input_t",input_t)
    i0 = tf.cast((tf.where(input_t < 1)[:,0]),tf.int32)
    i1 = tf.cast((tf.where(input_t > 0)[:,0]),tf.int32)
    #mask = np.logical_and(np.array(input_t)[:,-1] == 1,1)
    #print("concated_data",concated_data)
    #print("mask",mask)
    group_T = tf.gather(concated_data,i1)
    #print("group_T",group_T)
    group_C = tf.gather(concated_data,i0)


    return tf.constant(group_T),tf.constant(group_C),i0,i1

   

def config_pare_SITE(iterations,lr_rate,lr_weigh_decay,flag_early_stop,use_batch,
rep_alpha,out_dropout,GNN_dropout,rep_dropout,inp_dropout,rep_hidden_layer,rep_hidden_shape,
GNN_hidden_layer,GNN_hidden_shape,out_T_layer,out_C_layer,out_hidden_shape,activation,reg_lambda,k,GNN_alpha):

    cur_activation = activation
    config = {}
    config["iterations"] = iterations
    config["lr_rate"] = lr_rate
    config["lr_dc"] = lr_weigh_decay
    config["flag_early_stop"] = flag_early_stop
    config['use_batch'] = use_batch
    config['rep_alpha'] = rep_alpha
    config['GNN_alpha'] = GNN_alpha
    config['out_dropout'] = out_dropout
    config['GNN_dropout'] = GNN_dropout
    config['rep_dropout'] = rep_dropout
    config['inp_drop'] = inp_dropout
    config['reg_lambda'] = reg_lambda
    config['rep_hidden_layer'] = rep_hidden_layer
    config['rep_hidden_shape'] = rep_hidden_shape
    config['GNN_hidden_layer'] = GNN_hidden_layer
    config['GNN_hidden_shape'] = GNN_hidden_shape
    config['out_T_layer'] = out_T_layer
    config['out_C_layer'] = out_C_layer
    config['out_hidden_shape'] = out_hidden_shape
    config['k'] = k



    return config,cur_activation

def load_data(data_name):

    data = []

    if data_name == 'flickr':
        xs = np.load("data/flk/flk_x.npy")
        adjs= np.load("data/flk/flk_A.npy")
        Ts = np.load("data/flk/flk_t.npy")
        all_yfs = np.load("data/flk/flk_yf.npy")
        all_y1s = np.load("data/flk/flk_y1.npy")
        all_y0s = np.load("data/flk/flk_y0.npy")
        
    data.append(xs)
    data.append(adjs)
    data.append(Ts)
    data.append(all_yfs)
    data.append(all_y1s)
    data.append(all_y0s)
  

    return data


def split_train_val_test(data,train_ratio,val_ratio,test_ratio):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    train_set_size = int(len(data) * train_ratio)
    val_set_size = int(len(data) * val_ratio)
    train_indices = shuffled_indices[:train_set_size]
    val_indices = shuffled_indices[train_set_size:train_set_size+val_set_size]
    test_indices = shuffled_indices[train_set_size+val_set_size:]
    return train_indices,val_indices,test_indices
 

def normalize_adj(mx):
        rowsum = tf.reduce_sum(mx, axis = -1)
        msq_rosum = tf.pow(rowsum,-0.5)
        #msq_rosum[np.isinf(msq_rosum)] = 0
        msq_D = tf.raw_ops.MatrixDiag(diagonal = msq_rosum )
        L = tf.matmul(tf.matmul(msq_D,mx),msq_D)
        return L

def train(Model_name,cur_train_input,agg_features_train,cur_train_yf,cur_val_input,agg_features_val,cur_val_yf,config_hyperparameters, max_iterations,train_indices,flag_early_stop=False,activation = tf.nn.relu):

    cur_model = Model_name(config_hyperparameters,activation=activation) 
    count = 0
    losslist_CV = []
    sum_loss = 0
    sum_CV_loss = 0
    losslist = []
    for i in range(max_iterations):
        print("iter",i)
        I = random.sample(range(0, len(train_indices)), 1024)
        bath_input = tf.cast(np.array(cur_train_input)[I],tf.float32)
        batch_y = tf.cast(np.array(cur_train_yf)[I],tf.float32)
        batch_agg = tf.cast(np.array(agg_features_train)[I],tf.float32)
        total_loss = cur_model.network_learn(bath_input, batch_agg, batch_y)
        tr_loss = cur_model.CV_y(cur_train_input,agg_features_train, cur_train_yf)
        CV_loss = cur_model.CV_y(cur_val_input,agg_features_val, cur_val_yf)
        print("train loss",tr_loss)
        print("val loss",CV_loss)
        
        sum_loss += tr_loss
        sum_CV_loss += CV_loss
        if (i+1) % 20 == 0:
            if len(losslist_CV) > 0 and sum_CV_loss/20 >= losslist_CV[-1]:
                count += 1
            else:
                count = 0
            if flag_early_stop:
                if i > 400 and count >= 1:
                    break
            losslist.append(sum_loss/20)
            losslist_CV.append(sum_CV_loss/20)
            sum_loss = 0
            sum_CV_loss = 0

    return cur_model

 
def save_mymodel(save_path,save_name,need_save_model):
    cur_path = save_path + '/' + save_name
    need_save_model.save_weights(cur_path )
    print("Already saved the model's weights in file" + cur_path  )

def load_mymodel(load_path,load_name,need_load_model,config_hyperparameters,activation,init_A):
    cur_model = need_load_model(config_hyperparameters,activation,init_A)
    cur_path = load_path + '/' + load_name
    cur_model.load_weights(cur_path)
    print("load model")
    return cur_model




def implement(config,data_name,Model_name,activation):
    data = load_data(data_name)
    x = data[0]
    adj = data[1]
    T = data[2]
    all_yf = data[3]
    y1 = data[4]
    y0 = data[5]
    T = T.reshape(len(T),1)
    y1 = y1.reshape(len(y1),1)
    y0 = y0.reshape(len(y0),1)
    all_yf = all_yf.reshape(len(all_yf),1)
    cur_all_input = np.concatenate([x,T],axis=1)
    cur_ite_true = y1 - y0
    train_indices,val_indices,test_indices = split_train_val_test(x,0.7,0.15,0.15)

    init_AplusI = ((adj>0)+0.0).T + np.eye(adj.shape[0])
    L = normalize_adj(init_AplusI) 
    final_A = L
    for i in range(config['k']-1):
        final_A =  np.matmul(final_A,L) 
    agg_features = np.matmul(final_A,np.array(cur_all_input))

    cur_all_input = tf.cast(cur_all_input,tf.float32)
    cur_train_input = tf.gather(cur_all_input,train_indices)
    cur_val_input = tf.gather(cur_all_input,val_indices)
    cur_test_input = tf.gather(cur_all_input,test_indices)
    cur_yf = tf.cast(all_yf,tf.float32)
    cur_train_yf = tf.gather(cur_yf,train_indices)
    cur_val_yf = tf.gather(cur_yf,val_indices)
    cur_test_yf = tf.gather(cur_yf,test_indices)
    agg_features_train = tf.gather(agg_features ,train_indices)
    agg_features_val = tf.gather(agg_features ,val_indices)
    agg_features_test = tf.gather(agg_features ,test_indices)
    train_ITE = cur_ite_true[train_indices]
    val_ITE = cur_ite_true[val_indices]
    test_ITE = cur_ite_true[test_indices]

    for cur_i in range(10):


        cur_model = train(Model_name,cur_train_input,agg_features_train,cur_train_yf,cur_val_input,agg_features_val,cur_val_yf,config,config["iterations"] ,train_indices,config["flag_early_stop"],activation = activation)
        cur_save_model_name = "model"
        cur_save_path = './save_Models/data_'+ data_name + "_" + str(Model_name)[8:-2]   + "_" "repeat_" + str(cur_i)
        os.makedirs(cur_save_path,exist_ok=True)
        save_mymodel(cur_save_path,cur_save_model_name,cur_model)

        cur_val_results = []
        pehe,msey= evaluation.evaluate_msey_pehe(cur_model,cur_val_input,agg_features_val,cur_val_yf,val_ITE)
        cur_val_results.append(pehe)
        cur_val_results.append(msey)

        cur_val_results_name = './results/val_results_'+ data_name + str(Model_name)[8:-2]+'_'+"reapted_" + str(cur_i)
        save_results(cur_val_results,cur_val_results_name)

        cur_test_results = []
        pehe,msey= evaluation.evaluate_msey_pehe(cur_model,cur_test_input,agg_features_test,cur_test_yf,test_ITE)
        cur_test_results.append(pehe)
        cur_test_results.append(msey)

        cur_test_results_name = './results/test_results_'+ data_name + str(Model_name)[8:-2]+'_'+"reapted_" + str(cur_i)
        save_results(cur_test_results,cur_test_results_name)

  

def save_results(save_result,save_name):
    np.save(save_name,save_result) 
    print("saved all results ")



