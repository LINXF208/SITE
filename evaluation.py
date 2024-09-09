import numpy as np   
import tensorflow as tf 
import model

def evaluate_msey_pehe(Model,cur_test_input,agg_features_test,cur_test_yf,test_ITE):

    pre_T,pre_C = Model(cur_test_input,agg_features_test, False)
    
    ITE = pre_T - pre_C
    print("ITE pre shape",ITE.shape)
    print("ITE true shape",test_ITE.shape)
    yf = Model.pre_yf(cur_test_input,agg_features_test, False)
    print("yf pre shape",yf.shape)
    print("yf true shape",cur_test_yf.shape)

    pehe = np.mean((ITE - test_ITE)**2)
    msey = np.mean((cur_test_yf - yf)**2)
                              
    return pehe,msey