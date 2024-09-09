
import tensorflow as tf
import utils
import warnings
warnings.filterwarnings('ignore')
import numpy as np



from tensorflow import keras


SQRT_CONST = 1e-10
VERY_SMALL_NUMBER = 1e-10

class reprelayer(keras.layers.Layer):
    def __init__(self, num_outputs,activation=tf.nn.relu,reg = 0):
        super(reprelayer,self).__init__()
        self.num_outputs = num_outputs
        self.activation = activation
        self.reg = reg
    def build(self,input_shape):
        self.kernel = self.add_weight("kernel",shape = [int(input_shape[-1]),self.num_outputs],
                                      dtype=tf.float32,initializer=tf.keras.initializers.glorot_uniform(),
                                     regularizer=keras.regularizers.l2(self.reg))
        
        self.bias = self.add_weight("bias",shape=[self.num_outputs], initializer=keras.initializers.Zeros())
        
    def call(self,features,flag=False):
        output = tf.matmul(features,self.kernel)+self.bias
        output = self.activation(output)
     
        self.result = output
        return output
    
class SGCNLayer(keras.layers.Layer):
    def __init__(self,out_features,batch_norm = False,activation=tf.nn.relu):
        super(SGCNLayer,self).__init__()
        self.out_features = out_features

        if batch_norm:
            self.batch_norm = batch_norm
        else:
            self.batch_norm = None
        self.activation = activation
            
    def build(self,input_shape):
        self.kernel = self.add_variable("kernel", shape = [int(input_shape[-1]),self.out_features],dtype=tf.float32,initializer=tf.keras.initializers.glorot_uniform())
        self.bias = self.add_variable("bias",shape=[self.out_features])
        
    def call(self,features):
        output = tf.matmul(features,self.kernel)+self.bias
        output = self.activation(output)
        return output
    
class SITE(keras.Model):
    def __init__(self,config,activation=tf.nn.relu):
        super(SITE, self).__init__()
        
        print("Initialization ...")
        self.rep_layers = []
        self.gnn_layers = []
        self.out_T_layers = []
        self.out_C_layers = []
        self.activation = activation
        self.GNN_alpha = config['GNN_alpha']
        self.rep_alpha = config['rep_alpha']
        self.reg_lambda = config['reg_lambda']
        self.out_dropout = config['out_dropout']
        self.GNN_dropout = config['GNN_dropout']
        self.rep_dropout = config['rep_dropout']
        self.out_dropout = config['out_dropout']
        self.inp_drop = config['inp_drop']
        self.use_batch = config['use_batch']
        self.optimizer= keras.optimizers.Adam(lr=config['lr_rate'],decay = config['lr_dc'])
        
        for i in range(config['rep_hidden_layer']):
            h = reprelayer(config['rep_hidden_shape'][i],activation = self.activation,reg = 0.0)
           
            self.rep_layers.append(h)

        for i in range(config['GNN_hidden_layer']):
            g = SGCNLayer(out_features = config['GNN_hidden_shape'][i], 
                                     activation = self.activation)
            self.gnn_layers.append(g)
 
        for i in range(config['out_T_layer']):
            out_T = keras.layers.Dense(config['out_hidden_shape'][i], activation = self.activation,kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                       kernel_regularizer = keras.regularizers.l2(0.0))

            self.out_T_layers.append(out_T)

        for i in range(config['out_C_layer']):
            out_C = keras.layers.Dense(config['out_hidden_shape'][i], activation = self.activation,kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                      kernel_regularizer = keras.regularizers.l2(0.0))

            self.out_C_layers.append(out_C)
        
        self.layer_6_T = keras.layers.Dense(1)
        self.layer_6_C = keras.layers.Dense(1)
   
 
        
    def call(self, inputtensor,aggreted_results,training = False):
        print("Call ...")
        input_tensor = inputtensor[:,:-1]
        #input_tensor = input_tensor/tf.sqrt( tf.clip_by_value(tf.reduce_sum(tf.square(input_tensor),axis = 1,keepdims=True),SQRT_CONST,np.inf))
        
        input_t = tf.constant(inputtensor[:,-1],shape = [input_tensor.shape[0],1])
        features = input_tensor
        

       
        hidden = input_tensor
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden,flag = training)
       
       
        GNN = aggreted_results
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i](GNN)
           
        
        concated_data = tf.concat([hidden,GNN],axis = 1)

       
    
        
        
        group_t,group_c=concated_data,concated_data
        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
           
        output_T = self.layer_6_T(outnn_T)

        outnn_C = group_c
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
            

        output_C = self.layer_6_C(outnn_C)
        
        
 
        
        return output_T,output_C
            
            
            

    
    def get_loss(self,inputtensor,aggreted_results,train_y,training = True):
        input_tensor = inputtensor[:,:-1]

        input_t = tf.constant(inputtensor[:,-1],shape = [input_tensor.shape[0],1])
        features = input_tensor

        regularization_1 = 0
        hidden = input_tensor
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden)
            hidden = tf.nn.dropout(hidden,self.rep_dropout)
            regularization_1 += tf.nn.l2_loss(self.rep_layers[i].kernel)


        GNN = aggreted_results
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i](GNN)
            GNN = tf.nn.dropout(GNN,self.GNN_dropout)
            regularization_1 += tf.nn.l2_loss(self.gnn_layers[i].kernel)

        concated_data = tf.concat([hidden,GNN ],axis = 1)

     
        group_t,group_c,i_0,i_1= utils.divide_TC(concated_data ,input_t)
        
        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
            outnn_T = tf.nn.dropout(outnn_T,self.out_dropout)
            regularization_1 += tf.nn.l2_loss(self.out_T_layers[i].kernel)
        output_T = self.layer_6_T(outnn_T)

        outnn_C = group_c
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
            outnn_C = tf.nn.dropout(outnn_C,self.out_dropout)
            regularization_1 += tf.nn.l2_loss(self.out_C_layers[i].kernel)

        output_C = self.layer_6_C(outnn_C)
        
        y_pre = tf.dynamic_stitch([i_0,i_1],[output_C,output_T])
        
        p = tf.divide(tf.reduce_sum(input_t),input_t.shape[0])
        clf_error_1_pri = tf.reduce_mean(tf.square(train_y - y_pre))


        pred_error_1 = clf_error_1_pri      
        rep_error_1 = self.rep_alpha*tf.sqrt(tf.clip_by_value(utils.mmd2_lin(hidden,input_t,p),SQRT_CONST,tf.cast(np.inf,tf.float32)))
        GNN_error_1 = self.rep_alpha*tf.sqrt(tf.clip_by_value(utils.mmd2_lin(GNN,input_t,p),SQRT_CONST,tf.cast(np.inf,tf.float32)))
        

        L_1 =   rep_error_1 + pred_error_1 + self.reg_lambda*regularization_1 + GNN_error_1 

        
        
        
    
        return L_1
    
 
    def get_grad(self,inputtensor,aggreted_results,y):
        with tf.GradientTape() as tape:
            tape.watch(self.variables)
            L = self.get_loss(inputtensor,aggreted_results,y)
            self.train_loss = L
            g = tape.gradient(L,self.variables)
        return g
        
    def network_learn(self, inputtensor,aggreted_results,y):
        g = self.get_grad(inputtensor,aggreted_results,y)
        self.optimizer.apply_gradients(zip(g,self.variables))
        return self.train_loss
        
 
    
   
 
    def CV_y(self,inputtensor,aggreted_results,train_y,training = False):
        print("CV ...")
        input_tensor = inputtensor[:,:-1]

        input_t = tf.constant(inputtensor[:,-1],shape = [input_tensor.shape[0],1])
        features = input_tensor
        

        regularization_1 = 0
        hidden = input_tensor
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden,flag = training)

        

        GNN = aggreted_results
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i](GNN)
          
   
       
        concated_data = tf.concat([hidden,GNN ],axis = 1)

     
        group_t,group_c,i_0,i_1= utils.divide_TC(concated_data ,input_t)
        
        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
            
        output_T = self.layer_6_T(outnn_T)

        outnn_C = group_c
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
           
        output_C = self.layer_6_C(outnn_C)
        
        y_pre = tf.dynamic_stitch([i_0,i_1],[output_C,output_T])
        
        
        clf_error_1_pri = tf.reduce_mean(tf.square(train_y - y_pre))


        pred_error_1 = clf_error_1_pri 

        return clf_error_1_pri
    def pre_yf(self, inputtensor,aggreted_results,training = False):
        input_tensor = inputtensor[:,:-1]
        
        input_t = tf.constant(inputtensor[:,-1],shape = [input_tensor.shape[0],1])
        features = input_tensor
        

        regularization_1 = 0
        hidden = input_tensor
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden,flag = training)
       
     

        GNN = aggreted_results
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i](GNN)
           
      
        concated_data = tf.concat([hidden,GNN ],axis = 1)

       
        

        group_t,group_c=concated_data,concated_data
        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
           
        output_T = self.layer_6_T(outnn_T)

        outnn_C = group_c
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
            

        output_C = self.layer_6_C(outnn_C)
        
        output = input_t*output_T + (1-input_t)*output_C
 
        
        return output
            
             
 
    
 
