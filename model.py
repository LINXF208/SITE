import warnings
warnings.filterwarnings('ignore')

from tensorflow import keras
import tensorflow as tf
import numpy as np

import utils


class RepLayer(keras.layers.Layer):
    def __init__(self, num_outputs, activation=tf.nn.relu, reg=0.0):
        super(RepLayer, self).__init__()
        self.num_outputs = num_outputs
        self.activation = activation
        self.reg = reg

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel",
            shape=[int(input_shape[-1]),
            self.num_outputs],
            dtype=tf.float32,
            initializer=tf.keras.initializers.glorot_uniform(),
            regularizer=keras.regularizers.l2(self.reg)
        )
        
        self.bias = self.add_weight("bias", shape=[self.num_outputs], initializer=keras.initializers.Zeros())
        
    def call(self, features):
        output = tf.matmul(features, self.kernel) + self.bias
        output = self.activation(output)
        
        return output


class SITE(keras.Model):
    def __init__(self, config, activation=tf.nn.relu):
        super(SITE, self).__init__()
        
        print("Initialization ...")
        
        self.rep_layers = []
        self.gnn_layers = []
        self.out_T_layers = []
        self.out_C_layers = []
        
        self.train_loss = None
        self.activation = activation
        self.optimizer= keras.optimizers.Adam(lr=config['lr_rate'], decay=config['lr_dc'])
        self.use_batch = config['use_batch']

        self.rep_alpha = config['rep_alpha']
        self.reg_lambda = config['reg_lambda']
        
        self.out_dropout = config['out_dropout']
        self.GNN_dropout = config['GNN_dropout']
        self.rep_dropout = config['rep_dropout']
        self.inp_drop = config['inp_dropout']

        for i in range(config['rep_hidden_layer']):
            h = RepLayer(config['rep_hidden_shape'][i], activation=self.activation, reg=0.0)
            self.rep_layers.append(h)

        for i in range(config['GNN_hidden_layer']):
            g = RepLayer(config['GNN_hidden_shape'][i], activation=self.activation, reg=0.0)
            self.gnn_layers.append(g)
 
        for i in range(config['out_T_layer']):
            out_T = keras.layers.Dense(
                config['out_hidden_shape'][i],
                activation=self.activation,
                kernel_initializer=tf.keras.initializers.glorot_uniform(),
                kernel_regularizer=keras.regularizers.l2(0.0)
            )
            self.out_T_layers.append(out_T)

        for i in range(config['out_C_layer']):
            out_C = keras.layers.Dense(
                config['out_hidden_shape'][i], 
                activation=self.activation,
                kernel_initializer=tf.keras.initializers.glorot_uniform(),
                kernel_regularizer=keras.regularizers.l2(0.0)
            )
            self.out_C_layers.append(out_C)

        self.final_out_y1 = keras.layers.Dense(1)
        self.final_out_y0 = keras.layers.Dense(1)

    def call(self, input_tensor, aggreted_results, training=False):
        input_x = input_tensor[:, :-1]
        input_t = tf.constant(input_tensor[:, -1], shape=[input_x.shape[0], 1])

        hidden = input_x
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden)

        GNN = aggreted_results
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i](GNN)

        concated_data = tf.concat([hidden, GNN], axis=1)
        group_t, group_c = concated_data, concated_data

        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
        output_T = self.final_out_y1(outnn_T)

        outnn_C = group_c
        for i in range(len(self.out_C_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
        output_C = self.final_out_y0(outnn_C)

        return output_T, output_C

    def get_loss(self, input_tensor, aggreted_results, train_y, training=True):
        input_x = input_tensor[:, :-1]
        input_t = tf.constant(input_tensor[:, -1], shape=[input_x.shape[0], 1])
        p = tf.divide(tf.reduce_sum(input_t), input_t.shape[0])

        regularization = 0

        hidden = input_x
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden)
            hidden = tf.nn.dropout(hidden, self.rep_dropout)
            regularization += tf.nn.l2_loss(self.rep_layers[i].kernel)

        GNN = aggreted_results
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i](GNN)
            GNN = tf.nn.dropout(GNN, self.GNN_dropout)
            regularization += tf.nn.l2_loss(self.gnn_layers[i].kernel)

        concated_data = tf.concat([hidden, GNN], axis=1)
        group_t, group_c, i_0, i_1 = utils.divide_t_c(concated_data, input_t)
        
        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
            outnn_T = tf.nn.dropout(outnn_T, self.out_dropout)
            regularization += tf.nn.l2_loss(self.out_T_layers[i].kernel)
        output_T = self.final_out_y1(outnn_T)

        outnn_C = group_c
        for i in range(len(self.out_C_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
            outnn_C = tf.nn.dropout(outnn_C, self.out_dropout)
            regularization += tf.nn.l2_loss(self.out_C_layers[i].kernel)
        output_C = self.final_out_y0(outnn_C)
        y_pre = tf.dynamic_stitch([i_0, i_1], [output_C, output_T])

        pred_error = tf.reduce_mean(tf.square(train_y - y_pre))
   
        rep_error = self.rep_alpha * tf.sqrt(
            tf.clip_by_value(
                utils.mmd2_lin(hidden, input_t, p), 
                1e-10,
                tf.cast(np.inf, tf.float32)
            )
        )
        
        GNN_error = self.rep_alpha * tf.sqrt(
            tf.clip_by_value(
                utils.mmd2_lin(GNN, input_t, p), 
                1e-10, 
                tf.cast(np.inf, tf.float32)
            )
        )

        L =   rep_error + pred_error + self.reg_lambda * regularization + GNN_error

        return L
    
    def get_grad(self, input_tensor, aggreted_results, y):
        with tf.GradientTape() as tape:
            tape.watch(self.variables)
            self.train_loss = self.get_loss(input_tensor, aggreted_results, y)
            g = tape.gradient(self.train_loss, self.variables)
            
        return g

    def network_learn(self, input_tensor, aggreted_results, y):
        g = self.get_grad(input_tensor, aggreted_results, y)
        self.optimizer.apply_gradients(zip(g, self.variables))

        return self.train_loss

    def val_y(self, input_tensor, aggreted_results, train_y, training=False):
        input_x = input_tensor[:,:-1]
        input_t = tf.constant(input_tensor[:,-1], shape=[input_x.shape[0], 1])

        hidden = input_x
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden)

        GNN = aggreted_results
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i](GNN)

        concated_data = tf.concat([hidden, GNN], axis = 1)
        group_t, group_c, i_0, i_1= utils.divide_t_c(concated_data, input_t)
        
        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
        output_T = self.final_out_y1(outnn_T)

        outnn_C = group_c
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
        output_C = self.final_out_y0(outnn_C)
        
        y_pre = tf.dynamic_stitch([i_0, i_1], [output_C, output_T])
        
        pred_error = tf.reduce_mean(tf.square(train_y - y_pre))

        return pred_error

    def pre_yf(self, input_tensor, aggreted_results, training=False):
        input_x = input_tensor[:, :-1]
        input_t = tf.constant(input_tensor[:, -1], shape=[input_x.shape[0], 1])

        hidden = input_x
        for i in range(len(self.rep_layers)):
            hidden = self.rep_layers[i](hidden)

        GNN = aggreted_results
        for i in range(len(self.gnn_layers)):
            GNN = self.gnn_layers[i](GNN)

        concated_data = tf.concat([hidden, GNN], axis=1)
        group_t, group_c=concated_data, concated_data

        outnn_T = group_t
        for i in range(len(self.out_T_layers)):
            outnn_T = self.out_T_layers[i](outnn_T)
        output_T = self.final_out_y1(outnn_T)

        outnn_C = group_c
        for i in range(len(self.out_T_layers)):
            outnn_C = self.out_C_layers[i](outnn_C)
        output_C = self.final_out_y0(outnn_C)
        
        output = input_t * output_T + (1 - input_t) * output_C

        return output
