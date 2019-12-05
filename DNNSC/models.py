#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 22:31:15 2018

@author: Yupeng Shi
@creating time: 7/23/2018
"""

#from __future__ import division
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout,
    merge
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)

from keras.layers import (Conv1D, TimeDistributed, Bidirectional, SimpleRNN, GRU, LSTM, MaxPooling1D)

from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda,  Reshape
#from keras.regularizers import l2
from keras import regularizers
from keras import backend as K
#from keras.layers import Input, Dense
#from keras.models import Model
import keras
from tcn import TCN

# from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.models import Sequential

from spectrum import lpc
from scipy import signal
import numpy as np

def func(data_frames, n_freqs):
    #win=np.hanning(frame_length)
    print("computing the weight factor...")
    nfft = (n_freqs-1)*2
            
    ak, _ = lpc(data_frames, 10)
    a_numerator = np.insert(ak, 0, 1)
    a_denominator = np.ones_like(a_numerator)
    for ii in range(1, len(a_numerator)):
        a_denominator[ii] = (0.8 ** ii) * a_numerator[ii]

    _, H = signal.freqz(a_numerator, a_denominator, worN=nfft, whole=True)
    weight_factor = 20*np.log10(abs(H[0:n_freqs]))
     
    return weight_factor

def get_slice(x,index):
        return x[:, index, :]

def dnn(n_hid, n_concat, n_freq):
    
# 
    inputs_batch = Input(shape=(n_concat, n_freq))
    flatten1= Flatten()(inputs_batch)
    dense1 = Dense(n_hid, kernel_regularizer=regularizers.l2(l=0.0001))(flatten1)
    bn1 = BatchNormalization()(dense1)
    activation1 = keras.layers.PReLU()(bn1)
    dropout1 = Dropout(0.2)(activation1)

    dense2 = Dense(n_hid, kernel_regularizer=regularizers.l2(l=0.0001))(dropout1)
    bn2 = BatchNormalization()(dense2)
    activation2 = keras.layers.PReLU()(bn2)
    dropout2 = Dropout(0.2)(activation2)
    
    

    dense3 = Dense(n_hid, kernel_regularizer=regularizers.l2(l=0.0001))(dropout2)
    bn3 = BatchNormalization()(dense3)
    activation3 = Activation('linear')(bn3)
    dropout3 = Dropout(0.2)(activation3)

    dense4 = Dense(n_freq, activation='linear')(dropout3)

    model = Model(inputs=inputs_batch, outputs=dense4)
    
    return model

def sdnn1(n_hid, n_concat, n_freq):
    
    #short_cut = K.variable(value=x)
    inputs_batch = Input(shape=(n_concat, n_freq))
    #lambda1 = Lambda(lambda x: x[3, :], output_shape=(1, n_freq))(inputs_batch)
    lambda1 = Lambda(get_slice,output_shape=(1,n_freq),arguments={'index':3})(inputs_batch)
    #lambda_flatten = Flatten()(K.lambda1)
    #sliced_reshape = K.reshape(input_sliced, [None, 1, n_freq])
    #K.int_shape(sliced_reshape)
    #input2_flatten = Flatten()(input_sliced)
    
    flatten1= Flatten()(inputs_batch)
    dense1 = Dense(n_hid, kernel_regularizer=regularizers.l2(l=0.0001))(flatten1)
    bn1 = BatchNormalization()(dense1)
    activation1 = keras.layers.PReLU()(bn1)
    dropout1 = Dropout(0.2)(activation1)

    dense2 = Dense(n_hid, kernel_regularizer=regularizers.l2(l=0.0001))(dropout1)
    bn2 = BatchNormalization()(dense2)
    activation2 = keras.layers.PReLU()(bn2)
    dropout2 = Dropout(0.2)(activation2)

    dense3 = Dense(n_hid, kernel_regularizer=regularizers.l2(l=0.0001))(dropout2)
    bn3 = BatchNormalization()(dense3)
    activation3 = Activation('linear')(bn3)
    dropout3 = Dropout(0.2)(activation3)
    
    dense4 = Dense(n_freq, activation='linear')(dropout3)
    lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    dense4_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(dense4)
    residual_layer = keras.layers.add([dense4_reshape, lambda1_reshape])
    
    final_output = Flatten()(residual_layer)
    dense5 = Dense(n_freq, activation='linear')(final_output)
    #dropout3 = Dropout(0.2)(activation3)
    
    #dense4 = Dense(n_freq, kernel_regularizer=regularizers.l2(l=0.0001))(activation3)
    #bn4 = BatchNormalization()(dense4)
    #activation4 = Activation('relu')(bn4)
    
    #dense4 = Dense(n_freq, activation='linear')(dropout3)
    #def expanDims(ins): 
    #    return K.expand_dims(ins, 1)
    #dense4_reshpae = Reshape((1, n_freq))(activation4)
    #lambda1_reshape = Reshape((1, n_freq))(lambda1)
    #residual_layer = add([dense4_reshpae, lambda1])
    #residual_reshape = Reshape((n_freq))(residual_layer)
    #flatten2 = Flatten()(residual_layer)
    #dense5 = Dense(n_freq, activation='linear')(flatten2)
    #out_final = Flatten()(residual_layer)
    
    #dense5 = Dense(n_freq, activation='linear')(Flatten()(one_skip_output))
    #one_skip_output = merge([dense4, lambda1], mode='sum')
    model = Model(inputs=inputs_batch, outputs=dense5)
    return model


def sdnn2(n_hid, n_concat, n_freq):
    
    #short_cut = K.variable(value=x)
    inputs_batch = Input(shape=(n_concat, n_freq))
    #lambda1 = Lambda(lambda x: x[3, :], output_shape=(1, n_freq))(inputs_batch)
    lambda1 = Lambda(get_slice,output_shape=(1,n_freq),arguments={'index':3})(inputs_batch)
    #lambda_flatten = Flatten()(K.lambda1)
    #sliced_reshape = K.reshape(input_sliced, [None, 1, n_freq])
    #K.int_shape(sliced_reshape)
    #input2_flatten = Flatten()(input_sliced)
    
    flatten1= Flatten()(inputs_batch)
    dense1 = Dense(n_hid, kernel_regularizer=regularizers.l2(l=0.0001))(flatten1)
    bn1 = BatchNormalization()(dense1)
    activation1 = Activation('relu')(bn1)
    #dropout1 = Dropout(0.2)(activation1)


    dense2 = Dense(n_hid, kernel_regularizer=regularizers.l2(l=0.0001))(activation1)
    bn2 = BatchNormalization()(dense2)
    activation2 = Activation('relu')(bn2)
    #dropout2 = Dropout(0.2)(activation2)
    
    activation_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_hid))(activation2)
    extra1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_hid))(activation1)
    residual_layer1 = add([extra1_reshape, activation_reshape])
    
    residual1_flatten = Flatten()(residual_layer1)

    dense3 = Dense(n_hid, kernel_regularizer=regularizers.l2(l=0.0001))(residual1_flatten)
    bn3 = BatchNormalization()(dense3)
    activation3 = Activation('relu')(bn3)
    
    dense4 = Dense(n_freq, activation='linear')(activation3)
    lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    dense4_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(dense4)
    residual_layer = keras.layers.add([dense4_reshape, lambda1_reshape])
    
    final_output = Flatten()(residual_layer)
    dense5 = Dense(n_freq, activation='linear')(final_output)
    #dropout3 = Dropout(0.2)(activation3)
    
    #dense4 = Dense(n_freq, kernel_regularizer=regularizers.l2(l=0.0001))(activation3)
    #bn4 = BatchNormalization()(dense4)
    #activation4 = Activation('relu')(bn4)
    
    #dense4 = Dense(n_freq, activation='linear')(dropout3)
    #def expanDims(ins): 
    #    return K.expand_dims(ins, 1)
    #dense4_reshpae = Reshape((1, n_freq))(activation4)
    #lambda1_reshape = Reshape((1, n_freq))(lambda1)
    #residual_layer = add([dense4_reshpae, lambda1])
    #residual_reshape = Reshape((n_freq))(residual_layer)
    #flatten2 = Flatten()(residual_layer)
    #dense5 = Dense(n_freq, activation='linear')(flatten2)
    #out_final = Flatten()(residual_layer)
    
    #dense5 = Dense(n_freq, activation='linear')(Flatten()(one_skip_output))
    #one_skip_output = merge([dense4, lambda1], mode='sum')
    model = Model(inputs=inputs_batch, outputs=dense5)
    return model

def sdnn3(n_hid, n_concat, n_freq):
    
    # n_hid = 1024
    inputs_batch = Input(shape=(n_concat, n_freq))
    lambda1 = Lambda(get_slice,output_shape=(1,n_freq),arguments={'index':3})(inputs_batch)
    
    flatten1= Flatten()(inputs_batch)
    dense1 = Dense(n_hid, kernel_regularizer=regularizers.l2(l=0.0001))(flatten1)
    bn1 = BatchNormalization()(dense1)
    activation1 = Activation('relu')(bn1)
    #dropout1 = Dropout(0.1)(activation1)
    extra_dense1 = Dense(n_freq, activation='linear')(activation1)
    lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    extra1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(extra_dense1)
    residual_layer1 = add([extra1_reshape, lambda1_reshape])
    
    residual1_flatten = Flatten()(residual_layer1)
    dense2 = Dense(n_hid, kernel_regularizer=regularizers.l2(l=0.0001))(residual1_flatten)
    bn2 = BatchNormalization()(dense2)
    activation2 = Activation('relu')(bn2)
    #dropout2 = Dropout(0.1)(activation2)
    extra_dense2 = Dense(n_freq, activation='linear')(activation2)
    
    extral2_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(extra_dense2)
    residual_layer2 = add([residual_layer1, extral2_reshape])
    
    residual2_flatten = Flatten()(residual_layer2)
    dense3 = Dense(n_hid, kernel_regularizer=regularizers.l2(l=0.0001))(residual2_flatten)
    bn3 = BatchNormalization()(dense3)
    activation3 = Activation('relu')(bn3)
    #dropout3 = Dropout(0.1)(activation3)
    extra_dense3 = Dense(n_freq, activation='linear')(activation3)
    extral3_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(extra_dense3)
    residual_layer3 = add([residual_layer2, extral3_reshape])
    
    final_output = Flatten()(residual_layer3)
    dense4 = Dense(n_freq, activation='linear')(final_output)
    #dense4 = Dense(n_freq, activation='linear')(residual_layer3)
    #one_skip_output = add(dense4, input2)
    model = Model(inputs=inputs_batch, outputs=dense4)
    return model

def fcn(n_concat, n_freq):
    
    inputs_batch = Input(shape=(n_concat, n_freq))
    input_reshape = Lambda(lambda x: K.expand_dims(x,3), output_shape=(n_concat, n_freq, 1))(inputs_batch)
    cov1 = Conv2D(12, input_shape=[n_concat, n_freq],
           kernel_size=(n_concat, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(input_reshape)
    bn1 = BatchNormalization(axis=3)(cov1)
    relu1 = Activation('relu')(bn1)
    drop1 = Dropout(0.25)(relu1)
    
    cov2 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop1)
    bn2 = BatchNormalization(axis=3)(cov2)
    relu2 = Activation('relu')(bn2)
    drop2 = Dropout(0.25)(relu2)
    
    cov3 = Conv2D(20, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop2)
    bn3 = BatchNormalization(axis=3)(cov3)
    relu3 = Activation('relu')(bn3)
    drop3 = Dropout(0.25)(relu3)
    
    cov4 = Conv2D(24, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop3)
    bn4 = BatchNormalization(axis=3)(cov4)
    relu4 = Activation('relu')(bn4)
    drop4 = Dropout(0.25)(relu4)
    
    cov5 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop4)
    bn5 = BatchNormalization(axis=3)(cov5)
    relu5 = Activation('relu')(bn5)
    drop5 = Dropout(0.25)(relu5)

    cov6 = Conv2D(24, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop5)
    bn6 = BatchNormalization(axis=3)(cov6)
    relu6 = Activation('relu')(bn6)
    drop6 = Dropout(0.25)(relu6)
    
    cov7 = Conv2D(20, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop6)
    bn7 = BatchNormalization(axis=3)(cov7)
    relu7 = Activation('relu')(bn7)
    drop7 = Dropout(0.25)(relu7)
    
    cov8 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop7)
    bn8 = BatchNormalization(axis=3)(cov8)
    relu8 = Activation('relu')(bn8)
    drop8 = Dropout(0.25)(relu8)
    
    cov9 = Conv2D(12, input_shape=[1, n_freq],
           kernel_size=(1, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop8)
    bn9 = BatchNormalization(axis=3)(cov9)
    relu9 = Activation('relu')(bn9)
    drop9 = Dropout(0.25)(relu9)
    
    cov10 = Conv2D(1, input_shape=[1, n_freq],
           kernel_size=(1, 3),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           activation='relu')(drop9)
    cov10_flatten = Flatten()(cov10)
    dense10 = Dense(n_freq, activation='linear')(cov10_flatten)
    
    model = Model(inputs=inputs_batch, outputs=dense10)
    return model


def fcn1_re(n_concat, n_freq):
    
    inputs_batch = Input(shape=(n_concat, n_freq))
    lambda1 = Lambda(get_slice,output_shape=(1,n_freq),arguments={'index':3})(inputs_batch)
    
    input_reshape = Lambda(lambda x: K.expand_dims(x,3), output_shape=(n_concat, n_freq, 1))(inputs_batch)
    cov1 = Conv2D(12, input_shape=[n_concat, n_freq],
           kernel_size=(n_concat, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(input_reshape)
    bn1 = BatchNormalization(axis=3)(cov1)
    relu1 = Activation('relu')(bn1)
    drop1 = Dropout(0.25)(relu1)
    
    cov2 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop1)
    bn2 = BatchNormalization(axis=3)(cov2)
    relu2 = Activation('relu')(bn2)
    drop2 = Dropout(0.25)(relu2)
    
    cov3 = Conv2D(20, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop2)
    bn3 = BatchNormalization(axis=3)(cov3)
    relu3 = Activation('relu')(bn3)
    drop3 = Dropout(0.25)(relu3)
    
    cov4 = Conv2D(24, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop3)
    bn4 = BatchNormalization(axis=3)(cov4)
    relu4 = Activation('relu')(bn4)
    drop4 = Dropout(0.25)(relu4)
    
    cov5 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop4)
    bn5 = BatchNormalization(axis=3)(cov5)
    relu5 = Activation('relu')(bn5)
    drop5 = Dropout(0.25)(relu5)

    cov6 = Conv2D(24, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop5)
    bn6 = BatchNormalization(axis=3)(cov6)
    relu6 = Activation('relu')(bn6)
    drop6 = Dropout(0.25)(relu6)
    
    cov7 = Conv2D(20, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop6)
    bn7 = BatchNormalization(axis=3)(cov7)
    relu7 = Activation('relu')(bn7)
    drop7 = Dropout(0.25)(relu7)
    
    cov8 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop7)
    bn8 = BatchNormalization(axis=3)(cov8)
    relu8 = Activation('relu')(bn8)
    drop8 = Dropout(0.25)(relu8)
    
    cov9 = Conv2D(12, input_shape=[1, n_freq],
           kernel_size=(1, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop8)
    bn9 = BatchNormalization(axis=3)(cov9)
    relu9 = Activation('relu')(bn9)
    drop9 = Dropout(0.25)(relu9)
    
    #residual_layer1 = keras.layers.add([drop1, drop9])
    
    cov10 = Conv2D(1, input_shape=[1, n_freq],
           kernel_size=(1, 3),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           activation='relu')(drop9)
    
    cov10_flatten = Flatten()(cov10)
    
    #lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    cov10_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(cov10_flatten)
    residual_layer = keras.layers.add([cov10_reshape, lambda1_reshape])
    
    final_output = Flatten()(residual_layer)
    dense10 = Dense(n_freq, activation='linear')(final_output)
    
    model = Model(inputs=inputs_batch, outputs=dense10)
    return model


def fcn1(n_concat, n_freq):
    
    inputs_batch = Input(shape=(n_concat, n_freq))
    #lambda1 = Lambda(get_slice,output_shape=(1,n_freq),arguments={'index':3})(inputs_batch)
    
    input_reshape = Lambda(lambda x: K.expand_dims(x,3), output_shape=(n_concat, n_freq, 1))(inputs_batch)
    cov1 = Conv2D(12, input_shape=[n_concat, n_freq],
           kernel_size=(n_concat, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(input_reshape)
    bn1 = BatchNormalization(axis=3)(cov1)
    relu1 = Activation('relu')(bn1)
    drop1 = Dropout(0.25)(relu1)
    
    cov2 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop1)
    bn2 = BatchNormalization(axis=3)(cov2)
    relu2 = Activation('relu')(bn2)
    drop2 = Dropout(0.25)(relu2)
    
    cov3 = Conv2D(20, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop2)
    bn3 = BatchNormalization(axis=3)(cov3)
    relu3 = Activation('relu')(bn3)
    drop3 = Dropout(0.25)(relu3)
    
    cov4 = Conv2D(24, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop3)
    bn4 = BatchNormalization(axis=3)(cov4)
    relu4 = Activation('relu')(bn4)
    drop4 = Dropout(0.25)(relu4)
    
    cov5 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop4)
    bn5 = BatchNormalization(axis=3)(cov5)
    relu5 = Activation('relu')(bn5)
    drop5 = Dropout(0.25)(relu5)

    cov6 = Conv2D(24, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop5)
    bn6 = BatchNormalization(axis=3)(cov6)
    relu6 = Activation('relu')(bn6)
    drop6 = Dropout(0.25)(relu6)
    
    cov7 = Conv2D(20, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop6)
    bn7 = BatchNormalization(axis=3)(cov7)
    relu7 = Activation('relu')(bn7)
    drop7 = Dropout(0.25)(relu7)
    
    cov8 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop7)
    bn8 = BatchNormalization(axis=3)(cov8)
    relu8 = Activation('relu')(bn8)
    drop8 = Dropout(0.25)(relu8)
    
    cov9 = Conv2D(12, input_shape=[1, n_freq],
           kernel_size=(1, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop8)
    bn9 = BatchNormalization(axis=3)(cov9)
    relu9 = Activation('relu')(bn9)
    drop9 = Dropout(0.25)(relu9)
    
    residual_layer1 = keras.layers.add([drop1, drop9])
    
    cov10 = Conv2D(1, input_shape=[1, n_freq],
           kernel_size=(1, 3),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           activation='relu')(residual_layer1)
    
    cov10_flatten = Flatten()(cov10)
    
    #lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    #lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    #cov10_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(cov10_flatten)
    #residual_layer = keras.layers.add([cov10_reshape, lambda1_reshape])
    
    #final_output = Flatten()(residual_layer)
    dense10 = Dense(n_freq, activation='linear')(cov10_flatten)
    
    model = Model(inputs=inputs_batch, outputs=dense10)
    return model
'''
def fcn2(n_concat, n_freq):
    
    inputs_batch = Input(shape=(n_concat, n_freq))
    #lambda1 = Lambda(get_slice,output_shape=(1,n_freq),arguments={'index':3})(inputs_batch)
    
    input_reshape = Lambda(lambda x: K.expand_dims(x,3), output_shape=(n_concat, n_freq, 1))(inputs_batch)
    cov1 = Conv2D(12, input_shape=[n_concat, n_freq],
           kernel_size=(n_concat, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(input_reshape)
    bn1 = BatchNormalization(axis=3)(cov1)
    relu1 = Activation('relu')(bn1)
    drop1 = Dropout(0.25)(relu1)
    
    cov2 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop1)
    bn2 = BatchNormalization(axis=3)(cov2)
    relu2 = Activation('relu')(bn2)
    drop2 = Dropout(0.25)(relu2)
    
    cov3 = Conv2D(20, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop2)
    bn3 = BatchNormalization(axis=3)(cov3)
    relu3 = Activation('relu')(bn3)
    drop3 = Dropout(0.25)(relu3)
    
    cov4 = Conv2D(24, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop3)
    bn4 = BatchNormalization(axis=3)(cov4)
    relu4 = Activation('relu')(bn4)
    drop4 = Dropout(0.25)(relu4)
    
    cov5 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop4)
    bn5 = BatchNormalization(axis=3)(cov5)
    relu5 = Activation('relu')(bn5)
    drop5 = Dropout(0.25)(relu5)

    cov6 = Conv2D(24, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop5)
    bn6 = BatchNormalization(axis=3)(cov6)
    relu6 = Activation('relu')(bn6)
    drop6 = Dropout(0.25)(relu6)
    
    cov7 = Conv2D(20, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop6)
    bn7 = BatchNormalization(axis=3)(cov7)
    relu7 = Activation('relu')(bn7)
    drop7 = Dropout(0.25)(relu7)
    
    residual_layer3 = keras.layers.add([drop3, drop7])
    
    cov8 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(residual_layer3)
    bn8 = BatchNormalization(axis=3)(cov8)
    relu8 = Activation('relu')(bn8)
    drop8 = Dropout(0.25)(relu8)
    
    cov9 = Conv2D(12, input_shape=[1, n_freq],
           kernel_size=(1, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop8)
    bn9 = BatchNormalization(axis=3)(cov9)
    relu9 = Activation('relu')(bn9)
    drop9 = Dropout(0.25)(relu9)
    
    residual_layer1 = keras.layers.add([drop1, drop9])
    
    cov10 = Conv2D(1, input_shape=[1, n_freq],
           kernel_size=(1, 3),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           activation='relu')(residual_layer1)
    
    cov10_flatten = Flatten()(cov10)
    
    #lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    #lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    #cov10_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(cov10_flatten)
    #residual_layer = keras.layers.add([cov10_reshape, lambda1_reshape])
    
    #final_output = Flatten()(residual_layer)
    dense10 = Dense(n_freq, activation='linear')(cov10_flatten)
    
    model = Model(inputs=inputs_batch, outputs=dense10)
    return model
'''
def fcn2(n_concat, n_freq):
    
    inputs_batch = Input(shape=(n_concat, n_freq))
    lambda1 = Lambda(get_slice,output_shape=(1,n_freq),arguments={'index':3})(inputs_batch)
    
    input_reshape = Lambda(lambda x: K.expand_dims(x,3), output_shape=(n_concat, n_freq, 1))(inputs_batch)
    cov1 = Conv2D(12, input_shape=[n_concat, n_freq],
           kernel_size=(n_concat, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(input_reshape)
    bn1 = BatchNormalization(axis=3)(cov1)
    relu1 = Activation('relu')(bn1)
    drop1 = Dropout(0.25)(relu1)
    
    cov2 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop1)
    bn2 = BatchNormalization(axis=3)(cov2)
    relu2 = Activation('relu')(bn2)
    drop2 = Dropout(0.25)(relu2)
    
    cov3 = Conv2D(20, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop2)
    bn3 = BatchNormalization(axis=3)(cov3)
    relu3 = Activation('relu')(bn3)
    drop3 = Dropout(0.25)(relu3)
    
    cov4 = Conv2D(24, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop3)
    bn4 = BatchNormalization(axis=3)(cov4)
    relu4 = Activation('relu')(bn4)
    drop4 = Dropout(0.25)(relu4)
    
    cov5 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop4)
    bn5 = BatchNormalization(axis=3)(cov5)
    relu5 = Activation('relu')(bn5)
    drop5 = Dropout(0.25)(relu5)

    cov6 = Conv2D(24, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop5)
    bn6 = BatchNormalization(axis=3)(cov6)
    relu6 = Activation('relu')(bn6)
    drop6 = Dropout(0.25)(relu6)
    
    #residual_layer4 = keras.layers.add([drop4, drop6])
    
    cov7 = Conv2D(20, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop6)
    bn7 = BatchNormalization(axis=3)(cov7)
    relu7 = Activation('relu')(bn7)
    drop7 = Dropout(0.25)(relu7)
    
    #residual_layer3 = keras.layers.add([drop3, drop7])
    
    cov8 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop7)
    bn8 = BatchNormalization(axis=3)(cov8)
    relu8 = Activation('relu')(bn8)
    drop8 = Dropout(0.25)(relu8)
    
    #residual_layer2 = keras.layers.add([drop2, drop8])
    
    cov9 = Conv2D(12, input_shape=[1, n_freq],
           kernel_size=(1, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop8)
    bn9 = BatchNormalization(axis=3)(cov9)
    relu9 = Activation('relu')(bn9)
    drop9 = Dropout(0.25)(relu9)
    
    #residual_layer1 = keras.layers.add([drop1, drop9])
    residual_layer1 = keras.layers.add([drop1, drop9])
    
    cov10 = Conv2D(1, input_shape=[1, n_freq],
           kernel_size=(1, 3),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           activation='relu')(residual_layer1)
    
    cov10_flatten = Flatten()(cov10)
    
    #lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    #lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    #cov10_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(cov10_flatten)
    #residual_layer = keras.layers.add([cov10_reshape, lambda1_reshape])
    
    #final_output = Flatten()(residual_layer)
    lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    cov10_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(cov10_flatten)
    residual_layer = keras.layers.add([cov10_reshape, lambda1_reshape])
    
    final_output = Flatten()(residual_layer)
    dense10 = Dense(n_freq, activation='linear')(final_output)
    
    model = Model(inputs=inputs_batch, outputs=dense10)
    return model

def fcn3(n_concat, n_freq):
    
    inputs_batch = Input(shape=(n_concat, n_freq))
    lambda1 = Lambda(get_slice,output_shape=(1,n_freq),arguments={'index':3})(inputs_batch)
    
    input_reshape = Lambda(lambda x: K.expand_dims(x,3), output_shape=(n_concat, n_freq, 1))(inputs_batch)
    cov1 = Conv2D(12, input_shape=[n_concat, n_freq],
           kernel_size=(n_concat, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(input_reshape)
    bn1 = BatchNormalization(axis=3)(cov1)
    relu1 = Activation('relu')(bn1)
    drop1 = Dropout(0.25)(relu1)
    
    cov2 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop1)
    bn2 = BatchNormalization(axis=3)(cov2)
    relu2 = Activation('relu')(bn2)
    drop2 = Dropout(0.25)(relu2)
    
    cov3 = Conv2D(20, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop2)
    bn3 = BatchNormalization(axis=3)(cov3)
    relu3 = Activation('relu')(bn3)
    drop3 = Dropout(0.25)(relu3)
    
    cov4 = Conv2D(24, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop3)
    bn4 = BatchNormalization(axis=3)(cov4)
    relu4 = Activation('relu')(bn4)
    drop4 = Dropout(0.25)(relu4)
    
    cov5 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop4)
    bn5 = BatchNormalization(axis=3)(cov5)
    relu5 = Activation('relu')(bn5)
    drop5 = Dropout(0.25)(relu5)

    cov6 = Conv2D(24, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop5)
    bn6 = BatchNormalization(axis=3)(cov6)
    relu6 = Activation('relu')(bn6)
    drop6 = Dropout(0.25)(relu6)
    
    
    cov7 = Conv2D(20, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop6)
    bn7 = BatchNormalization(axis=3)(cov7)
    relu7 = Activation('relu')(bn7)
    drop7 = Dropout(0.25)(relu7)
    
    residual_layer3 = keras.layers.add([drop3, drop7])
    
    cov8 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(residual_layer3)
    bn8 = BatchNormalization(axis=3)(cov8)
    relu8 = Activation('relu')(bn8)
    drop8 = Dropout(0.25)(relu8)
    
    cov9 = Conv2D(12, input_shape=[1, n_freq],
           kernel_size=(1, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop8)
    bn9 = BatchNormalization(axis=3)(cov9)
    relu9 = Activation('relu')(bn9)
    drop9 = Dropout(0.25)(relu9)
    
    residual_layer1 = keras.layers.add([drop1, drop9])
    
    cov10 = Conv2D(1, input_shape=[1, n_freq],
           kernel_size=(1, 3),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           activation='relu')(residual_layer1)
    
    cov10_flatten = Flatten()(cov10)
    
    #lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    #lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    #cov10_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(cov10_flatten)
    #residual_layer = keras.layers.add([cov10_reshape, lambda1_reshape])
    
    #final_output = Flatten()(residual_layer)
    lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    cov10_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(cov10_flatten)
    residual_layer = keras.layers.add([cov10_reshape, lambda1_reshape])
    
    final_output = Flatten()(residual_layer)
    dense10 = Dense(n_freq, activation='linear')(final_output)
    
    model = Model(inputs=inputs_batch, outputs=dense10)
    return model


def fcn4(n_concat, n_freq):
    
    inputs_batch = Input(shape=(n_concat, n_freq))
    lambda1 = Lambda(get_slice,output_shape=(1,n_freq),arguments={'index':3})(inputs_batch)
    
    input_reshape = Lambda(lambda x: K.expand_dims(x,3), output_shape=(n_concat, n_freq, 1))(inputs_batch)
    cov1 = Conv2D(12, input_shape=[n_concat, n_freq],
           kernel_size=(n_concat, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(input_reshape)
    bn1 = BatchNormalization(axis=3)(cov1)
    relu1 = Activation('relu')(bn1)
    drop1 = Dropout(0.25)(relu1)
    
    cov2 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop1)
    bn2 = BatchNormalization(axis=3)(cov2)
    relu2 = Activation('relu')(bn2)
    drop2 = Dropout(0.25)(relu2)
    
    cov3 = Conv2D(20, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop2)
    bn3 = BatchNormalization(axis=3)(cov3)
    relu3 = Activation('relu')(bn3)
    drop3 = Dropout(0.25)(relu3)
    
    cov4 = Conv2D(24, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop3)
    bn4 = BatchNormalization(axis=3)(cov4)
    relu4 = Activation('relu')(bn4)
    drop4 = Dropout(0.25)(relu4)
    
    cov5 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop4)
    bn5 = BatchNormalization(axis=3)(cov5)
    relu5 = Activation('relu')(bn5)
    drop5 = Dropout(0.25)(relu5)

    cov6 = Conv2D(24, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop5)
    bn6 = BatchNormalization(axis=3)(cov6)
    relu6 = Activation('relu')(bn6)
    drop6 = Dropout(0.25)(relu6)
    
    residual_layer4 = keras.layers.add([drop4, drop6])
    
    cov7 = Conv2D(20, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(residual_layer4)
    bn7 = BatchNormalization(axis=3)(cov7)
    relu7 = Activation('relu')(bn7)
    drop7 = Dropout(0.25)(relu7)
    
    residual_layer3 = keras.layers.add([drop3, drop7])
    
    cov8 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(residual_layer3)
    bn8 = BatchNormalization(axis=3)(cov8)
    relu8 = Activation('relu')(bn8)
    drop8 = Dropout(0.25)(relu8)
    
    residual_layer2 = keras.layers.add([drop2, drop8])
    
    cov9 = Conv2D(12, input_shape=[1, n_freq],
           kernel_size=(1, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(residual_layer2)
    bn9 = BatchNormalization(axis=3)(cov9)
    relu9 = Activation('relu')(bn9)
    drop9 = Dropout(0.25)(relu9)
    
    residual_layer1 = keras.layers.add([drop1, drop9])
    
    cov10 = Conv2D(1, input_shape=[1, n_freq],
           kernel_size=(1, 3),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           activation='relu')(residual_layer1)
    
    cov10_flatten = Flatten()(cov10)
    
    #lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    cov10_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(cov10_flatten)
    residual_layer = keras.layers.add([cov10_reshape, lambda1_reshape])
    
    final_output = Flatten()(residual_layer)
    dense10 = Dense(n_freq, activation='linear')(final_output)
    
    model = Model(inputs=inputs_batch, outputs=dense10)
    return model


def m_vgg(n_concat, n_freq):
    
    inputs_batch = Input(shape=(n_concat, n_freq))
    input_reshape = Lambda(lambda x: K.expand_dims(x,3), output_shape=(n_concat, n_freq, 1))(inputs_batch)
    cov1 = Conv2D(16, input_shape=[n_concat, n_freq],
           kernel_size=(n_concat, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(input_reshape)
    bn1 = BatchNormalization(axis=3)(cov1)
    relu1 = Activation('relu')(bn1)
    drop1 = Dropout(0.5)(relu1)
    
    cov2 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop1)
    bn2 = BatchNormalization(axis=3)(cov2)
    relu2 = Activation('relu')(bn2)
    drop2 = Dropout(0.5)(relu2)
    
    cov3 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop2)
    bn3 = BatchNormalization(axis=3)(cov3)
    relu3 = Activation('relu')(bn3)
    drop3 = Dropout(0.5)(relu3)
    
    cov4 = Conv2D(64, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop3)
    bn4 = BatchNormalization(axis=3)(cov4)
    relu4 = Activation('relu')(bn4)
    drop4 = Dropout(0.5)(relu4)
    
    cov5 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop4)
    bn5 = BatchNormalization(axis=3)(cov5)
    relu5 = Activation('relu')(bn5)
    drop5 = Dropout(0.5)(relu5)

    cov6 = Conv2D(64, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop5)
    bn6 = BatchNormalization(axis=3)(cov6)
    relu6 = Activation('relu')(bn6)
    drop6 = Dropout(0.5)(relu6)
    
    cov7 = Conv2D(128, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop6)
    bn7 = BatchNormalization(axis=3)(cov7)
    relu7 = Activation('relu')(bn7)
    drop7 = Dropout(0.5)(relu7)
    
    cov8 = Conv2D(64, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop7)
    bn8 = BatchNormalization(axis=3)(cov8)
    relu8 = Activation('relu')(bn8)
    drop8 = Dropout(0.5)(relu8)
    
    
    cov9 = Conv2D(1, input_shape=[1, n_freq],
           kernel_size=(1, 3),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           activation='relu')(drop8)
    cov9_flatten = Flatten()(cov9)
    dense10 = Dense(n_freq, activation='linear')(cov9_flatten)
    model = Model(inputs=inputs_batch, outputs=dense10)
    return model


def m_vgg1(n_concat, n_freq):
    
    inputs_batch = Input(shape=(n_concat, n_freq))
    lambda1 = Lambda(get_slice,output_shape=(1,n_freq),arguments={'index':3})(inputs_batch)
    
    input_reshape = Lambda(lambda x: K.expand_dims(x,3), output_shape=(n_concat, n_freq, 1))(inputs_batch)
    cov1 = Conv2D(16, input_shape=[n_concat, n_freq],
           kernel_size=(n_concat, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(input_reshape)
    bn1 = BatchNormalization(axis=3)(cov1)
    relu1 = Activation('relu')(bn1)
    drop1 = Dropout(0.25)(relu1)
    
    cov2 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop1)
    bn2 = BatchNormalization(axis=3)(cov2)
    relu2 = Activation('relu')(bn2)
    drop2 = Dropout(0.25)(relu2)
    
    cov3 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop2)
    bn3 = BatchNormalization(axis=3)(cov3)
    relu3 = Activation('relu')(bn3)
    drop3 = Dropout(0.25)(relu3)
    
    cov4 = Conv2D(64, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop3)
    bn4 = BatchNormalization(axis=3)(cov4)
    relu4 = Activation('relu')(bn4)
    drop4 = Dropout(0.25)(relu4)
    
    cov5 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop4)
    bn5 = BatchNormalization(axis=3)(cov5)
    relu5 = Activation('relu')(bn5)
    drop5 = Dropout(0.25)(relu5)

    cov6 = Conv2D(64, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop5)
    bn6 = BatchNormalization(axis=3)(cov6)
    relu6 = Activation('relu')(bn6)
    drop6 = Dropout(0.25)(relu6)
    
    cov7 = Conv2D(128, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop6)
    bn7 = BatchNormalization(axis=3)(cov7)
    relu7 = Activation('relu')(bn7)
    drop7 = Dropout(0.25)(relu7)
    
    cov8 = Conv2D(64, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop7)
    bn8 = BatchNormalization(axis=3)(cov8)
    relu8 = Activation('relu')(bn8)
    drop8 = Dropout(0.25)(relu8)
    
    
    cov9 = Conv2D(1, input_shape=[1, n_freq],
           kernel_size=(1, 3),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           activation='relu')(drop8)
    cov9_flatten = Flatten()(cov9)
    
    lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    #lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    cov9_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(cov9_flatten)
    residual_layer = keras.layers.add([cov9_reshape, lambda1_reshape])
    
    final_output = Flatten()(residual_layer)
    dense10 = Dense(n_freq, activation='linear')(final_output)
    model = Model(inputs=inputs_batch, outputs=dense10)
    return model

def m_vgg2(n_concat, n_freq):
    
    inputs_batch = Input(shape=(n_concat, n_freq))
    lambda1 = Lambda(get_slice,output_shape=(1,n_freq),arguments={'index':3})(inputs_batch)
    
    input_reshape = Lambda(lambda x: K.expand_dims(x,3), output_shape=(n_concat, n_freq, 1))(inputs_batch)
    cov1 = Conv2D(16, input_shape=[n_concat, n_freq],
           kernel_size=(n_concat, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(input_reshape)
    bn1 = BatchNormalization(axis=3)(cov1)
    relu1 = Activation('relu')(bn1)
    drop1 = Dropout(0.5)(relu1)
    
    cov2 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop1)
    bn2 = BatchNormalization(axis=3)(cov2)
    relu2 = Activation('relu')(bn2)
    drop2 = Dropout(0.5)(relu2)
    
    re0 = keras.layers.add([drop1, drop2])
    
    cov3 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(re0)
    bn3 = BatchNormalization(axis=3)(cov3)
    relu3 = Activation('relu')(bn3)
    drop3 = Dropout(0.5)(relu3)
    
    cov4 = Conv2D(64, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop3)
    bn4 = BatchNormalization(axis=3)(cov4)
    relu4 = Activation('relu')(bn4)
    drop4 = Dropout(0.5)(relu4)
    
    cov5 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop4)
    bn5 = BatchNormalization(axis=3)(cov5)
    relu5 = Activation('relu')(bn5)
    drop5 = Dropout(0.5)(relu5)

    re1 = keras.layers.add([drop3, drop5])
    
    cov6 = Conv2D(64, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(re1)
    bn6 = BatchNormalization(axis=3)(cov6)
    relu6 = Activation('relu')(bn6)
    drop6 = Dropout(0.5)(relu6)
    
    cov7 = Conv2D(128, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop6)
    bn7 = BatchNormalization(axis=3)(cov7)
    relu7 = Activation('relu')(bn7)
    drop7 = Dropout(0.5)(relu7)
    
    cov8 = Conv2D(64, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop7)
    bn8 = BatchNormalization(axis=3)(cov8)
    relu8 = Activation('relu')(bn8)
    drop8 = Dropout(0.5)(relu8)
    
    re2 = keras.layers.add([drop6, drop8])
    
    cov9 = Conv2D(1, input_shape=[1, n_freq],
           kernel_size=(1, 3),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           activation='relu')(re2)
    cov9_flatten = Flatten()(cov9)
    
    lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    #lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    cov9_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(cov9_flatten)
    residual_layer = keras.layers.add([cov9_reshape, lambda1_reshape])
    
    final_output = Flatten()(residual_layer)
    dense10 = Dense(n_freq, activation='linear')(final_output)
    model = Model(inputs=inputs_batch, outputs=dense10)
    return model

def m_vgg3(n_concat, n_freq):
    
    inputs_batch = Input(shape=(n_concat, n_freq))
    lambda1 = Lambda(get_slice,output_shape=(1,n_freq),arguments={'index':3})(inputs_batch)
    
    input_reshape = Lambda(lambda x: K.expand_dims(x,3), output_shape=(n_concat, n_freq, 1))(inputs_batch)
    cov1 = Conv2D(16, input_shape=[n_concat, n_freq],
           kernel_size=(n_concat, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(input_reshape)
    bn1 = BatchNormalization(axis=3)(cov1)
    relu1 = Activation('relu')(bn1)
    drop1 = Dropout(0.25)(relu1)
    
    cov2 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop1)
    bn2 = BatchNormalization(axis=3)(cov2)
    relu2 = Activation('relu')(bn2)
    drop2 = Dropout(0.25)(relu2)
    
    re0 = keras.layers.add([cov1, drop2])
    
    cov3 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(re0)
    bn3 = BatchNormalization(axis=3)(cov3)
    relu3 = Activation('relu')(bn3)
    drop3 = Dropout(0.25)(relu3)
    
    cov4 = Conv2D(64, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop3)
    bn4 = BatchNormalization(axis=3)(cov4)
    relu4 = Activation('relu')(bn4)
    drop4 = Dropout(0.25)(relu4)
    
    cov5 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop4)
    bn5 = BatchNormalization(axis=3)(cov5)
    relu5 = Activation('relu')(bn5)
    drop5 = Dropout(0.25)(relu5)

    re1 = keras.layers.add([cov3, drop5])
    
    cov6 = Conv2D(64, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(re1)
    bn6 = BatchNormalization(axis=3)(cov6)
    relu6 = Activation('relu')(bn6)
    drop6 = Dropout(0.25)(relu6)
    
    cov7 = Conv2D(128, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop6)
    bn7 = BatchNormalization(axis=3)(cov7)
    relu7 = Activation('relu')(bn7)
    drop7 = Dropout(0.25)(relu7)
    
    cov8 = Conv2D(64, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop7)
    bn8 = BatchNormalization(axis=3)(cov8)
    relu8 = Activation('relu')(bn8)
    drop8 = Dropout(0.25)(relu8)
    
    re2 = keras.layers.add([cov6, drop8])
    
    cov9 = Conv2D(1, input_shape=[1, n_freq],
           kernel_size=(1, 3),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           activation='relu')(re2)
    cov9_flatten = Flatten()(cov9)
    
    lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    #lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    cov9_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(cov9_flatten)
    residual_layer = keras.layers.add([cov9_reshape, lambda1_reshape])
    
    final_output = Flatten()(residual_layer)
    dense10 = Dense(n_freq, activation='linear')(final_output)
    model = Model(inputs=inputs_batch, outputs=dense10)
    return model

def m_vgg4(n_concat, n_freq):
    
    inputs_batch = Input(shape=(n_concat, n_freq))
    lambda1 = Lambda(get_slice,output_shape=(1,n_freq),arguments={'index':3})(inputs_batch)
    
    input_reshape = Lambda(lambda x: K.expand_dims(x,3), output_shape=(n_concat, n_freq, 1))(inputs_batch)
    cov1 = Conv2D(16, input_shape=[n_concat, n_freq],
           kernel_size=(n_concat, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(input_reshape)
    bn1 = BatchNormalization(axis=3)(cov1)
    relu1 = Activation('relu')(bn1)
    drop1 = Dropout(0.25)(relu1)
    
    cov2 = Conv2D(16, input_shape=[1, n_freq],
           kernel_size=(1, 13),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop1)
    bn2 = BatchNormalization(axis=3)(cov2)
    relu2 = Activation('relu')(bn2)
    drop2 = Dropout(0.25)(relu2)
    
    re0 = K.concatenate([drop1, drop2], axis=2)
    
    cov3 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 2),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(re0)
    bn3 = BatchNormalization(axis=3)(cov3)
    relu3 = Activation('relu')(bn3)
    drop3 = Dropout(0.25)(relu3)
    
    cov4 = Conv2D(64, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop3)
    bn4 = BatchNormalization(axis=3)(cov4)
    relu4 = Activation('relu')(bn4)
    drop4 = Dropout(0.25)(relu4)
    
    cov5 = Conv2D(32, input_shape=[1, n_freq],
           kernel_size=(1, 11),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop4)
    bn5 = BatchNormalization(axis=3)(cov5)
    relu5 = Activation('relu')(bn5)
    drop5 = Dropout(0.25)(relu5)

    #re1 = keras.layers.add([drop3, drop5])
    re1 = K.concatenate([drop3, drop5], axis=2)
    
    cov6 = Conv2D(64, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 2),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(re1)
    bn6 = BatchNormalization(axis=3)(cov6)
    relu6 = Activation('relu')(bn6)
    drop6 = Dropout(0.25)(relu6)
    
    cov7 = Conv2D(128, input_shape=[1, n_freq],
           kernel_size=(1, 7),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop6)
    bn7 = BatchNormalization(axis=3)(cov7)
    relu7 = Activation('relu')(bn7)
    drop7 = Dropout(0.25)(relu7)
    
    cov8 = Conv2D(64, input_shape=[1, n_freq],
           kernel_size=(1, 9),
           strides=(n_concat, 1),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           kernel_regularizer=regularizers.l2(l=0.0001))(drop7)
    bn8 = BatchNormalization(axis=3)(cov8)
    relu8 = Activation('relu')(bn8)
    drop8 = Dropout(0.25)(relu8)
    
    #re2 = keras.layers.add([drop6, drop8])
    re2 = K.concatenate([drop6, drop8], axis=2)
    
    cov9 = Conv2D(1, input_shape=[1, n_freq],
           kernel_size=(1, 3),
           strides=(n_concat, 2),
           padding='same',
           #data_format='channels_first',
           kernel_initializer='glorot_uniform',
           activation='relu')(re2)
    cov9_flatten = Flatten()(cov9)
    
    lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    #lambda1_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(lambda1)
    cov9_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_freq))(cov9_flatten)
    residual_layer = keras.layers.add([cov9_reshape, lambda1_reshape])
    
    final_output = Flatten()(residual_layer)
    dense10 = Dense(n_freq, activation='linear')(final_output)
    model = Model(inputs=inputs_batch, outputs=dense10)
    return model


def CapsNet(n_concat, n_freq, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    inputs_batch = Input(shape=(n_concat, n_freq))
    x_reshape = Lambda(lambda x: K.expand_dims(x,3), output_shape=(n_concat, n_freq, 1))(inputs_batch)
    # Layer 1: Just a conventional Conv2D layer
    conv1 = Conv2D(filters=256, kernel_size=(7,9), strides=1, padding='valid', activation='relu', name='conv1')(x_reshape)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=(1, 40), strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_freq, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    #out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    #y = layers.Input(shape=(n_class,))
    y = Lambda(get_slice,output_shape=(1,n_freq),arguments={'index':3})(inputs_batch)
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    #masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder_dense1 = Dense(512, kernel_regularizer=regularizers.l2(l=0.0001))(masked_by_y)
    bn1 = BatchNormalization()(decoder_dense1)
    activation1 = Activation('relu')(bn1)
    
    decoder_dense2 = Dense(1024, kernel_regularizer=regularizers.l2(l=0.0001))(activation1)
    bn2 = BatchNormalization()(decoder_dense2)
    activation2 = Activation('relu')(bn2)
    
    decoder_dense3 = Dense(n_freq, activation='linear')(activation2)
    
    #decoder = Sequential(name='decoder')
    #decoder.add(Dense(512, activation='relu', input_dim=16*n_freq))
    #decoder.add(Dense(1024, activation='relu'))
    #decoder.add(Dense(n_freq, activation='linear'))
    #decoder.add(Reshape(target_shape=n_freq, name='out_recon'))

    # Models for training and evaluation (prediction)
    #train_model = Model(inputs=inputs_batch, outputs=decoder(masked_by_y))
    #train_model = Model([inputs_batch, y], [out_caps, decoder_dense3])
    #eval_model = Model(x, [out_caps, masked])
    model = Model(inputs=inputs_batch, outputs=decoder_dense3)
    return model

def rnn(n_concat, input_dim, output_dim):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(n_concat, input_dim))
    flatten= Flatten()(input_data)
    flatten_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_concat*input_dim))(flatten)
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, activation='relu', 
                 implementation=2, name='rnn')(flatten_reshape)
    
    bn_rnn = BatchNormalization()(simp_rnn)
    
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    flatten2 = Flatten()(time_dense)
    final_dense = Dense(input_dim, activation='linear')(flatten2)
    
    model = Model(inputs=input_data , outputs=final_dense)
    return model

def brnn(n_concat, input_dim, units, recur_layers, output_dim):
    
     input_data = Input(name='the_input', shape=(n_concat, input_dim))
     flatten= Flatten()(input_data)
     flatten_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(1, n_concat*input_dim))(flatten)
     # TODO: Add recurrent layers, each with batch normalization
     brnn = Bidirectional(GRU(units, activation='relu',
                 return_sequences=True, implementation=2, recurrent_dropout=0.01, 
                 kernel_regularizer=regularizers.l2(l=0.0001), name='brnn'))(flatten_reshape)
     
     bn_rnn = BatchNormalization()(brnn)
     
     # Loop for additional layers
     for i in range(recur_layers - 1):
        name = 'brnn_' + str(i + 1)
        brnn = Bidirectional(GRU(units, activation='relu', 
        return_sequences=True, implementation=2, name=name,
        kernel_regularizer=regularizers.l2(l=0.0001)))(bn_rnn)
        bn_rnn = BatchNormalization()(brnn)
        
     # TimeDistributed Dense layer
     time_distributed_dense = TimeDistributed(Dense(1024, kernel_regularizer=regularizers.l2(l=0.0001)))(bn_rnn)
     time_dense = TimeDistributed(Dense(output_dim))(time_distributed_dense)
     flatten2 = Flatten()(time_dense)
     final_dense = Dense(input_dim, activation='linear')(flatten2)
    
     model = Model(inputs=input_data , outputs=final_dense)
    
     return model


def tcn(n_concat, input_dim):
    
    input_data = Input(name='the_input', shape=(n_concat, input_dim))
    #flatten= Flatten()(input_data)
    #flatten_reshape = Lambda(lambda x: K.expand_dims(x,1), output_shape=(n_concat*input_dim, 1))(flatten)
    
    tcn1 = TCN(return_sequences=True, name='TCN_1', nb_filters=13, kernel_size=2, 
            nb_stacks=2, dilations=[2 ** i for i in range(8)], activation='relu', 
            use_skip_connections=True, dropout_rate=0.1)(input_data)
    tcn2 = TCN(return_sequences=True, name='TCN_2', nb_filters=31, kernel_size=3, 
            nb_stacks=2, dilations=[2 ** i for i in range(8)], activation='relu', 
            use_skip_connections=True, dropout_rate=0.1)(tcn1)
    tcn3 = TCN(return_sequences=True, name='TCN_3', nb_filters=64, kernel_size=5, 
            nb_stacks=2, dilations=[2 ** i for i in range(8)], activation='relu', 
            use_skip_connections=True, dropout_rate=0.1)(tcn2) 
    tcn4 = TCN(return_sequences=True, name='TCN_4', nb_filters=31, kernel_size=3, 
            nb_stacks=2, dilations=[2 ** i for i in range(8)], activation='relu', 
            use_skip_connections=True, dropout_rate=0.1)(tcn3)
    tcn5 = TCN(return_sequences=True, name='TCN_5', nb_filters=13, kernel_size=2, 
            nb_stacks=2, dilations=[2 ** i for i in range(8)], activation='relu', 
            use_skip_connections=True, dropout_rate=0.1)(tcn4)
    
    flatten2 = Flatten()(tcn5)
    final_dense = Dense(input_dim, activation='linear')(flatten2)
    
    model = Model(inputs=input_data , outputs=final_dense)
    
    return model
    
    
   