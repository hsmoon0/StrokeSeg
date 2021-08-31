#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: hsm
"""

import numpy as np
#import keras
import os
import sys
from fnmatch import fnmatch
import tensorflow as tf
import scipy.io as sio
import tensorflow.keras as keras
from keras import backend as K
from matplotlib import pyplot as plt
 
 
from sklearn.model_selection import KFold # tool for getting random folds in K-fold cross validation
from keras.models import Model
from keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, Dropout, BatchNormalization, concatenate, Add, ReLU, Softmax
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import Constant
from keras.models import load_model
import h5py
import nibabel as nib
from tensorflow.python.client import device_lib 

from sklearn.model_selection import train_test_split
from pytictoc import TicToc




# tf.config.experimental.list_physical_devices('GPU') 


#strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])


#os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"]="3"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])





print("Number of Devices : {}".format(strategy.num_replicas_in_sync))


# print(device_lib.list_local_devices())
# config = tf.ConfigProto( device_count = {'GPU': 1} )
# sess = tf.Session(config=config)
# keras.backend.set_session(sess)



def loss_fn(y_true, y_pred):
    # Assign more weights to ROIs than the background
    # tmp weights are stroke, periventricular lesions and background respectively
    tmp_weights_1 = y_true[:,:,:,:,0]
    tmp_weights_2 = y_true[:,:,:,:,1]

    weight_tot = tf.add(tmp_weights_1,tmp_weights_2)
    class_weights = tf.add(tf.cast(tf.multiply(tf.cast(tf.greater(tmp_weights_1, 0), tf.float32), 2), tf.float32), tf.cast(tf.equal(tmp_weights_1, 0), tf.float32))
 
    y_true = tf.stop_gradient(y_true)
    return tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred), class_weights))


def loss_fn_2(y_true, y_pred):
    return tf.compat.v1.losses.softmax_cross_entropy(
            onehot_labels=y_true, logits=y_pred, weights=1.0, label_smoothing=0, scope=None,
            reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
            )


def fuzzy_dice_coef(y_true, y_pred):
 
    y_pred = tf.nn.softmax(y_pred)
               
    y_pred_1 = y_pred[:,:,:,:,0]
               
    y_pred_back_1 = 1 - y_pred_1
   
    y_true_1 = y_true[:,:,:,:,0]
   
    y_true_back_1 = 1 - y_true_1
   
    
    tmp_TP_1 = tf.minimum(y_pred_1,y_true_1)
    TP_1 = tf.reduce_sum(tmp_TP_1,[1,2,3])
    tmp_FP_1 = tf.maximum(y_pred_1-y_true_1, 0)
    FP_1 = tf.reduce_sum(tmp_FP_1,[1,2,3])
    tmp_FN_1 = tf.maximum(y_pred_back_1-y_true_back_1, 0)
    FN_1 = tf.reduce_sum(tmp_FN_1,[1,2,3])
               
    nominator_1 = tf.multiply(TP_1,2)
    tmp_denominator_1 = tf.add(FP_1,FN_1)
    denominator_1 = tf.add(tmp_denominator_1, tf.multiply(TP_1,2))
    fuzzy_dice_1 = tf.reduce_mean(tf.divide(nominator_1,denominator_1))

 

    
    fuzzy_dice = fuzzy_dice_1
 
    return fuzzy_dice


def dice_metric(y_true, y_pred):


    threshold = 0.5

    mask_pred = y_pred > threshold
    mask_pred = tf.cast(mask_pred, dtype=tf.float32)
    y_pred = tf.multiply(y_pred, mask_pred)
    mask_true = y_true > threshold
    mask_true = tf.cast(mask_true, dtype=tf.float32)
    y_true = tf.multiply(y_true, mask_true)

    inse = tf.reduce_sum(tf.multiply(y_pred, y_true))
    l = tf.reduce_sum(y_pred)
    r = tf.reduce_sum(y_true)

    hard_dice = (2. * inse) / (l + r)

    hard_dice = tf.reduce_mean(hard_dice)

    if tf.math.is_nan(hard_dice) is True:
        hard_dice = 0

    return hard_dice


def dice_metric_softmax(y_true_raw, y_pred_raw):
    y_pred_softmax = tf.nn.softmax(y_pred_raw)
    y_pred = y_pred_softmax[:,:,:,:,0]

    y_true = y_true_raw[:,:,:,:,0]
    


    threshold = 0.3

    mask = y_pred > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_pred = tf.multiply(y_pred, mask)
    mask = y_true > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_true = tf.multiply(y_true, mask)

    inse = tf.reduce_sum(tf.multiply(y_pred, y_true))
    l = tf.reduce_sum(y_pred)
    r = tf.reduce_sum(y_true)

    hard_dice = (2. * inse) / (l + r)

    hard_dice = tf.reduce_mean(hard_dice)

    if tf.math.is_nan(hard_dice) is True:
        hard_dice = 0

    return hard_dice


def dice_coe(y_true,y_pred, loss_type='sorensen', smooth=0.01):

    threshold = 0.5
    mask = y_pred > threshold
    mask = tf.cast(mask, dtype=tf.float32)
    y_pred = tf.multiply(y_pred, mask)

    y_true_f = tf.reshape(y_true,[-1])
    y_pred_f = tf.reshape(y_pred,[-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)

    return (2. * intersection + smooth) / (union + smooth)


def dice_loss(y_true,y_pred, loss_type='jaccard', smooth=0.01):

    y_true_f = tf.cast(tf.reshape(y_true,[-1]),tf.float32)
    y_pred_f =tf.cast(tf.reshape(y_pred,[-1]),tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred_f)) + tf.reduce_sum(tf.square(y_true_f))

    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred_f) + tf.reduce_sum(y_true_f)

    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)

    return (1-(2. * intersection + smooth) / (union + smooth))


def intensity_normalizer(image):
    img_normalized = np.copy(image)
    for i in range(0, len(image)):
        for j in range(0,2):
            img_tmp = image[i,:,:,:,j]
            img_tmp_norm = (img_tmp-np.mean(img_tmp))/np.std(img_tmp)
            img_normalized[i,:,:,:,j] = img_tmp_norm
    return img_normalized



def data_organizer_random_3d(fold_num):
    
    image = nib.load('/home/alex/Stroke_multi/data/img_full.nii')
    data_img = image.get_fdata()

    mask = nib.load('/home/alex/Stroke_multi/data/mask_full.nii')
    data_mask = mask.get_fdata()

    data_norm = intensity_normalizer(data_img)

    #init_list = list(range(0, len(data_norm)))
    #train_num, test_num= train_test_split(init_list , test_size=0.3, random_state=42)  
    #test_num = list(range(0,86,6))
    #test_num = list(range(1,86,6))
    #test_num = list(range(2,86,6))
    #test_num = list(range(3,86,6))
    #test_num = list(range(4,86,6))
    #test_num = list(range(5,86,6))

    test_num = list(range(int(fold_num)-1,86,6))



    train_num_pre = list(range(0,86))

    for i in range(0, len(test_num)):
        train_num_pre[test_num[i]] = []

    train_num = res = [ele for ele in train_num_pre if ele != []]

    x_test_pre = data_img[test_num,:,:,:,:]
    #x_test_rot = np.rot90(x_test_pre,axes=(1,3))
    #x_test_rot = np.rot90(x_test_rot,axes=(2,3))
    #x_test_rot = np.rot90(x_test_rot,axes=(2,3))
    #x_test = np.reshape(x_test_rot, [len(x_test_pre)*32,512,512,2])

    y_test_pre = data_mask[test_num,:,:,:,:]
    #y_test_rot = np.rot90(y_test_pre,axes=(1,3))
    #y_test_rot = np.rot90(y_test_rot,axes=(2,3))
    #y_test_rot = np.rot90(y_test_rot,axes=(2,3))
    #y_test_pre_pre = np.reshape(y_test_rot, [len(y_test_pre)*32,512,512,2])
    #y_test = np.reshape(y_test, [len(y_test_pre_pre),512,512,1])
    y_test_bg = 1 - y_test_pre[:,:,:,:,0]
    y_test_pre[:,:,:,:,1] = y_test_bg
    y_test = y_test_pre[:,:,:,:,0]
    y_test = np.reshape(y_test, [len(y_test),512,512,32,1]) 


    x_train_pre = data_img[train_num,:,:,:,:]
    #x_train_rot = np.rot90(x_train_pre,axes=(1,3))
    #x_train_rot = np.rot90(x_train_rot,axes=(2,3))
    #x_train_rot = np.rot90(x_train_rot,axes=(2,3))
    #x_train = np.reshape(x_train_rot, [len(x_train_pre)*32,512,512,2])

    y_train_pre = data_mask[train_num,:,:,:,:]
    #y_train_rot = np.rot90(y_train_pre,axes=(1,3))
    #y_train_rot = np.rot90(y_train_rot,axes=(2,3))
    #y_train_rot = np.rot90(y_train_rot,axes=(2,3))
    #y_train_pre_pre = np.reshape(y_train_rot, [len(y_train_pre)*32,512,512,2])
    #y_train = y_train_pre_pre[:,:,:,0]
    #y_train = np.reshape(y_train, [len(y_train_pre_pre),512,512,1])
    y_train_bg = 1 - y_train_pre[:,:,:,:,0]
    y_train_pre[:,:,:,:,1] = y_train_bg
    y_train = y_train_pre[:,:,:,:,0]
    y_train = np.reshape(y_train, [len(y_train),512,512,32,1]) 


    return x_train_pre, y_train, x_test_pre, y_test



def add_conv_layer(num_filter, filter_size, stride_size, input_layer, bias_ct=0.03, leaky_alpha=0.01):
    layer = Conv3D(num_filter, (filter_size, filter_size, filter_size), # num. of filters and kernel size
                   strides=stride_size,
                   padding='same',
                   use_bias=True,
                   kernel_initializer='glorot_normal', # Xavier init
                   bias_initializer=Constant(value=bias_ct))(input_layer)
    layer = BatchNormalization(axis=-1)(layer) 
    #layer = ReLU()(layer) # activation func.
    layer = LeakyReLU(alpha=0.01)(layer)
 
    return layer
 
def add_conv_layer_ini(num_filter, filter_size, stride_size, input_layer, bias_ct=0.03, leaky_alpha=0.01):
    layer = Conv3D(num_filter, (filter_size, filter_size, filter_size), # num. of filters and kernel size
                   strides=stride_size,
                   padding='same',
                   use_bias=True,
                   kernel_initializer='glorot_normal', # Xavier init            
                   bias_initializer=Constant(value=bias_ct))(input_layer)
    layer = BatchNormalization(axis=-1)(layer) 
    #layer = ReLU()(layer) # activation func.
    layer = LeakyReLU(alpha=0.01)(layer)
 
    return layer
 
def add_conv_layer_lower(num_filter, filter_size, stride_size, input_layer, bias_ct=0.03, leaky_alpha=0.01):
    layer = Conv3D(num_filter, (filter_size, filter_size, filter_size), # num. of filters and kernel size
                   strides=stride_size,
                   padding='valid',
                   use_bias=True,
                   kernel_initializer='glorot_normal', # Xavier init
                   bias_initializer=Constant(value=bias_ct))(input_layer)
    layer = BatchNormalization(axis=-1)(layer) 
    #layer = ReLU()(layer) # activation func.
    layer = LeakyReLU(alpha=0.01)(layer)
 
    return layer

def add_conv_layer_output(num_filter, filter_size, stride_size, input_layer, bias_ct=0.03, leaky_alpha=0.01):
    layer = Conv3D(num_filter, (filter_size, filter_size, filter_size), # num. of filters and kernel size
                   strides=stride_size,
                   padding='valid',
                   use_bias=True,
                   kernel_initializer='glorot_normal', # Xavier init
                   bias_initializer=Constant(value=bias_ct))(input_layer)
    layer = BatchNormalization(axis=-1)(layer) 
    #layer = ReLU()(layer) # activation func.
    layer = LeakyReLU(alpha=0.01)(layer)
 
    return layer
 
def add_transposed_conv_layer(num_filter, filter_size, stride_size, input_layer, bias_ct=0.03, leaky_alpha=0.01):
    layer = Conv3DTranspose(num_filter, (filter_size, filter_size, filter_size), # num. of filters and kernel size
                   strides=stride_size,
                   padding='same',
                   use_bias=True,
                   kernel_initializer='glorot_normal', # Xavier init
                   bias_initializer=Constant(value=bias_ct))(input_layer)
    layer = BatchNormalization(axis=-1)(layer)
    #layer = ReLU()(layer) # activation func.
    layer = LeakyReLU(alpha=0.01)(layer)
    
    return layer
   
def add_transposed_conv_layer_higher(num_filter, filter_size, stride_size, input_layer, bias_ct=0.03, leaky_alpha=0.01):
    layer = Conv3DTranspose(num_filter, (filter_size, filter_size, filter_size), # num. of filters and kernel size
                   strides=stride_size,
                   padding='valid',
                   use_bias=True,
                   kernel_initializer='glorot_normal', # Xavier init
                   bias_initializer=Constant(value=bias_ct))(input_layer)
    layer = BatchNormalization(axis=-1)(layer)
    #layer = ReLU()(layer) # activation func.
    layer = LeakyReLU(alpha=0.01)(layer)
 
    return layer




def cnn(start_filter_num, filter_size, stride_size, dropout_ratio, fold_num):
    with strategy.scope():

        input_layer = keras.layers.Input(shape=(512, 512, 32, 2))
        conv1 = add_conv_layer_ini(start_filter_num, filter_size, (1, 1, 1), input_layer)
        down1 = add_conv_layer(start_filter_num, filter_size, (stride_size, stride_size, stride_size), conv1)
               
        conv2 = add_conv_layer(start_filter_num*2, filter_size, (1, 1, 1), down1)
        down2 = add_conv_layer(start_filter_num*2, filter_size, (stride_size, stride_size, stride_size), conv2)
               
        conv3 = add_conv_layer(start_filter_num*4, filter_size, (1, 1, 1), down2)
        down3 = add_conv_layer(start_filter_num*4, filter_size, (stride_size, stride_size, stride_size), conv3)
               
        conv4 = add_conv_layer(start_filter_num*8, filter_size, (1, 1, 1), down3)
               
        drop5 = Dropout(dropout_ratio)(conv4)
               

        up7 = add_transposed_conv_layer(start_filter_num*4, filter_size, (stride_size, stride_size, stride_size), drop5)
        up7 = Add()([up7, conv3])
        conv7 = add_conv_layer(start_filter_num*4, filter_size, (1, 1, 1), up7)
               
        up8 = add_transposed_conv_layer(start_filter_num*2, filter_size, (stride_size, stride_size, stride_size), conv7)
        up8 = Add()([up8, conv2])
        conv8 = add_conv_layer(start_filter_num*2, filter_size, (1, 1, 1), up8)
               
        up9 = add_transposed_conv_layer(start_filter_num, filter_size, (stride_size, stride_size, stride_size), conv8)
        up9 = Add()([up9, conv1])
   
        conv9 = add_conv_layer(start_filter_num, filter_size, (1, 1, 1), up9)
        # conv10 = add_conv_layer(2, filter_size, (1, 1, 1), conv9)

        output = Conv3D(1, 1, activation = 'sigmoid')(conv9)

        model = keras.models.Model(inputs=input_layer, outputs=output)

        #lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.0005,decay_steps=8*100,decay_rate=1,staircase=False)
        opt = keras.optimizers.Adam(learning_rate=8e-4)
        model.compile(optimizer=opt, loss=dice_loss, metrics=[dice_metric])
        #model.summary()
    
        x_train, y_train, x_val, y_val = data_organizer_random_3d(fold_num)
        model.summary()

        #cp_callback = tf.keras.callbacks.ModelCheckpoint(
        #    filepath='/home/alex/Stroke_multi/model', 
        #    save_freq=int(20*8))
        batch_size = 8

        train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))


        train_data = train_data.batch(batch_size)
        val_data = val_data.batch(batch_size, drop_remainder=True)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_data = train_data.with_options(options)
        val_data = val_data.with_options(options)


    #    class CustomCallback(keras.callbacks.Callback):
    #        def on_epoch_end(self, epoch, logs=None):
    #            a = model.evaluate(x=x_val, y=y_val, verbose=0, workers=4, return_dict=True)
    #            print(a)


        # history = model.fit(x_train,y_train, epochs=500, batch_size=8, callbacks=[CustomCallback()])
        # history = model.fit(x_train,y_train, epochs=100, validation_data=(x_val, y_val))
        history = model.fit(train_data, epochs=200, validation_data=val_data)

        train_loss = history.history['loss']
        np.save('/home/alex/Stroke_multi/results/train_loss_3d_fold_' + str(fold_num) + '_1.npy',train_loss)
        train_acc = history.history['dice_metric']
        np.save('/home/alex/Stroke_multi/results/train_acc_3d_fold_' + str(fold_num) + '_1.npy',train_acc)
        val_loss = history.history['val_loss']
        np.save('/home/alex/Stroke_multi/results/val_loss_3d_fold_' + str(fold_num) + '_1.npy',val_loss)
        val_acc = history.history['val_dice_metric']
        np.save('/home/alex/Stroke_multi/results/val_acc_3d_fold_' + str(fold_num) + '_1.npy',val_acc)

        model.save('/home/alex/Stroke_multi/results/model_3d_fold_' + str(fold_num) + '_1.h5')
    
        pred = model.predict(x_val)
        #pred = K.eval(K.argmax(K.softmax(pred),axis=4))
        #np.save('/home/alex/Stroke_multi/results/pred_test_3d_13.npy',pred)

        nif_img = nib.Nifti1Image(pred, affine=np.eye(4))
        nib.save(nif_img, '/home/alex/Stroke_multi/results/pred_3d_fold_' + str(fold_num) + '_1.nii')



if __name__=='__main__':
    cnn(32, 5, 2, 0, 1)
    cnn(32, 5, 2, 0, 2)
    cnn(32, 5, 2, 0, 3)
    cnn(32, 5, 2, 0, 4)
    cnn(32, 5, 2, 0, 5)
    cnn(32, 5, 2, 0, 6)
