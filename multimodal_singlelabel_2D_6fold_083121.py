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
    mean_dice = tf.reduce_mean(hard_dice)
    # tf.debugging.check_numerics(mean_dice, 'NaN found', name=None)
    value_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(mean_dice)), dtype=tf.float32)
    mean_dice_no_nan = tf.math.multiply_no_nan(mean_dice, value_not_nan)
    return mean_dice_no_nan










def dice_metric_3d(y_true, y_pred):


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


def train_test_index(kfold, nth_fold, data_img):
    data_length = len(data_img)
    test_num = list(range(int(nth_fold)-1,data_length,kfold))
    train_num_pre = list(range(0,data_length))

    for i in range(0, len(test_num)):
        train_num_pre[test_num[i]] = []
    train_num = [ele for ele in train_num_pre if ele != []]
    return test_num, train_num


def data_organizer_random_2d(kfold, nth_fold):
    
    image = nib.load('/home/alex/Stroke_multi/data/img_full.nii')
    data_img = image.get_fdata()
    mask = nib.load('/home/alex/Stroke_multi/data/mask_full.nii')
    data_mask = mask.get_fdata()

    data_norm = intensity_normalizer(data_img)

    test_num, train_num = train_test_index(kfold, nth_fold, data_mask)
    

    x_test_pre = data_norm[test_num,:,:,:,:]
    x_test_rot = np.rot90(x_test_pre,axes=(1,3))
    x_test_rot = np.rot90(x_test_rot,axes=(2,3))
    x_test_rot = np.rot90(x_test_rot,axes=(2,3))
    x_test = np.reshape(x_test_rot, [len(x_test_pre)*32,512,512,2])

    y_test_pre = data_mask[test_num,:,:,:,:]
    y_test_rot = np.rot90(y_test_pre,axes=(1,3))
    y_test_rot = np.rot90(y_test_rot,axes=(2,3))
    y_test_rot = np.rot90(y_test_rot,axes=(2,3))
    y_test_pre_pre = np.reshape(y_test_rot, [len(y_test_pre)*32,512,512,2])
    y_test = y_test_pre_pre[:,:,:,0]
    y_test = np.reshape(y_test, [len(y_test_pre_pre),512,512,1])

    x_train_pre = data_norm[train_num,:,:,:,:]
    x_train_rot = np.rot90(x_train_pre,axes=(1,3))
    x_train_rot = np.rot90(x_train_rot,axes=(2,3))
    x_train_rot = np.rot90(x_train_rot,axes=(2,3))
    x_train = np.reshape(x_train_rot, [len(x_train_pre)*32,512,512,2])

    y_train_pre = data_mask[train_num,:,:,:,:]
    y_train_rot = np.rot90(y_train_pre,axes=(1,3))
    y_train_rot = np.rot90(y_train_rot,axes=(2,3))
    y_train_rot = np.rot90(y_train_rot,axes=(2,3))
    y_train_pre_pre = np.reshape(y_train_rot, [len(y_train_pre)*32,512,512,2])
    y_train = y_train_pre_pre[:,:,:,0]
    y_train = np.reshape(y_train, [len(y_train_pre_pre),512,512,1])

    return x_train, y_train, x_test, y_test




def cnn(fliter_num, kernel_size, kfold, nth_fold):
    with strategy.scope():
        input_layer = keras.layers.Input(shape=(512, 512, 2))
        conv1a = keras.layers.Conv2D(filters=fliter_num, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(input_layer)
        conv1b = keras.layers.Conv2D(filters=fliter_num, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(conv1a)
        pool1 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv1b)
        conv2a = keras.layers.Conv2D(filters=fliter_num*2, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(pool1)
        conv2b = keras.layers.Conv2D(filters=fliter_num*2, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(conv2a)
        pool2 = keras.layers.MaxPool2D(pool_size=(2, 2))(conv2b)
        conv3a = keras.layers.Conv2D(filters=fliter_num*3, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(pool2)
        conv3b = keras.layers.Conv2D(filters=fliter_num*3, kernel_size=(kernel_size, kernel_size), activation='relu', padding='same')(conv3a)

        dconv3a = keras.layers.Conv2DTranspose(filters=fliter_num*3, kernel_size=(kernel_size, kernel_size), padding='same')(conv3b)
        dconv3b = keras.layers.Conv2DTranspose(filters=fliter_num*3, kernel_size=(kernel_size, kernel_size), padding='same')(dconv3a)
        unpool2 = keras.layers.UpSampling2D(size=(2, 2))(dconv3b)
        cat2 = keras.layers.concatenate([conv2b, unpool2])
        dconv2a = keras.layers.Conv2DTranspose(filters=fliter_num*2, kernel_size=(kernel_size, kernel_size), padding='same')(cat2)
        dconv2b = keras.layers.Conv2DTranspose(filters=fliter_num*2, kernel_size=(kernel_size, kernel_size), padding='same')(dconv2a)
        unpool1 = keras.layers.UpSampling2D(size=(2, 2))(dconv2b)
        cat1 = keras.layers.concatenate([conv1b, unpool1])
        dconv1a = keras.layers.Conv2DTranspose(filters=fliter_num, kernel_size=(kernel_size, kernel_size), padding='same')(cat1)
        dconv1b = keras.layers.Conv2DTranspose(filters=fliter_num, kernel_size=(kernel_size, kernel_size), padding='same')(dconv1a)

        output = keras.layers.Conv2D(filters=1, kernel_size=(kernel_size, kernel_size), activation='sigmoid', padding='same')(dconv1b)

        model = keras.models.Model(inputs=input_layer, outputs=output)

        #lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.0001,decay_steps=8*100,decay_rate=1,staircase=False)

        opt = keras.optimizers.Adam(learning_rate=1e-5)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[dice_metric])
        #model.summary()

        x_train, y_train, x_val, y_val = data_organizer_random_2d(kfold, nth_fold)

        history = model.fit(x_train,y_train, epochs=200, batch_size = 100, validation_data=(x_val, y_val))


        train_loss = history.history['loss']
        np.save('/home/alex/Stroke_multi/results/train_loss_2d_fold_' + str(nth_fold) + '_2.npy',train_loss)
        train_acc = history.history['dice_metric']
        np.save('/home/alex/Stroke_multi/results/train_acc_2d_fold_' + str(nth_fold) + '_2.npy',train_acc)
        val_loss = history.history['val_loss']
        np.save('/home/alex/Stroke_multi/results/val_loss_2d_fold_' + str(nth_fold) + '_2.npy',val_loss)
        val_acc = history.history['val_dice_metric']
        np.save('/home/alex/Stroke_multi/results/val_acc_2d_fold_' + str(nth_fold) + '_2.npy',val_acc)

        model.save('/home/alex/Stroke_multi/results/model_2d_fold_' + str(nth_fold) + '_2.h5')

        test_pred = model.predict(x_val)


        nif_pred = nib.Nifti1Image(test_pred, affine=np.eye(4))
        nif_test_b1000 = nib.Nifti1Image(x_val[:,:,:,0], affine=np.eye(4))
        nif_test_flair = nib.Nifti1Image(x_val[:,:,:,1], affine=np.eye(4))
        nif_test = nib.Nifti1Image(y_val, affine=np.eye(4))


        nib.save(nif_pred, '/home/alex/Stroke_multi/results/pred_2d_fold_' + str(nth_fold) + '_2.nii')
        nib.save(nif_test_b1000, '/home/alex/Stroke_multi/results/b1000_test_2d_fold_' + str(nth_fold) + '_2.nii')
        nib.save(nif_test_flair, '/home/alex/Stroke_multi/results/flair_test_2d_fold_' + str(nth_fold) + '_2.nii')
        nib.save(nif_test, '/home/alex/Stroke_multi/results/mask_test_2d_fold_' + str(nth_fold) + '_2.nii')


if __name__=='__main__':
    cnn(64,7,6,1)
    cnn(64,7,6,2)
    cnn(64,7,6,3)
    cnn(64,7,6,4)
    cnn(64,7,6,5)
    cnn(64,7,6,6)

