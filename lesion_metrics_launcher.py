#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:08:00 2022

@author: hsm
"""

import nibabel as nib
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import scipy.stats as st
from lesion_metrics import seg_metrics_calc lesion_load_calc weighted_lesion_load_calc lesion_vol_calc hausdorff_3D_calc


# Decision threshold value
threshold_volume = 0.4
voxel_size_cc = 0.4297*0.4297*6*0.001

vox_size_x = 0.43 # mm
vox_size_y = 0.43 # mm
vox_size_z = 6 # mm




def data_fetch(threshold_volume):

    pred_all = np.empty([79,512,512,32,1])
    mask_all = np.empty([79,512,512,32,1])

    pred_path = '/Volumes/alex/Desktop/hsm/Stroke/results/pred_2d_mm_full_data.nii'
    mask_path = '/Volumes/alex/Desktop/hsm/Stroke/results/mask_test_2d_mm_full_data.nii'
    pred_nib = nib.load(pred_path)
    mask_nib = nib.load(mask_path)
    pred_data = pred_nib.get_fdata()
    mask_data = mask_nib.get_fdata()

    pred_data[pred_data >= threshold_volume] = 1
    pred_data[pred_data < threshold_volume] = 0

    CST_path = '/Volumes/Data/Badea/Lab/hsm/Stroke/stroke_data_sorted_full_cc/stroke_data_sorted_9/FengAHA017/IIT_atlas_CST_registered.nii.gz'
    CST_nib = nib.load(CST_path)
    CST_data = CST_nib.get_fdata()

    for i in range(0, pred_all.shape[0]):
        mask_tmp_1 = mask_data[i*pred_all.shape[3]:i*pred_all.shape[3]+pred_all.shape[3],:,:,0]
        pred_tmp_1 = pred_data[i*pred_all.shape[3]:i*pred_all.shape[3]+pred_all.shape[3],:,:,0]

        img_rot_1 = np.rot90(pred_tmp_1,axes=(2,1))
        img_rot_2 = np.rot90(img_rot_1,axes=(2,1))
        img_rot_3 = np.rot90(img_rot_2,axes=(2,0))
        pred_all[i,:,:,:,0] = np.copy(img_rot_3)
        
        mask_rot_1 = np.rot90(mask_tmp_1,axes=(2,1))
        mask_rot_2 = np.rot90(mask_rot_1,axes=(2,1))
        mask_rot_3 = np.rot90(mask_rot_2,axes=(2,0))
        mask_all[i,:,:,:,0] = np.copy(mask_rot_3)

    return pred_all mask_all CST_data


def metrics_laucher(pred_all, mask_all, CST_mask, voxel_size_cc, vox_size_x, vox_size_y, vox_size_z)
    dice_pred = np.zeros(pred_all.shape[0])
    precision_pred = np.zeros(pred_all.shape[0])
    recall_pred = np.zeros(pred_all.shape[0])
    specificity_pred = np.zeros(pred_all.shape[0])
    accuracy_pred = np.zeros(pred_all.shape[0])
    lesion_vol_pred = np.zeros(pred_all.shape[0])
    lesion_load_pred = np.zeros(pred_all.shape[0])
    weighted_lesion_load_pred = np.zeros(pred_all.shape[0])
    hd_max = np.zeros(pred_all.shape[0])
    hd_95 = np.zeros(pred_all.shape[0])
    hd_avg = np.zeros(pred_all.shape[0])

    for i in range(0, pred_all.shape[0]):
        dice_pred[i], precision_pred[i], recall_pred[i], specificity_pred[i], accuracy_pred[i] = seg_metrics_calc(pred_all[i,:,:,:,0], mask_all[i,:,:,:,0])
        lesion_vol_pred[i] = lesion_vol_calc(pred_all[i,:,:,:,0], voxel_size_cc)
        lesion_load_pred[i] = lesion_load_calc(pred_all[i,:,:,:,0], CST_mask, voxel_size_cc)
        weighted_lesion_load_pred[i] = weighted_lesion_load_calc(pred_all[i,:,:,:,0], CST_mask, voxel_size_cc)
        hd_max[i], hd_95[i], hd_avg[i] = hausdorff_dist_3D(pred_all[i,:,:,:,0], mask_all[i,:,:,:,0], vox_size_x, vox_size_y, vox_size_z)
    
    return dice_pred, precision_pred, recall_pred, specificity_pred, accuracy_pred, lesion_vol_pred, lesion_load_pred \
    weighted_lesion_load_pred, hd_max, hd_95, hd_avg









if __name__ == "__main__":
    pred_all, mask_all, CST_mask = data_fetch(threshold_volume)
    dice_pred, precision_pred, recall_pred, specificity_pred, accuracy_pred, lesion vol_pred, lesion load_pred \
    weighted_lesion_load_pred = metrics_laucher(pred_all, mask_all, CST_mask, voxel_size_cc, vox_size_x, vox_size_y, vox_size_z) 
















