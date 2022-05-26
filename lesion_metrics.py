#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 20:18:26 2022

@author: haso
"""

import nibabel as nib
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import scipy.stats as st
import time as time
from scipy.ndimage.morphology import binary_dilation


def seg_metrics_calc(pred, mask):
    TP = sum(sum(sum(pred*mask)))
    TN = pred.shape[0]*pred.shape[1]*pred.shape[2]-sum(sum(sum(pred+mask-(pred*mask))))
    FP =sum(sum(sum(pred-mask*pred)))
    FN = sum(sum(sum(mask-mask*pred)))
    dice = (2*TP)/(FP+FN+(2*TP))
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specificity = TN/(TN+FP)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    return dice, precision, recall, specificity, accuracy

def lesion_load_calc(pred, CST_mask, voxel_size_cc):
    lesion_load = sum(sum(sum(pred*CST_mask)))*voxel_size_cc
    return lesion_load

def weighted_lesion_load_calc(pred, CST_mask, voxel_size_cc):
    cst_overlap = pred*CST_mask
    ind_weight = np.ones([CST_mask.shape[0], CST_mask.shape[1], CST_mask.shape[2]])*sum(sum(CST_mask))
    max_weight = max(sum(sum(CST_mask)))
    total_weight = max_weight/ind_weight
    total_weight[np.isinf(total_weight)]=0
    total_weight[np.isnan(total_weight)]=0
    weigthed_lesion_load = sum(sum(sum(cst_overlap*total_weight)))*voxel_size_cc
    return weigthed_lesion_load

def lesion_vol_calc(pred, voxel_size_cc):
    lesion_vol = sum(sum(sum(pred)))*voxel_size_cc
    return lesion_vol

def hausdorff_3D_calc(vol_a, vol_b, vox_size_x, vox_size_y, vox_size_z):
    vol_a=vol_a.astype(int)
    vol_b=vol_b.astype(int)
    k = np.zeros((3,3),dtype=int); k[1] = 1; k[:,1] = 1
    surface_a = np.empty([vol_b.shape[0],vol_b.shape[1],vol_b.shape[2]])
    surface_b = np.empty([vol_b.shape[0],vol_b.shape[1],vol_b.shape[2]])

    for i in range(0,vol_b.shape[2]):
        surface_slice_a = binary_dilation(vol_a[:,:,i]==0, k) & vol_a[:,:,i]
        surface_a[:,:,i] = np.copy(surface_slice_a)
        surface_slice_b = binary_dilation(vol_b[:,:,i]==0, k) & vol_b[:,:,i]
        surface_b[:,:,i] = np.copy(surface_slice_b)

    surf_a_coord = np.nonzero(surface_a)
    surf_b_coord = np.nonzero(surface_b)
    tot_dist_point_ = np.empty(len(surf_b_coord[0]))

    tot_dist_a = np.empty(len(surf_a_coord[0]))
    tot_dist_b = np.empty(len(surf_b_coord[0]))
   
    for ii in range(0,len(surf_a_coord[0])):
        new_tot_dist = 1000
        for jj in range(0,len(surf_b_coord[0])):
            x_dist = np.abs(surf_a_coord[0][ii]-surf_b_coord[0][jj])*vox_size_x
            y_dist = np.abs(surf_a_coord[1][ii]-surf_b_coord[1][jj])*vox_size_y
            z_dist = np.abs(surf_a_coord[2][ii]-surf_b_coord[2][jj])*vox_size_z
            tmp_dist = (x_dist**2+y_dist**2+z_dist**2)**0.5
            new_tot_dist = np.min([tmp_dist,new_tot_dist])
        tot_dist_a[ii] = new_tot_dist


    for kk in range(0,len(surf_b_coord[0])):
        new_tot_dist = 1000
        for ll in range(0,len(surf_a_coord[0])):
            x_dist = np.abs(surf_b_coord[0][kk]-surf_a_coord[0][ll])*vox_size_x
            y_dist = np.abs(surf_b_coord[1][kk]-surf_a_coord[1][ll])*vox_size_y
            z_dist = np.abs(surf_b_coord[2][kk]-surf_a_coord[2][ll])*vox_size_z
            tmp_dist = (x_dist**2+y_dist**2+z_dist**2)**0.5
            new_tot_dist = np.min([tmp_dist,new_tot_dist])
        tot_dist_b[kk] = new_tot_dist
    tot_dist_both = np.append(tot_dist_a, tot_dist_b)
    tot_mean = (np.mean(tot_dist_a) + np.mean(tot_dist_b))/2
    return np.max(tot_dist_both), np.percentile(tot_dist_both,95), tot_mean  

    
    
    
    

 