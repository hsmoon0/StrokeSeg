#!/bin/bash
echo \#
#SBATCH
export ANTSPATH=${ANTSPATH:="$HOME/ants/bin/bin/"}  
cd '/Users/hsm/ANTs/bin/bin'

template=/Users/hsm/Stroke/KirbyMultiModal/S_template2.nii.gz
prior=/Users/hsm/Stroke/KirbyMultiModal/S_template_BrainCerebellumProbabilityMask.nii.gz


image_1=/Users/hsm/Stroke/stroke_data_sorted/FengAHA012/FLAIR_AX/6_flair_ax.nii.gz
fileout_1=/Users/hsm/Stroke/stroke_data_sorted/Registered/test1_patient_012_sc_KirbyMultiModal_2_
antscmd_1="./antsBrainExtraction.sh -d 3 -a $image_1 -e $template -m $prior -o $fileout_1"
echo $antscmd_1
$antscmd_1


image_2=/Users/hsm/Stroke/stroke_data_sorted/FengAHA013/FLAIR_AX/6_flair_ax.nii.gz
fileout_2=/Users/hsm/Stroke/stroke_data_sorted/Registered/test1_patient_013_sc_KirbyMultiModal_2_
antscmd_2="./antsBrainExtraction.sh -d 3 -a $image_2 -e $template -m $prior -o $fileout_2"
echo $antscmd_2
$antscmd_2

