#!/bin/bash
echo \#
#SBATCH 
cd '/Users/hsm/ANTs/bin/bin'



flair_1=/Users/hsm/Stroke/stroke_data_sorted/Registered/test1_patient_006_sc_KirbyMultiModal_2_BrainExtractionBrain.nii.gz 
dw_1=/Users/hsm/Stroke/stroke_data_sorted/FengAHA006/B1000/100_b1000.nii.gz
fileout_1_1=/Users/hsm/Stroke/stroke_data_sorted/Registered/patient_006_1_
fileout_1_2=/Users/hsm/Stroke/stroke_data_sorted/Registered/patient_006_1_registered_Warped.nii.gz
fileout_1_3=/Users/hsm/Stroke/stroke_data_sorted/Registered/patient_006_1_registered_InverseWarped.nii.gz

antscmd_1="./antsRegistration -v 1 -d 3 -r [ $flair_1,$dw_1,1 ] \
					-t Rigid[ 0.1 ] \
					-m MI[ $flair_1,$dw_1,1,32,None ] \
					-c [ 500x500x100x0,1e-8,20 ] \
					-s 0x0x0x0vox \
					-f 6x4x2x1 \
					-t Affine[ 0.2 ] \
					-m MI[ $flair_1,$dw_1,1,32,None ] \
					-c [ 50x50x10x0,1e-8,20 ] \
					-s 0x0x0x0vox \
					-f 6x4x2x1 \
-o [ $fileout_1_1,$fileout_1_2,$fileout_1_3 ]"

echo $antscmd_1
$antscmd_1

