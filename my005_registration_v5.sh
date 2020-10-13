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



flair_2=/Users/hsm/Stroke/stroke_data_sorted/Registered/test1_patient_008_sc_KirbyMultiModal_2_BrainExtractionBrain.nii.gz 
dw_2=/Users/hsm/Stroke/stroke_data_sorted/FengAHA008/b1000/100_b1000.nii.gz
fileout_2_1=/Users/hsm/Stroke/stroke_data_sorted/Registered/patient_008_1_
fileout_2_2=/Users/hsm/Stroke/stroke_data_sorted/Registered/patient_008_1_registered_Warped.nii.gz
fileout_2_3=/Users/hsm/Stroke/stroke_data_sorted/Registered/patient_008_1_registered_InverseWarped.nii.gz

antscmd_2="./antsRegistration -v 1 -d 3 -r [ $flair_2,$dw_2,1 ] \
					-t Rigid[ 0.1 ] \
					-m MI[ $flair_2,$dw_2,1,32,None ] \
					-c [ 500x500x100x0,1e-8,20 ] \
					-s 0x0x0x0vox \
					-f 6x4x2x1 \
					-t Affine[ 0.2 ] \
					-m MI[ $flair_2,$dw_2,1,32,None ] \
					-c [ 50x50x10x0,1e-8,20 ] \
					-s 0x0x0x0vox \
					-f 6x4x2x1 \
-o [ $fileout_2_1,$fileout_2_2,$fileout_2_3 ]"

echo $antscmd_2
$antscmd_2



flair_3=/Users/hsm/Stroke/stroke_data_sorted/Registered/test1_patient_012_sc_KirbyMultiModal_2_BrainExtractionBrain.nii.gz 
dw_3=/Users/hsm/Stroke/stroke_data_sorted/FengAHA012/B1000/100_b1000.nii.gz
fileout_3_1=/Users/hsm/Stroke/stroke_data_sorted/Registered/patient_012_1_
fileout_3_2=/Users/hsm/Stroke/stroke_data_sorted/Registered/patient_012_1_registered_Warped.nii.gz
fileout_3_3=/Users/hsm/Stroke/stroke_data_sorted/Registered/patient_012_1_registered_InverseWarped.nii.gz

antscmd_3="./antsRegistration -v 1 -d 3 -r [ $flair_3,$dw_3,1 ] \
					-t Rigid[ 0.1 ] \
					-m MI[ $flair_3,$dw_3,1,32,None ] \
					-c [ 500x500x100x0,1e-8,20 ] \
					-s 0x0x0x0vox \
					-f 6x4x2x1 \
					-t Affine[ 0.2 ] \
					-m MI[ $flair_3,$dw_3,1,32,None ] \
					-c [ 50x50x10x0,1e-8,20 ] \
					-s 0x0x0x0vox \
					-f 6x4x2x1 \
-o [ $fileout_3_1,$fileout_3_2,$fileout_3_3 ]"

echo $antscmd_3
$antscmd_3




flair_4=/Users/hsm/Stroke/stroke_data_sorted/Registered/test1_patient_013_sc_KirbyMultiModal_2_BrainExtractionBrain.nii.gz 
dw_4=/Users/hsm/Stroke/stroke_data_sorted/FengAHA013/B1000/100_b1000.nii.gz
fileout_4_1=/Users/hsm/Stroke/stroke_data_sorted/Registered/patient_013_1_
fileout_4_2=/Users/hsm/Stroke/stroke_data_sorted/Registered/patient_013_1_registered_Warped.nii.gz
fileout_4_3=/Users/hsm/Stroke/stroke_data_sorted/Registered/patient_013_1_registered_InverseWarped.nii.gz

antscmd_4="./antsRegistration -v 1 -d 3 -r [ $flair_4,$dw_4,1 ] \
					-t Rigid[ 0.1 ] \
					-m MI[ $flair_4,$dw_4,1,32,None ] \
					-c [ 500x500x100x0,1e-8,20 ] \
					-s 0x0x0x0vox \
					-f 6x4x2x1 \
					-t Affine[ 0.2 ] \
					-m MI[ $flair_4,$dw_4,1,32,None ] \
					-c [ 50x50x10x0,1e-8,20 ] \
					-s 0x0x0x0vox \
					-f 6x4x2x1 \
-o [ $fileout_4_1,$fileout_4_2,$fileout_4_3 ]"

echo $antscmd_4
$antscmd_4








