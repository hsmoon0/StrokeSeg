#!/bin/bash
echo \#
#SBATCH 
if [ ! -f /Users/hsm/Stroke/stroke_data_sorted/FengAHA005/FLAIR_AX/7_flair_ax.nii.gz ]; then
    echo "File not found!"

else 

	echo "File found!"
fi


./antsRegistrationSyN.sh -d 3 -f /Users/hsm/Stroke/stroke_data_sorted/FengAHA005/FLAIR_AX/7_flair_ax.nii.gz -m /Users/hsm/Stroke/stroke_data_sorted/FengAHA005/FLAIR_AX/7_flair_ax.nii.gz -t a  -s 0x0x0x0vox  -z 1 -o /Users/hsm/Stroke/stroke_data_sorted/FengAHA005/Registered/005_b1000_registered_1


