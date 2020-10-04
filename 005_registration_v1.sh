#!/bin/bash
echo \#
#SBATCH 
if [ ! -f /Users/hsm/Stroke/stroke_data_sorted/FengAHA005/FLAIR_AX/_flair_ax.nii.gz ]; then
    echo "File not found!"

else 

	echo "File found!"
fi


./antsRegistration -v 1 -d 3 -r [‎⁨/Users/hsm/Stroke/stroke_data_sorted/FengAHA005/FLAIR_AX/7_flair_ax.nii.gz,/Users/hsm/Stroke/stroke_data_sorted/FengAHA005/b1000/100_b1000.nii.gz,1] -m MI[/Users/hsm/Stroke/stroke_data_sorted/FengAHA005/FLAIR_AX/7_flair_ax.nii.gz,/Users/hsm/Stroke/stroke_data_sorted/FengAHA005/b1000/100_b1000.nii.gz,,1,32,None] -t rigid[0.1] -c [3000x3000x3000x0,1e-8,20] -s 0x0x0x0vox -f 6x4x2x1 -u 1 -z 1 -o /Users/hsm/Stroke/stroke_data_sorted/FengAHA005/Registered/005_b1000_registered_1


