#!/bin/bash
#Path example: "./data/Rnet/tar_fourier_real/*.mhd"

python3 ./src/DeepFLASH_training.py --im_src_realpart  "Path of directory for reaf frequecies of source images" \
--im_tar_realpart "Path of directory for reaf frequecies of target images" \
--im_vel_realX "Path of directory for reaf frequecies of velocity fields (X direction)" \
--im_vel_realY "Path of directory for reaf frequecies of velocity fields (Y direction)" \
--im_vel_realZ "Path of directory for reaf frequecies of velocity fields (Z direction)" \
--im_src_imaginarypart "Path of directory for reaf frequecies of source images" \
--im_tar_imaginarypart "Path of directory for reaf frequecies of target images" \
--im_vel_imagX "Path of directory for reaf frequecies of velocity fields (X direction)" \
--im_vel_imagY "Path of directory for reaf frequecies of velocity fields (Y direction)" \
--im_vel_imagZ "Path of directory for reaf frequecies of velocity fields (Z direction)" 