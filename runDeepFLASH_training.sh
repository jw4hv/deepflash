#!/bin/bash
#Path example: "./data/Rnet/tar_fourier_real/*.mhd"

python3 ./src/DeepFLASH_training.py --im_src_realpart  "Directory path of real frequencies for source images" \
--im_tar_realpart "Directory path of real frequencies for target images" \
--im_vel_realX "Directory path of real frequencies for velocity fields (X direction)" \
--im_vel_realY "Directory path of real frequencies for velocity fields (Y direction)" \
--im_vel_realZ "Directory path of real frequencies for velocity fields (Z direction)" \
--im_src_imaginarypart "Directory path of imag frequencies for source images" \
--im_tar_imaginarypart "Directory path of imag frequencies for target images" \
--im_vel_imagX "Directory path of imag frequencies for velocity fields (X direction)" \
--im_vel_imagY "Directory path of imag frequencies for velocity fields (Y direction)" \
--im_vel_imagZ "Directory path of imag frequencies for velocity fields (Z direction)" 