#!/bin/bash
source_image_real_root = "../data/Rnet/src_fourier_real/*.mhd"
target_image_real_root = "../data/Rnet/tar_fourier_real/*.mhd"
velocity_x_real_root = "../data/Rnet/velo_fourier_real/*.mhd"
velocity_y_real_root = "../data/Rnet/velo_fourier_real_y/*.mhd"
velocity_z_real_root = "../data/Rnet/velo_fourier_real_z/*.mhd"

source_image_imag_root = "../data/Inet/src_fourier_real/*.mhd"
target_image_imag_root = "../data/Inet/src_fourier_real/*.mhd"
velocity_x_imag_root = "../data/Inet/src_fourier_real/*.mhd"
velocity_y_imag_root = "../data/Inet/src_fourier_real/*.mhd"
velocity_z_imag_root = "../data/Inet/src_fourier_real/*.mhd"


python3 DeepFLASH_test --im_src_realpart  ${source_image_real_root} \
                       --im_tar_realpart ${target_image_real_root} \ 
                       --im_vel_realX ${velocity_x_real_root} \
                       --im_vel_realY ${velocity_y_real_root}\
                       --im_vel_realZ ${velocity_z_real_root}\

                       --im_src_imaginarypart source_image_imag_root \
                       --im_tar_imaginarypart target_image_imag_root \
                       --im_vel_imagX velocity_x_imag_root \
                       --im_vel_imagY velocity_y_imag_root \
                       --im_vel_imagZ velocity_z_imag_root 
