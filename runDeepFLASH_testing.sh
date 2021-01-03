#!/bin/bash
#Network type: "deepflash" or "deepflash3D";
#Path example: "./data/Rnet/tar_fourier_real/*.mhd".

python3 ./src/DeepFLASH_testing.py --network_type "network type" \
--saved_model "saved model path" \
--im_src_realpart  "Directory path of real frequencies for source images" \
--im_tar_realpart "Directory path of real frequencies for target images" \
--im_src_imaginarypart "Directory path of imag frequencies for source images" \
--im_tar_imaginarypart "Directory path of imag frequencies for target images" \
