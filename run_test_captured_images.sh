#!/bin/bash

# Set the checkpoint and dataset paths
checkpoints=/path/to/checkpoints
dataset_dir=/path/to/dataset/RAW_synset_ISO8000_EXP10000/

# Change --eval_dir paramater if needed
# Proposed Joint Architecture
python test_captured_images.py --device=1 --dataset_dir=$dataset_dir --dataset_name=imagenet \
    --checkpoint_path=$checkpoints/joint128/2to200lux/model.ckpt-232721 \
    --model_name=mobilenet_isp --noise_channel=True --use_anscombe=True \
    --isp_model_name=isp --eval_image_size=224 --sensor=Pixel --eval_dir joint_real_2to200lux

# Proposed Joint Architecture (no Anscombe layers)
python test_captured_images.py --device=1 --dataset_dir=$dataset_dir --dataset_name=imagenet \
    --checkpoint_path=$checkpoints/joint128/2to200lux_no_ansc/model.ckpt-215307 \
    --model_name=mobilenet_isp --noise_channel=False --use_anscombe=False \
    --isp_model_name=isp --eval_image_size=224 --sensor=Pixel --eval_dir joint_no_anscombe_real_2to200lux

# # From Scratch MobileNet-v1
python test_captured_images.py --device=1 --dataset_dir=$dataset_dir --dataset_name=imagenet \
    --checkpoint_path=$checkpoints/mobilenet_v1_128/2to200lux/model.ckpt-325357 \
    --model_name=mobilenet_v1 --eval_image_size=224 --preprocessing_name=mobilenet_isp \
    --eval_dir mobilenet_v1_real_2to200lux