#!/bin/bash

checkpoints_dir=/path/to/checkpoints
dataset_dir=/path/to/imagenet_validation
eval_dir=/path/to/output_dir

noise=3lux
python test_synthetic_images.py --device=1 \
    --checkpoint_path=$checkpoints_dir'/joint128/'$noise'/model.ckpt-216759'  \
    --dataset_dir=$dataset_dir  --dataset_name=imagenet --mode=$noise \
    --model_name=mobilenet_isp --eval_dir=$eval_dir/$noise

noise=6lux
python test_synthetic_images.py --device=1 \
    --checkpoint_path=$checkpoints_dir'/joint128/'$noise'/model.ckpt-222267'  \
    --dataset_dir=$dataset_dir  --dataset_name=imagenet --mode=$noise \
    --model_name=mobilenet_isp --eval_dir=$eval_dir/$noise

noise=2to20lux
python test_synthetic_images.py --device=1 \
    --checkpoint_path=$checkpoints_dir'/joint128/'$noise'/model.ckpt-232718'  \
    --dataset_dir=$dataset_dir  --dataset_name=imagenet --mode=$noise \
    --model_name=mobilenet_isp --eval_dir=$eval_dir/$noise

noise=2to200lux
python test_synthetic_images.py --device=1 \
--checkpoint_path=$checkpoints_dir'/joint128/'$noise'/model.ckpt-232721'  \
    --dataset_dir=$dataset_dir  --dataset_name=imagenet --mode=$noise \
    --model_name=mobilenet_isp --eval_dir=$eval_dir/$noise

