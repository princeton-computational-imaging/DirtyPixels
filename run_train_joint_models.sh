#!/bin/bash

TRAIN_DIR=/path/to/train_dir
IMAGENET_TFRECORDS=/path/to/imagenetTFRecords
CHECKPOINTS=/path/to/checkpoints

# Train with 3lux noisy images
# Set number of clones and device according to machine resources
python train_image_classifier.py --train_dir=$TRAIN_DIR/3lux \
    --dataset_dir=$IMAGENET_TFRECORDS  --ll_low=0.0015 \
    --ll_high=0.0015  --batch_size=256 --model_name=mobilenet_isp --num_readers=8 \
    --num_preprocessing_threads=8 --isp_checkpoint_path=$CHECKPOINTS/multires128/6lux/model.ckpt-27000  \
    --checkpoint_path=$CHECKPOINTS/mobilenet_v1_128/mobilenet_v1_1.0_128.ckpt --noise_channel=True \
    --use_anscombe=True --num_clones=4 --isp_model_name=isp --num_iters=1 --device=0,1,2,3 \
    --learning_rate=0.00045 --num_epochs_per_decay=2 --train_image_size=128


# Train with 6lux noisy images 
python train_image_classifier.py --train_dir=$TRAIN_DIR/6lux \
    --dataset_dir=$IMAGENET_TFRECORDS  --ll_low=0.003 \
    --ll_high=0.003  --batch_size=256 --model_name=mobilenet_isp --num_readers=8 \
    --num_preprocessing_threads=8 --isp_checkpoint_path=$CHECKPOINTS/multires128/6lux/model.ckpt-27000  \
    --checkpoint_path=$CHECKPOINTS/mobilenet_v1_128/mobilenet_v1_1.0_128.ckpt --noise_channel=True \
    --use_anscombe=True --num_clones=4 --isp_model_name=isp --num_iters=1 --device=0,1,3,4 \
    --learning_rate=0.00045 --num_epochs_per_decay=2 --train_image_size=128


# Train with 2to20lux noisy images
python train_image_classifier.py --train_dir=$TRAIN_DIR/2to20lux \
    --dataset_dir=$IMAGENET_TFRECORDS  --ll_low=0.001 \
    --ll_high=0.010  --batch_size=256 --model_name=mobilenet_isp --num_readers=8 \
    --num_preprocessing_threads=8 --isp_checkpoint_path=$CHECKPOINTS/multires128/6lux/model.ckpt-27000  \
    --checkpoint_path=$CHECKPOINTS/mobilenet_v1_128/mobilenet_v1_1.0_128.ckpt --noise_channel=True \
    --use_anscombe=True --num_clones=4 --isp_model_name=isp --num_iters=1 --device=0,1,2,3 \
    --learning_rate=0.00045 --num_epochs_per_decay=2 --train_image_size=128

# Train with 2to200lux noisy images
python train_image_classifier.py --train_dir=$TRAIN_DIR/2to200lux \
    --dataset_dir=$IMAGENET_TFRECORDS  --ll_low=0.001 \
    --ll_high=0.100  --batch_size=256 --model_name=mobilenet_isp --num_readers=8 \
    --num_preprocessing_threads=8 --isp_checkpoint_path=$CHECKPOINTS/multires128/6lux/model.ckpt-27000  \
    --checkpoint_path=$CHECKPOINTS/mobilenet_v1_128/mobilenet_v1_1.0_128.ckpt --noise_channel=True \
    --use_anscombe=True --num_clones=4 --isp_model_name=isp --num_iters=1 --device=0,1,3,4 \
    --learning_rate=0.00045 --num_epochs_per_decay=2 --train_image_size=128