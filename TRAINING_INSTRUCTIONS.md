
## Training new models over noisy RAW data
Download the [Imagenet][in] (training) dataset.
As described in the supplemental document,
our joint models were trained in two stages. 
In the first stage, we train the anscombe 
and MobileNet components separately on ImageNet. 
In this stage, we use L1 norm to train the 
anscombe networks. In the second stage, 
the joint (MobileNet + Anscombe) model is trained using only 
the high level (classification) 
loss and the checkpoints obtained in the first stage. 
To facilitate training new models, we provide the  
checkpoints obtained from the first stage. The checkpoints 
can be downloaded following the instruction in 
[EVALUATION_INSTRUCTIONS.md](EVALUATION_INSTRUCTIONS.md).

## Generating TFRecords for training
In order to generate TFRecord files for training, 
run the `build_imagenet_data.py` script in the `datasets` 
folder:

```
cd datasets
python build_imagenet_data.py --train_directory=$IMAGENET_TRAIN_DIR \
    --output_directory=$OUT_DIR \
    --num_threads 8
```
where `$IMAGENET_TRAIN_DIR` is the path to the Imagenet training dataset,
`$OUT_DIR` is the path to the directory where the TFRecord files will 
be exported, and `--num_threads` defines the number of threads to 
preprocess the images.



##  Training command example
In order to train our proposed joint architecture
on a `6lux` noise profile run:

```
python train_image_classifier.py --train_dir=$TRAIN_DIR \
    --dataset_dir=$IMAGENET_TFRECORDS  --ll_low=0.003 \
    --ll_high=0.003  --batch_size=256 --model_name=mobilenet_isp --num_readers=8 \
    --num_preprocessing_threads=8 --isp_checkpoint_path=$CHECKPOINTS/multires128/6lux/model.ckpt-27000  \
    --checkpoint_path=$CHECKPOINTS/mobilenet_v1_128/mobilenet_v1_1.0_128.ckpt --noise_channel=True \
    --use_anscombe=True --num_clones=2 --isp_model_name=isp --num_iters=1 --device=0,1 \
    --learning_rate=0.00045 --num_epochs_per_decay=2 --train_image_size=128
```
where `$IMAGENET_TFRECORDS` is set to the directory with the Imagenet TFRecords, and `$CHECKPOINTS` is set to the downloaded checkpoints directory. The paramaters `--checkpoint_path` 
and `--isp_checkpoint_path` are set to the checkpoints obtained in the first training stage.
For training over other noisy profiles, see 
`run_train_joint_models.sh`. For more details about the specific training parameters, 
see the main manuscript and supplemental document. To visualise the training 
progress run `tensorboard --logdir=$TRAIN_DIR`.

[in]: http://image-net.org/index

