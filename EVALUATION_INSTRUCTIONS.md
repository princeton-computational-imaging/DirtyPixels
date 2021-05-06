# Evaluating pre-trained models
In order to reproduce the results presented in the 
paper, first, download the [pre-trained models](https://drive.google.com/file/d/1kBTRAS2W5Ayf2DOxKIgIBmPv5OHaMbCD/view?usp=sharing).

## Evaluate our joint models on real data
Download and extract the real captured (low-light) images [dataset](https://drive.google.com/file/d/1fj2u8t_wVdNVUmcjyeK8VuqDfTAd7RJA/view?usp=sharing).

To run our `2to200lux` joint model over the captured data (Table 2 of the paper), 
run
```
python test_captured_images.py --device=1 --dataset_dir=$DATASET_DIR --dataset_name=imagenet \
    --checkpoint_path=$CHECKPOINTS/joint128/2to200lux/model.ckpt-232721 \
    --model_name=mobilenet_isp --noise_channel=True --use_anscombe=True \
    --isp_model_name=isp --eval_image_size=224 --sensor=Pixel --eval_dir $OUT_DIR
```
where `--device` is the GPU where the model will run on, 
and `$DATASET_DIR` and `$CHECKPOINTS` should be set to the downloaded dataset 
and checkpoint directories, respectively. `$OUT_DIR` can be set to an
arbitrary output directory path. See `run_test_captured_images.sh` for 
additional parameters to evaluate baseline models.

## Evaluate our joint models on synthetic data
Download the [Imagenet][in] (validation) dataset.
To evaluate our joint model over noisy images with a `6lux` noise profile, run
```
python test_synthetic_images.py --device=1 --checkpoint_path=$CHECKPOINTS/joint128/6lux/model.ckpt-222267  \
    --dataset_dir=$IMAGENET_DATASET_DIR  --dataset_name=imagenet --mode=6lux \
    --model_name=mobilenet_isp --eval_dir=$OUT_DIR
```

where  `$IMAGENET_DATASET_DIR` is the path to the ImageNet (validation) dataset,
`$CHECKPOINTS` is set to the downloaded checkpoints directory,
`--device` is the GPU where the model will run on, 
and `$OUT_DIR` is an arbitrary directory where the results are written to. 

For both the synthetic and captured image evaluation scripts, generated results include
the noisy input raw images, anscombe network output images, 
and lists of correctly and wrongly classified images. 
To run the trained models over different noise profiles, 
modify the checkpoint paths and `--mode` parameter (3lux, 6lux, 2to20lux, or 2to200lux)
accordingly. See `run_test_synthetic_images.sh` for the specific parameters for each noise profile.


<!-- ## Evaluate baseline models -->
<!-- ## Download Datasets -->
<!-- To evaluate the models trained from scratch and models that use 
U-net, run `eval_and_log_filenames_with_isp.py` with the appropiate 
checkpoints and and noisy profiles, as done with our models. -->

<!-- To evaluate models trained from scratch, finetuned in 
Movidius and Darktable ISP output, and U-Net learnable ISP, 
first download the simulated RAW noisy, and ISP processed datasets by running
```
tools/download_checkpoints.sh
```
The script downloads extracts the data into the `noisy_imagenet` folder.

To evaluate models that use Movidius and Darktable ISP, use 
```
noise=6lux
checkpoints_dir=/mnt/storage03-mtl/users/frank/dirty_pixels/checkpoints
dataset_dir=/media/frank.julca-aguilar/daimler_data/dirty_pixels/dataset
python eval_and_log_filenames_with_isp.py --device=1 --checkpoint_path=$checkpoints_dir'/joint128/'$noise'/model.ckpt-222267'  \
    --dataset_dir=$dataset_dir'/validation_clean'  --dataset_name=imagenet --mode=$noise \
    --model_name=mobilenet_isp --target_name=/media/frank.julca-aguilar/daimler_data/dirty_pixels/evaluations_tog_final/ours 
``` -->

[in]: http://image-net.org/index