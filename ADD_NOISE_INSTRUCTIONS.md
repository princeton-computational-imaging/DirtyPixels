# Simulating noisy raw images from Imagenet
In order to evaluate and train new ISP or perception 
models on noisy images, we provide the noisy images 
that we used for evaluating a the Hardware ISP of Movidius Myraid 2
evaluation board at https://drive.google.com/drive/folders/1f9B319TDtFpZSi7HEXnrPa31rtPm54iH?usp=sharing.

We also provide the code to simulate noisy raw images from 
the ImageNet dataset, using the image formation model 
described in the manuscript.

In order to introduce `2to20lux` noise to a the ImageNet dataset run 

```
python simulate_raw_images.py --ll_low=0.001 --ll_high=0.010 \
    --input_dir=$IMAGENET_DIR --output_dir=$OUT_DIR 
```
where `$IMAGENET_DIR` is the ImageNet directory, training or evaluation sets,
`$OUT_DIR` is the directory where the noisy images are written to, and 
`ll_low` and `ll_high` are the lowest and highest light level, respectively.
To generate images with other noise profiles, adapt the `ll_low` and 
`ll_high` accordingly (see more examples in the `run_train_joint_models.sh` script).
