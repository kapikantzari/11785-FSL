# Few Shot Learning Image Classification (Intro 2 DL 11-785 @ CMU)

- We proposed a **Weighted-distribution Calibration (WC)** to alleviate bias of distribution of novel classes by generating more data from a calibrated distribution, which takes advantage of transferred statistics of all base classes.
- With a backbone of ResNet-12 and a logistic regression classifier, WC successfully improves the model's performance from a base accuracy of **57.33\%** to a surprising accuracy of **63.87\%**.
- We experimented on Vision Transformer (ViT) based backbone feature extractor. We expected ViT to pay more attention to target objects, thus decreasing the redundancy in extracted features. Unfortunately, it did not work well in our experiments, possibly, due to its loss of reception field.

> The implementations of backbone networks are adapted from [here](https://github.com/RL-VIG/LibFewShot). 
>
> Note due to the size limit, the pretrained model, miniImageNet dataset are not included but they can be downloaded from the following links:
>
> - [miniImageNet Dataset](https://drive.google.com/file/d/1wnm7AqyuKWS-XvDfbm0qxF2O2uUZ191V/view?usp=sharing)



## File Structure

- The `config` folder contains all `.yaml` files to configure models, loggers, and miscellaneous parameters.
- The `core` folder contains all the code for initializing models, training, and testing them. In particular `train.py` and `test.py` are used for training and testing the ResNet-12 backbone. `ViTtrainer.py` and `ViTtest.py` are used for training and testing the ViT backbone.


## Steps to run code

1. Download [miniImageNet Dataset](https://drive.google.com/file/d/1wnm7AqyuKWS-XvDfbm0qxF2O2uUZ191V/view?usp=sharing) and extract the dataset.
2. Move dataset to a designated location, and match data_root in `/config/headers/data.yaml` to this location.
3. install all dependencies in `requirements.txt`

### Run the Pretrained ResNet 12 Baseline

1. Download the [Pretrained ResNet 12 Backbone](https://drive.google.com/drive/folders/1BDIeiQuSNqAfksurLtZ3FL6Ds3v41da0?usp=sharing)
2. Modify `PATH` variable in `run_test.py` to point to the downloaded backbone.
3. `python run_test.py`

### Train the Backbone

1. Modify configuration in corresponding `[backbone_name].yaml` file
2. Start training: `python run_trainer.py`

### Run the Test with Weighted-distribution Calibration (WC)

1. Modify test configuration in  `config.yaml`  file at the root path
2. Start testing: `python run_test.py`

## Experiments on Vision Transformer-based Feature Extractor

### Train the Vision Transformer

1. Modify configuration in `vitconfig.yaml`
2. Start training: `python run_vit_trainer.py`

### Run the Vision Transformer Test with Weighted-distribution Calibration (WC)

1. Modify test configuration in `vitconfig.yaml` file at the root path
2. Start testing: `python run_vit_test.py`
