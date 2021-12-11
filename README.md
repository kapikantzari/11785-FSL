# 11785-FSL
Use `simple-baseline.ipynb` and `libfewshot-baseline.ipynb`  to run the baseline models.

The implementations are adapted from the implementation of the original papers found [here](https://github.com/RL-VIG/LibFewShot). Note due to the size limit, the pretrained model, miniImageNet dataset are not included but they can be downloaded from the following links:

- [Pretrained ResNet 12 Backbone](https://drive.google.com/drive/folders/1BDIeiQuSNqAfksurLtZ3FL6Ds3v41da0?usp=sharing)
- [miniImageNet Dataset](https://drive.google.com/file/d/1wnm7AqyuKWS-XvDfbm0qxF2O2uUZ191V/view?usp=sharing)

#### File Structure
- The `config` folder contains all `.yaml` files to configure models, loggers, and miscellaneous parameters.
- The `core` folder contains all the code for initializing models, training, and testing them. In particular `train.py` and `test.py` are used for training and testing the ResNet-12 backbone. `ViTtrainer.py` and `ViTtest.py` are used for training and testing the ViT backbone.


#### Steps to run code
1. Download [miniImageNet Dataset](https://drive.google.com/file/d/1wnm7AqyuKWS-XvDfbm0qxF2O2uUZ191V/view?usp=sharing) and extract the dataset.
2. Move dataset to a designated location, and match data_root in `/config/headers/data.yaml` to this location.
3. install all dependencies in `requirements.txt`

##### Run the Pretrained ResNet 12 Baseline
1. Download the [Pretrained ResNet 12 Backbone](https://drive.google.com/drive/folders/1BDIeiQuSNqAfksurLtZ3FL6Ds3v41da0?usp=sharing)
2. Modify `PATH` variable in `run_test.py` to point to the downloaded backbone.
3. `python run_test.py`

##### Train the Vision Transformer
1. Configure parameters in `vitconfig.yaml`
2. Start training: `python run_vit_trainer.py`

##### Run the Vision Transformer Baseline
1. Modify `PATH` variable in `run_test.py` to point to the trained backbone.
2. `python run_vit_test.py`