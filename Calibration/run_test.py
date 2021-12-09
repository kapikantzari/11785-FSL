# -*- coding: utf-8 -*-
import sys
import os
import numpy as np

sys.dont_write_bytecode = True

from core.config import Config
from core import Test

PATH = "/home/dl_g33/dataset/miniImageNet--ravi/trained"
VAR_DICT = {
    "test_epoch": 5,
    "device_ids": "2",
    "n_gpu": 1,
    "test_episode": 600,
    "episode_size": 1,
    "test_way": 5,
}

if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()
    test = Test(config, result_path=PATH,load_data=True)
    checkpoint_dir = os.path.join(PATH, 'features')

    # Extract features for base and novel class
    test.logger.info("============ Extract features and generating base class stats ============")
    output_dict_base = test.extract_features_loop(checkpoint_dir, tag='last', loader_type='base') #dataloader, vit class
    test.logger.info("Base set features saved!")
    output_dict_novel = test.extract_features_loop(checkpoint_dir, tag='last',loader_type='novel')
    test.logger.info("Novel features saved!")

    # Generate base class stats
    test.logger.info("============ Generate base class stats ============")
    base_means = []
    base_cov = []
    for key in output_dict_base.keys():
        feature = np.array(output_dict_base[key])
        mean = np.mean(feature, axis=0)
        cov = np.cov(feature.T)
        base_means.append(mean)
        base_cov.append(cov)
    test.logger.info("Finish generating base class stats!")

    test.logger.info("============ Test Loop ============")
    test.test_loop(output_dict_novel, base_means, base_cov)
