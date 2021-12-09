# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import pickle

sys.dont_write_bytecode = True

from core.config import Config
from core import Test, ViTTest

PATH = "/home/dl_g33/dataset/miniImageNet--ravi/trained"
VAR_DICT = {
    "test_epoch": 5,
    "device_ids": "2",
    "n_gpu": 1,
    "test_episode": 600,
    "episode_size": 1,
    "test_way": 5,
}


#to modify: .yaml file 

if __name__ == "__main__":
    # config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()
    config = Config("vitconfig.yaml", VAR_DICT).get_config_dict()
    print("="*40)
    print(config)
    print("="*40)
    test = ViTTest(config, result_path=PATH)
    checkpoint_dir = os.path.join(PATH, 'features')

    # Extract features for base and novel class
    # test.logger.info("============ Extract features and generating base class stats ============")
    # output_dict_base = test.extract_features_loop(checkpoint_dir, tag='vit', loader_type='base')
    # test.logger.info("Base set features saved!")
    # output_dict_novel = test.extract_features_loop(checkpoint_dir, tag='vit', loader_type='novel')
    # test.logger.info("Novel features saved!")
    test.logger.info("Load features...")
    basepath = os.path.join(checkpoint_dir, "vit/base_features.plk")
    with open(basepath, "rb") as fin:
        output_dict_base = pickle.load(fin)
    novelpath = os.path.join(checkpoint_dir, "vit/novel_features.plk")
    with open(novelpath, "rb") as fin:
        output_dict_novel = pickle.load(fin)
    
    # test.logger.info("base features: {}" .format(output_dict_base.shape))
    # test.logger.info("novel features: {}" .format(output_dict_novel.shape))
    test.logger.info("Generate base class stats...")
    # Generate base class stats
    base_means = []
    base_cov = []
    for key in output_dict_base.keys():
        feature = np.array(output_dict_base[key])
        mean = np.mean(feature, axis=0)
        cov = np.cov(feature.T)
        base_means.append(mean)
        base_cov.append(cov)
        # test.logger.info("calss {}, mean {}, cov {}" .format(key, mean.shape, cov.shape))
    test.logger.info("Finish generating base class stats!")

    base_means = np.array(base_means)
    base_cov = np.array(base_cov)
    test.logger.info("Test...")
    test.test_loop(output_dict_novel, base_means, base_cov)
    # test.test_loop()
    
