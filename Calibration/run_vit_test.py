# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import pickle
import yaml

sys.dont_write_bytecode = True

from core.config import Config
from core import Test, ViTTest

PATH = "/home/dl_g33/dataset/miniImageNet--ravi/trained"



#to modify: .yaml file 

if __name__ == "__main__":
    # config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()
    config = Config("vitconfig.yaml").get_config_dict()
    print("="*40)
    print(config)
    print("="*40)
    test = ViTTest(config, result_path=PATH)
    checkpoint_dir = os.path.join(PATH, 'features')
    with open(os.path.join(checkpoint_dir, "%s/config.yaml" %(config['log_tag'])), "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(config))
    # Extract features for base and novel class

    basepath = os.path.join(checkpoint_dir, "%s/base_features.plk" %(config['log_tag']))
    novelpath = os.path.join(checkpoint_dir, "%s/novel_features.plk" %(config['log_tag']))
    if os.path.exists(basepath) and os.path.exists(novelpath):
        test.logger.info("Load features...")
        with open(basepath, "rb") as fin:
            output_dict_base = pickle.load(fin)
        test.logger.info("Base features loaded!")
        with open(novelpath, "rb") as fin:
            output_dict_novel = pickle.load(fin)
        test.logger.info("Novel features loaded!")
    else:
        test.logger.info("Extract features and generating base class stats...")
        output_dict_base = test.extract_features_loop(checkpoint_dir, tag=config['log_tag'], loader_type='base')
        test.logger.info("Base set features saved!")
        output_dict_novel = test.extract_features_loop(checkpoint_dir, tag=config['log_tag'], loader_type='novel')
        test.logger.info("Novel features saved!")
        
    
    # test.logger.info("base features: {}" .format(output_dict_base.shape))
    # test.logger.info("novel features: {}" .format(output_dict_novel.shape))
    test.logger.info("Generate base class stats...")
    # Generate base class stats
    base_means = []
    base_cov = []
    # support_data = np.power(support_data * (support_data>0), beta)
    # query_data = np.power(query_data * (query_data>0), beta)
    for key in output_dict_base.keys():
        feature = np.array(output_dict_base[key])
        feature = np.power(feature * (feature > 0), 0.5)
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

    
