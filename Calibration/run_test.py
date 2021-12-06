# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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
    test = Test(config, result_path=PATH)
    checkpoint_dir = os.path.join(PATH, 'features')

    # Extract features for base and novel class
    test.logger.info("============ Extract features and generating base class stats ============")
    output_dict_base = test.extract_features_loop(checkpoint_dir, tag='last', loader_type='base')
    test.logger.info("Base set features saved!")
    output_dict_novel = test.extract_features_loop(checkpoint_dir, tag='last',loader_type='novel')
    test.logger.info("Novel features saved!")

    # Generate base class stats
    base_means = []
    base_cov = []
    for key in output_dict_base.keys():
        feature = np.array(output_dict_base[key])
        mean = np.mean(feature, axis=0)
        cov = np.cov(feature.T)
        base_means.append(mean)
        base_cov.append(cov)
    test.logger.info("Finish generating base class stats!")

    # test.test_loop(output_dict_novel, base_means, base_cov)

    # Generate plots
    acc = []
    for num_sampled in range(0, 3000, 100):
        acc.append(test.test_loop(output_dict_novel, base_means, base_cov, num_sampled=num_sampled))
    
    print(acc)
    fig, axs = plt.subplots()

    # LR all num_sampled acc
    # acc = [0.5533333333333333, 0.55, 0.56, 0.5533333333333333, 0.5466666666666666, 0.5833333333333334, 0.5566666666666666, 0.5733333333333334, 0.5733333333333333, 0.5466666666666667, 0.5466666666666666, 0.57, 0.5566666666666666, 0.5633333333333334, 0.5566666666666666, 0.5866666666666667, 0.5633333333333332, 0.55, 0.5433333333333333, 0.5433333333333333, 0.5766666666666667, 0.5666666666666667, 0.55, 0.5333333333333333, 0.5566666666666666, 0.5533333333333333, 0.5599999999999999, 0.5433333333333333, 0.5533333333333333, 0.5533333333333333]

    # SVM all num_sampled acc
    # acc = [0.51, 0.5133333333333333, 0.54, 0.5333333333333333, 0.5366666666666667, 0.5366666666666667, 0.5266666666666666, 0.5433333333333333, 0.54, 0.55, 0.5633333333333334, 0.5166666666666667, 0.5333333333333333, 0.54, 0.54, 0.5233333333333333, 0.5133333333333333, 0.52, 0.5333333333333333, 0.5366666666666666, 0.53, 0.5366666666666666, 0.56, 0.5233333333333333, 0.5466666666666667, 0.5333333333333334, 0.54, 0.5233333333333333, 0.5333333333333333, 0.5233333333333333]

    axs.plot(np.arange(30) * 100, acc, label='5-way 1-shot Accuracy vs. Number of Samples', linewidth=2.5)
    plt.xlabel('Number of samples', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.savefig('acc_num_sampled_{}.png'.format(config['test_classifier']), bbox_inches='tight')
    plt.close(fig)
