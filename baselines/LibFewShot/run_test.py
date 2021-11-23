# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os

from core.config import Config
from core import Test

PATH = "/content/Baseline-miniImageNet--ravi-resnet12-5-1-Nov-03-2021-12-42-37"
VAR_DICT = {
    "test_epoch": 5,
    "device_ids": "2",
    "n_gpu": 1,
    "test_episode": 600,
    "episode_size": 1,
    "test_way": 6,
}

if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()
    test = Test(config, PATH)
    test.test_loop()
