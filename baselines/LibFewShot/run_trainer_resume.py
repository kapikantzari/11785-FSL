# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os

from core.config import Config
from core import Trainer

PATH = "/content/gdrive/MyDrive/results/Baseline-miniImageNet--ravi-resnet12-5-1-Nov-03-2021-21-28-37"

if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), is_resume=True).get_config_dict()
    trainer = Trainer(config)
    trainer.train_loop()
