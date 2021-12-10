# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

from core.config import Config
from core import Trainer, ViTTrainer

if __name__ == "__main__":
    config = Config("./config/vit.yaml").get_config_dict()
    trainer = ViTTrainer(config)
    print("= = "*30)
    trainer._train()
