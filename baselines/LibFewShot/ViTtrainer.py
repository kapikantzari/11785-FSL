# -*- coding: utf-8 -*-
import datetime
import os
from logging import getLogger
from time import time

import torch
import yaml
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from queue import Queue
# import model as arch
from .data import get_dataloader, get_vit_dataloader
from .utils import (
    accuracy,
    AverageMeter,
    ModelType,
    SaveType,
    TensorboardWriter,
    count_parameters,
    create_dirs,
    get_local_time,
    init_logger,
    init_seed,
    prepare_device,
    save_model,
    get_instance,
    data_prefetcher,
)

import transformers
from transformers import ViTFeatureExtractor, ViTForImageClassification

ViTbase = 'google/vit-base-patch16-224-in21k'



class ViTTrainer(object):
    """
    The trainer.

    Build a trainer from config dict, set up optimizer, model, etc. Train/test/val and log.
    """


    def __init__(self, config):
        self.config = config
        self.device, self.list_ids = self._init_device(config)
        (
            self.result_path,
            self.log_path,
            self.checkpoints_path,
            self.viz_path,
        ) = self._init_files(config)
        self.writer = TensorboardWriter(self.viz_path)
        self.train_meter, self.val_meter, self.test_meter = self._init_meter()
        self.logger = getLogger(__name__)
        self.logger.info(config)

        # self.model, self.model_type = self._init_model(config)
        #
        # (
        #     self.train_loader,
        #     self.val_loader,
        #     self.test_loader,
        # ) = self._init_dataloader(config)
        # print("returning from Trainer()")
        # self.optimizer, self.scheduler, self.from_epoch = self._init_optim(config)
        """
            ViT model
        """
        self.model, self.model_type = self._init_vit_model(config)
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = self._init_vit_dataloader(config)
        print("returning from Trainer()")
        self.optimizer, self.scheduler, self.from_epoch = self._init_vit_optim(config)


    """
        ViT train
    """
    def _init_vit_model(self, config):
        """
                Init model(backbone+classifier) from the config dict and load the pretrained params or resume from a
                checkpoint, then parallel if necessary .

                Args:
                    config (dict): Parsed config file.

                Returns:
                    tuple: A tuple of the model and model's type.
        """

        model = ViTForImageClassification.from_pretrained(ViTbase)
        print("Model:")
        print(model)

        self.logger.info("Trainable params in the model: {}".format(count_parameters(model)))
        model = model.to(self.device)

        return model, ModelType.FINETUNING

    def _init_vit_optim(self, config):
        no_decay = ["bias", "LayerNorm.weight"]
        ACCUMULATE_GRAD_BATCHES = 1
        WARMUP_PROPORTION = 0.1
        LR = 4e-5
        WEIGHT_DECAY = 0.01
        ADAM_EPSILON = 1e-8
        PRECISION = 16
        MAX_LENGTH = 16
        train_steps = self.config["epoch"] * (
                len(self.train_loader) // ACCUMULATE_GRAD_BATCHES + 1)
        warmup_steps = int(train_steps * WARMUP_PROPORTION)
        self.model_hparams = {'lr': LR,
                              'warmup_steps': warmup_steps,
                              'train_steps': train_steps,
                              'weight_decay': WEIGHT_DECAY,
                              'adam_epsilon': ADAM_EPSILON,
                              }
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": self.model_hparams['weight_decay']},
            {"params": [p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]

        optimizer = transformers.AdamW(
            optimizer_grouped_parameters,
            lr=self.model_hparams['lr'], eps=self.model_hparams['adam_epsilon'])

        lr_scheduler = {
            'scheduler': transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.model_hparams['warmup_steps'],
                num_training_steps=self.model_hparams['train_steps']),
            'interval': 'step'}

        from_epoch = -1
        # if self.config["resume"]:
        #     resume_path = os.path.join(self.config["resume_path"], "checkpoints", "model_last.pth")
        #     self.logger.info(
        #         "load the optimizer, lr_scheduler and epoch checkpoints dict from {}.".format(
        #             resume_path
        #         )
        #     )
        #     all_state_dict = torch.load(resume_path, map_location="cpu")
        #     state_dict = all_state_dict["optimizer"]
        #     optimizer.load_state_dict(state_dict)
        #     state_dict = all_state_dict["lr_scheduler"]
        #     lr_scheduler.load_state_dict(state_dict)
        #     from_epoch = all_state_dict["epoch"]
        #     self.logger.info("model resume from the epoch {}".format(from_epoch))
        self.logger.info(optimizer)
        self.logger.info(lr_scheduler)

        return optimizer, lr_scheduler, from_epoch

    def _init_vit_dataloader(self, config):
        feature_extractor = ViTFeatureExtractor.from_pretrained(ViTbase)
        print("feature extractor:")
        print(feature_extractor)
        train_loader = get_vit_dataloader(config, "train", self.model_type, feature_extractor)
        val_loader = get_vit_dataloader(config, "val", self.model_type, feature_extractor)
        test_loader = get_vit_dataloader(config, "test", self.model_type, feature_extractor)
        return train_loader, val_loader, test_loader

    def _train_trainer(self):
        # ACCUMULATE_GRAD_BATCHES = 1
        # VAL_CHECK_INTERVAL = 1. / 4
        batch_size = 16
        log_dir = '../results/ViT_exp1_dec_1'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.logger.info('Trainer log dir: %s' %(log_dir))
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'{log_dir}/',
            filename='best_model',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            verbose=True)

        trainer = pl.Trainer(
            callbacks=checkpoint_callback,
            max_epochs=self.config['epoch'],
            accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
            val_check_interval=VAL_CHECK_INTERVAL,
            gpus=1)

        trainer.fit(
            model=self.model,
            train_dataloader=self.train_loader,
            val_dataloaders=self.val_loader)

        # ViTTrainingArguments = transformers.TrainingArguments(
        #     output_dir=output_dir,
        #     do_train=True,
        #     do_eval=True,
        #     evaluation_strategy="steps",
        #     per_device_train_batch_size=batch_size,
        #     per_device_eval_batch_size=batch_size,
        #     learning_rate=5e-5,
        #     weight_decay=5e-6,
        #     save_strategy='steps',
        #     gradient_accumulation_steps=1,
        #     num_train_epochs=10,
        #     logging_steps=500
        # )
        # ViTTrainer = transformers.Trainer(
        #     model = self.model,
        #     args=ViTTrainingArguments,
        #     train_dataset = self.train_set,
        #     eval_dataset = self.val_set
        # )
        # self.logger.info(ViTTrainingArguments)
        # self.logger.info(ViTTrainer)
        # ViTTrainer.train()

