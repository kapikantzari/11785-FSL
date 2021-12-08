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
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTModel, ViTConfig

ViTbase = 'google/vit-base-patch16-224-in21k'

# 1. Vit feature extractor -> input data,f extractor x -> ignore
# 2. ViTModel -> self.embed_func (load params from vitforclassifier)
# 3. ViTForImageClassification -> self.model (trained! param)


class VisionTransformerClassifier(pl.LightningModule): # customized class
    def __init__(self, ViTmodel, config):
        super(VisionTransformerClassifier, self).__init__()
        self.config = config
        self._model = ViTForImageClassification.from_pretrained(ViTmodel, num_labels=64)
        self.model_type = ModelType.FINETUNING
        self.train_loader, self.val_loader, self.test_loader = self.train_dataloader()
        print("- - - [Vision Transformer Classifier initialized] - - ")

        # TODO:
        print(ViTmodel)
        self.emb_func = ViTmodel.from_pretrained(ViTbase) # pretrained param -> finetuned params
        # 1. feature_extractor(emb_func)
        # 2. few shot def _validate()

    def train_dataloader(self):
        feature_extractor = ViTFeatureExtractor.from_pretrained(ViTbase)
        print("- - - feature extractor - - -")
        print(feature_extractor)
        print("- - - data loader - - -")
        train_loader = get_vit_dataloader(self.config, "train", feature_extractor)
        val_loader = get_vit_dataloader(self.config, "val", feature_extractor)
        test_loader = get_vit_dataloader(self.config, "test", feature_extractor)

        return train_loader, val_loader, test_loader

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        ACCUMULATE_GRAD_BATCHES = 1
        WARMUP_PROPORTION = 0.1
        LR = 4e-5
        WEIGHT_DECAY = 0.01
        ADAM_EPSILON = 1e-8
        MAX_LENGTH = 16
        train_steps = self.config["epoch"] * (
                len(self.train_loader) // ACCUMULATE_GRAD_BATCHES + 1)
        warmup_steps = int(train_steps * WARMUP_PROPORTION)
        self._model_hparams = {'lr': LR,
                              'warmup_steps': warmup_steps,
                              'train_steps': train_steps,
                              'weight_decay': WEIGHT_DECAY,
                              'adam_epsilon': ADAM_EPSILON,
                              }
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self._model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": self._model_hparams['weight_decay']},
            {"params": [p for n, p in self._model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]

        print("- - -  optimizer -  - -")
        optimizer = transformers.AdamW(
            optimizer_grouped_parameters,
            lr=self._model_hparams['lr'], eps=self._model_hparams['adam_epsilon'])

        print(optimizer)


        print("- - - scheduler - - -")
        lr_scheduler = {
            'scheduler': transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self._model_hparams['warmup_steps'],
                num_training_steps=self._model_hparams['train_steps']),
            'interval': 'step'}
        print(lr_scheduler)

        return [optimizer], [lr_scheduler]

    def forward(self, pixel_values, labels):
        return self._model(pixel_values=pixel_values, labels=labels)

    #def set_forward(): return self.embed_func

    def training_step(self, batch, batch_idx):
        if batch.pixel_values is None:
            return torch.tensor(0., requires_grad=True, device=self.device)

        outputs = self._model(pixel_values=batch.pixel_values, labels=batch.labels)
        logits = outputs.logits
        # print(logits.size())
        loss = outputs.loss
        # print(loss)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.training_step(batch=batch, batch_idx=None)

        self.log('val_loss', loss)


class ViTTrainer(object):
    """
    The trainer.

    Build a trainer from config dict, set up optimizer, model, etc. Train/test/val and log.
    """


    def __init__(self, config):
        self.config = config
        # self.device, self.list_ids = self._init_device(config)
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

        """
            ViT model
        """
        self.model = VisionTransformerClassifier(ViTbase, self.config)
        self.logger.info("Trainable params in the model: {}".format(count_parameters(self.model)))
        self.train_loader = self.model.train_loader
        self.val_loader = self.model.val_loader
        self.test_loader = self.model.test_loader


    """
        ViT train
    """
    def _train(self):
        ACCUMULATE_GRAD_BATCHES = 1
        VAL_CHECK_INTERVAL = 0.5
        batch_size = 16
        log_dir = '../results/ViT_exp1_dec_1'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.logger.info('Trainer log dir: %s' %(log_dir))
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'{log_dir}/',
            filename='vit-mini-{epoch:02d}-{loss:.2f}',
            monitor='train_loss',
            mode='min',
            save_top_k=1,
            verbose=False)
        # best model

        trainer = pl.Trainer(
            callbacks=checkpoint_callback,
            max_epochs=self.config['epoch'],
            accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
            val_check_interval=VAL_CHECK_INTERVAL,
            # devices=4,
            accelerator='gpu',
            gpus=1)

        self.logger.info(trainer)
        print("*"*40)
        print("Start Train ...")
        print("*"*40)
        trainer.fit(
            model=self.model,
            train_dataloader=self.train_loader,
            val_dataloaders=self.train_loader)

    def _init_files(self, config):
        """
        Init result_path(checkpoints_path, log_path, viz_path) from the config dict.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of (result_path, log_path, checkpoints_path, viz_path).
        """
        # you should ensure that data_root name contains its true name
        base_dir = "{}-{}-{}-{}-{}".format(
            config["classifier"]["name"],
            config["data_root"].split("/")[-1],
            config["backbone"]["name"],
            config["way_num"],
            config["shot_num"],
        )
        result_dir = (
            base_dir
            + "{}-{}".format(
                ("-" + config["tag"]) if config["tag"] is not None else "", get_local_time()
            )
            if config["log_name"] is None
            else config["log_name"]
        )
        if not os.path.exists(config["result_root"]):
            os.mkdir(config["result_root"])

        result_path = os.path.join(config["result_root"], result_dir)
        # self.logger.log("Result DIR: " + result_path)
        checkpoints_path = os.path.join(result_path, "checkpoints")
        log_path = os.path.join(result_path, "log_files")
        viz_path = os.path.join(log_path, "tfboard_files")
        create_dirs([result_path, log_path, checkpoints_path, viz_path])

        with open(os.path.join(result_path, "config.yaml"), "w", encoding="utf-8") as fout:
            fout.write(yaml.dump(config))

        init_logger(
            config["log_level"],
            log_path,
            config["classifier"]["name"],
            config["backbone"]["name"],
        )

        return result_path, log_path, checkpoints_path, viz_path

    def _init_meter(self):
        """
        Init the AverageMeter of train/val/test stage to cal avg... of batch_time, data_time,calc_time ,loss and acc1.

        Returns:
            tuple: A tuple of train_meter, val_meter, test_meter.
        """
        train_meter = AverageMeter(
            "train",
            ["batch_time", "data_time", "calc_time", "loss", "acc1"],
            self.writer,
        )
        val_meter = AverageMeter(
            "val",
            ["batch_time", "data_time", "calc_time", "acc1"],
            self.writer,
        )
        test_meter = AverageMeter(
            "test",
            ["batch_time", "data_time", "calc_time", "acc1"],
            self.writer,
        )

        return train_meter, val_meter, test_meter

    def _cal_time_scheduler(self, start_time, epoch_idx):
        """
        Calculate the remaining time and consuming time of the training process.

        Returns:
            str: A string similar to "00:00:00/0 days, 00:00:00". First: comsuming time; Second: total time.
        """
        total_epoch = self.config["epoch"] - self.from_epoch - 1
        now_epoch = epoch_idx - self.from_epoch

        time_consum = datetime.datetime.now() - datetime.datetime.fromtimestamp(start_time)
        time_consum -= datetime.timedelta(microseconds=time_consum.microseconds)
        time_remain = (time_consum * (total_epoch - now_epoch)) / (now_epoch)

        res_str = str(time_consum) + "/" + str(time_remain + time_consum)

        return res_str