# -*- coding: utf-8 -*-
import datetime
import os
from logging import getLogger
from time import time
import pickle
import numpy as np

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
print("ViTModel:\n ========================================")
print(ViTModel)

ViTbase = 'google/vit-base-patch16-224-in21k'

class VisionTransformerClassifier(pl.LightningModule): # customized class
    def __init__(self, vitmodelpath, config):
        super(VisionTransformerClassifier, self).__init__()
        self.config = config
        self._model = ViTForImageClassification.from_pretrained(vitmodelpath, num_labels=64)
        self.model_type = ModelType.FINETUNING
        self.train_loader, self.val_loader, self.test_loader = self.train_dataloader()
        print("- - - [Vision Transformer Classifier initialized] - - ")

        # TODO:
        # pretrained param -> finetuned params
        self.emb_func = ViTModel.from_pretrained(vitmodelpath)
        print("ViTmodel:")
        print(self.emb_func)
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


class ViTTest1(object):
    """
    The tester.

    Build a tester from config dict, set up model from a saved checkpoint, etc. Test and log.
    """

    def __init__(self, config, result_path=None):
        self.config = config
        self.result_path = result_path
        self.device, self.list_ids = self._init_device(config)
        self.viz_path, self.state_dict_path = self._init_files(config)
        self.writer = TensorboardWriter(self.viz_path)
        self.test_meter = self._init_meter()
        self.logger = getLogger(__name__)
        self.logger.info(config)
        self.model = VisionTransformerClassifier(ViTbase, self.config)
        self.model_type = ModelType.METRIC

        self.train_loader = self.model.train_loader
        self.val_loader = self.model.val_loader
        self.test_loader = self.model.test_loader

    def test_loop(self):
        """
        The normal test loop: test and cal the 0.95 mean_confidence_interval.
        """
        total_accuracy = 0.0
        total_h = np.zeros(self.config["test_epoch"])
        total_accuracy_vector = []

        for epoch_idx in range(self.config["test_epoch"]):
            self.logger.info("============ Testing on the test set ============")
            _, accuracies = self._validate(epoch_idx)
            test_accuracy, h = mean_confidence_interval(accuracies)
            self.logger.info("Test Accuracy: {:.3f}\t h: {:.3f}".format(test_accuracy, h))
            total_accuracy += test_accuracy
            total_accuracy_vector.extend(accuracies)
            total_h[epoch_idx] = h

        aver_accuracy, h = mean_confidence_interval(total_accuracy_vector)
        self.logger.info("Aver Accuracy: {:.3f}\t Aver h: {:.3f}".format(aver_accuracy, h))
        self.logger.info("............Testing is end............")

    def _validate(self, epoch_idx):
        """
        The test stage.

        Args:
            epoch_idx (int): Epoch index.

        Returns:
            float: Acc.
        """
        # switch to evaluate mode
        self.model.eval()
        print("model type: ")
        print(self.model)
        #self.model.reverse_setting_info()
        meter = self.test_meter
        meter.reset()
        episode_size = self.config["episode_size"]
        accuracies = []

        end = time()
        if self.model_type == ModelType.METRIC:
            enable_grad = False
        else:
            enable_grad = True

        with torch.set_grad_enabled(enable_grad):
            for episode_idx, batch in enumerate(self.test_loader):
                self.writer.set_step(epoch_idx * len(self.test_loader) + episode_idx * episode_size)

                meter.update("data_time", time() - end)

                # calculate the output
                outputs = self.model.forward(batch.pixel_values,batch.labels)

                #debugging output
                print('outputs: ')
                print(outputs)
                logits=outputs[1]
                print('labels: ')
                print(batch.labels)
                print(batch.labels.shape)
                print('logits: ')
                print(logits)
                print(logits.shape)
                pred=torch.argmax(logits,dim=1)
                print('pred:')
                print(pred)
                acc=np.mean(pred==labels)
                accuracies.append(acc)
                # measure accuracy and record loss
                meter.update("acc", acc)

                # measure elapsed time
                meter.update("batch_time", time() - end)
                end = time()

                if (
                    episode_idx != 0 and (episode_idx + 1) % self.config["log_interval"] == 0
                ) or episode_idx * episode_size + 1 >= len(self.test_loader):
                    info_str = (
                        "Epoch-({}): [{}/{}]\t"
                        "Time {:.3f} ({:.3f})\t"
                        "Data {:.3f} ({:.3f})\t"
                        "Acc@1 {:.3f} ({:.3f})".format(
                            epoch_idx,
                            (episode_idx + 1) * episode_size,
                            len(self.test_loader),
                            meter.last("batch_time"),
                            meter.avg("batch_time"),
                            meter.last("data_time"),
                            meter.avg("data_time"),
                            meter.last("acc"),
                            meter.avg("acc"),
                        )
                    )
                    self.logger.info(info_str)
       # self.model.reverse_setting_info()
        return meter.avg("acc"), accuracies

    def _init_files(self, config):
        """
        Init result_path(log_path, viz_path) from the config dict.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of (result_path, log_path, checkpoints_path, viz_path).
        """
        if self.result_path is not None:
            result_path = self.result_path
        else:
            result_dir = "{}-{}-{}-{}-{}".format(
                config["classifier"]["name"],
                # you should ensure that data_root name contains its true name
                config["data_root"].split("/")[-1],
                config["backbone"]["name"],
                config["way_num"],
                config["shot_num"],
            )
            result_path = os.path.join(config["result_root"], result_dir)
        # self.logger.log("Result DIR: " + result_path)
        log_path = os.path.join(result_path, "log_files")
        viz_path = os.path.join(log_path, "tfboard_files")

        init_logger(
            config["log_level"],
            log_path,
            config["classifier"]["name"],
            config["backbone"]["name"],
            is_train=False,
        )

        state_dict_path = os.path.join(result_path, "checkpoints", "model_best.pth")

        return viz_path, state_dict_path

    def _init_device(self, config):
        """
        Init the devices from the config file.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of deviceand list_ids.
        """
        init_seed(config["seed"], config["deterministic"])
        device, list_ids = prepare_device(config["device_ids"], config["n_gpu"])
        return device, list_ids

    def _init_meter(self):
        """
        Init the AverageMeter of test stage to cal avg... of batch_time, data_time,calc_time ,loss and acc1.

        Returns:
            tuple: A tuple of train_meter, val_meter, test_meter.
        """
        test_meter = AverageMeter("test", ["batch_time", "data_time", "acc"], self.writer)

        return test_meter


    #added from test.py (use set_forward_loss, or set_forward)
    def _extract_features(self):
        self.model.eval()
        print("train loader")
        for i, batch in enumerate(self.train_loader):
            print(i)
            # # compute output
            if loader_type == 'base':
                output = self.model.set_forward_loss(batch, return_embedding=True)
            else:
                output = self.model.set_forward(batch, return_embedding=True)
            print(output)
            output = output.cpu().data.numpy()

            for out, label in zip(output, torch.reshape(batch[1], (-1,))):
                 outs = output_dict.get(label.item(), [])
                 outs.append(out)
                 output_dict[label.item()] = outs

        if loader_type != 'base':
            self.model.reverse_setting_info()
        return output_dict
    

    #added from test.py
    def extract_features_loop(self, checkpoint_dir, tag='last',loader_type='base'):
        print("calling")
        save_dir = '{}/{}'.format(checkpoint_dir, tag)
        if os.path.isfile(save_dir + '/%s_features.plk'%loader_type):
            with open(save_dir + '/%s_features.plk'%loader_type, 'rb') as f:
                data = pickle.load(f)
            return data
        else:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

        if loader_type == 'base':
            loader = self.train_loader
        else:
            loader = self.test_loader
        with torch.no_grad():
            output_dict = collections.defaultdict(list)
            print(loader_type)
            output_dict = self._extract_features(loader, output_dict, loader_type)
            with open(save_dir + '/%s_features.plk'%loader_type, 'wb') as f:
                pickle.dump(output_dict, f)
        
            return output_dict
