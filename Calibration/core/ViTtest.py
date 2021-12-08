# -*- coding: utf-8 -*-
import datetime
import os
from logging import getLogger
from time import time
import pickle
import collections
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
# print("ViTModel:\n ========================================")
# print(ViTModel)

ViTbase = 'google/vit-base-patch16-224-in21k'

class VisionTransformerClassifier(pl.LightningModule): # customized class
    def __init__(self, vitmodelpath, config):
        super(VisionTransformerClassifier, self).__init__()
        self.config = config
        self._model = ViTForImageClassification.from_pretrained(vitmodelpath, num_labels=64)
        self.model_type = ModelType.FINETUNING
        self.train_loader, self.val_loader, self.test_loader = self.train_dataloader()
        # print("- - - [Vision Transformer Classifier initialized] - - ")

        # TODO:
        # pretrained param -> finetuned params
        self.emb_func = ViTModel.from_pretrained(vitmodelpath)
        print("- - - ViTmodel embedding func - - -")
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


class ViTTest(object):
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

    def test_loop(self, output_dict_novel, base_means, base_cov, is_test=False):
        """
        The normal test loop: test and cal the 0.95 mean_confidence_interval.
        """
        print("Base mean:", base_means.shape)
        print("Base cov:", base_cov.shape)
        acc_list = []
        self.logger.info("test way {} test shot {} test query {}" .format(self.config["test_way"], self.config["test_shot"],self.config["test_query"]))

        for epoch_idx in range(4):
            self.logger.info("============ Testing on the test set ============")
            acc = self._validate(output_dict_novel, base_means, base_cov, epoch_idx=epoch_idx)
            acc_list.append(acc)
            self.logger.info("[Epoch {}] Test Accuracy: {:.3f}".format(epoch_idx, acc))
        self.logger.info("{} way {} shot ACC : {}".format(self.config["test_way"], self.config["test_shot"], float(np.mean(acc_list))))
        self.logger.info("............Testing is end............")

    def _validate(self, output_dict_novel, base_means, base_cov, epoch_idx=None):
        """
        The test stage.

        Args:
            epoch_idx (int): Epoch index.

        Returns:
            float: Acc.
        """
        X_aug = []
        Y_aug = []
        query_data_all = []
        query_label_all = []
        k = 2
        if epoch_idx != None:
            classes = sorted(list(output_dict_novel.keys()))[self.config["test_way"]*epoch_idx:self.config["test_way"]*(epoch_idx+1)]
        else:
            classes = np.random.permutation(list(output_dict_novel.keys()))[:self.config["test_way"]]
        
        for label in classes:
            data = output_dict_novel[label]
            # idxs = np.random.permutation(len(data))[:self.config["test_shot"]+self.config["test_query"]]
            idxs = np.arange(self.config["test_shot"]+self.config["test_query"])
            support_data = np.array(data)[idxs[:self.config["test_shot"]]]
            support_label = np.array([[label] * self.config["test_shot"]])
            query_data = np.array(data)[idxs[self.config["test_shot"]:]]
            query_label = np.array([[label] * (len(idxs) - self.config["test_shot"])])

            if self.num_sampled != 0:
                # Tukey's transform
                beta = 0.5
                # softmax = nn.Softmax(dim=1)
                # support_data = np.power(softmax(torch.tensor(support_data)).numpy(), beta)
                # query_data = np.power(softmax(torch.tensor(query_data)).numpy(), beta)
                support_data = np.power(support_data * (support_data>0), beta)
                query_data = np.power(query_data * (query_data>0), beta)
                
                # distribution calibration and feature sampling
                mean, cov = self.distribution_calibration(support_data, base_means, base_cov, k=k)
                sampled_data = np.random.multivariate_normal(mean=mean, cov=cov, size=self.num_sampled)
                sampled_label = [support_label[0]] * self.num_sampled
                print("Label {}: length {}, k={}, first entry {}".format(label, len(sampled_data), k, sampled_data[0][:10]))
                X_aug.append(np.concatenate([support_data, sampled_data]))
                Y_aug.append(np.concatenate([support_label, sampled_label]))
                query_data_all.append(query_data)
                query_label_all.append(query_label.reshape(-1,1))
            else:
                X_aug.append(support_data)
                Y_aug.append(support_label)
                query_data_all.append(query_data)
                query_label_all.append(query_label.reshape(-1,1))
        X_aug = np.vstack(X_aug)
        Y_aug = np.vstack(Y_aug)
        query_data = np.vstack(query_data_all)
        query_label = np.vstack(query_label_all)
        classifier = LogisticRegression(max_iter=1000).fit(X=X_aug, y=Y_aug.reshape(-1,))
        predicts = classifier.predict(query_data)
        acc = np.mean(predicts == query_label.reshape(-1,))
        return acc

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



    def _extract_features(self, loader, output_dict, loader_type):
        self.model.eval()
        print("loader type: ", loader_type)
        self.logger.info(f"-- output of batch 0/{len(loader)} --")
        for idx, batch in enumerate(loader):
            if (idx+1) % 50 == 0:
                self.logger.info(f"-- output of batch {idx + 1}/{len(loader)} --")
                break
            # self.logger.info("intput size: {}" .format(batch.pixel_values.shape))
            pixel_values = batch.pixel_values
            labels = batch.labels
            output = self.model.emb_func(pixel_values=pixel_values)[0]
            # relucount = np.array(output[:,0,:])
            # relucount[relucount > 0] = 1
            # relucount[relucount < 0] = 0
            # self.logger.info(relucount)
            # self.logger.info("output: {}/{}, {}, {}" .format(np.sum(relucount), relucount.shape[0]*relucount.shape[1], output.shape, output))
            # self.logger.info("output[:,0,:]: {}, {}" .format(output[:,0,:].shape, output[:,0,:]))

            bs = labels.shape[0]
            # for out, label in zip(output.last_hidden_state, labels):
            for i in range(bs):
                out = output[i]
                label = labels[i]
                # self.logger.info("{}, out shape {}" .format(i, out.shape))
                outs = output_dict.get(label.item(), [])
                outs.append(out)
                output_dict[label.item()] = outs
                # self.logger.info("{}, {}" .format(label, out))
                # self.logger.info("output dict[{}] has {} samples" .format(label.item(), len(output_dict[label.item()])))
        return output_dict
    


    def extract_features_loop(self, checkpoint_dir, tag, loader_type):
        save_dir = '{}/{}'.format(checkpoint_dir, tag)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if loader_type == 'base':
            loader = self.train_loader
            print("Extract features: base")
        else:
            loader = self.test_loader
            print("Extract features: test")
        
        with torch.no_grad():
            output_dict = collections.defaultdict(list)
            output_dict = self._extract_features(loader, output_dict, loader_type)

            self.logger.info("save dir: {}" .format(save_dir))
            with open(save_dir + '/%s_features.plk'%loader_type, 'wb') as f:
                pickle.dump(output_dict, f)
            
            self.logger.info("output dict\n{}" .format(output_dict))
            print("{} features extraction done!" .format(loader_type))
            return output_dict
