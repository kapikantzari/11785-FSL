# -*- coding: utf-8 -*-
import datetime
import os
from logging import getLogger
from time import time
import pickle
import collections
import numpy as np
import yaml

import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.linear_model import LogisticRegression

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



"""
    # pretrained
    # class VisionTransformerClassifier(pl.LightningModule): # customized class
    #     def __init__(self, vitmodelpath, config):
    #         super(VisionTransformerClassifier, self).__init__()
    #         self.config = config
    #         self._model = ViTForImageClassification.from_pretrained(vitmodelpath, num_labels=64)
    #         self.model_type = ModelType.FINETUNING
    #         self.train_loader, self.val_loader, self.test_loader = self.train_dataloader()
    #         # print("- - - [Vision Transformer Classifier initialized] - - ")

    #         # TODO:
    #         # pretrained param -> finetuned params
    #         self.emb_func = ViTModel.from_pretrained(vitmodelpath)
    #         # print("- - - ViTmodel embedding func - - -")
    #         # print(self.emb_func)

    #     def train_dataloader(self):
    #         feature_extractor = ViTFeatureExtractor.from_pretrained(ViTbase)
    #         print("- - - feature extractor - - -")
    #         print(feature_extractor)
    #         print("- - - data loader - - -")
    #         train_loader = get_vit_dataloader(self.config, "train", feature_extractor)
    #         val_loader = get_vit_dataloader(self.config, "val", feature_extractor)
    #         test_loader = get_vit_dataloader(self.config, "test", feature_extractor)
            
    #         return train_loader, val_loader, test_loader

    #     def configure_optimizers(self):
    #         no_decay = ["bias", "LayerNorm.weight"]
    #         ACCUMULATE_GRAD_BATCHES = 1
    #         WARMUP_PROPORTION = 0.1
    #         LR = 4e-5
    #         WEIGHT_DECAY = 0.01
    #         ADAM_EPSILON = 1e-8
    #         MAX_LENGTH = 16
    #         train_steps = self.config["epoch"] * (
    #                 len(self.train_loader) // ACCUMULATE_GRAD_BATCHES + 1)
    #         warmup_steps = int(train_steps * WARMUP_PROPORTION)
    #         self._model_hparams = {'lr': LR,
    #                               'warmup_steps': warmup_steps,
    #                               'train_steps': train_steps,
    #                               'weight_decay': WEIGHT_DECAY,
    #                               'adam_epsilon': ADAM_EPSILON,
    #                               }
    #         optimizer_grouped_parameters = [
    #             {"params": [p for n, p in self._model.named_parameters()
    #                         if not any(nd in n for nd in no_decay)],
    #              "weight_decay": self._model_hparams['weight_decay']},
    #             {"params": [p for n, p in self._model.named_parameters()
    #                         if any(nd in n for nd in no_decay)],
    #              "weight_decay": 0.0}]

    #         print("- - -  optimizer -  - -")
    #         optimizer = transformers.AdamW(
    #             optimizer_grouped_parameters,
    #             lr=self._model_hparams['lr'], eps=self._model_hparams['adam_epsilon'])

    #         print(optimizer)


    #         print("- - - scheduler - - -")
    #         lr_scheduler = {
    #             'scheduler': transformers.get_linear_schedule_with_warmup(
    #                 optimizer,
    #                 num_warmup_steps=self._model_hparams['warmup_steps'],
    #                 num_training_steps=self._model_hparams['train_steps']),
    #             'interval': 'step'}
    #         print(lr_scheduler)

    #         return [optimizer], [lr_scheduler]

    #     def forward(self, pixel_values, labels):
    #         return self._model(pixel_values=pixel_values, labels=labels)

    #     def training_step(self, batch, batch_idx):
    #         if batch.pixel_values is None:
    #             return torch.tensor(0., requires_grad=True, device=self.device)

    #         outputs = self._model(pixel_values=batch.pixel_values, labels=batch.labels)
    #         logits = outputs.logits

    #         loss = outputs.loss

    #         self.log('train_loss', loss)

    #         return loss

    #     def validation_step(self, batch, batch_idx):
    #         with torch.no_grad():
    #             loss = self.training_step(batch=batch, batch_idx=None)

    #         self.log('val_loss', loss)
"""

class VisionTransformerClassifier(pl.LightningModule): # customized class
    def __init__(self, config):
        super(VisionTransformerClassifier, self).__init__()
        self.config = config
        self._vitconfig = ViTConfig()
        self._vitconfig.num_labels = config['num_labels']
        self._vitconfig.image_size = config['image_size']
        self._vitconfig.num_hidden_layers = config['num_hidden_layers']
        self._vitconfig.num_attention_heads = config['num_attention_heads']
        self._vitconfig.hidden_size = config['hidden_size']
        self._vitconfig.intermediate_size = config['intermediate_size']

        self._model = ViTForImageClassification(self._vitconfig)
        # self._load_ckpt()
        self.model_type = ModelType.FINETUNING

        self.train_loader, self.val_loader, self.test_loader = self.train_dataloader()
        # print("- - - [Vision Transformer Classifier initialized] - - ")

        self.emb_func = self._model.vit
        print("- - - embed func - - -")
        print(self.emb_func)

    def set_to_ckpt(self):
        print("- - - set embed func to ckpt - - -")
        self.emb_func = self._model.vit
    # def _load_ckpt(self):
    #     checkpoint_callback = ModelCheckpoint(dirpath=self.config['ckpt_path'])
    #     print(checkpoint_callback.dirpath)
    #     trainer = pl.Trainer(
    #         callbacks=[checkpoint_callback],
    #         max_epochs=1,
    #         accelerator='cpu')
    #     trainer.fit(self._model)
    #     print(checkpoint_callback.best_model_path)

    def train_dataloader(self):
        # feature_extractor = ViTFeatureExtractor.from_pretrained(ViTbase)
        feature_extractor = ViTFeatureExtractor(size=84)
        # print("- - - feature extractor - - -")
        # print(feature_extractor)
        # print("- - - data loader - - -")
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

        loss = outputs.loss

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
        # self.logger.info(config)
        self.model = VisionTransformerClassifier(self.config)
        self.model_type = ModelType.METRIC

        self.train_loader = self.model.train_loader
        self.val_loader = self.model.val_loader
        self.test_loader = self.model.test_loader
        if config['trainvit']:
            self._train()
        print("trainvit: {}" .format(config['trainvit']))

        self.rng = np.random.default_rng(seed=42)
        self.num_sampled = self.config["num_sampled"]

    """
        few-shot test
    """
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
        calibrationtype = 'standard'
        if epoch_idx != None:
            classes = sorted(list(output_dict_novel.keys()))[self.config["test_way"]*epoch_idx:self.config["test_way"]*(epoch_idx+1)]
        else:
            classes = np.random.permutation(list(output_dict_novel.keys()))[:self.config["test_way"]]
        
        for label in classes:
            data = output_dict_novel[label]
            # bias = np.random.randint(len(data)-(self.config["test_shot"]+self.config["test_query"])-1)
            bias = 0
            idxs = np.random.permutation(len(data))[bias:bias+self.config["test_shot"]+self.config["test_query"]]
            # idxs = np.arange(self.config["test_shot"]+self.config["test_query"])
            support_data = np.array(data)[idxs[:self.config["test_shot"]]]
            support_label = np.array([[label] * self.config["test_shot"]])
            query_data = np.array(data)[idxs[self.config["test_shot"]:]]
            query_label = np.array([[label] * (len(idxs) - self.config["test_shot"])])

            if self.num_sampled != 0:
                # Tukey's transform
                beta = 0.5
                shift = np.min(support_data)
                print("Shift = ", shift)
                support_data -= shift
                query_data -= shift
                support_data = np.power(support_data * (support_data>0), beta)
                query_data = np.power(query_data * (query_data>0), beta)
                support_data += shift
                query_data += shift
                # softmax = nn.Softmax(dim=1)
                # support_data = np.power(softmax(torch.tensor(support_data)).numpy(), beta)
                # query_data = np.power(softmax(torch.tensor(query_data)).numpy(), beta)
                
                
                # distribution calibration and feature sampling
                # mean, cov = self.distribution_calibration(support_data, base_means, base_cov, k, calibrationtype)
                mean, cov = self.weighted_calibration(support_data, base_means, base_cov)
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


    # distribution calibraion
    def distribution_calibration(self, query, base_means, base_cov, k, type='standard', alpha=0.21):
        dist = []
        for i in range(len(base_means)):
            dist.append(np.linalg.norm(query-base_means[i]))
        index = np.argpartition(dist, k)[:k]
        mean = np.concatenate([np.array(base_means)[index], query])
        calibrated_mean = np.mean(mean, axis=0)
        calibrated_cov = np.mean(np.array(base_cov)[index], axis=0)+alpha

        return calibrated_mean, calibrated_cov

    def weighted_calibration(self, query, base_means, base_cov, alpha=0.21):
        base_means = np.array(base_means)
        base_cov = np.array(base_cov)
        dist = []
        for i in range(len(base_means)):
            dist.append(-1 * np.linalg.norm(base_means[i]-query))
        softmax = nn.Softmax()
        dist = softmax(torch.tensor(dist)).numpy()
        mean = []
        cov = []
        for i in range(64):
            mean.append(np.dot(base_means[i], dist[i]))
            cov.append(np.dot(base_cov[i], dist[i]))
        calibrated_mean = np.squeeze(np.transpose(np.sum(mean, axis=0)/2 + query/2))
        calibrated_cov = np.sum(cov, axis=0) + alpha
        return calibrated_mean, calibrated_cov
    def _train(self):
        ACCUMULATE_GRAD_BATCHES = 1
        VAL_CHECK_INTERVAL = 0.5
        batch_size = 8
        log_dir = os.path.join('../results/', self.config['log_tag'])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.logger.info('Trainer log dir: %s' %(log_dir))
        checkpoint_callback = ModelCheckpoint(
            dirpath=f'{log_dir}/',
            filename='vit-mini-{epoch:02d}-{train_loss:.2f}',
            monitor='train_loss',
            mode='min',
            save_top_k=1,
            verbose=False)
        # best model

        trainer = pl.Trainer(
            callbacks=checkpoint_callback,
            max_epochs=self.config['epoch'],
            # max_epochs=20,
            accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
            val_check_interval=VAL_CHECK_INTERVAL,
            accelerator='gpu',
            devices=4,
            gpus=1)

        self.logger.info(trainer)
        print("*"*40)
        print("Start Train ...")
        print("*"*40)
        trainer.fit(
            model=self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.train_loader)
        print("*"*40)
        print("End Train ...")
        print("*"*40)


    def _load_ckpt(self):
        checkpoint_callback = ModelCheckpoint(dirpath=self.config['ckpt_path'])
        print(checkpoint_callback.best_model_path)
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            max_epochs=1,
            accelerator='cpu')
        trainer.fit(
            model=self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.train_loader)
        print(checkpoint_callback.best_model_path)
        self.logger.info("Successfully load ckpt!")

    """
        configuration init
    """
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
                config["backbone"],
                config["test_classifier"],
                # you should ensure that data_root name contains its true name
                config["data_root"].split("/")[-1],
                config["way_num"],
                config["shot_num"],
                config["num_sampled"]
            )
            result_path = os.path.join(config["result_root"], result_dir)
        # self.logger.log("Result DIR: " + result_path)
        log_path = os.path.join(result_path, "log_files")
        viz_path = os.path.join(log_path, "tfboard_files")

        init_logger(
            config["log_level"],
            log_path,
            config["backbone"],
            config["test_classifier"],
            is_train=False,
        )
        print("log path: ", log_path)

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


    """
        feature extraction using trained basemodel
    """
    def _extract_features(self, loader, output_dict, loader_type):
        self.model.eval()
        print("loader type: ", loader_type)
        self.logger.info(f"-- output of batch 0/{len(loader)} --")
        for idx, batch in enumerate(loader):
            if (idx+1) % 50 == 0:
                self.logger.info(f"-- output of batch {idx + 1}/{len(loader)} --")

            # self.logger.info("intput size: {}" .format(batch.pixel_values.shape))
            pixel_values = batch.pixel_values
            labels = batch.labels
            outputs = self.model.emb_func(pixel_values=pixel_values)
            output = outputs.last_hidden_state[:,0,:].cpu().numpy()
            # self.logger.info("output[:,0,:]: {}" .format(output.shape))
            bs = labels.shape[0]
            # self.logger.info("batch size = {}" .format(bs))
            for i in range(bs):
                out = output[i]
                label = labels[i]
                # if i == 0:
                    # self.logger.info("out shape {}" .format(out.shape))
                outs = output_dict.get(label.item(), [])
                outs.append(out)
                output_dict[label.item()] = outs
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
        
        self.model.set_to_ckpt()
        self.logger.info("Set embed fun to trained parameters!")

        with torch.no_grad():
            output_dict = collections.defaultdict(list)
            output_dict = self._extract_features(loader, output_dict, loader_type)

            self.logger.info("save dir: {}" .format(save_dir))
            with open(save_dir + '/%s_features.plk'%loader_type, 'wb') as f:
                pickle.dump(output_dict, f)
            f.close()
            # self.logger.info("output dict\n{}" .format(output_dict))
            print("{} features extraction done!" .format(loader_type))
            return output_dict
