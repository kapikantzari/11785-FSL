# -*- coding: utf-8 -*-
import os
from logging import getLogger
from time import time
import collections
import pickle
import random

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
import dcor

import core.model as arch
from core.data import get_dataloader
from core.utils import (
    init_logger,
    prepare_device,
    init_seed,
    AverageMeter,
    count_parameters,
    ModelType,
    TensorboardWriter,
    mean_confidence_interval,
    get_local_time,
    get_instance,
)


class Test(object):
    """
    The tester.
    Build a tester from config dict, set up model from a saved checkpoint, etc. Test and log.
    """

    def __init__(self, config, result_path=None, load_data=False):
        self.config = config
        self.result_path = result_path
        self.device, self.list_ids = self._init_device(config)
        self.viz_path, self.state_dict_path = self._init_files(config)
        self.writer = TensorboardWriter(self.viz_path)
        self.test_meter = self._init_meter()
        self.logger = getLogger(__name__)
        # self.logger.info(config)
        if load_data:
            self.model, self.model_type = self._init_model(config)
            self.model.reverse_setting_info()
            (
                self.train_loader,
                self.test_loader,
            ) = self._init_dataloader(config)
        self.rng = np.random.default_rng(seed=20)
        self.num_sampled = self.config["num_sampled"]

    def test_loop(self, output_dict_novel, base_means, base_cov, is_round_test=True, is_test=False):
        """
        The normal test loop: test and cal the 0.95 mean_confidence_interval.
        """
        acc_list = []

        print("num sumpled: {}" .format(self.num_sampled))
        print("is round test: {}".format(is_round_test))
        # self.logger.info("============ Testing on the test set ============")
        if is_round_test:
            for epoch_idx in range(4):
                acc = self._validate(output_dict_novel, base_means, base_cov, epoch_idx=epoch_idx)
                acc_list.append(acc)
                self.logger.info("[Epoch {}] Test Accuracy: {:.3f} Running Acc: {:.3f}".format(epoch_idx, acc, float(np.mean(acc_list))))
        else:
            for epoch_idx in range(self.config['test_epoch']):
                acc = self._validate(output_dict_novel, base_means, base_cov)
                acc_list.append(acc)
                self.logger.info("[Epoch {}] Test Accuracy: {:.3f} Running Acc: {:.3f}".format(epoch_idx, acc, float(np.mean(acc_list))))
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
            # classes = np.random.permutation(list(output_dict_novel.keys()))[:self.config["test_way"]]
            classes = set()
            while len(classes)<self.config["test_way"]:
                classes.add(random.choice(list(output_dict_novel.keys())))
        
        self.logger.info("Classes: {}" .format(classes))

        for label in classes:
            data = output_dict_novel[label]
            # print("label {}, idx [{}]" .format(label, idxs))
            idxs = np.random.permutation(len(data))[:self.config["test_shot"]+self.config["test_query"]]
            # idxs = np.arange(self.config["test_shot"]+self.config["test_query"])

            support_data = np.array(data)[idxs[:self.config["test_shot"]]]
            support_label = np.array([[label] * self.config["test_shot"]])
            query_data = np.array(data)[idxs[self.config["test_shot"]:]]
            query_label = np.array([[label] * (len(idxs) - self.config["test_shot"])])

            if self.num_sampled != 0:
                # Tukey's transform
                # beta = 0.5
                # alpha = 0.21
                beta = 0.9
                alpha = 0.0
                support_data = np.power(support_data * (support_data>0), beta)
                query_data = np.power(query_data * (query_data>0), beta)

                # distribution calibration and feature sampling
                # mean, cov = self.distribution_calibration(support_data, base_means, base_cov, k=k)
                mean, cov = self.weighted_calibration(support_data, base_means, base_cov)
                # mean, cov = self.cosine_wc(support_data, base_means, base_cov, alpha)
                # mean, cov = self.dcor_wc(support_data, base_means, base_cov)

                sampled_data = np.random.multivariate_normal(mean=mean, cov=cov, size=self.num_sampled)
                sampled_label = [support_label[0]] * self.num_sampled
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
        # self.logger.info("Result DIR: " + result_path)
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

    def _init_dataloader(self, config):
        """
        Init dataloaders.(train_loader, val_loader and test_loader)
        Args:
            config (dict): Parsed config file.
        Returns:
            tuple: A tuple of (train_loader, val_loader and test_loader).
        """
        train_loader = get_dataloader(config, "train", self.model_type)
        test_loader = get_dataloader(config, "test", self.model_type)

        return train_loader, test_loader

    def _init_model(self, config):
        """
        Init model(backbone+classifier) from the config dict and load the best checkpoint, then parallel if necessary .
        Args:
            config (dict): Parsed config file.
        Returns:
            tuple: A tuple of the model and model's type.
        """
        emb_func = get_instance(arch, "backbone", config)
        model_kwargs = {
            "way_num": config["way_num"],
            "shot_num": config["shot_num"] * config["augment_times"],
            "query_num": config["query_num"],
            "test_way": config["test_way"],
            "test_shot": config["test_shot"] * config["augment_times"],
            "test_query": config["test_query"],
            "emb_func": emb_func,
            "device": self.device,
        }
        model = get_instance(arch, "classifier", config, **model_kwargs)

        # self.logger.info(model)
        # self.logger.info("Trainable params in the model: {}".format(count_parameters(model)))

        self.logger.info("load the state dict from {}.".format(self.state_dict_path))
        state_dict = torch.load(self.state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)

        model = model.to(self.device)
        if len(self.list_ids) > 1:
            parallel_list = self.config["parallel_part"]
            if parallel_list is not None:
                for parallel_part in parallel_list:
                    if hasattr(model, parallel_part):
                        setattr(
                            model,
                            parallel_part,
                            nn.DataParallel(
                                getattr(model, parallel_part),
                                device_ids=self.list_ids,
                            ),
                        )

        return model, model.model_type

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

    def extract_features_loop(self, checkpoint_dir, tag='last',loader_type='base'):
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
            output_dict = self._extract_features(loader, output_dict, loader_type)
            with open(save_dir + '/%s_features.plk'%loader_type, 'wb') as f:
                pickle.dump(output_dict, f)

            return output_dict

    def _extract_features(self, loader, output_dict, loader_type):
        self.model.eval()
        if loader_type != 'base':
            self.model.reverse_setting_info()
        for i, batch in enumerate(loader):
            # compute output
            if loader_type == 'base':
                output = self.model.set_forward_loss(batch, return_embedding=True)
            else:
                output = self.model.set_forward(batch, return_embedding=True)
            output = output.cpu().data.numpy()

            for out, label in zip(output, torch.reshape(batch[1], (-1,))):
                outs = output_dict.get(label.item(), [])
                outs.append(out)
                output_dict[label.item()] = outs

        if loader_type != 'base':
            self.model.reverse_setting_info()
        return output_dict

    def distribution_calibration(self, query, base_means, base_cov, k, alpha=0.21):
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
    
    def cosine_wc(self, query, base_means, base_cov, alpha=0.21):
        base_means = np.array(base_means)
        base_cov = np.array(base_cov)
        dist = []
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for i in range(len(base_means)):
            dist.append(cos(torch.tensor(base_means[i]), torch.tensor(query)))
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

    def dcor_wc(self, query, base_means, base_cov, alpha=0.21):
        base_means = np.array(base_means)
        base_cov = np.array(base_cov)
        dist = []
        for i in range(len(base_means)):
            dist.append(dcor.distance_correlation(base_means[i], query))
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