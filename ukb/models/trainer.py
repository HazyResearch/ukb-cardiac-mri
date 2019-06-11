"""
Simple random grid search
"""
import os
import sys
import glob
import copy
import time
import torch
import logging
import numpy as np
import pandas as pd
from itertools import product

import metrics
from metrics import *

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import timeit, format_time

logger = logging.getLogger(__name__)

def tune_threshold(y_true, y_prob, metric='roc_auc_score'):
    """
    Function for tuning the threshold
    """
    try:
        logger.info("The tune_metric chosen is {}".format(metric))
        metric = getattr(metrics, metric) if type(metric) is str else metric

        thresholds = np.arange(0.01, 1, 0.01)
        best_score = 0.0
        best_threshold = 0.5
        for threshold in thresholds:
            y_pred = np.array([1 if p > threshold else 0 for p in y_prob])
            auc_score = metric(y_true, y_pred)
            if auc_score > best_score:
                best_score = auc_score
                best_threshold = threshold
        dev_threshold = best_threshold

        return dev_threshold
    except:
        try:
            logger.info("The tune_metric chosen is disabled.\n"
                        "Fixed threshold chosen: {}".format(metric))
            return float(metric)
        except:
            sys.exit("Invalid tune_metric input!\n"
                     "Valid option1: `str` eg. roc_auc_score\n"
                     "Valid option2: `float` eg.  0.7\n")


class Trainer(object):

    def __init__(self, model_class, model_class_params, noise_aware=False, use_cuda=False, seed=1234):
        """
        :param model_class:
        :param model_class_params:
        :param noise_aware:
        :param use_cuda:
        :param seed:
        """
        self.model_class        = model_class
        self.model_class_params = model_class_params
        self.noise_aware        = noise_aware
        self.use_cuda           = use_cuda
        self.seed               = seed
        self.model_class_params.update({"use_cuda": use_cuda})

    @timeit
    def fit(self, train, dev, test=None, update_freq=5, checkpoint_freq=10, checkpoint_dir=".", **kwargs):
        """
        Fit target model

        :param train:
        :param dev:
        :param test:
        :param update_freq:
        :param checkpoint_freq:
        :param checkpoint_dir:
        :param kwargs:
        :return:
        """
        self._set_seed()

        lr                = kwargs.get('lr', 0.001)
        momentum          = kwargs.get('momentum', 0.9)
        n_epochs          = kwargs.get('n_epochs', 10)
        metric            = kwargs.get('metric', 'roc_auc_score')
        tune_metric       = kwargs.get('tune_metric', 'roc_auc_score')
        batch_size        = kwargs.get('batch_size', 4)
        num_workers       = kwargs.get('num_workers', 1)
        threshold         = kwargs.get('threshold', 0.5)
        use_class_weights = kwargs.get('class_weights', False)
        use_scheduler     = kwargs.get('scheduler', True)
        l2_penalty        = kwargs.get('l2_penalty', 0.001)
        num_frames        = kwargs.get('num_frames', None)
        binary            = self.model_class_params.get('n_classes', 2) == 2
        verbose           = kwargs.get('verbose', False)
        checkpoint_burn   = kwargs.get('checkpoint_burn', 1)

        logger.info("============================")
        logger.info("Trainer Config")
        logger.info("============================")
        logger.info("lr:                {}".format(lr))
        logger.info("momentum:          {}".format(momentum))
        logger.info("tune_metric:       {}".format(tune_metric))
        logger.info("batch_size:        {}".format(batch_size))
        logger.info("l2_penalty:        {}".format(l2_penalty))
        logger.info("num_frames:        {}".format(num_frames))
        logger.info("use_scheduler:     {}".format(use_scheduler))
        logger.info("checkpoint_burn:   {}".format(checkpoint_burn))

        # get metric function
        metric = getattr(metrics, metric) if type(metric) is str else metric

        # build model params dictionary
        params = {name:v if name not in kwargs else kwargs[name] for name,v in self.model_class_params.items()}
        model  = self.model_class(**params) if not self.use_cuda else self.model_class(**params).cuda()

        # setup checkpointing / model state name
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        filelist = glob.glob("{}/{}*".format(checkpoint_dir, model.name))
        checkpoint_name = "{}{}".format(model.name, len(filelist))

        train_loader, dev_loader, test_loader = self._get_data_loaders(train, dev, test, batch_size, num_workers)
        criterion = self._get_criterion(train_loader, use_weights=use_class_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=l2_penalty)
        if use_scheduler:
            scheduler = ReduceLROnPlateau(optimizer, 'min')
        
        best_score = -0.01
        best_threshold = threshold
        best_model = None
        start_time = time.time()

        for epoch in range(n_epochs):
            train_loss, correct, total = 0.0, 0.0, 0.0
            for i, batch in enumerate(train_loader):
                x,y = batch
                if num_frames is not None:
                    if self.noise_aware:
                        y = y.view(-1)
                    else:
                        y = np.repeat(y, num_frames)
                if isinstance(x, list):
                    x = [Variable(x_) if not self.use_cuda else Variable(x_).cuda() for x_ in x]
                    h0 = model.init_hidden(x[0].size(0))
                else:
                    x = Variable(x) if not self.use_cuda else Variable(x).cuda()
                    h0 = model.init_hidden(x.size(0))
                y = Variable(y) if not self.use_cuda else Variable(y).cuda()

                optimizer.zero_grad()
                outputs = model(x, h0)

                # BCELoss assumes binary classification and relies on probability of second class
                if self.noise_aware:
                    loss = criterion(outputs[:,1], y.float())
                    y = (y-best_threshold+0.5).round().long()
                else:
                    loss = criterion(outputs, y)
                
                loss.backward()
                optimizer.step()
                 
                train_loss += loss.data[0]
                total += y.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(y.data).cpu().sum()
           
            # progress update
            if (epoch + 1) % update_freq == 0:
                elapsed = time.time() - start_time
                msg = 'Epoch {:>8} | {:>10} | Loss: {:2.3f} | Acc. {:>5}% ({}/{})'
                avg_loss = train_loss / (i+1)
                acc = 100.0 * correct / total
                logger.info(msg.format("[{}/{}]".format(epoch + 1, n_epochs), format_time(elapsed),
                                       avg_loss, "{:2.1f}".format(acc), int(correct), int(total)))

            # dev set checkpointing
            if epoch + 1 >= checkpoint_burn and ((epoch + 1) % checkpoint_freq == 0 or epoch + 1 == n_epochs):
                dev_true, dev_pred, dev_prob, dev_threshold = self._evaluate(model, dev_loader, "DEV", 
                                                                             binary=binary, 
                                                                             threshold=best_threshold,
                                                                             tune_metric=tune_metric)
                try:
                    score = metric(dev_true, dev_prob)
                except:
                    score = metric(dev_true, dev_pred)

                if test:
                    test_true, test_pred, test_prob, _ = self._evaluate(model, test_loader, "TEST",
                                                                        binary=binary, threshold=dev_threshold,
                                                                        tune_metric=tune_metric)

                    if verbose:
                        y_proba, y_pred = model.predict(test_loader, threshold=threshold,
                                                        binary=binary, return_proba=True)
                        classification_summary(test_true, test_pred, [], test_prob)
                        print(test_prob)

                if (score > 0.0 and score > best_score) or best_score == -0.01:
                    best_score = score
                    best_threshold = dev_threshold
                    best_model = {
                        'epoch': epoch,
                        'model': copy.deepcopy(model),
                        'state_dict': copy.deepcopy(model.state_dict()),
                        'best_score': best_score,
                        'optimizer': optimizer.state_dict()
                    }
            torch.cuda.empty_cache()

            if use_scheduler:
                score = train_loss / (i+1)
                scheduler.step(score)
                #for ii,group in enumerate(optimizer.param_groups):
                #    logger.info("group lr {} {} {}".format(group['lr'], score, epoch))


        # load best model
        #model.load_state_dict(best_model['state_dict'])
        return best_model['model'], best_score, best_threshold

    def save(self, state, checkpoint_root_dir, checkpoint_name):
        """
        Dump model & optimizer state_dict to file
        :param state:
        :param checkpoint_root_dir:
        :param checkpoint_name:
        :return:
        """
        filename = "{}/{}".format(checkpoint_root_dir, checkpoint_name)
        torch.save(state, filename)
        logger.info("Saved model to {}".format(filename))

    def load(self, model, checkpoint_root_dir, checkpoint_name):
        """
        Load saved model. Assumes only state_dict is saved to file.
        :param model:
        :param checkpoint_root_dir:
        :param checkpoint_name:
        :return:
        """
        filename = "{}/{}".format(checkpoint_root_dir, checkpoint_name)
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            logger.info("No checkpoint found at '{}'".format(filename))
        return model

    ################################################################################
    # INTERNAL
    ################################################################################

    def _set_seed(self):
        """
        Set seed for deterministic random behavior for PyTorch on CPU and GPU
        :return:
        """
        torch.cuda.random.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(seed=int(self.seed))

    def _get_criterion(self, train_loader, use_weights=False):
        """
        NOTE: Noise aware loss assumes binary classes
        :param train_loader:
        :param use_weights:
        :return:
        """
        if use_weights and not self.noise_aware:
            class_weights = []
            return nn.CrossEntropyLoss(weight=class_weights)

        elif not self.noise_aware:
            return nn.CrossEntropyLoss()

        return nn.BCEWithLogitsLoss(size_average=False)

    def _get_data_loaders(self, train, dev, test=None, batch_size=4, num_workers=1):
        """
        Initialize dataloaders here so we can tune over batch_size
        :param train:
        :param dev:
        :param test:
        :param batch_size:
        :param num_workers:
        :return:
        """
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        dev_loader   = DataLoader(dev, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader  = None if not test else DataLoader(test, batch_size=batch_size,
                                                       shuffle=False, num_workers=num_workers)

        return train_loader, dev_loader, test_loader

    def _scorer(self, y_true, y_pred_prob, y_pred, name, binary=False, pos_label=1):
        """
        Print performance metrics
        :param y_true:
        :param y_pred:
        :param name:
        :return:
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob, pos_label=pos_label)
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        roc_score = auc(fpr, tpr) * 100.0
        prc_score = auc(recall, precision) * 100.0
        logloss   = log_loss(y_true, y_pred_prob)

        average = "binary" if binary else "micro"
        precision = precision_score(y_true, y_pred, average=average) * 100.0
        recall    = recall_score(y_true, y_pred, average=average) * 100.0
        f1        = f1_score(y_true, y_pred, average=average) * 100.0
        acc       = accuracy_score(y_true, y_pred) * 100.0

        msg = "{:<6} log loss: {:2.3f} | ROC: {:>5} | PRC: {:>5} | accuracy: {:>5} | P/R/F1: {:>5}"
        prf1 = "{:>5} / {:>5} / {:>5}".format("%2.1f" % precision, "%2.1f" % recall, "%2.1f" % f1)
        return msg.format("  [%s]" % name, logloss, 
                          "%2.1f" % roc_score, 
                          "%2.1f" % prc_score, 
                          "%2.1f" % acc, prf1)

    def _evaluate(self, model, data_loader, name, binary=False, threshold=0.5, tune_metric='roc_auc_score'):
        """
        Generate label predictions
        :param model:
        :param data_loader:
        :param name:
        :return:
        """
        # save rng state, seed for deterministic evaluation
        rng_gpu = torch.cuda.random.get_rng_state_all()
        rng_cpu = torch.random.get_rng_state()
        torch.cuda.random.manual_seed_all(self.seed)
        torch.random.manual_seed(self.seed)

        y_true = np.hstack([y.numpy() for x,y in data_loader])
        y_prob, y_pred = model.predict(data_loader, binary=binary, pos_label=1, 
                                       threshold=threshold, return_proba=True)

        if name == "DEV":
            threshold = tune_threshold(y_true, y_prob, tune_metric)
            logger.info("Tuned threshold: {:.2f}".format(threshold))
            if binary:
                y_pred = np.array([1 if p > threshold else 0 for p in y_prob])
            else:
                y_pred = np.argmax(y_prob, 1)

        msg = self._scorer(y_true, y_prob, y_pred, name, binary)
        logger.info(msg)

        # restore rng state to all devices
        torch.cuda.set_rng_state_all(rng_gpu)
        torch.random.set_rng_state(rng_cpu)

        return y_true, y_pred, y_prob, threshold



class GridSearchTrainer(Trainer):

    def __init__(self, model_class, model_class_params, param_grid, n_model_search, 
                 noise_aware=False, use_cuda=False, seed=1234):
        """
        Single-threaded random grid search
        :param model_class:
        :param model_class_params:
        :param param_grid:
        :param n_model_search:
        :param seed:
        """
        super(GridSearchTrainer, self).__init__(model_class, model_class_params, 
                                                noise_aware=noise_aware, use_cuda=use_cuda,
                                                seed=seed)
        # use fixed random seed for consistent parameter grid
        self.rng            = np.random.RandomState(1234)
        self.param_grid     = param_grid
        self.param_names    = [name for name in self.param_grid]
        self.n_model_search = n_model_search

    @timeit
    def fit(self, train, dev, test=None, update_freq=5, checkpoint_freq=10, checkpoint_dir=".",
              metric='roc_auc_score', **kwargs):
        """
        Random grid search
        :param train:
        :param dev:
        :param test:
        :param update_freq:
        :param checkpoint_freq:
        :param checkpoint_dir:
        :param metric:  scikit-learn metric (function or str) or custom score function
        :param kwargs:
        :return:
        """
        hyperparams = self.get_hyperparams(self.n_model_search)
        self._print_parameter_space()
        metric = getattr(metrics, metric) if type(metric) is str else metric

        scores = {}
        curr_best = -0.01
        tuned_threshold = 0.5
        best_model = None
        for i, params in enumerate(hyperparams):
            params = dict(zip(self.param_names, params))
            model_name = "MODEL [{}]".format(i)
            logger.info(model_name)

            # override any grid search params
            params.update(kwargs)
            logger.info(params)

            fit_time, model, score, dev_threshold = super(GridSearchTrainer, self).fit(
                train=train, dev=dev, test=test, update_freq=update_freq, metric=metric,
                checkpoint_freq=checkpoint_freq, checkpoint_dir=checkpoint_dir, **params
            )
            scores[model_name] = [score, model_name, params]

            if score > curr_best:
                curr_best = score
                tuned_threshold = dev_threshold
                best_model = {
                    'model': copy.deepcopy(model),
                    'state_dict': copy.deepcopy(model.state_dict()),
                    'best_score': curr_best,
                    'params': copy.deepcopy(params),
                    'threshold': copy.deepcopy(tuned_threshold)
                }
                checkpoint_name = "{}_{}".format(model.name, i)
                logger.info("NEW BEST: {} {}".format(metric.__name__, curr_best))
                self.save(best_model, checkpoint_dir, checkpoint_name)
                self.save(best_model, checkpoint_dir, "{}_BEST".format(model.name))
            logger.info(("#" * 90) + "\n")
            del model
            torch.cuda.empty_cache()

        # print performance summary
        logger.info("=" * 90)
        self._print_grid_search_summary(scores, metric)
        logger.info("Best [DEV] score: {} {}\n".format(metric.__name__, curr_best))
        model = best_model['model']
        logger.info(model)
        logger.info("=" * 90)

        return model, curr_best, tuned_threshold

    def search_space(self):
        """
        Get full parameter grid
        :return:
        """
        return product(*[self.param_grid[name] for name in self.param_grid])

    def get_hyperparams(self, n_model_search=5):
        """
        Fetch n_model_search parameter sets
        :param n_model_search:
        :return:
        """
        ss = sorted(list(set(self.search_space())))
        n = min(len(ss), n_model_search)
        self.rng.seed(1234)
        self.rng.shuffle(ss)
        return ss[0:n]


    ################################################################################
    # INTERNAL
    ################################################################################

    def _print_grid_search_summary(self, scores, metric):
        """ Print sorted grid search results """
        df = []
        for score, name, params in sorted(scores.values(), reverse=1):
            params.update({metric.__name__: score, "model": name})
            df.append(params)
        print(pd.DataFrame(df))

    def _print_parameter_space(self):
        """
        Show full hyperparameter search space
        :param name:
        :return:
        """
        name = "{}".format(self.model_class.__name__)
        ss = self.get_hyperparams(self.n_model_search)
        logger.info("=" * 90)
        n,N = len(ss), len(list(self.search_space()))
        logger.info("Model Parameter Space {}/{} {:2.2f}% (grid search seed={}):".format(n, N, float(n)/N * 100, 1234))
        logger.info("=" * 90)
        padding = 10
        for i, params in enumerate(ss):
            param_set = [": ".join(x) for x in zip(self.param_names, map(str, params))]
            tmpl = ("{:<" + str(padding) + "} ") * len(param_set)
            logger.info("{}_{} | {}".format(name, i, tmpl.format(*param_set)))
        logger.info("")
