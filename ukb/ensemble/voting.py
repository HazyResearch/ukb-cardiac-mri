import glob
import numpy as np
import pandas as pd
from collections import OrderedDict
#from . import metrics
import metrics
from .csv_reader import csv_node

__all__ = ['tune_threshold',
           'assemble_node',
           'assemble_dev_threshold',
           'metric_reading',
           'Ensemble']

def tune_threshold(y_true, y_prob, metric="f1_score"):
    if isinstance(metric, str):
        metric = getattr(metrics, metric)

    thresholds = np.arange(0.01, 1, 0.01)
    best_score = 0.0
    best_threshold = 0.5
    for threshold in thresholds:
        y_pred = np.array([1 if p > threshold else 0 for p in y_prob])
        cur_score = metric(y_true, y_pred)
        if cur_score > best_score:
            best_score = cur_score
            best_threshold = threshold

    print("Tuned threshold: {:.4f}".format(best_threshold))
    return best_threshold


def assemble_node(nodes, key="Y_PROBA", method="median", PIDs=None):
    if isinstance(method, str):
        method = getattr(np, method)

    if PIDs is None:
        PIDs = nodes[0].PID

    probas = []
    for pid in PIDs:
        proba = method([x.data[pid][key] for x in nodes])
        probas.append(proba)

    return np.array(probas)


def assemble_dev_threshold(nodes, method="median", metric="f1_score", PIDs=None):
    y_prob = assemble_node(nodes, key="Y_PROBA", method=method, PIDs=PIDs)
    y_true = nodes[0].extract("Y_TRUE", PIDs)
    threshold = tune_threshold(y_true, y_prob, metric)

    return threshold

def metric_reading(y_true, y_pred, y_proba):
    if isinstance(y_true, list):
        readings = [metric_reading(y_true_, y_pred_, y_proba_) 
                    for y_true_,y_pred_,y_proba_ in zip(y_true, y_pred, y_proba)]
        return readings
    else:
        scores = metrics.classification_summary(y_true, y_pred, [0,1], y_proba, verbose=False)
        reading = OrderedDict([('Pos.Acc',scores['pos_acc']*100.0),
                               ('Neg.Acc',scores['neg_acc']*100.0),
                               ('Precision',scores['precision']*100.0),
                               ('Recall',scores['recall']*100.0),
                               ('F1',scores['f1']*100.0),
                               ('ROC',scores['roc']*100.0),
                               ('PRC',scores['prc']*100.0),
                               ('NDCG',scores['ndcg']*100.0),
                               ('TP',scores['tp']),
                               ('FP',scores['fp']),
                               ('TN',scores['tn']),
                               ('FN',scores['fn'])])
        return reading



class Ensemble(object):
    def __init__(self, results_csvs, dev_csvs, pids=None):
        self.results_csvs = results_csvs
        self.dev_csvs = dev_csvs
        self.build(pids)
    
    @classmethod
    def from_keyword(klass, test_keyword, dev_keyword, pids=None):
        test_csvs = glob.glob(test_keyword, recursive=True)
        dev_csvs = glob.glob(dev_keyword, recursive=True)
        return klass(test_csvs, dev_csvs, pids)

    @classmethod
    def from_folder(klass, results_folder, dev_folder, pids=None):
        results_csvs = glob.glob("{}/**/predictions*.csv".format(results_folder), recursive=True)
        dev_csvs = glob.glob("{}/**/predictions*.csv".format(dev_folder), recursive=True)
        return klass(results_csvs, dev_csvs, pids)

    def build(self, pids=None):
        self.results = [csv_node.from_csv(x) for x in self.results_csvs]
        self.devs = [csv_node.from_csv(x) for x in self.dev_csvs]
        self.results = sorted(self.results, key=lambda x: x.seed)
        self.devs = sorted(self.devs, key=lambda x: x.seed)
        if pids is None:
            self.pids = list(self.results[0].PID)
        else:
            self.pids = pids
        try:
            self.score_list = self.get_seeds_score_list()
            self.score = True
        except:
            self.score = False
        self.proba_list = self.get_seeds_proba_list()
        self.pred_list = self.get_seeds_pred_list()

    @property
    def score_dataframe(self):
        return pd.DataFrame(OrderedDict(self.score_list_head+self.score_list))

    @property
    def proba_dataframe(self):
        return pd.DataFrame(OrderedDict(self.proba_list_head+self.proba_list))

    @property
    def pred_dataframe(self):
        return pd.DataFrame(OrderedDict(self.pred_list_head+self.pred_list))

    def get_df_by_seed(self, key="Y_PROBA"):
        seeds = [x.seed for x in self.results]
        probas = [x.extract(key, self.pids) for x in self.results]
        df_dict = OrderedDict([("PID", self.pids)] + \
                              [("SEED_{}".format(seed), proba) for seed, proba in zip(seeds, probas)])
        df = pd.DataFrame(df_dict)

        return df

    def get_score_by_seed(self, seed=0):
        idx     = [x.seed for x in self.results].index(seed)
        node    = self.results[idx]

        y_true  = node.extract("Y_TRUE")
        y_pred  = node.extract("Y_PRED")
        y_proba = node.extract("Y_PROBA")
        score   = metric_reading(y_true, y_pred, y_proba)

        return score

    def score2pair(self, key, score):
        val = ["{:.2f}".format(score[key]) for key in self.score_keys]
        return (key, val)
        

    def get_seeds_score_list(self):
        seeds   = [x.seed for x in self.results]
        scores  = [self.get_score_by_seed(x) for x in seeds]

        self.score_keys = list(scores[0].keys())
        self.score_list_head = [("Experiment", self.score_keys)]
        df_list = []
        
        for seed, score in zip(seeds, scores):
            pair = self.score2pair("SEED_{}".format(seed), score)
            df_list.append(pair)

        mean_score = OrderedDict([(key, np.mean([score[key] for score in scores])) for key in self.score_keys])
        std_score = OrderedDict([(key, np.std([score[key] for score in scores])) for key in self.score_keys])
        df_list.append(self.score2pair("AVERAGE", mean_score))
        df_list.append(self.score2pair("STD", std_score))

        return df_list

    def get_seeds_proba_list(self):
        seeds   = [x.seed for x in self.results]
        probas  = [x.extract("Y_PROBA", self.pids) for x in self.results]
        self.proba_list_head = [("PID", self.pids)]
        proba_list = [("SEED_{}".format(seed), proba) for seed, proba in zip(seeds, probas)]

        return proba_list
        
    def get_seeds_pred_list(self):
        seeds   = [x.seed for x in self.results]
        preds  = [x.extract("Y_PRED", self.pids) for x in self.results]
        self.pred_list_head = [("PID", self.pids)]
        pred_list = [("SEED_{}".format(seed), pred) for seed, pred in zip(seeds, preds)]

        return pred_list

    def median_vote(self, metric="f1_score"):
        dev_threshold = assemble_dev_threshold(self.devs, method="median", 
                                               metric=metric, PIDs=self.devs[0].PID)
        voted_y_proba = assemble_node(self.results, key="Y_PROBA", 
                                      method="median", PIDs=self.pids)
        voted_y_pred = np.array([1 if p > dev_threshold else 0 for p in voted_y_proba])
        y_true = self.results[0].extract("Y_TRUE", self.pids)
        #df_dict = OrderedDict([("PID", self.pids),
        #                       ("Y_PROBA", voted_y_proba),
        #                       ("Y_PRED", voted_y_pred)])
        #df = pd.DataFrame(df_dict)
        proba_pair = ("MEDIAN", voted_y_proba)
        self.proba_list.append(proba_pair)
        proba_df = pd.DataFrame(OrderedDict(self.proba_list_head+[proba_pair]))

        pred_pair = ("MEDIAN", voted_y_pred)
        self.pred_list.append(pred_pair)
        pred_df = pd.DataFrame(OrderedDict(self.pred_list_head+[pred_pair]))

        if self.score:
            score = metric_reading(y_true, voted_y_pred, voted_y_proba)
            score_pair = self.score2pair("MEDIAN", score)
            self.score_list.append(score_pair)
            score_df = pd.DataFrame(OrderedDict(self.score_list_head+[score_pair]))
        else:
            score_df = None

        return proba_df, pred_df, score_df



    def mv_vote(self):
        voted_y_proba = assemble_node(self.results, key="Y_PRED", 
                                      method="mean", PIDs=self.pids)
        voted_y_pred  = np.round(voted_y_proba)
        y_true = self.results[0].extract("Y_TRUE", self.pids)

        proba_pair = ("MV", voted_y_proba)
        self.proba_list.append(proba_pair)
        proba_df = pd.DataFrame(OrderedDict(self.proba_list_head+[proba_pair]))

        pred_pair = ("MV", voted_y_pred)
        self.pred_list.append(pred_pair)
        pred_df = pd.DataFrame(OrderedDict(self.pred_list_head+[pred_pair]))

        if self.score:
            score = metric_reading(y_true, voted_y_pred, voted_y_proba)
            score_pair = self.score2pair("MV", score)
            self.score_list.append(score_pair)
            score_df = pd.DataFrame(OrderedDict(self.score_list_head+[score_pair]))
        else:
            score_df = None

        return proba_df, pred_df, score_df


