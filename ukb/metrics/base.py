from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, precision_score, recall_score, \
     accuracy_score, confusion_matrix, classification_report, log_loss, \
     roc_auc_score, roc_curve, precision_recall_curve, auc


__all__ = ["binary_scores_from_counts",
           "print_metricatk",
           "print_scores",
           "classification_summary",
           "prc_auc_score",
           "dcg_score",
           "ndcg_score",
           "ndcg_score2",
           "f1_score",
           "precision_score",
           "recall_score",
           "accuracy_score",
           "confusion_matrix",
           "classification_report",
           "log_loss",
           "roc_auc_score",
           "roc_curve",
           "precision_recall_curve",
           "auc"]

# wrappers for using data loaders to compute standard metrics


def binary_scores_from_counts(ntp, nfp, ntn, nfn):
    """
    Precision, recall, and F1 scores from counts of TP, FP, TN, FN.
    Example usage:
        p, r, f1 = binary_scores_from_counts(*map(len, error_sets))
    """
    prec = ntp / float(ntp + nfp) if ntp + nfp > 0 else 0.0
    rec  = ntp / float(ntp + nfn) if ntp + nfn > 0 else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def print_metricatk(y_true, y_pred, y_proba):
    """
    print out the F1/Precision/Recall at k=5,10..
    """
    sorted_indexes = np.argsort(y_proba)

    print("========================================")
    print("Metric at K (5, 10, ...)")
    print("========================================")
    for k in range(5, y_true.shape[0], 5):
        target          = sorted_indexes[-k:]
        prec            = y_true[target].sum()/float(k)
        rec             = y_true[target].sum()/float(y_true.sum())
        f1   = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0.0
        print("At {:>4}: | Precision: {:5.1f} | Recall: {:5.1f} | F1: {:5.1f}".format(k, prec*100.0, rec*100.0, f1*100.0))
        if rec == 1:
            break

def print_scores(ntp, nfp, ntn, nfn,
                 pos_acc, neg_acc, 
                 prec, rec, f1, roc, prc,
                 ndcg, title='Scores'):
    print("========================================")
    print(title)
    print("========================================")
    print("Pos. class accuracy: {:2.1f}".format(pos_acc * 100))
    print("Neg. class accuracy: {:2.1f}".format(neg_acc * 100))
    print("----------------------------------------")
    print("AUC:                 {:2.1f}".format(roc * 100))
    print("PRC:                 {:2.1f}".format(prc * 100))
    print("NDCG:                {:2.1f}".format(ndcg * 100))
    print("----------------------------------------")
    print("Precision:           {:2.1f}".format(prec * 100))
    print("Recall:              {:2.1f}".format(rec * 100))
    print("F1:                  {:2.1f}".format(f1 * 100))
    print("----------------------------------------")
    print("TP: {} | FP: {} | TN: {} | FN: {}".format(ntp, nfp, ntn, nfn))
    print("========================================\n")


def classification_summary(y_true, y_pred, classes, y_proba, verbose=True):
    """
    Assumes binary classification

    :param model:
    :param data_loader:
    :return:
    """
    #print_metricatk(y_true, y_pred, y_proba)
    roc = roc_auc_score(y_true, y_proba)
    prc = prc_auc_score(y_true, y_proba)
    if len(classes) <= 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        # compute metrics
        prec, rec, f1 = binary_scores_from_counts(tp, fp, tn, fn)
        pos_acc = tp / float(tp + fn) if tp + fn > 0 else 0.0
        neg_acc = tn / float(tn + fp) if tn + fp > 0 else 0.0
        ndcg = ndcg_score(y_true, y_proba)
        if verbose:
            print_scores(tp, fp, tn, fn, pos_acc, neg_acc, prec, rec, f1, roc, prc, ndcg)
        header = ["ndcg", "roc", "prc", "precision", "recall", "f1", "pos_acc", "neg_acc", "tp", "fp", "tn", "fn"]
        return dict(zip(header,(ndcg, roc, prc, prec, rec, f1, pos_acc, neg_acc, tp, fp, tn, fn)))
    else:
        print(classification_report(y_true, y_pred, target_names=classes, digits=3))
        return {}


def prc_auc_score(y_true, y_prob):
    """
    Precision-Recall-Curve Area-Under-Score
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    prc_auc = auc(recall, precision)
    return prc_auc



def dcg_score(y_true, y_score, k=None):
    """
    Function for Discounted Cumulative Gain
    """
    k = len(y_true) if k is None else k
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

def ndcg_score(y_true, y_score, k=None):
    """
    Function for Normalized Discounted Cumulative Gain
    """
    y_true, y_score = np.squeeze(y_true), np.squeeze(y_score)
    k = len(y_true) if k is None else k

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score have different shapes.")

    IDCG    = dcg_score(y_true, y_true)
    DCG     = dcg_score(y_true, y_score)

    return DCG/IDCG


def ndcg_score2(y_true, y_score, k=2):
    """
    Function for Normalized Discounted Cumulative Gain
    Only accepts if y_score is shaped [n_samples, n_classes]
    """
    y_true, y_score = np.array(y_true), np.array(y_score)
    if y_true.ndim == 1:
        y_true = np.expand_dims(y_true, axis=-1)

    enc = OneHotEncoder(sparse=False)
    oneHot_y_true = enc.fit_transform(y_true)

    if oneHot_y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score have different value ranges")

    scores = []

    # Iterate over each y_value_true and compute the DCG score
    for y_value_true, y_value_score in zip(oneHot_y_true, y_score):
        actual = dcg_score(y_value_true, y_value_score, k)
        best = dcg_score(y_value_true, y_value_true, k)
        scores.append(actual / best)

    return np.mean(scores)


