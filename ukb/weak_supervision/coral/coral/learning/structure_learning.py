from numba import jit
import numpy as np
import random
from numbskull.udf import *
from numbskull.numbskulltypes import *


class CoralDependencySelector(object):
    """
    Fast method for identifying dependencies among labeling functions.

    :param seed: seed for initializing state of Numbskull variables
    """
    def __init__(self, seed=271828):
        self.rng = random.Random()
        self.rng.seed(seed)

    def select(self, V, cardinality, L, UDF_SET, threshold=0.05, truncation=10):
        """
        Identifies a dependency structure among labeling functions for a given data set.

        By default searches for correlations, i.e., the DEP_SIMILAR dependency type.

        :param L: labeling function output matrix
        :param higher_order: bool indicating whether to additionally search for higher order
                             fixing and reinforcing dependencies (DEP_FIXING and DEP_REINFORCING)
        :param propensity: bool indicating whether to include LF propensity dependencies during learning
        :param threshold: minimum magnitude weight a dependency must have to be returned (in log scale), also
                          regularization strength
        :param truncation: number of iterations between truncation step for regularization
        :return: collection of tuples of the format (LF 1 index, LF 2 index, dependency type),
                 see snorkel.learning.constants
        """

        n_data = V.shape[0]
        n_vocab = V.shape[1]
        n_lf = len(L)

        # Initializes data structures
        deps = set()
        n_weights = n_lf + n_vocab
        weights = np.zeros((n_weights,))
        joint = np.zeros((6,))
        # joint[0] = P(Y = -1, V_j = 0)
        # joint[1] = P(Y = -1, V_j = 1)
        # joint[2] = P(Y = -1, V_j = 2)
        # joint[3] = P(Y =  1, V_j = 0)
        # joint[4] = P(Y =  1, V_j = 1)
        # joint[5] = P(Y =  1, V_j = 2)

        Lstart = np.cumsum(np.array([0] + [len(l) for l in L]))
        L = [item for sublist in L for item in sublist]
        for j in range(n_vocab):
            ## Initializes weights
            # Accuracy
            for k in range(n_lf):
                weights[k] = 1.1 - .2 * self.rng.random()
            # Similarity
            for k in range(n_lf, len(weights)):
                weights[k] = 0.0

            _fit_deps(n_data, n_lf, n_vocab, j, V, cardinality, L, Lstart, UDF_USAGE[UDF_SET], weights, joint, threshold, truncation)

            for k in range(n_vocab):
                if abs(weights[n_lf + k]) > threshold:
                    deps.add((j, k) if j < k else (k, j))

        return deps


@jit(nopython=True, cache=True, nogil=True)
def eval_udf(i, udf_index, V, L, var_samp, value):
    var_copy = 0
    var_value = V[i:(i + 1), :]
    fmap = np.empty(len(L), FactorToVar)
    for i in range(len(L)):
        fmap[i]["vid"] = L[i]
        fmap[i]["dense_equal_to"] = 0 # not used
    ftv_start = 0
    return udf(udf_index, var_samp, value, var_copy, var_value, fmap, ftv_start)


@jit(nopython=True, cache=True, nogil=True)
def _fit_deps(n_data, n_lf, n_vocab, j, V, cardinality, L, Lstart, udf, weights, joint, regularization, truncation):
    step_size = 1.0 / n_data
    epochs = 100
    l1delta = regularization * step_size * truncation

    for t in range(epochs):
        for i in range(n_data):
            # Processes a training example
            # First, computes joint and conditional distributions
            joint[:] = 0, 0, 0, 0, 0, 0
            for k in range(n_lf):
                # Accuracy
                for value in range(cardinality[j]):
                    u = eval_udf(i, udf[k], V, L[Lstart[k]:Lstart[k + 1]], j, value)
                    joint[0 + value] += weights[k] * -1 * u
                    joint[3 + value] += weights[k] * +1 * u
            for k in range(n_vocab):
                # Similarity
                if j != k:
                    if cardinality[j] == cardinality[k]:
                        joint[0 + V[i, k]] += weights[n_lf + k]
                        joint[3 + V[i, k]] += weights[n_lf + k]
                    elif cardinality[j] == 2 and cardinality[k] == 3:
                        if V[i, k] == 0:
                            joint[0] += weights[n_lf + k]
                            joint[3] += weights[n_lf + k]
                        elif V[i, k] == 1:
                            pass
                        elif V[i, k] == 2:
                            joint[1] += weights[n_lf + k]
                            joint[4] += weights[n_lf + k]
                    elif cardinality[j] == 3 and cardinality[k] == 2:
                        if V[i, k] == 0:
                            joint[0] += weights[n_lf + k]
                            joint[3] += weights[n_lf + k]
                        elif V[i, k] == 1:
                            joint[2] += weights[n_lf + k]
                            joint[5] += weights[n_lf + k]
                    else:
                        raise ValueError("cardinality not valid")
            if cardinality[j] == 2:
                joint[2] = -np.inf
                joint[5] = -np.inf


            joint = np.exp(joint)
            joint /= np.sum(joint)

            marginal_pos = np.sum(joint[3:6])
            marginal_neg = np.sum(joint[0:3])

            conditional_pos = joint[3 + V[i, j]] / (joint[0 + V[i, j]] + joint[3 + V[i, j]])
            conditional_neg = joint[0 + V[i, j]] / (joint[0 + V[i, j]] + joint[3 + V[i, j]])

            # Second, takes likelihood gradient step

            for k in range(n_lf):
                for value in range(cardinality[j]):
                    # decrease marginal
                    u = eval_udf(i, udf[k], V, L[Lstart[k]:Lstart[k + 1]], j, value)
                    weights[k] -= step_size * joint[0 + value] * -1 * u
                    weights[k] -= step_size * joint[3 + value] * +1 * u

                # increase conditional
                value = V[i, j]
                u = eval_udf(i, udf[k], V, L[Lstart[k]:Lstart[k + 1]], j, value)
                weights[k] += step_size * -1 * u * conditional_neg
                weights[k] += step_size * +1 * u * conditional_pos

            for k in range(n_vocab):
                # Similarity
                if j != k:
                    if cardinality[j] == cardinality[k]:
                        weights[n_lf + k] -= step_size * (joint[0 + V[i, k]] + joint[3 + V[i, k]])

                        if V[i, j] == V[i, k]:
                            weights[n_lf + k] += step_size
                    elif cardinality[j] == 2 and cardinality[k] == 3:
                        if V[i, k] == 0:
                            weights[n_lf + k] -= step_size * (joint[0] + joint[3])
                        if V[i, k] == 2:
                            weights[n_lf + k] -= step_size * (joint[1] + joint[4])

                        if (V[i, j] == 0 and V[i, k] == 0) or (V[i, j] == 1 and V[i, k] == 2):
                            weights[n_lf + k] += step_size
                    elif cardinality[j] == 3 and cardinality[k] == 2:
                        if V[i, k] == 0:
                            weights[n_lf + k] -= step_size * (joint[0] + joint[3])
                        if V[i, k] == 1:
                            weights[n_lf + k] -= step_size * (joint[2] + joint[5])

                        if (V[i, j] == 0 and V[i, k] == 0) or (V[i, j] == 2 and V[i, k] == 1):
                            weights[n_lf + k] += step_size
                    else:
                        raise ValueError("cardinality not valid")


            # Third, takes regularization gradient step
            if (t * n_data + i) % truncation == 0:
                # do not regularize accuracy
                # only regularize dependencies
                for k in range(n_lf, len(weights)):
                    weights[k] = max(0, weights[k] - l1delta) if weights[k] > 0 else min(0, weights[k] + l1delta)