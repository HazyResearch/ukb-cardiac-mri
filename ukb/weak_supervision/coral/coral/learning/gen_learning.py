from numbskull import NumbSkull
from numbskull.inference import *
from numbskull.numbskulltypes import Weight, Variable, Factor, FactorToVar
from numbskull.udf import *
import numpy as np
import random

class CoralModel(object):
    def __init__(self, class_prior=False, lf_prior=False, lf_propensity=False, lf_class_propensity=False, seed=271828):
        self.class_prior = class_prior
        self.lf_prior = lf_prior
        self.lf_propensity = lf_propensity
        self.lf_class_propensity = lf_class_propensity
        self.weights = None

        self.rng = random.Random()
        self.rng.seed(seed)

    def train(self, V, cardinality, L, L_offset, y=None, deps=(), init_acc = 1.0, init_deps=0.0, init_class_prior=-1.0, epochs=100, step_size=None, decay=0.99, reg_param=0.1, reg_type=2, verbose=False,
              truncation=10, burn_in=50, timer=None):

        n_data = V.shape[0]
        step_size = step_size or 1.0 / n_data
        reg_param_scaled = reg_param / n_data
        # self._process_dependency_graph(L, deps)
        weight, variable, factor, ftv, domain_mask, n_edges = self._compile(V, cardinality, L, L_offset, y, deps, init_acc, init_deps) # , init_deps, init_class_prior)

        fg = NumbSkull(n_inference_epoch=0, n_learning_epoch=epochs, stepsize=step_size, decay=decay,
                       reg_param=reg_param_scaled, regularization=reg_type, truncation=truncation,
                       quiet=(not verbose), verbose=verbose, learn_non_evidence=True, burn_in=burn_in)
        fg.loadFactorGraph(weight, variable, factor, ftv, domain_mask, n_edges)

        if timer is not None:
            timer.start()
        fg.learning(out=False)
        if timer is not None:
            timer.end()

        self.weights = fg.factorGraphs[0].weight_value[0][:len(L)]
        self.dep_weights = fg.factorGraphs[0].weight_value[0][len(L):]
        self.lf_accuracy = 1. / (1. + np.exp(-self.weights[:len(L)]))
        # self._process_learned_weights(L, fg)

    def marginals(self, V, cardinality, L, L_offset, deps=(), init_acc = 1.0, init_deps=1.0, init_class_prior=-1.0, epochs=100, step_size=None, decay=0.99, verbose=False,
              burn_in=50, timer=None):
        if self.weights is None:
            raise ValueError("Must fit model with train() before computing marginal probabilities.")

        y = None
        weight, variable, factor, ftv, domain_mask, n_edges = self._compile(V, cardinality, L, L_offset, y, deps, self.weights, self.dep_weights)

        fg = NumbSkull(n_inference_epoch=epochs, n_learning_epoch=0, stepsize=step_size, decay=decay,
                       quiet=(not verbose), verbose=verbose, learn_non_evidence=True, burn_in=burn_in,
                       sample_evidence=False)
        fg.loadFactorGraph(weight, variable, factor, ftv, domain_mask, n_edges)


        fg.inference(out=False)
        marginals = fg.factorGraphs[0].marginals[:V.shape[0]]

        return marginals

    def _compile(self, V, cardinality, L, L_offset, y, deps, init_acc, init_deps):
        """
        Compiles a generative model based on L and the current labeling function dependencies.
        """

        ### Error Checking ###

        # Check L_offset is valid
        index = np.flatnonzero(UdfStart == L_offset)
        if len(index) == 0:
            raise ValueError("L_offset " + str(L_offset) + " does not correspond to a known application")
        if len(index) > 1:
            raise ValueError("L_offset " + str(L_offset) + " found multiple times")
        index = index[0]

        # Check L is the right size
        if len(L) != LfCount[index]:
            raise ValueError("Wrong number of LFs passed: (" + str(len(L)) + " given and " + str(LfCount[index]) + " in udf.py)")

        # Check cardinality of each LF is right
        for i in range(len(L)):
            if len(L[i]) != UdfCardinality[UdfCardinalityStart[index] + i]:
                raise ValueError("LF " + str(i) + " has the wrong cardinality: (" + str(len(L[i])) + " given and " + str(UdfCardinality[UdfCardinalityStart[index] + i]) + " in udf.py)")

        # Check that there are enough vocab terms
        for i in range(len(L)):
            for j in range(len(L[i])):
                if L[i][j] >= V.shape[1]:
                    raise ValueError("LF " + str(i) + " uses vocab " + str(L[i][j]) + " when there are only " + str(V.shape[1]) + " terms")


        ### Set up factor graph ###

        n_data = V.shape[0]
        n_vocab = V.shape[1]
        n_lf = len(L)

        n_weights = n_lf + len(deps)
        n_vars = n_data * (n_vocab + 1)
        n_factors = n_data * n_weights
        n_edges = n_data * (sum([len(l) + 1 for l in L]) + 2 * len(deps))

        weight = np.zeros(n_weights, Weight)
        variable = np.zeros(n_vars, Variable)
        factor = np.zeros(n_factors, Factor)
        ftv = np.zeros(n_edges, FactorToVar)
        domain_mask = np.zeros(n_vars, np.bool)

        #
        # Compiles weight matrix
        #
        for i in range(n_weights):
            weight[i]['isFixed'] = False
            if i < n_lf:
                if type(init_acc) == int or type(init_acc) == float:
                    weight[i]['initialValue'] = np.float64(init_acc)
                else:
                    weight[i]['initialValue'] = init_acc[i]
            else:
                if type(init_deps) == int or type(init_deps) == float:
                    weight[i]['initialValue'] = np.float64(init_deps)
                else:
                    weight[i]['initialValue'] = init_deps[i - n_lf]

        #
        # Compiles variable matrix
        #

        # True Label y
        for i in range(n_data):
            variable[i]['isEvidence'] = False if (y is None) else True
            variable[i]['initialValue'] = self.rng.randrange(0, 2) if (y is None) else (1 if y[i] == 1 else 0)
            variable[i]["dataType"] = 0
            variable[i]["cardinality"] = 2

        # Vocabulary
        for i in range(n_data):
            for j in range(n_vocab):
                variable[n_data + i * n_vocab + j]["isEvidence"] = True
                variable[n_data + i * n_vocab + j]["initialValue"] = V[i, j]
                variable[n_data + i * n_vocab + j]["dataType"] = 0
                variable[n_data + i * n_vocab + j]["cardinality"] = cardinality[j]
                if V[i, j] >= cardinality[j]:
                    raise ValueError("Vocab " + str(j) + " contains " + str(V[i, j]) + " even though it has a cardinality of " + str(cardinality[j]))

        #
        # Compiles factor and ftv matrices
        #
        index = 0
        # Accuracy
        for i in range(n_data):
            for j in range(n_lf):
                factor[i * n_lf + j]["factorFunction"] = L_offset + j
                factor[i * n_lf + j]["weightId"] = j
                factor[i * n_lf + j]["featureValue"] = 1.0
                factor[i * n_lf + j]["arity"] = len(L[j]) + 1
                factor[i * n_lf + j]["ftv_offset"] = index
                for k in range(len(L[j])):
                    ftv[index]["vid"] = n_data + i * n_vocab + L[j][k]
                    ftv[index]["dense_equal_to"] = 0 # not actually used
                    index += 1
                ftv[index]["vid"] = i
                ftv[index]["dense_equal_to"] = 0 # not actually used
                index += 1

        # Dependencies
        for i in range(n_data):
            for j in range(len(deps)):
                factor[n_lf * n_data + i * len(deps) + j]["factorFunction"] = FUNC_CORAL_GEN_DEP_SIMILAR
                factor[n_lf * n_data + i * len(deps) + j]["weightId"] = n_lf + j
                factor[n_lf * n_data + i * len(deps) + j]["featureValue"] = 1.0
                factor[n_lf * n_data + i * len(deps) + j]["arity"] = 2
                factor[n_lf * n_data + i * len(deps) + j]["ftv_offset"] = index

                ftv[index + 0]["vid"] = n_data + i * n_vocab + deps[j][0]
                ftv[index + 0]["dense_equal_to"] = 0 # not actually used
                ftv[index + 1]["vid"] = n_data + i * n_vocab + deps[j][1]
                ftv[index + 1]["dense_equal_to"] = 0 # not actually used
                index += 2

        return weight, variable, factor, ftv, domain_mask, n_edges