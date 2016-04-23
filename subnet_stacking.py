#!/usr/bin/env python
"""The Subnetwork based ensemble of classifiers..."""
import collections
from collections import defaultdict
from copy import deepcopy

import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
import sklearn.base
from scipy.special import logit
from scipy.special import expit
import scipy.optimize
from joblib import Parallel, delayed

__author__ = "Stelios Sfakianakis"
__email__ = "sgsfak@gmail.com"

class SubnetBaseClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, seed_gene, neighbors, delegate_estimator, max_distance=1.):
        """

        Parameters
        ----------
        seed_gene : str
            the gene that we always start expanding from

        neighbors : dict of (gene, distance)
            the list of neighbors with their distance.

        delegate_estimator : a classifier
            Used as the underlying estimator..
        """
        self.seed = seed_gene
        self.neighbors = neighbors
        self.estimator = delegate_estimator #sklearn.base.clone(estimator)
        self.max_distance = max_distance


    def get_nearest_neighbors(self, d=None):
        """Returns the 'neighbors' located at most `d` in distance"""
        if not d:
            d = self.max_distance
        return [g for g,v in self.neighbors.items() if v <= d]

    def get_params(self, deep=True):
        return {"seed_gene": self.seed,
                "neighbors": self.neighbors,
                "delegate_estimator": self.estimator,
                "max_distance": self.max_distance}

    def fit(self, X, y):
        """Trains a classifier using only the nodes (genes) in the subset of the given
        `seed`. The X argument should be a DataFrame in order to select only the cols
        that correspond to the genes of the subnet"""
        nodes = self.get_nearest_neighbors()
        ## We need to take only the columns for the specific nodes in the subnet:
        #### print "----> ", nodes
        X_train = X.loc[:,nodes].values
        self.estimator.fit(X_train, y)
        return self

    def predict_proba(self, X):
        nodes = self.get_nearest_neighbors()
        X_test = X.loc[:,nodes].values
        base_predict = self.estimator.predict_proba(X_test)
        return base_predict

    def predict(self, X):
        nodes = self.get_nearest_neighbors()
        X_test = X.loc[:,nodes].values
        base_predict = self.estimator.predict(X_test)
        return base_predict

class SubnetStackingClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, seeds_with_neighbors, scoring='matthews'):
        self.seeds_with_neighbors = seeds_with_neighbors
        self.scoring = scoring
        if scoring == 'matthews':
            self.scoring = metrics.make_scorer(metrics.matthews_corrcoef)
        self.normalized_seeds_with_neighbors_ = deepcopy(seeds_with_neighbors)
        # The supplied `seed_with_neighbors` contain for each 'neighbor'
        # (i.e. gene) its weight in the induced subnetwork. The "distance"
        # of each neighbor is determined by their -log(1/Sum_i w_i)
        for seed, neighbors in self.normalized_seeds_with_neighbors_.items():
            if seed not in neighbors:
                # Give the seed the most weight:
                m = max(neighbors.values()) + 1
                neighbors[ self.seed ] = m
            s = sum(neighbors.values())
            for n,w in neighbors.items():
                neighbors[n] = np.log(s) - np.log(w) # i.e. -np.log(w/s)
        self.base_clfs_ = [self._create_base_clf(seed) for seed in
                seeds_with_neighbors.keys()]

    def distance_percentiles(self, seed, n=[3, 5, 10, 20, 30]):
        ## We want to retrieve various number of neighbors at different
        # distances. I.e. with the default `n` we should find the
        # distance that contains 3 genes, then the distance that
        # contains 5, etc. For that we use the corresponding percentiles
        # of the distribution of "distances"
        distances = self.normalized_seeds_with_neighbors_[seed].values()
        q = np.array(n) * 100.0 / len(distances)
        return np.percentile(list(distances), q=q)

    def get_params(self, deep=True):
        out = { 'seeds_with_neighbors': self.seeds_with_neighbors,
                #'C' : self.C
                'scoring': self.scoring}
        if deep:
            for c in self.base_clfs_:
                for m,v in c.get_params(deep).items():
                    out["subnet%s__%s" % (c.seed, m)] = v
        return out

    def _features(self):
        return set([n for neighbors in self.seeds_with_neighbors.values()
            for n in neighbors.keys()])

    def _create_base_clf(self, seed, max_distance=1.):
        ## bc = tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=5, class_weight='balanced',max_depth=5)
        bc = linear_model.LogisticRegression(C=1.0)
        neighbors = self.normalized_seeds_with_neighbors_[seed]
        base_clf = SubnetBaseClassifier(seed, neighbors, bc, max_distance=max_distance)
        return base_clf

    def _find_best_distance(self, seed, X, y):
        base_clf = self._create_base_clf(seed)
        tuned_parameters = {'max_distance': self.distance_percentiles(seed)}
        grid_clf = GridSearchCV(base_clf, tuned_parameters, cv=4,
                scoring=self.scoring)
        grid_clf.fit(X,y)
        return grid_clf.best_params_

    def fit(self, X, y):
        if np.unique(y).shape[0] != 2 or np.max(y) != 1:
            raise ValueError("This is 2-class classifier and the (pos) class label should be 1")
        if not hasattr(X, 'iloc'):
            raise ValueError("The input dataset should be a DataFrame!")
        # "Stacked generalization" by Wolpert advices to perform a Leave-Out-One cross validation
        # in order to create the data set for the second level classifier
        base_predictions = defaultdict(list)
        inner_cv = cross_validation.LeaveOneOut(len(y))
        ##X = X_all.loc[:, self._features()]
        # We perform an initial grid search for each subnetwork classifier
        # in order to find the best hyperparameter 'max_distance'
        hyperparams = {g: self._find_best_distance(g, X, y)['max_distance']
                for g in self.normalized_seeds_with_neighbors_.keys()}
        for loo_train_index, loo_test_index in inner_cv:
            element_to_test = loo_test_index[0]
            loo_X_train, loo_y_train = X.iloc[loo_train_index], y[loo_train_index]
            loo_X_test, loo_y_test = X.iloc[loo_test_index], y[loo_test_index]
            ## Create a base classifier from each subnet:
            for seed in self.normalized_seeds_with_neighbors_.keys():
                base_clf = self._create_base_clf(seed, hyperparams[seed])
                base_clf.fit(loo_X_train, loo_y_train)
                base_predict = base_clf.predict_proba(loo_X_test)
                # we are looking for the probability of class '1' and
                # since -1 or 0 < 1, we take the second element of the
                # returned probalities array:
                base_predict = base_predict[0,1]
                base_predictions[element_to_test].append( (seed, base_predict) )
        # now we need to go through their predictions and train the second level classifier
        X2 = np.array([[i[1] for i in v] for tested_elem, v in base_predictions.items()])
        Y2 = np.array([ y[tested_elem] for tested_elem in base_predictions.keys()])
        self.meta_clf_ = ensemble.RandomForestClassifier(n_estimators=200,
                criterion='entropy', max_features=None,
                #max_depth=None, min_samples_split=1,
                max_depth=5, #min_samples_split=1,
                bootstrap=True, oob_score=False, n_jobs=-1)
        self.meta_clf_.fit(X2, Y2)
        self.base_clfs_ = [self._create_base_clf(seed, hyperparams[seed]).fit(X,y)
                for seed in self.normalized_seeds_with_neighbors_.keys()]
        return self

    def predict(self, X):
        base_predictions = np.array([ base_clf.predict_proba(X)[:,1]
            for base_clf in self.base_clfs_]).transpose()
        return self.meta_clf_.predict(base_predictions)

    def predict_proba(self, X):
        base_predictions = np.array([ base_clf.predict_proba(X)[:,1]
            for base_clf in self.base_clfs_]).transpose()
        return self.meta_clf_.predict_proba(base_predictions)


class GeneSubsetTransformer(sklearn.base.TransformerMixin):
    def __init__(self, features):
        self.features = features
    def fit(self, X,y):
        if not hasattr(X, 'iloc'):
            raise ValueError("The input dataset should be a DataFrame!")
        return self
    def transform(self, X):
        return X.loc[:,self.features]



def measure_classification(clf, X_test, y_true, pos_label=1):
    y_pred = clf.predict(X_test)
    tp = np.sum(y_pred[np.array(y_true)==pos_label] == pos_label)
    tn = np.sum(y_pred[np.array(y_true)!=pos_label] != pos_label)
    fp = np.sum(y_pred[np.array(y_true)!=pos_label] == pos_label)
    fn = np.sum(y_pred[np.array(y_true)==pos_label] != pos_label)
    scores = {name: metric(y_true, y_pred) for name, metric in
            [('precision', metrics.precision_score), # or Positive Predictive value
                ('recall', metrics.recall_score), # Sensitivity or True Positive rate
                ('accuracy', metrics.accuracy_score),
                ('matthews_corrcoef', metrics.matthews_corrcoef),
                ('average_precision', metrics.average_precision_score),
                ('roc_auc', metrics.roc_auc_score),
                ('f1', metrics.f1_score)]}
    scores['_true_pos'] = tp
    scores['_true_neg'] = tn
    scores['_false_neg'] = fn
    scores['_false_pos'] = fp
    scores['specificity'] = tn * 1. / (tn + fp) # True Negative rate
    scores['balanced_accuracy'] = (scores['recall'] + scores['specificity'])/2.0 # mean of TPR and TNR
    scores['informedness'] = scores['recall'] + scores['specificity'] - 1
    return scores

def str_clf(clf):
    s = clf.__class__.__name__
    if hasattr(clf, 'seed'):
        s = "%s (seed='%s')" % (s, clf.seed)
    return s

def evaluate_classifiers(clfs, X_test, y_true, pos_label=1):
    if not isinstance(clfs, collections.Iterable):
        clfs = [clfs]
    # scores = {clf.__class__.__name__ : measure_classification(clf, X_test, y_true, pos_label) for clf in clfs}
    scores = [ dict( measure_classification(clf, X_test, y_true, pos_label),
                     **{'classifier': str_clf(clf)}) for clf in clfs ]
    cols = scores[0].keys()
    from pandas import DataFrame
    dd = DataFrame.from_dict(data={m: [sc[m] for sc in scores] for m in
        cols})
    dd.set_index('classifier', inplace=True)
    return dd


def score_clfs_in_df(clfs, X, Y, n_iter=10, test_size=0.3, random_state=0xBABE, pos_label=1, verbose=True):
    scores = []
    fitted_clfs = []
    i = 1
    for training,testing in sklearn.cross_validation.StratifiedShuffleSplit(Y, n_iter=n_iter, test_size=test_size, random_state=random_state):
        fitted_clfs_i = [clf.fit(X.iloc[training], Y[training]) for clf in clfs]
        #scores.append([measure_classification(clf, X.iloc[testing], Y[testing]) for clf in fitted_clfs])
        scores.append( evaluate_classifiers(fitted_clfs_i, X.iloc[testing], Y[testing], pos_label) )
        fitted_clfs.append(fitted_clfs_i)
        if verbose: print("* %d / %d" % (i, n_iter))
        i += 1
    return scores, fitted_clfs

def train_test_job(X,Y, training, testing, clfs, pos_label=1):
    fitted_clfs_i = [clf.fit(X.iloc[training], Y[training]) for clf in clfs]
    #scores.append([measure_classification(clf, X.iloc[testing], Y[testing]) for clf in fitted_clfs])
    score = evaluate_classifiers(fitted_clfs_i, X.iloc[testing], Y[testing], pos_label)
    return (score, fitted_clfs_i)

def score_clfs_in_df_parallel(clfs, X, Y, n_iter=10, test_size=0.3, random_state=0xBABE, pos_label=1, n_jobs=-1, verbose=True):
    iterable = sklearn.cross_validation.StratifiedShuffleSplit(Y, n_iter=n_iter, test_size=test_size, random_state=random_state)
    splits = [(training, testing) for training,testing in iterable]
    verbosity = 0
    if verbose:
        verbosity = 10
    results = Parallel(n_jobs=n_jobs, verbose=verbosity)(delayed(train_test_job)(X, Y, training, testing, clfs, pos_label=pos_label) for training,testing in splits)
    scores = [r[0] for r in results]
    fitted_clfs = [r[1] for r in results]
    return scores, fitted_clfs

# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
