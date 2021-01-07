from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble._forest import ForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample, gen_batches, check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

import numpy as np
import pandas as pd

from Constants import LABEL_COL


def random_feature_subsets(array, batch_size, random_state=1234):
    """ Generate K subsets of the features in X """
    random_state = check_random_state(random_state)
    features = list(range(array.shape[1]))
    random_state.shuffle(features)
    for batch in gen_batches(len(features), batch_size):
        yield features[batch]


class RotationTreeClassifier(DecisionTreeClassifier):
    # https://github.com/joshloyal/RotationForest/blob/master/rotation_forest/rotation_forest.py
    def __init__(self,
                 n_features_per_subset=3,
                 rotation_algo='pca',
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=1.0,
                 random_state=None,
                 max_leaf_nodes=None,
                 class_weight=None,
                 presort=False):

        self.n_features_per_subset = n_features_per_subset
        self.rotation_algo = rotation_algo
        self.presort = presort
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state)

    def rotate(self, X):
        if not hasattr(self, 'rotation_matrix'):
            raise NotFittedError('The estimator has not been fitted')

        return safe_sparse_dot(X, self.rotation_matrix)

    def pca_algorithm(self):
        """ Deterimine PCA algorithm to use. """
        if self.rotation_algo == 'pca':
            return PCA()
        else:
            raise ValueError("`rotation_algo` must be either "
                             "'pca' or 'randomized'.")

    def _fit_rotation_matrix(self, X):
        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        self.rotation_matrix = np.zeros((n_features, n_features),
                                        dtype=np.float32)
        for i, subset in enumerate(
                random_feature_subsets(X, self.n_features_per_subset,
                                       random_state=self.random_state)):
            # take a 75% bootstrap from the rows
            x_sample = resample(X, n_samples=int(n_samples * 0.75), random_state=10 * i)
            pca = self.pca_algorithm()
            pca.fit(x_sample[:, subset])
            self.rotation_matrix[np.ix_(subset, subset)] = pca.components_

    def fit(self, X, y, sample_weight=None, check_input=True):
        self._fit_rotation_matrix(X)
        super().fit(self.rotate(X), y,
                    sample_weight, check_input)

    def predict_proba(self, X, check_input=True):
        return super().predict_proba(self.rotate(X), check_input)

    def predict(self, X, check_input=True):
        return super().predict(self.rotate(X),
                               check_input)

    def apply(self, X, check_input=True):
        return super().apply(self.rotate(X),
                             check_input)

    def decision_path(self, X, check_input=True):
        return super().decision_path(self.rotate(X),
                                     check_input)


class RotationForestClassifier(ForestClassifier):
    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 n_features_per_subset=3,
                 rotation_algo='pca',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=1.0,
                 max_leaf_nodes=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None):
        super().__init__(
            base_estimator=RotationTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("n_features_per_subset", "rotation_algo",
                              "criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.n_features_per_subset = n_features_per_subset
        self.rotation_algo = rotation_algo
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes


class MulticlassToManyBinaries(object):
    def __init__(self,
                 estimator_name="mnb",
                 estimator_class=MultinomialNB,
                 estimator_args={},
                 random_state=0
                 ):
        self.random_state = random_state
        self.estimator_name = estimator_name
        self.estimator_class = estimator_class
        self.estimator_args = estimator_args
        self.estimators = {}

    def fit(self, X, y,validation_X=None, validation_y = None):
        df = pd.DataFrame(X).copy()
        df[LABEL_COL] = y
        labels = list(df[LABEL_COL].unique())
        for label_a in labels:
            apply_train = False  # to remove duplicate comparisons
            for label_b in labels:
                if label_a == label_b:
                    apply_train = True
                    continue
                if apply_train:
                    print(f"fit {label_a} vs {label_b}")
                    df_ = df[df[LABEL_COL].isin([label_a, label_b])]

                    y_ = df_[LABEL_COL]
                    X_ = df_.drop(columns=LABEL_COL)

                    estimator_ = self.estimator_class(**self.estimator_args)
                    if isinstance(estimator_,CatBoostClassifier):
                        estimator_ = fit_catboost(estimator_, X, y, validation_X, validation_y)
                    else:
                        estimator_.fit(X_, y_)
                    self.estimators[f"{self.estimator_name}_{label_a}_{label_b}"] = estimator_

    def predict_proba(self, X):
        probs = []
        for estimator_name, estimator_ in self.estimators.items():
            prob = estimator_.predict_proba(X)[:, 0]
            prob = pd.DataFrame(prob, columns=[estimator_name], index=X.index)
            probs.append(prob)

        probs_df = pd.concat(probs, axis=1)
        return probs_df


class StackedModel(object):
    def __init__(self,
                 stacked_class=LogisticRegression,
                 stacked_args={},
                 estimators_args={
                     "mNB": {"class": MultinomialNB, "args": {}},
                     "ada": {"class": AdaBoostClassifier, "args": {"n_estimators": 100, "random_state": 0}}
                 },
                 random_state=0):

        self.random_state = random_state
        self.estimators_args = estimators_args
        self.estimators = {}
        self.stacked_class = stacked_class
        self.stacked_args = stacked_args
        self.stacked_model = None
        self.classes_ = []

    def fit_estimator(self,X,y, estimator_name,estimator_class, estimator_args,validation_X=None, validation_y = None):
        print("fit estimator")
        estimator = MulticlassToManyBinaries(estimator_name=estimator_name,
                                 estimator_class=estimator_class,
                                 estimator_args=estimator_args)
        estimator.fit(X,y,validation_X,validation_y)
        self.estimators[estimator_name] = estimator
        return estimator

    def fit_stacked(self,X,y):
        stacked_X_preds_df = self.predict_subs(X)
        self.stacked_model = self.stacked_class(**self.stacked_args)
        self.stacked_model.fit(stacked_X_preds_df, y)

    def fit(self, X, y):
        self.classes_ = list(y.unique())
        sub_estimators_X, stacked_X, sub_estimators_y, stacked_y = train_test_split(X, y, test_size=0.2,
                                                                                    random_state=self.random_state)

        for estimator_name, estimator_args in self.estimators_args.items():
            print(f'fit {estimator_name}')
            self.fit_estimator(sub_estimators_X, sub_estimators_y, estimator_name, estimator_args["class"],estimator_args["args"])

        self.fit_stacked(stacked_X, stacked_y)

    def predict_subs(self, X):
        stacked_X_preds = []
        for estimator_name, estimator_ in self.estimators.items():
            preds_ = estimator_.predict_proba(X)
            stacked_X_preds.append(preds_)
        stacked_X_preds_df = pd.concat(stacked_X_preds, axis=1)
        return stacked_X_preds_df

    def predict_proba(self, X):
        stacked_X_preds_df = self.predict_subs(X)
        return self.stacked_model.predict_proba(stacked_X_preds_df)

    def predict(self, X, check_input=True):
        stacked_X_preds_df = self.predict_subs(X)
        return self.stacked_model.predict(stacked_X_preds_df)


def fit_catboost(model, X,y,X_eval, y_eval,early_stopping_rounds = 10 ):
    print("fit catboost")
    eval_pool = Pool(X_eval, y_eval)
    model.fit(X, y, eval_set=eval_pool, early_stopping_rounds=early_stopping_rounds)
    return model
