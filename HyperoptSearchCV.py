from hyperopt import tpe, space_eval
from hyperopt.fmin import fmin
import numpy as np
import sklearn


class HyperoptSearchCV(sklearn.base.BaseEstimator):
    def __init__(self, estimator, search_space, param_types={},
                 n_iter=25, scoring='accuracy', cv=None,
                 verbose=False, seed=42):
        """ Constructor for model to be optimized using hyperopt

        Keyword arguments:
            :param BaseEstimator estimator: -- model with sklearns conventional interface (fit(), predict())
            :param dict search_space: -- dictionary with search space of parameters for hyperopt
            param_types -- dictionary with types to cast, `None` - for no casting
            n_iter -- integer max number of evaluations
            scoring -- string or function
            cv -- int, cross-validation generator or an iterable, optional
            print_log -- boolean, True for printing log
            seed -- seed for hyperopt `tpe` optimization function
        """
        self.estimator = estimator
        self.search_space = search_space
        self.param_types = param_types
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose
        self.n_iter = n_iter
        self.seed = seed
        self.best_params_ = None
        self.best_score_ = None
        self.X_train = None
        self.y_train = None

    def __cast_params(self, recv_params):
        # cast the parameters stored in `recv_params` to
        # types stored in `self.param_types`
        casted_params = {}
        for param_name, param_value in recv_params.items():
            param_type = self.param_types.get(param_name, None)  # if type for casting not found, skip
            if param_type is None:
                casted_params[param_name] = param_value
            else:
                casted_params[param_name] = param_type(param_value)

        return casted_params

    def __objective(self, recv_params):
        casted_params = self.__cast_params(recv_params)
        updated_model = self.estimator.set_params(**casted_params)
        score = sklearn.model_selection.cross_val_score(updated_model,
                                                        self.X_train,
                                                        self.y_train,
                                                        scoring=self.scoring,
                                                        cv=self.cv,
                                                        n_jobs=-1).mean()
        if self.verbose:
            print("{:.3f} - mean score on CV with params {}".format(score, recv_params))

        if score > self.best_score_:
            self.best_score_ = score

        return score

    def fit(self, X_train, y_train):

        if self.best_score_ is None:
            self.best_score_ = -float('inf')

        self.X_train = X_train
        self.y_train = y_train

        self.estimator = sklearn.base.clone(self.estimator)

        best_params = fmin(fn=self.__objective,
                           space=self.search_space,
                           algo=tpe.suggest,
                           max_evals=self.n_iter,
                           rstate=np.random.RandomState(42))
        evaluated = space_eval(self.search_space, best_params)
        if self.verbose:
            print('best params: ', evaluated)

        casted_best_params = self.__cast_params(evaluated)
        self.best_params_ = casted_best_params

        self.estimator.set_params(**casted_best_params)
        self.estimator.fit(self.X_train, self.y_train)

        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
