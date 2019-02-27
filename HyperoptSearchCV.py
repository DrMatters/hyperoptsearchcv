from hyperopt import hp, tpe, space_eval
from hyperopt.fmin import fmin
import numpy as np
import sklearn


class HyperoptSearchCV(sklearn.base.BaseEstimator):
    def __init__(self, estimator=None, search_space=None, param_types=None, scoring='accuracy', maximize=False,
                 print_log=False, max_evals=25, seed=42):
        """ Constructor for model to be optimized using hyperopt

        Keyword arguments:
            estimator -- model with sklearn's conventional interface (fit(), predict())
            search_space -- dictionary with search space of parameters for hyperopt
            search_types -- dictionary with types to cast, `None` - for no casting
            scoring -- string or function to feed into `sklearn.model_selection.cross_val_score()`
            maximize -- boolean, set True to maximize scoring function
            print_log -- boolean, True for printing log
            seed -- seed for hyperopt `tpe` optimization function
        """
        self.estimator = estimator
        self.search_space = search_space
        self.param_types = param_types
        self.scoring = scoring
        self.maximize = maximize
        self.print_log = print_log
        self.max_evals = max_evals
        self.seed = seed
        self.best_params_ = None
        self.best_score_ = None

    def cast_params(self, recv_params):
        # cast the parameters stored in `recv_params` to 
        # types stored in `self.param_types`
        casted_params = {}
        for param_name, param_value in recv_params.items():
            param_type = self.param_types.get(param_name, None)  # if type for casting not found, skip
            if param_type is None:
                casted_params[param_name] = param_value
            else:
                casted_params[param_name] = (param_type)(param_value)

        return casted_params

    def universal_objective(self, recv_params):
        casted_params = self.cast_params(recv_params)
        updated_model = self.estimator.set_params(**casted_params)
        score = cross_val_score(updated_model, self.X_train, self.y_train,
                                scoring=self.scoring,
                                cv=StratifiedKFold(n_splits=3),
                                n_jobs=-1).mean()
        if self.print_log:
            print("{:.3f} - mean score on 3-fold CV with params {}".format(score, recv_params))

        if self.maximize:
            if score > self.best_score_:
                self.best_score_ = score
            score = -score
        else:
            if score < self.best_score_:
                self.best_score_ = score

        return score

    def fit(self, X_train, y_train):
        if maximize:
            self.best_score_ = -999999
        else:
            self.best_score_ = 999999
        self.X_train = X_train
        self.y_train = y_train

        self.estimator = sklearn.base.clone(self.estimator)

        best_params = fmin(fn=self.universal_objective,
                           space=self.search_space,
                           algo=tpe.suggest,
                           max_evals=self.max_evals,
                           rstate=np.random.RandomState(42))
        evaluated = space_eval(self.search_space, best_params)
        if self.print_log:
            print('best params: ', evaluated)

        casted_best_params = self.cast_params(evaluated)
        self.best_params_ = casted_best_params

        self.estimator.set_params(**casted_best_params)
        self.estimator.fit(self.X_train, self.y_train)

        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)