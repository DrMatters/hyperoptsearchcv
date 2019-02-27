from HyperoptSearchCV import HyperoptSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from hyperopt import hp

rf_space = {
    'n_estimators': hp.quniform('n_estimators', 25, 1525, 50),
    'max_depth': hp.quniform('max_depth', 7, 20, 1),
    'min_samples_split': hp.choice('min_samples_split', [2, 5, 20, 50]),
    'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4]),
}
rf_cast = {
    'n_estimators': int,
    'max_depth': int,
    'min_samples_split': None,
    # parameter can be omitted if cast is not required
    # 'min_samples_leaf': None,
}
metric = 'accuracy'
text_log = True
max_evals = 3

rf_hyper = HyperoptSearchCV(RandomForestClassifier(n_jobs=-1, random_state=42), rf_space,
                            rf_cast, metric, text_log, max_evals)

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


rf_hyper.fit(X_train, y_train)
print(rf_hyper.best_score_)
