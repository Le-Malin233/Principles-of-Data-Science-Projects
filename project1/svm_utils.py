import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def train_svm_with_cv(X_train, y_train):

    param_grid = {
        "C": [0.01]
    }

    svm = LinearSVC(max_iter=5000)

    grid = GridSearchCV(
        svm,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_c = grid.best_params_["C"]

    return best_model, best_c


def evaluate(model, X_test, y_test):

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    return acc