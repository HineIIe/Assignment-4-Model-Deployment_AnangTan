import optuna
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


RANDOM_STATE = 42


def objective_lr(trial, X, y):
    

    params = {
        "C": trial.suggest_float("C", 0.001, 100, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "solver": trial.suggest_categorical("solver", ["liblinear", "saga"]),
        "max_iter": trial.suggest_int("max_iter", 200, 2000),
        "random_state": RANDOM_STATE
    }

    model = LogisticRegression(**params)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    return scores.mean()


def tune_logistic_regression(X, y, n_trials=30):
    

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler
    )

    study.optimize(
        lambda trial: objective_lr(trial, X, y),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print("\nBest Accuracy:", study.best_value)
    print("Best Parameters:", study.best_params)

    return study.best_params


def train_final_model(X, y, best_params):
    

    model = LogisticRegression(
        **best_params,
        random_state=RANDOM_STATE
    )

    model.fit(X, y)

    print("\nFinal Logistic Regression model trained!")

    return model