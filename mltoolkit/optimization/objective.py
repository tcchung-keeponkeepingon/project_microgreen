"""Model-agnostic Optuna objective function factory."""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def create_cv_objective(X, y, model_fn, n_splits=5, random_state=42,
                        use_cv=True, scoring='neg_mae'):
    """
    Create an Optuna objective function for any model type.

    Parameters
    ----------
    X : np.ndarray
        Training features.
    y : np.ndarray
        Training targets.
    model_fn : callable
        Factory function: model_fn(trial, X_train, y_train) -> fitted model with .predict(X).
        The function should use trial.suggest_* to define hyperparameters,
        then build, fit, and return the model.
    n_splits : int, default=5
        Number of CV folds.
    random_state : int, default=42
        Random seed for KFold shuffle.
    use_cv : bool, default=True
        If True, use cross-validation. If False, train on full data.
    scoring : str, default='neg_mae'
        Scoring metric. Options: 'neg_mae', 'neg_mse', 'neg_rmse', 'r2'.
        For 'neg_*' metrics, the objective returns the negative (for minimization).
        For 'r2', the objective returns negative R² (since Optuna minimizes by default).

    Returns
    -------
    callable
        Optuna objective function.

    Examples
    --------
    >>> from mltoolkit.optimization import create_cv_objective, suggest_xgboost_params
    >>> from xgboost import XGBRegressor
    >>>
    >>> def model_fn(trial, X_train, y_train):
    ...     params = suggest_xgboost_params(trial)
    ...     model = XGBRegressor(**params, random_state=42, verbosity=0)
    ...     model.fit(X_train, y_train)
    ...     return model
    >>>
    >>> objective = create_cv_objective(X, y, model_fn, scoring='neg_mae')
    >>> study = optuna.create_study(direction='minimize')
    >>> study.optimize(objective, n_trials=50)
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()

    score_fns = {
        'neg_mae': lambda yt, yp: mean_absolute_error(yt, yp),
        'neg_mse': lambda yt, yp: mean_squared_error(yt, yp),
        'neg_rmse': lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)),
        'r2': lambda yt, yp: -r2_score(yt, yp),  # negate so minimization works
    }

    if scoring not in score_fns:
        raise ValueError(f"Unknown scoring: {scoring}. Choose from {list(score_fns.keys())}")

    compute_score = score_fns[scoring]

    def objective(trial):
        if use_cv:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            scores = []

            for train_idx, val_idx in cv.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                model = model_fn(trial, X_tr, y_tr)
                y_pred = model.predict(X_val)
                scores.append(compute_score(y_val, y_pred))

            mean_score = float(np.mean(scores))
            trial.set_user_attr(f"cv_{scoring}", mean_score)
            return mean_score

        else:
            model = model_fn(trial, X, y)
            y_pred = model.predict(X)
            score = compute_score(y, y_pred)
            trial.set_user_attr(f"train_{scoring}", float(score))
            return float(score)

    return objective
