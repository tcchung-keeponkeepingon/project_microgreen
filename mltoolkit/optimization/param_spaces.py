"""Optuna search space templates for common model types."""


def suggest_ann_params(trial):
    """Suggest ANN hyperparameters for Optuna trial.

    Returns dict compatible with ANNTrainer.build_model().
    """
    n_layers = trial.suggest_int("n_layers", 2, 6)
    params = {
        'n_layers': n_layers,
        'dropout': trial.suggest_float("dropout", 0.0, 0.5),
        'activation': trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu"]),
        'lr': trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        'batch_size': trial.suggest_categorical("batch_size", [8, 16]),
        'epochs': trial.suggest_int("epochs", 100, 500),
    }
    for i in range(n_layers):
        params[f'hidden_{i}'] = trial.suggest_int(f"hidden_{i}", 8, 64)
    return params


def suggest_svr_params(trial):
    """Suggest SVR hyperparameters for Optuna trial."""
    return {
        'C': trial.suggest_float("C", 1e-2, 1e3, log=True),
        'epsilon': trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
        'kernel': trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
        'gamma': trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


def suggest_svc_params(trial):
    """Suggest SVC hyperparameters for Optuna trial."""
    return {
        'C': trial.suggest_float("C", 1e-2, 1e3, log=True),
        'kernel': trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
        'gamma': trial.suggest_categorical("gamma", ["scale", "auto"]),
    }


def suggest_rf_params(trial):
    """Suggest Random Forest hyperparameters for Optuna trial."""
    return {
        'n_estimators': trial.suggest_int("n_estimators", 50, 500),
        'max_depth': trial.suggest_int("max_depth", 3, 30),
        'min_samples_split': trial.suggest_int("min_samples_split", 2, 20),
        'min_samples_leaf': trial.suggest_int("min_samples_leaf", 1, 10),
    }


def suggest_xgboost_params(trial):
    """Suggest XGBoost hyperparameters for Optuna trial."""
    return {
        'n_estimators': trial.suggest_int("n_estimators", 50, 500),
        'max_depth': trial.suggest_int("max_depth", 3, 12),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float("subsample", 0.5, 1.0),
        'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 1.0),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
    }


def suggest_catboost_params(trial):
    """Suggest CatBoost hyperparameters for Optuna trial."""
    return {
        'iterations': trial.suggest_int("iterations", 100, 1000),
        'depth': trial.suggest_int("depth", 3, 10),
        'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
    }
