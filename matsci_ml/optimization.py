"""Optuna objective function factory for hyperparameter optimization."""

import numpy as np
import torch
from sklearn.model_selection import KFold

from .models.trainer import ANN, train_model


def create_cv_objective(X, y, target_points=None, target_weights=None,
                        target_weight_in_loss=-10.0, n_splits=5, random_state=42,
                        use_cv=True, input_dim=5):
    """
    Create an Optuna objective function for ANN hyperparameter search.

    Args:
        X: Training features (numpy array)
        y: Training targets (numpy array)
        target_points: Optional target points to evaluate (for BO/AL)
        target_weights: Optional weights for target points
        target_weight_in_loss: Weight for target prediction in loss (default -10.0)
        n_splits: Number of CV folds (default 5)
        random_state: Random seed (default 42)
        use_cv: If True, use cross-validation; if False, train on full data
        input_dim: Input dimension (default 5)

    Returns:
        Optuna objective function
    """
    # Convert target points/weights to tensors if provided
    target_points_tensor = None
    target_weights_tensor = None
    if target_points is not None:
        target_points_tensor = torch.tensor(target_points, dtype=torch.float32)
    if target_weights is not None:
        target_weights_tensor = torch.tensor(target_weights, dtype=torch.float32)

    def objective(trial):
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Suggest hyperparameters
        n_layers = trial.suggest_int("n_layers", 2, 6)
        hidden_dims = [trial.suggest_int(f"hidden_{i}", 8, 64) for i in range(n_layers)]
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu"])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [8, 16])
        epochs = trial.suggest_int("epochs", 100, 500)

        if use_cv:
            # Cross-validation mode
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            cv_mae_scores = []
            cv_target_preds = []

            for train_idx, val_idx in cv.split(X, y):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]

                model = ANN(input_dim=input_dim, hidden_dims=hidden_dims,
                            dropout=dropout, activation=activation)
                train_model(model, X_tr, y_tr, lr, weight_decay, batch_size, epochs, seed=random_state)

                results = model.evaluate(X_val, y_val, target_points_tensor, target_weights_tensor)
                cv_mae_scores.append(results['mae'])
                if 'weighted_target' in results:
                    cv_target_preds.append(results['weighted_target'])

            mean_mae = np.mean(cv_mae_scores)
            trial.set_user_attr("cv_mae", float(mean_mae))

            if cv_target_preds:
                mean_target_pred = np.mean(cv_target_preds)
                trial.set_user_attr("target_pred", float(mean_target_pred))
                return float(mean_mae + target_weight_in_loss * mean_target_pred)
            else:
                return float(mean_mae)

        else:
            # Train on full data mode
            model = ANN(input_dim=input_dim, hidden_dims=hidden_dims,
                        dropout=dropout, activation=activation)
            train_model(model, X, y, lr, weight_decay, batch_size, epochs, seed=random_state)

            results = model.evaluate(X, y, target_points_tensor, target_weights_tensor)
            train_mae = results['mae']
            trial.set_user_attr("train_mae", float(train_mae))

            if 'weighted_target' in results:
                target_pred = results['weighted_target']
                trial.set_user_attr("target_pred", float(target_pred))
                return float(train_mae + target_weight_in_loss * target_pred)
            else:
                return float(train_mae)

    return objective
