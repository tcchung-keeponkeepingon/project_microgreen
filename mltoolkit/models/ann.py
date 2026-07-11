"""PyTorch ANN trainer for regression tasks."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error


# Activation function mapping
ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU
}


class ANN(nn.Module):
    """Feedforward neural network for regression."""

    def __init__(self, input_dim, hidden_dims, dropout, activation='relu'):
        super().__init__()
        activation_fn = ACTIVATION_MAP[activation]
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                activation_fn(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            return self(X).numpy().flatten()

    def evaluate(self, X, y, target_points=None, target_weights=None):
        self.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            if not isinstance(y, np.ndarray):
                y = np.array(y)

            preds = self(X).numpy().flatten()
            results = {
                'predictions': preds,
                'mae': mean_absolute_error(y, preds)
            }

            if target_points is not None:
                if not isinstance(target_points, torch.Tensor):
                    target_points = torch.tensor(target_points, dtype=torch.float32)
                target_preds = self(target_points).numpy().flatten()
                results['target_predictions'] = target_preds

                if target_weights is not None:
                    if not isinstance(target_weights, torch.Tensor):
                        target_weights = torch.tensor(target_weights, dtype=torch.float32)
                    results['weighted_target'] = (target_weights.numpy() * target_preds).sum()

            return results


def train_model(model, X_tr, y_tr, lr, weight_decay, batch_size, epochs, seed=42,
                X_val=None, y_val=None, patience=None, loss_fn='l1'):
    """Train the ANN model.

    Parameters
    ----------
    model : ANN
        Model to train.
    X_tr, y_tr : array-like
        Training data.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularization.
    batch_size : int
        Batch size.
    epochs : int
        Maximum number of epochs.
    seed : int, default=42
        Random seed.
    X_val, y_val : array-like or None
        Validation data for early stopping.
    patience : int or None
        Early stopping patience. None disables early stopping.
    loss_fn : str, default='l1'
        Loss function: 'l1' or 'mse'.

    Returns
    -------
    dict
        Training history with 'train_losses' and optionally 'val_losses'.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed if seed is not None else 42)
    dataset = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if loss_fn == 'l1':
        criterion = nn.L1Loss()
    elif loss_fn == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss_fn: {loss_fn}. Choose 'l1' or 'mse'.")

    # Prepare validation if provided
    has_val = X_val is not None and y_val is not None
    if has_val:
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    history = {'train_losses': [], 'val_losses': []}
    best_val_loss = float('inf')
    best_state = None
    wait = 0

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        history['train_losses'].append(epoch_loss / n_batches)

        if has_val:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()
            history['val_losses'].append(val_loss)
            model.train()

            if patience is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

    # Restore best model if early stopping was used
    if patience is not None and best_state is not None:
        model.load_state_dict(best_state)

    return history


class ANNTrainer:
    """Wrapper class for ANN training with clean interface."""

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None
        self.params = None

    def build_model(self, params):
        """Build model from hyperparameters dict (e.g., from Optuna best_params)."""
        self.params = params
        hidden_dims = [params[f'hidden_{i}'] for i in range(params['n_layers'])]
        self.model = ANN(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            dropout=params['dropout'],
            activation=params['activation']
        )
        return self

    def fit(self, X, y, seed=42):
        """Train the model."""
        if self.model is None:
            raise ValueError("Must call build_model() before fit()")
        train_model(
            self.model, X, y,
            lr=self.params['lr'],
            weight_decay=self.params['weight_decay'],
            batch_size=self.params['batch_size'],
            epochs=self.params['epochs'],
            seed=seed,
        )
        return self

    def predict(self, X):
        """Get predictions."""
        if self.model is None:
            raise ValueError("Must call build_model() and fit() before predict()")
        return self.model.predict(X)

    def evaluate(self, X, y, target_points=None, target_weights=None):
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Must call build_model() and fit() before evaluate()")
        return self.model.evaluate(X, y, target_points, target_weights)

    def save(self, filepath):
        """Save model state and parameters."""
        if self.model is None:
            raise ValueError("No model to save")
        hidden_dims = [self.params[f'hidden_{i}'] for i in range(self.params['n_layers'])]
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'params': self.params,
            'hidden_dims': hidden_dims,
            'input_dim': self.input_dim
        }, filepath)

    def load(self, filepath):
        """Load model state and parameters."""
        checkpoint = torch.load(filepath, weights_only=False)
        self.params = checkpoint.get('params') or checkpoint.get('best_params')
        self.input_dim = checkpoint.get('input_dim', self.input_dim)
        hidden_dims = checkpoint['hidden_dims']
        self.model = ANN(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            dropout=self.params['dropout'],
            activation=self.params['activation']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self


class AugmentedANNTrainer(ANNTrainer):
    """ANN trainer with Gaussian data augmentation applied during fit().

    Augmentation happens only on training data passed to fit(), so when used
    inside KFold CV, validation folds remain unaugmented (no data leakage).
    """

    def __init__(self, input_dim, n_augment=5, feature_scale=0.05, label_scale=0.05):
        super().__init__(input_dim)
        self.n_augment = n_augment
        self.feature_scale = feature_scale
        self.label_scale = label_scale

    def fit(self, X, y, seed=42):
        """Train the model with augmented data."""
        if self.model is None:
            raise ValueError("Must call build_model() before fit()")
        from ..data.augmentation import data_aug_gaussian
        X_aug, y_aug = data_aug_gaussian(
            X, y,
            feature_scale=self.feature_scale,
            label_scale=self.label_scale,
            n_augment=self.n_augment,
            seed=seed,
        )
        train_model(
            self.model, X_aug, y_aug,
            lr=self.params['lr'],
            weight_decay=self.params['weight_decay'],
            batch_size=self.params['batch_size'],
            epochs=self.params['epochs'],
            seed=seed,
        )
        return self
