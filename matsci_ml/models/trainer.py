"""PyTorch ANN trainer for materials science optimization."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error


# Activation function mapping (from original notebook)
ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU
}


class ANN(nn.Module):
    """Feedforward neural network for regression (from original notebook)."""

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


def train_model(model, X_tr, y_tr, lr, weight_decay, batch_size, epochs, seed=42):
    """Train the ANN model (from original notebook)."""
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
    criterion = nn.L1Loss()

    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()


class PyTorchANNTrainer:
    """Wrapper class for ANN training with clean interface."""

    def __init__(self, input_dim=5):
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
        self.input_dim = checkpoint.get('input_dim', 5)
        hidden_dims = checkpoint['hidden_dims']
        self.model = ANN(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            dropout=self.params['dropout'],
            activation=self.params['activation']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self


class AugmentedANNTrainer(PyTorchANNTrainer):
    """ANN trainer with Gaussian data augmentation applied during fit().

    Augmentation happens only on training data passed to fit(), so when used
    inside KFold CV, validation folds remain unaugmented (no data leakage).
    """

    def __init__(self, input_dim=5, n_augment=5, feature_scale=0.05, label_scale=0.05):
        super().__init__(input_dim)
        self.n_augment = n_augment
        self.feature_scale = feature_scale
        self.label_scale = label_scale

    def fit(self, X, y, seed=42):
        """Train the model with augmented data."""
        if self.model is None:
            raise ValueError("Must call build_model() before fit()")
        from utils.data_augmentation import data_aug_gaussian
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
