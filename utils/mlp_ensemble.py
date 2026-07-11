"""MLP Ensemble for smooth, continuous predictions with reduced variance."""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Dict, Any


class MLPEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensemble of MLPRegressors with different random seeds.

    Averaging predictions from multiple neural networks reduces variance
    while maintaining smooth, continuous predictions over the design space.

    Parameters
    ----------
    n_models : int, default=5
        Number of MLP models in the ensemble
    hidden_layer_sizes : tuple, default=(64, 32)
        Hidden layer sizes for each MLP
    activation : str, default='relu'
        Activation function ('relu', 'tanh', 'logistic')
    solver : str, default='adam'
        Optimizer ('adam', 'sgd', 'lbfgs')
    alpha : float, default=0.001
        L2 regularization strength
    learning_rate : str, default='adaptive'
        Learning rate schedule ('constant', 'invscaling', 'adaptive')
    learning_rate_init : float, default=0.001
        Initial learning rate
    max_iter : int, default=1000
        Maximum number of iterations
    early_stopping : bool, default=True
        Whether to use early stopping
    validation_fraction : float, default=0.1
        Fraction of data for early stopping validation
    n_iter_no_change : int, default=20
        Epochs without improvement before stopping
    base_random_state : int, default=0
        Base random state; each model uses base + model_index
    verbose : bool, default=False
        Whether to print training progress

    Attributes
    ----------
    models_ : list of MLPRegressor
        Fitted MLP models
    scaler_ : StandardScaler
        Feature scaler (shared across models)

    Examples
    --------
    >>> from utils.mlp_ensemble import MLPEnsemble
    >>> ensemble = MLPEnsemble(n_models=5)
    >>> ensemble.fit(X_train, y_train)
    >>> y_pred = ensemble.predict(X_test)
    >>> y_pred, y_std = ensemble.predict_with_uncertainty(X_test)
    """

    def __init__(
        self,
        n_models: int = 5,
        hidden_layer_sizes: tuple = (64, 32),
        activation: str = 'relu',
        solver: str = 'adam',
        alpha: float = 0.001,
        learning_rate: str = 'adaptive',
        learning_rate_init: float = 0.001,
        max_iter: int = 1000,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 20,
        base_random_state: int = 0,
        verbose: bool = False,
    ):
        self.n_models = n_models
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.base_random_state = base_random_state
        self.verbose = verbose

        self.models_: List[MLPRegressor] = []
        self.scaler_: Optional[StandardScaler] = None

    def _create_model(self, random_state: int) -> MLPRegressor:
        """Create a single MLP with specified random state."""
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=random_state,
            verbose=self.verbose,
        )

    def fit(self, X, y):
        """
        Fit all MLP models in the ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        # Fit shared scaler
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Fit each model with different random state
        self.models_ = []
        for i in range(self.n_models):
            random_state = self.base_random_state + i
            model = self._create_model(random_state)
            model.fit(X_scaled, y)
            self.models_.append(model)

            if self.verbose:
                print(f"Model {i+1}/{self.n_models} trained "
                      f"(iterations: {model.n_iter_})")

        return self

    def predict(self, X) -> np.ndarray:
        """
        Predict using ensemble mean.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Mean predictions from all models
        """
        predictions = self._get_all_predictions(X)
        return np.mean(predictions, axis=0)

    def predict_with_uncertainty(self, X) -> tuple:
        """
        Predict with uncertainty estimates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features to predict

        Returns
        -------
        y_mean : ndarray of shape (n_samples,)
            Mean predictions
        y_std : ndarray of shape (n_samples,)
            Standard deviation of predictions (epistemic uncertainty)
        """
        predictions = self._get_all_predictions(X)
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)

    def _get_all_predictions(self, X) -> np.ndarray:
        """Get predictions from all models."""
        if not self.models_:
            raise ValueError("Ensemble has not been fitted yet")

        X = np.asarray(X)
        X_scaled = self.scaler_.transform(X)

        predictions = np.array([
            model.predict(X_scaled) for model in self.models_
        ])
        return predictions  # shape: (n_models, n_samples)

    def get_individual_predictions(self, X) -> np.ndarray:
        """
        Get predictions from each individual model.

        Useful for analyzing ensemble diversity and spread.

        Returns
        -------
        predictions : ndarray of shape (n_models, n_samples)
        """
        return self._get_all_predictions(X)

    def score(self, X, y) -> float:
        """Return R² score on given data."""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


class PyTorchEnsemble:
    """
    Ensemble of PyTorch ANN models from matsci_ml.

    Parameters
    ----------
    n_models : int, default=5
        Number of models in ensemble
    model_params : dict
        Parameters passed to PyTorchANNTrainer.build_model()
    input_dim : int, default=5
        Number of input features
    base_random_state : int, default=0
        Base random state for reproducibility
    """

    def __init__(
        self,
        n_models: int = 5,
        model_params: Optional[Dict[str, Any]] = None,
        input_dim: int = 5,
        base_random_state: int = 0,
    ):
        self.n_models = n_models
        self.model_params = model_params or {}
        self.input_dim = input_dim
        self.base_random_state = base_random_state
        self.trainers_ = []

    def fit(self, X, y, verbose: bool = False):
        """Fit all models in the ensemble.

        Note: epochs should be specified in model_params during init.
        """
        import torch
        from matsci_ml import PyTorchANNTrainer

        self.trainers_ = []
        for i in range(self.n_models):
            # Set seed for reproducibility
            seed = self.base_random_state + i
            torch.manual_seed(seed)
            np.random.seed(seed)

            trainer = PyTorchANNTrainer(input_dim=self.input_dim)
            trainer.build_model(self.model_params)
            trainer.fit(X, y)
            self.trainers_.append(trainer)

        return self

    def predict(self, X) -> np.ndarray:
        """Predict using ensemble mean."""
        predictions = self._get_all_predictions(X)
        return np.mean(predictions, axis=0)

    def predict_with_uncertainty(self, X) -> tuple:
        """Predict with uncertainty estimates."""
        predictions = self._get_all_predictions(X)
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)

    def _get_all_predictions(self, X) -> np.ndarray:
        """Get predictions from all models."""
        predictions = np.array([
            trainer.predict(X) for trainer in self.trainers_
        ])
        return predictions
