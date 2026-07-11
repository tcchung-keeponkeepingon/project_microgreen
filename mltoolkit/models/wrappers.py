"""Model wrappers for scaling and oversampling."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import check_is_fitted

from ..data.oversampling import oversample_by_hist


class ScaledModel(BaseEstimator, RegressorMixin):
    """Wrapper that handles feature and target scaling automatically.

    Parameters
    ----------
    model : estimator
        Model with fit() and predict() methods.
    feature_scaler : sklearn scaler or None
        Scaler for features. Fitted during fit().
    target_scaler : sklearn scaler or None
        Scaler for target. Fitted during fit().
    """

    def __init__(self, model, feature_scaler=None, target_scaler=None):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

    def fit(self, X, y):
        """Fit scalers and model."""
        X = np.asarray(X)
        y = np.asarray(y)

        if self.feature_scaler is not None:
            X = self.feature_scaler.fit_transform(X)

        if self.target_scaler is not None:
            y = self.target_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Scale features, predict, inverse-transform target."""
        X = np.asarray(X)

        if self.feature_scaler is not None:
            X = self.feature_scaler.transform(X)

        y_pred = self.model.predict(X)

        if self.target_scaler is not None:
            y_pred = self.target_scaler.inverse_transform(
                y_pred.reshape(-1, 1)
            ).ravel()

        return y_pred


class MLPRegressorOversampled(RegressorMixin, BaseEstimator):
    """MLP Regressor with histogram-based oversampling applied in fit().

    Applies oversampling ONLY to training data, making it safe for CV.

    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        Hidden layer sizes.
    alpha : float, default=0.0001
        L2 penalty parameter.
    learning_rate_init : float, default=0.001
        Initial learning rate.
    max_iter : int, default=200
        Maximum number of iterations.
    early_stopping : bool, default=False
        Whether to use early stopping.
    random_state : int or None, default=None
        Random state for the MLP.
    expand_factor : float, default=10.0
        Factor by which to expand the dataset.
    n_bins : int, default=100
        Number of histogram bins.
    smoothing_coeff : float, default=1.0
        Smoothing coefficient for histogram counts.
    oversample_random_state : int or None, default=42
        Random state for reproducible oversampling.
    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=False,
        random_state=None,
        expand_factor=10.0,
        n_bins=100,
        smoothing_coeff=1.0,
        oversample_random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.expand_factor = expand_factor
        self.n_bins = n_bins
        self.smoothing_coeff = smoothing_coeff
        self.oversample_random_state = oversample_random_state

    def fit(self, X, y):
        """Fit the MLP model with oversampled training data."""
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        X_os, y_os = oversample_by_hist(
            X_arr,
            y_arr,
            expand_factor=self.expand_factor,
            n_bins=self.n_bins,
            smoothing_coeff=self.smoothing_coeff,
            random_state=self.oversample_random_state,
        )

        y_os = np.asarray(y_os).ravel()

        self.mlp_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            random_state=self.random_state,
        )
        self.mlp_.fit(X_os, y_os)

        return self

    def predict(self, X):
        """Predict using the fitted MLP model."""
        check_is_fitted(self, "mlp_")
        return self.mlp_.predict(X)

    @property
    def n_iter_(self):
        """Number of iterations the solver has run."""
        check_is_fitted(self, "mlp_")
        return self.mlp_.n_iter_

    @property
    def loss_(self):
        """The current loss computed with the loss function."""
        check_is_fitted(self, "mlp_")
        return self.mlp_.loss_

    @property
    def coefs_(self):
        """The weight matrices for each layer."""
        check_is_fitted(self, "mlp_")
        return self.mlp_.coefs_

    @property
    def intercepts_(self):
        """The bias vectors for each layer."""
        check_is_fitted(self, "mlp_")
        return self.mlp_.intercepts_
