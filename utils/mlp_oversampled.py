"""
MLPRegressorOversampled: sklearn-compatible MLP with CV-safe histogram oversampling.

Applies oversampling ONLY inside fit() to prevent data leakage during cross-validation.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import check_is_fitted

from .oversampling_by_hist import oversample_by_hist_v2


class MLPRegressorOversampled(RegressorMixin, BaseEstimator):
    """MLP Regressor with histogram-based oversampling applied in fit().

    Applies oversampling ONLY to training data, making it safe for CV.
    This prevents data leakage that occurs when oversampling before CV splitting.

    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.

    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter.

    learning_rate_init : float, default=0.001
        The initial learning rate used.

    max_iter : int, default=200
        Maximum number of iterations.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation
        score is not improving.

    random_state : int, RandomState instance or None, default=None
        Random state for the MLP.

    expand_factor : float, default=10.0
        Factor by which to expand the dataset. An expand_factor of 10 means
        the oversampled dataset will have ~10x the original samples.

    n_bins : int, default=100
        Number of histogram bins for computing sampling weights.

    smoothing_coeff : float, default=1.0
        Smoothing coefficient added to histogram counts to prevent division by zero.

    oversample_random_state : int or None, default=None
        Random state for reproducible oversampling.

    Examples
    --------
    >>> from utils import MLPRegressorOversampled
    >>> from sklearn.model_selection import cross_val_score
    >>> mlp = MLPRegressorOversampled(
    ...     hidden_layer_sizes=(64, 32),
    ...     expand_factor=10.0,
    ...     random_state=42
    ... )
    >>> # CV is now safe - oversampling happens per-fold
    >>> scores = cross_val_score(mlp, X, y, cv=5)
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
        oversample_random_state=None,
    ):
        # MLP parameters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state
        # Oversampling parameters
        self.expand_factor = expand_factor
        self.n_bins = n_bins
        self.smoothing_coeff = smoothing_coeff
        self.oversample_random_state = oversample_random_state

    def fit(self, X, y):
        """Fit the MLP model with oversampled training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Convert to numpy arrays to avoid index alignment issues
        # (sklearn CV slicing can produce non-contiguous DataFrame indices)
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        # Apply oversampling to training data only
        X_os, y_os = oversample_by_hist_v2(
            X_arr,
            y_arr,
            expand_factor=self.expand_factor,
            n_bins=self.n_bins,
            smoothing_coeff=self.smoothing_coeff,
            random_state=self.oversample_random_state,
        )

        # Flatten y if needed (MLPRegressor expects 1D for single output)
        y_os = np.asarray(y_os).ravel()

        # Create and fit MLP
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
        """Predict using the fitted MLP model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values.
        """
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
