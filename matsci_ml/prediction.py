"""Batch prediction utilities for large design spaces."""

import numpy as np
import pandas as pd


class BatchPredictor:
    """Memory-efficient predictions for large design spaces (275k+ points)."""

    def __init__(self, models, chunk_size=10000):
        """
        Initialize batch predictor.

        Args:
            models: List of trained models (PyTorchANNTrainer or ANN instances)
            chunk_size: Number of samples per batch (default 10000)
        """
        self.models = models if isinstance(models, list) else [models]
        self.chunk_size = chunk_size

    def predict(self, X):
        """
        Get predictions with uncertainty estimates.

        Args:
            X: Feature array (numpy array or DataFrame)

        Returns:
            mu: Mean predictions across models
            std: Standard deviation across models (0 if single model)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.asarray(X, dtype=np.float32)
        n_samples = len(X)
        n_models = len(self.models)

        # Collect predictions from all models
        all_preds = np.zeros((n_models, n_samples), dtype=np.float32)

        for model_idx, model in enumerate(self.models):
            # Get predict function (handle both ANN and PyTorchANNTrainer)
            predict_fn = model.predict if hasattr(model, 'predict') else model.model.predict

            # Predict in chunks
            preds = np.zeros(n_samples, dtype=np.float32)
            for start in range(0, n_samples, self.chunk_size):
                end = min(start + self.chunk_size, n_samples)
                preds[start:end] = predict_fn(X[start:end])

            all_preds[model_idx] = preds

        # Compute mean and std across models
        mu = all_preds.mean(axis=0)
        std = all_preds.std(axis=0) if n_models > 1 else np.zeros(n_samples, dtype=np.float32)

        return mu, std

    def predict_to_dataframe(self, X_df, feature_cols=None, mu_col='mu', std_col='std'):
        """
        Predict and return as DataFrame with original features.

        Args:
            X_df: Feature DataFrame
            feature_cols: Columns to use for prediction (default: all columns)
            mu_col: Name for mean prediction column
            std_col: Name for std column

        Returns:
            DataFrame with original features plus predictions
        """
        if feature_cols is None:
            feature_cols = X_df.columns.tolist()

        X = X_df[feature_cols].values
        mu, std = self.predict(X)

        result = X_df.copy()
        result[mu_col] = mu
        result[std_col] = std

        return result
