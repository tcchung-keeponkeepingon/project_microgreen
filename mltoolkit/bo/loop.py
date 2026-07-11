"""Bayesian Optimization loop with ask/tell interface."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from ..acquisition.functions import (
    expected_improvement,
    upper_confidence_bound,
    probability_improvement,
    thompson_sampling,
)
from ..acquisition.batch import select_batch, estimate_lipschitz


class BOLoop:
    """Bayesian Optimization loop with ask/tell interface.

    Parameters
    ----------
    acquisition : str, default='ucb'
        Acquisition function: 'ucb', 'ei', 'pi', 'thompson'.
    acq_params : dict or None
        Parameters for the acquisition function.
        - UCB: {'beta': 3.0}
        - EI/PI: {'xi': 0.01}
    batch_size : int, default=4
        Number of points to select per ask() call.
    surrogate : sklearn estimator or None
        Custom surrogate model. If None, uses GP with Matern(nu=2.5).
    seed : int, default=42
        Random seed for GP and Thompson sampling.

    Examples
    --------
    >>> loop = BOLoop(acquisition='ucb', acq_params={'beta': 3.0}, batch_size=4)
    >>> loop.tell(X_initial, y_initial)
    >>> candidates = loop.ask(X_candidates)
    >>> # candidates['points'], candidates['indices'], candidates['acq_values']
    >>> loop.tell(X_new, y_new)
    """

    def __init__(self, acquisition='ucb', acq_params=None, batch_size=4,
                 surrogate=None, seed=42):
        self.acquisition = acquisition
        self.acq_params = acq_params or {}
        self.batch_size = batch_size
        self.seed = seed

        if surrogate is not None:
            self.surrogate = surrogate
        else:
            self.surrogate = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                n_restarts_optimizer=10,
                random_state=seed,
            )

        self.X_observed_ = None
        self.y_observed_ = None
        self.round_ = 0
        self.history_ = []

    def tell(self, X, y):
        """Provide observations and fit the surrogate.

        Parameters
        ----------
        X : array-like, shape (n, d)
            Observed feature values.
        y : array-like, shape (n,)
            Observed target values.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if self.X_observed_ is None:
            self.X_observed_ = X
            self.y_observed_ = y
        else:
            self.X_observed_ = np.vstack([self.X_observed_, X])
            self.y_observed_ = np.concatenate([self.y_observed_, y])

        self.surrogate.fit(self.X_observed_, self.y_observed_)
        self.round_ += 1

    def ask(self, X_candidates):
        """Select next batch of points to evaluate.

        Parameters
        ----------
        X_candidates : array-like, shape (n, d)
            Candidate points to choose from.

        Returns
        -------
        dict with keys:
            'points' : array, shape (batch_size, d)
            'indices' : list of int
            'acq_values' : list of float
            'diagnostics' : dict
        """
        if self.X_observed_ is None:
            raise ValueError("Must call tell() before ask()")

        X_candidates = np.asarray(X_candidates, dtype=np.float64)
        mu, std = self.surrogate.predict(X_candidates, return_std=True)
        y_best = float(self.y_observed_.max())

        # Compute acquisition values
        acq_values = self._compute_acquisition(mu, std, y_best)

        # Select batch with penalization
        L = estimate_lipschitz(self.X_observed_, self.y_observed_)
        result = select_batch(
            X_candidates, acq_values, mu, std, y_best,
            batch_size=self.batch_size,
            L=L,
            seed=self.seed,
        )

        # Store in history
        self.history_.append({
            'round': self.round_,
            'n_candidates': len(X_candidates),
            'selected_indices': result['indices'],
            'acq_values': result['acq_values'],
        })

        return result

    def _compute_acquisition(self, mu, std, y_best):
        """Compute acquisition values based on self.acquisition type."""
        if self.acquisition == 'ucb':
            beta = self.acq_params.get('beta', 2.0)
            return upper_confidence_bound(mu, std, beta=beta)
        elif self.acquisition == 'ei':
            xi = self.acq_params.get('xi', 0.01)
            return expected_improvement(mu, std, y_best, xi=xi)
        elif self.acquisition == 'pi':
            xi = self.acq_params.get('xi', 0.01)
            return probability_improvement(mu, std, y_best, xi=xi)
        elif self.acquisition == 'thompson':
            return thompson_sampling(mu, std, seed=self.seed + self.round_)
        else:
            raise ValueError(f"Unknown acquisition: {self.acquisition}. "
                             f"Choose from 'ucb', 'ei', 'pi', 'thompson'.")
