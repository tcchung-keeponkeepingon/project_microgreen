"""Target transformation utilities for handling skewed distributions."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LogTargetTransformer(BaseEstimator, TransformerMixin):
    """
    Log transformation for right-skewed target variables.

    Uses log1p (log(1+y)) for forward transform and expm1 (exp(y)-1) for inverse.
    This handles values near zero gracefully.

    Parameters
    ----------
    offset : float, default=0.0
        Additional offset to add before log transform. Useful if target
        has negative values: y_transformed = log1p(y + offset)

    Attributes
    ----------
    min_value_ : float
        Minimum value seen during fit (for validation)
    """

    def __init__(self, offset=0.0):
        self.offset = offset
        self.min_value_ = None

    def fit(self, y, X=None):
        """
        Fit the transformer.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values
        X : ignored
            Not used, present for API consistency

        Returns
        -------
        self
        """
        y = np.asarray(y).ravel()
        self.min_value_ = y.min()

        if self.min_value_ + self.offset < 0:
            raise ValueError(
                f"Minimum value ({self.min_value_}) + offset ({self.offset}) = "
                f"{self.min_value_ + self.offset} is negative. "
                f"Increase offset to at least {-self.min_value_}."
            )
        return self

    def transform(self, y):
        """
        Apply log1p transformation.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        y_transformed : ndarray of shape (n_samples,)
            Log-transformed values: log(1 + y + offset)
        """
        y = np.asarray(y).ravel()
        return np.log1p(y + self.offset)

    def inverse_transform(self, y_transformed):
        """
        Apply inverse transformation (expm1).

        Parameters
        ----------
        y_transformed : array-like of shape (n_samples,)
            Log-transformed values

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Original-scale values: exp(y_transformed) - 1 - offset
        """
        y_transformed = np.asarray(y_transformed).ravel()
        return np.expm1(y_transformed) - self.offset

    def fit_transform(self, y, X=None):
        """Fit and transform in one step."""
        return self.fit(y, X).transform(y)


class PowerTransformer(BaseEstimator, TransformerMixin):
    """
    Box-Cox like power transformation for right-skewed targets.

    Applies y^lambda transformation where lambda is chosen to minimize skewness
    or specified directly.

    Parameters
    ----------
    power : float or 'auto', default='auto'
        Power parameter. If 'auto', searches for optimal power to minimize skewness.
    power_range : tuple, default=(0.1, 2.0)
        Range to search for optimal power when power='auto'

    Attributes
    ----------
    power_ : float
        The power parameter used for transformation
    """

    def __init__(self, power='auto', power_range=(0.1, 2.0)):
        self.power = power
        self.power_range = power_range
        self.power_ = None

    def fit(self, y, X=None):
        """Fit the transformer, finding optimal power if needed."""
        from scipy import stats

        y = np.asarray(y).ravel()

        if y.min() <= 0:
            raise ValueError("PowerTransformer requires strictly positive values")

        if self.power == 'auto':
            # Search for power that minimizes skewness
            powers = np.linspace(self.power_range[0], self.power_range[1], 50)
            skewnesses = [abs(stats.skew(np.power(y, p))) for p in powers]
            self.power_ = powers[np.argmin(skewnesses)]
        else:
            self.power_ = self.power

        return self

    def transform(self, y):
        """Apply power transformation."""
        y = np.asarray(y).ravel()
        return np.power(y, self.power_)

    def inverse_transform(self, y_transformed):
        """Apply inverse power transformation."""
        y_transformed = np.asarray(y_transformed).ravel()
        return np.power(y_transformed, 1.0 / self.power_)

    def fit_transform(self, y, X=None):
        """Fit and transform in one step."""
        return self.fit(y, X).transform(y)
