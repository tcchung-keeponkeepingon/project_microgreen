"""Data utilities: outlier detection, augmentation, oversampling, splitting, summary."""

from .outlier_detection import detect_outliers, detect_outliers_multivariate
from .augmentation import data_aug_gaussian, data_aug_compositional
from .oversampling import oversample_by_hist
from .splitting import split_by_ids
from .summary import data_summary
