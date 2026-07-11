"""mltoolkit — Reusable ML toolkit for materials science and beyond."""

from .utils import set_all_seeds
from .evaluation.metrics import evaluation_metrics, compare_models
from .transforms.target import LogTargetTransformer, PowerTransformer
from .transforms.features import prepare_data
from .data.augmentation import data_aug_gaussian, data_aug_compositional
from .data.outlier_detection import detect_outliers, detect_outliers_multivariate
from .data.oversampling import oversample_by_hist
from .data.splitting import split_by_ids
from .data.summary import data_summary
from .design_space.grid import create_design_space
from .design_space.simplex import sample_sobol_simplex, sample_random_simplex
from .design_space.uniformity import evaluate_uniformity
from .models.ann import ANN, ANNTrainer, AugmentedANNTrainer, train_model
from .models.ensemble import MLPEnsemble, PyTorchEnsemble
from .models.wrappers import ScaledModel, MLPRegressorOversampled
from .models.prediction import BatchPredictor
from .acquisition.functions import (
    expected_improvement, log_expected_improvement,
    probability_improvement, upper_confidence_bound,
    lower_confidence_bound, thompson_sampling,
    rank_normalize, multi_objective_acquisition,
)
from .acquisition.batch import (
    select_batch, estimate_lipschitz, adaptive_local_penalization,
    calculate_batch_diversity, acquisition_diagnostics,
)
from .optimization.objective import create_cv_objective
from .optimization.param_spaces import (
    suggest_ann_params, suggest_svr_params, suggest_svc_params,
    suggest_rf_params, suggest_xgboost_params, suggest_catboost_params,
)
from .bo.loop import BOLoop
from .viz.heatmap import plot_prediction_heatmap, plot_acquisition_heatmap
from .viz.violin import plot_round_violin
from .viz.scatter3d import scatter_by_mass, scatter_by_mass_go
from .viz.defaults import PLOT_DEFAULTS
