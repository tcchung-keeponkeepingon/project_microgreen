from .evaluation_metrics import evaluation_metrics
from .outlier_detection import detect_outliers
from .oversampling_by_hist import oversample_by_hist, oversample_by_hist_v2
from .mlp_oversampled import MLPRegressorOversampled
from .scatter import scatter_by_mass, scatter_by_mass_go
from .composition_design_space import create_material_design_space
from .sampling_simplex import sample_random_simplex, sample_sobol_simplex, evaluate_uniformity, create_2d_projections, create_3d_plot
from .acquisition_functions import expected_improvement, upper_confidence_bound, probability_improvement, local_penalization, select_batch_with_penalization, calculate_batch_diversity, acquisition_score_dist
from .acquisition_functions_v2 import (
    log_expected_improvement,
    expected_improvement as expected_improvement_v2,
    probability_improvement as probability_improvement_v2,
    upper_confidence_bound as upper_confidence_bound_v2,
    select_batch,
    adaptive_local_penalization,
    estimate_lipschitz,
    multi_objective_acquisition as multi_objective_acquisition_v2,
    rank_normalize,
)
from .data_augmentation import data_aug_gaussian
from .mlp_oversampled import MLPRegressorOversampled
from .target_transforms import LogTargetTransformer, PowerTransformer
from .mlp_ensemble import MLPEnsemble, PyTorchEnsemble

# Version info
__version__ = '1.0.0'
__author__ = 'Peter Chung'