"""Design space generation: grid, simplex sampling, uniformity evaluation."""

from .grid import create_design_space
from .simplex import sample_sobol_simplex, sample_random_simplex
from .uniformity import evaluate_uniformity
