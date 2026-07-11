import numpy as np
import pandas as pd

def generate_composition_grid(n_materials=4, n_points=21, material_names=None):
    grid = np.linspace(0, 1, n_points)
    coords = np.stack([g.flatten() for g in np.meshgrid(*[grid] * (n_materials - 1))]).T
    
    # Keep compositions that sum to ≤1
    valid = coords[coords.sum(axis=1) <= 1.0]
    last_fraction = (1 - valid.sum(axis=1)).reshape(-1, 1)
    compositions = np.hstack([valid, last_fraction])
    
    if material_names is None:
        material_names = [f'Material_{i+1}' for i in range(n_materials)]
    
    return pd.DataFrame(compositions, columns=material_names)


def replicate_grid_with_mass_loadings(base_grid, mass_loadings):
    grids = [base_grid.copy().assign(MassLoading=loading) for loading in mass_loadings]
    return pd.concat(grids, ignore_index=True)


def create_material_design_space(n_materials=4, n_points=21, 
                                material_names=None, mass_loadings=None):
    grid = generate_composition_grid(n_materials, n_points, material_names)
    return grid if mass_loadings is None else replicate_grid_with_mass_loadings(grid, mass_loadings)