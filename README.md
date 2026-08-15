# Robotics–Machine Learning Accelerated Discovery of Aerogel-Induced Hydrogels for Extraterrestrial Microgreen Farming

Reproduce the figures/analysis for Robotics–Machine Learning Accelerated Discovery of
Aerogel-Induced Hydrogels for Extraterrestrial Microgreen Farming study.

```bash
conda env create -f environment.yml
conda activate proj_mg_pub  
jupyter notebook          
```

## Notebooks

| Notebook | Figures |
|----------|---------|
| `01_all_properties.ipynb` | Feasible-design-space maps: grade / water / stress raw + predicted, combined pass |
| `02_bo_v7_heatmaps_3d.ipynb` | 3-D BO surfaces: GP mean (18/24/30 pts) + UCB acquisition (init / R3) |
| `03_bo_v7_shap.ipynb` | SHAP feature importance for the 6-member ANN committee |
| `04_compression_robustness.ipynb` | Automated vs manual compression: mean±SD curves, Welch t-test, TOST equivalence |
| `05_sem_porosity.ipynb` | SEM pore-overlay figures (Otsu + dark-floor segmentation) |
| `06_o2_and_plant.ipynb` | Relative O₂ and Kale/Amaranth areal fresh weight |
| `07_microgravity.ipynb` | Accelerometer 2-D time series, 3-D trajectory, net acceleration |
| `08_material_frequency.ipynb` | Literature role-frequency bars + role-level corpus partition |

## Layout

- `data/` — CSVs and raw inputs for the notebooks:
  - `db_all_properties.csv` — 168 GUM/ALG/PVA/CMC formulations with mass loading, grade, water retention and stress at 25% (NB01)
  - `mustard_fresh_weight.csv` — 30 formulations with mustard areal fresh weight ± SD (NB02, NB03)
  - `o2_relative.csv` — relative O₂ (%), Pure GUM vs Best formulation, 5 reps each (NB06)
  - `kale_amaranth_afw.csv` — kale/amaranth areal fresh weight (mg/cm²), Pure GUM vs Best formulation, 3 reps each (NB06)
  - `accelerometer.txt` — WT901BLE microgravity trace (NB07)
- `models/` — pre-trained artifacts (`all_properties/`, `BO/`, stress ensemble; NB03 committee build-result under `BO/committee_v7pub/`)
- `utils/`, `matsci_ml/`, `mltoolkit/` — vendored project libraries (`matsci_ml/` also holds the `stress_ensemble` helper)
- `sem_porosity/`, `auto_compression/`, `material_frequency/` — notebook-specific modules + inputs
- `plots/` — figure outputs, one subfolder per notebook group: `citation_frequency/`, `all_properties/`, `bo/`, `compression/`, `microgravity/`, `o2/`
- `reports/` — corpus bibliography: `corpus_articles.csv` lists all literature in the NB08 corpus
