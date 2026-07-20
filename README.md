# proj_mg_pub

Self-contained repository reproducing the publication figures for the microgreen
material-optimization study. Every notebook runs end-to-end using only the data and
**pre-trained** models in this repo — no model training, no external paths.

```bash
conda env create -f environment.yml
conda activate proj_mg_pub
jupyter notebook          # run any 0N_*.ipynb from the repo root
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

- `data/` — CSVs and raw inputs (incl. `corpus.jsonl` for NB08, microgravity accelerometer trace for NB07)
- `models/` — pre-trained artifacts (`all_properties/`, `BO/`, stress ensemble; NB03 committee build-result under `BO/committee_v7pub/`)
- `utils/`, `matsci_ml/`, `mltoolkit/` — vendored project libraries (`matsci_ml/` also holds the `stress_ensemble` helper)
- `sem_porosity/`, `auto_compression/`, `material_frequency/` — notebook-specific modules + inputs
- `plots/` — figure outputs, one subfolder per notebook group: `citation_frequency/`, `all_properties/`, `bo/`, `compression/`, `microgravity/`, `o2/`
- `reports/` — corpus bibliography: `corpus_articles.csv` lists all literature in the NB08 corpus
