"""Load-only stress OOF ensemble prediction (no training).

Loads the Caruana-weighted stress ensemble (`stress_ensemble_v2_seed10.pkl`,
SVR + ANN + XGBoost members) and predicts in target (kPa) space from raw
FEATURE_COLS. Distilled from `train_ensembles` in proj_mg_v2.
"""
import joblib
import numpy as np
import pandas as pd


def load_stress_bundle(path):
    b = joblib.load(path)
    b['members'] = b.get('members', b.get('members_refit'))
    b.setdefault('feature_scaling_mode', 'all')
    return b


def ensemble_predict(bundle, X_scaled):
    out = np.zeros(len(X_scaled), dtype=np.float64)
    for model, w, _family in bundle['members']:
        out += w * model.predict(X_scaled)
    return out


def predict_target_space(bundle, X_raw):
    """X_raw: unscaled FEATURE_COLS values. Returns target-space (kPa) preds."""
    if bundle['feature_scaling_mode'] == 'mass_only':
        Xs = X_raw.copy() if isinstance(X_raw, pd.DataFrame) else pd.DataFrame(X_raw, columns=bundle['feature_cols'])
        Xs['MassLoading'] = bundle['feature_scaler'].transform(Xs[['MassLoading']])
        Xs = Xs.values
    else:  # 'all'
        Xs = bundle['feature_scaler'].transform(X_raw)
    pred_s = ensemble_predict(bundle, Xs)
    pred = bundle['target_scaler'].inverse_transform(pred_s.reshape(-1, 1)).ravel()
    if bundle.get('target_clip') is not None:
        pred = np.clip(pred, *bundle['target_clip'])
    return pred
