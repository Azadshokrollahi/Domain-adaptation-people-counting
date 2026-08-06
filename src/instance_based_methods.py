"""
Instance-Based Domain Adaptation Transfer Learning for People Counting
RNN/LSTM/Transformer + KMM / ULSIF / RULSIF / TrAdaBoostR2 / IWC
                    + KMM+TSBW / ULSIF+TSBW / RULSIF+TSBW / IWC+TSBW
                    + KMM+TSW  / ULSIF+TSW  / RULSIF+TSW  / IWC+TSW

Reads:
  • configs/rooms.yaml   → room paths, scenarios, feature_variants, selected_rows
  • src/config.py        → seeds, epochs, batch size, LR, paths

Outputs (per variant):
  • results/instance_based_transfer/all_instance_based_homogeneous.csv
  • results/instance_based_transfer/all_instance_based_heterogeneous.csv

Methods (15 per scenario × model × ws):
  0.  Target-Only         — No source, target train only (baseline)
  1.  Baseline            — Source + target, uniform weights (baseline)
  2.  KMM                 — Kernel Mean Matching (Huang et al., 2006)
  3.  ULSIF               — Unconstrained Least-Squares Importance Fitting (Kanamori 2009)
  4.  RULSIF              — Relative ULSIF (Yamada et al., 2013)
  5.  TrAdaBoostR2        — Transfer AdaBoost for Regression (Pardoe & Stone, 2010)
  6.  IWC                 — Importance Weighted Classifier (Bickel et al., 2009)
  7.  KMM + TSBW          — KMM density-ratio × TSBW label-aware weight
  8.  ULSIF + TSBW        — ULSIF density-ratio × TSBW label-aware weight
  9.  RULSIF + TSBW       — RULSIF density-ratio × TSBW label-aware weight
  10. IWC + TSBW          — IWC density-ratio × TSBW label-aware weight
  11. KMM + TSW           — KMM density-ratio × TSW label-aware weight
  12. ULSIF + TSW         — ULSIF density-ratio × TSW label-aware weight
  13. RULSIF + TSW        — RULSIF density-ratio × TSW label-aware weight
  14. IWC + TSW           — IWC density-ratio × TSW label-aware weight

Heterogeneous mode:
  • Each room uses its own feature list from rooms.yaml.
  • If a room has no `heterogeneous:`, fall back to its `homogeneous:` features.
  • Scenarios where src_dim ≠ tgt_dim are SKIPPED (instance-based weight
    computation requires matching input dimensions).

TSBW (matches parameter_based_methods.py and feature_based_methods.py):
  FREQUENCY_EPS=1.0, TSBW_FLOOR=0.3, TSBW_ABSENT=0.05,
  TSBW_USE_SQRT=True, WEIGHT_CLIP_MAX=10.0

TSW (matches parameter_based_methods.py):
  FREQUENCY_EPS=1.0, TSW_ABSENT=0.0,
  Linear ratio weighting — no sqrt, no floor.

Complexity timing (added per row):
  model_complexity_time_sec =
        source_train_time_sec
      + target_train_time_sec
      + weight_compute_time_sec
      + alignment_train_time_sec

  In this script:
      source_train_time_sec    = 0.0   (no source-only NN training)
      target_train_time_sec    = neural network training time
      weight_compute_time_sec  = KMM / ULSIF / RULSIF / TrAdaBoostR2 / IWC time
      alignment_train_time_sec = 0.0

Author: Azadshokrollahi
"""

import os, sys, logging, ast, traceback, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader

# ADAPT library — weight computation for KMM, ULSIF, RULSIF, IWC
from adapt.instance_based import KMM, ULSIF, RULSIF, IWC

# ─── Centralized config ─────────────────────────────────────────────────────
from config import (
    SEED, WINDOW_SIZES, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    RESULTS_ROOT,
    load_rooms,
)
from main_regression_pretrain import (
    RNNRegressor, LSTMRegressor, TransformerRegressor,
    compute_all_metrics, compute_occupied_metrics, compute_empty_metrics,
    compute_per_count_metrics, COLUMN_RENAMES, DROP_COLS,
    reset_seed, device,
)


##############################################################################
#  REPRODUCIBILITY
##############################################################################

def reset_all_seeds(seed=SEED):
    reset_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


reset_all_seeds(SEED)


def make_generator(seed=SEED):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


##############################################################################
#  TSBW CONSTANTS — must match parameter_based_methods.py exactly
##############################################################################

FREQUENCY_EPS   = 1.0
TSBW_FLOOR      = 0.3
TSBW_ABSENT     = 0.05
TSBW_USE_SQRT   = True
WEIGHT_CLIP_MAX = 10.0

# ── NEW: TSW ──────────────────────────────────────────────────────────────
TSW_ABSENT      = 0.05   # labels absent in target → weight 0 (hard exclusion)
# ── END NEW ───────────────────────────────────────────────────────────────


def compute_tsbw_label_weights(y_src, y_ref,
                               eps=FREQUENCY_EPS,
                               floor=TSBW_FLOOR,
                               absent=TSBW_ABSENT,
                               use_sqrt=TSBW_USE_SQRT):
    y_ref_int = np.clip(np.round(y_ref).astype(int), 0, None)
    max_y = int(y_ref_int.max()) if len(y_ref_int) > 0 else 0
    counts = np.bincount(y_ref_int, minlength=max_y + 1).astype(float)
    max_count = max(counts.max(), 1.0)

    table = {}
    for yv in range(max_y + 1):
        if counts[yv] > 0:
            ratio = (counts[yv] + eps) / (max_count + eps)
            w_val = np.sqrt(ratio) if use_sqrt else ratio
            table[int(yv)] = float(max(w_val, floor))
        else:
            table[int(yv)] = float(absent)

    y_src_int = np.round(y_src).astype(int)
    raw = np.empty(len(y_src_int), dtype=np.float64)
    for i, yv in enumerate(y_src_int):
        raw[i] = absent if (yv < 0 or yv > max_y) else table.get(int(yv), absent)
    return raw


# ── NEW: TSW ──────────────────────────────────────────────────────────────
def compute_tsw_label_weights(y_src, y_ref,
                              eps=FREQUENCY_EPS,
                              absent=TSW_ABSENT):
    """
    TSW: Target Sample Weighting.
    Linear proportional weighting based on target label frequency.
    No sqrt, no floor — raw frequency ratio.
    Source samples whose label is absent in target receive weight = TSW_ABSENT (0.0).

    Difference from TSBW:
      TSBW: sqrt(ratio) + floor=0.3   → smoothed, prevents extreme down-weighting
      TSW:  raw ratio,  no floor      → stronger emphasis on target-frequent labels
    """
    y_ref_int = np.clip(np.round(y_ref).astype(int), 0, None)
    max_y = int(y_ref_int.max()) if len(y_ref_int) > 0 else 0
    counts = np.bincount(y_ref_int, minlength=max_y + 1).astype(float)
    max_count = max(counts.max(), 1.0)

    table = {}
    for yv in range(max_y + 1):
        if counts[yv] > 0:
            table[int(yv)] = float((counts[yv] + eps) / (max_count + eps))
        else:
            table[int(yv)] = float(absent)

    y_src_int = np.round(y_src).astype(int)
    raw = np.empty(len(y_src_int), dtype=np.float64)
    for i, yv in enumerate(y_src_int):
        raw[i] = absent if (yv < 0 or yv > max_y) else table.get(int(yv), absent)
    return raw
# ── END NEW ───────────────────────────────────────────────────────────────


def normalize_and_clip(w, clip_max=WEIGHT_CLIP_MAX):
    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, 0)
    mean = max(w.mean(), 1e-8)
    return np.clip(w / mean, 0, clip_max).astype(np.float32)


##############################################################################
#  ADAPT LIBRARY WEIGHT EXTRACTION
##############################################################################

def compute_kmm_weights(X_src, X_tgt):
    # ── FIX ──────────────────────────────────────────────────────────────
    # cvxopt's QP solver (matrix()) accepts only float64. The pipeline feeds
    # float32 (load_room_raw → StandardScaler), which raises
    # "TypeError: buffer format not supported". Cast explicitly. (verified via check.py)
    X_src = np.ascontiguousarray(X_src, dtype=np.float64)
    X_tgt = np.ascontiguousarray(X_tgt, dtype=np.float64)
    # ── END FIX ──────────────────────────────────────────────────────────
    n_s = len(X_src)
    kmm = KMM(estimator=None, Xt=X_tgt, kernel="rbf", B=10,
              eps=(np.sqrt(n_s) - 1) / np.sqrt(n_s),
              max_size=2000, verbose=0, random_state=SEED)
    w = np.maximum(kmm.fit_weights(X_src, X_tgt), 0)
    print(f"      KMM: mean={w.mean():.4f}, max={w.max():.4f}")
    return w


def compute_ulsif_weights(X_src, X_tgt):
    ulsif = ULSIF(estimator=None, Xt=X_tgt, kernel="rbf",
                  gamma=[0.01, 0.1, 1.0, 10.0],
                  lambdas=[0.01, 0.1, 1.0, 10.0],
                  max_centers=100, verbose=0, random_state=SEED)
    w = np.maximum(ulsif.fit_weights(X_src, X_tgt), 0)
    print(f"      ULSIF: best_params={ulsif.best_params_}, mean={w.mean():.4f}, max={w.max():.4f}")
    return w


def compute_rulsif_weights(X_src, X_tgt):
    rulsif = RULSIF(estimator=None, Xt=X_tgt, kernel="rbf", alpha=0.1,
                    gamma=[0.01, 0.1, 1.0, 10.0],
                    lambdas=[0.01, 0.1, 1.0, 10.0],
                    max_centers=100, verbose=0, random_state=SEED)
    w = np.maximum(rulsif.fit_weights(X_src, X_tgt), 0)
    print(f"      RULSIF: best_params={rulsif.best_params_}, mean={w.mean():.4f}, max={w.max():.4f}")
    return w


def compute_iwc_weights(X_src, X_tgt):
    iwc = IWC(estimator=None, Xt=X_tgt, classifier=None,
              verbose=0, random_state=SEED)
    w = np.maximum(iwc.fit_weights(X_src, X_tgt), 0)
    print(f"      IWC: mean={w.mean():.4f}, max={w.max():.4f}")
    return w


##############################################################################
#  SELF-CONTAINED: TrAdaBoostR2
##############################################################################

def compute_tradaboost_weights(X_src, y_src, X_tgt, y_tgt,
                               n_estimators=10, max_depth=6):
    n_s, n_t = len(X_src), len(X_tgt)
    X_all = np.vstack([X_src, X_tgt])
    y_all = np.concatenate([y_src, y_tgt])

    w = np.ones(n_s + n_t) / (n_s + n_t)
    beta_t = 1.0 / (1.0 + np.sqrt(2.0 * np.log(n_s) / max(n_estimators, 1)))

    for t in range(n_estimators):
        w_norm = w / w.sum()
        learner = DecisionTreeRegressor(max_depth=max_depth, random_state=SEED + t)
        learner.fit(X_all, y_all, sample_weight=w_norm)

        errors = np.abs(learner.predict(X_all) - y_all) / max(y_all.max() - y_all.min(), 1e-8)
        tgt_err = errors[n_s:]
        avg_tgt_err = np.clip(
            (w_norm[n_s:] * tgt_err).sum() / max(w_norm[n_s:].sum(), 1e-8),
            1e-8, 1 - 1e-8)

        beta = avg_tgt_err / (1 - avg_tgt_err)
        if beta >= 1:
            break

        w[:n_s] *= np.power(beta_t, errors[:n_s])
        w[n_s:] *= np.power(beta, -tgt_err)
        w = np.maximum(w, 1e-10)

    print(f"      TrAdaBoostR2: T={n_estimators}, mean={w[:n_s].mean():.6f}, "
          f"max={w[:n_s].max():.6f}")
    return w[:n_s]


##############################################################################
#  WEIGHT DISPATCHER
##############################################################################

def compute_ib_weights_raw(method, src_scaled, tgt_scaled, src_y=None, tgt_y=None):
    dispatch = {
        'KMM':          lambda: compute_kmm_weights(src_scaled, tgt_scaled),
        'ULSIF':        lambda: compute_ulsif_weights(src_scaled, tgt_scaled),
        'RULSIF':       lambda: compute_rulsif_weights(src_scaled, tgt_scaled),
        'TrAdaBoostR2': lambda: compute_tradaboost_weights(src_scaled, src_y, tgt_scaled, tgt_y),
        'IWC':          lambda: compute_iwc_weights(src_scaled, tgt_scaled),
    }
    return dispatch[method]()


def compute_adaptation_weights(method, src_scaled, tgt_scaled, src_y=None, tgt_y=None,
                               apply_tsbw=False,
                               apply_tsw=False):        # ── NEW: TSW ──
    n_s, n_t = len(src_scaled), len(tgt_scaled)
    base_method = (method.replace('+TSBW', '')
                         .replace('+TSW', ''))          # ── NEW: TSW ──
    print(f"    Computing {method} weights (src={n_s}, tgt={n_t})...")

    t0 = time.perf_counter()
    sw_raw = compute_ib_weights_raw(base_method, src_scaled, tgt_scaled, src_y, tgt_y)

    if apply_tsbw:
        if src_y is None or tgt_y is None:
            raise ValueError("TSBW requires src_y and tgt_y")
        w_label = compute_tsbw_label_weights(src_y, tgt_y)
        sw = normalize_and_clip(sw_raw * w_label, WEIGHT_CLIP_MAX)
        print(f"      TSBW labels: mean={w_label.mean():.4f}, "
              f"min={w_label.min():.4f}, max={w_label.max():.4f}")

    # ── NEW: TSW ──────────────────────────────────────────────────────────
    elif apply_tsw:
        if src_y is None or tgt_y is None:
            raise ValueError("TSW requires src_y and tgt_y")
        w_label = compute_tsw_label_weights(src_y, tgt_y)
        sw = normalize_and_clip(sw_raw * w_label, WEIGHT_CLIP_MAX)
        print(f"      TSW labels: mean={w_label.mean():.4f}, "
              f"min={w_label.min():.4f}, max={w_label.max():.4f}")
    # ── END NEW ───────────────────────────────────────────────────────────

    else:
        sw = np.clip(np.maximum(sw_raw, 0) / max(sw_raw.mean(), 1e-8),
                     0, WEIGHT_CLIP_MAX)

    weight_compute_time_sec = time.perf_counter() - t0
    print(f"    Final {method} — mean: {sw.mean():.4f}, std: {sw.std():.4f}, "
          f"min: {sw.min():.4f}, max: {sw.max():.4f}")
    print(f"    Weight compute time: {weight_compute_time_sec:.3f}s")

    return sw, np.ones(n_t), weight_compute_time_sec


##############################################################################
#  CONFIGURATION  (now reads from rooms.yaml + config.py)
##############################################################################

MODEL_TYPES      = ['RNN', 'LSTM', 'Transformer']
EPOCHS           = NUM_EPOCHS
FEATURE_VARIANTS = ['homogeneous']

OUTPUT_ROOT = str(RESULTS_ROOT / "instance_based_transfer")
ROOMS       = load_rooms()

TRANSFER_SCENARIOS = [
    ("Room_A", "Room_B"), ("Room_B", "Room_C"), ("Room_C", "Room_A"),
    ("Room_A", "Room_P"), ("Room_B", "Room_P"), ("Room_C", "Room_P"),
    ("Room_P", "Room_A"), ("Room_P", "Room_B"), ("Room_P", "Room_C"),
]

ADAPTATION_METHODS = [
    ('Target-Only',  'target_only',  'none'),
    ('Baseline',     'baseline',     'none'),
    ('KMM',          'KMM',          'none'),
    ('ULSIF',        'ULSIF',        'none'),
    ('RULSIF',       'RULSIF',       'none'),
    ('TrAdaBoostR2', 'TrAdaBoostR2', 'none'),
    ('IWC',          'IWC',          'none'),
    ('KMM+TSBW',     'KMM',          'tsbw'),
    ('ULSIF+TSBW',   'ULSIF',        'tsbw'),
    ('RULSIF+TSBW',  'RULSIF',       'tsbw'),
    ('IWC+TSBW',     'IWC',          'tsbw'),
    # ── NEW: TSW variants ─────────────────────────────────────────────────
    ('KMM+TSW',      'KMM',          'tsw'),
    ('ULSIF+TSW',    'ULSIF',        'tsw'),
    ('RULSIF+TSW',   'RULSIF',       'tsw'),
    ('IWC+TSW',      'IWC',          'tsw'),
    # ── END NEW ───────────────────────────────────────────────────────────
]


##############################################################################
#  VARIANT HELPERS  (with homogeneous fallback)
##############################################################################

def get_variant_features(room_id, variant):
    fv = ROOMS[room_id].get("feature_variants", {})
    if variant in fv:
        return fv[variant]
    if variant == "heterogeneous" and "homogeneous" in fv:
        return fv["homogeneous"]
    return None


def variant_scenario_supported(src_id, tgt_id, variant):
    return (get_variant_features(src_id, variant) is not None
            and get_variant_features(tgt_id, variant) is not None)


def build_room_cfg(room_id, variant):
    base = ROOMS[room_id]
    return {
        "name":          base["name"],
        "path":          base["path"],
        "scenario":      base["scenario"],
        "selected_rows": base["selected_rows"],
        "features":      get_variant_features(room_id, variant),
    }


##############################################################################
#  UTILITIES
##############################################################################

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def slugify(text):
    return (text.lower().replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')
            .replace('/', '_').replace('-', '_').replace('.', ''))


##############################################################################
#  DATA LOADING
##############################################################################

def load_room_raw(csv_path, features, room_name, selected_rows=None):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={k: v for k, v in COLUMN_RENAMES.items() if k in df.columns})

    if 'peoplecount_value' in df.columns:
        def _extract(val):
            try:
                return ast.literal_eval(val)[0] if isinstance(val, str) else val
            except Exception:
                return np.nan
        df['target'] = df['peoplecount_value'].apply(_extract)

    if 'target' not in df.columns:
        raise ValueError(f"No 'target' in {room_name}. Cols: {list(df.columns)}")

    if selected_rows is not None:
        s, e = selected_rows
        df = df.iloc[s:e].reset_index(drop=True)

    df_full = df.copy()
    drop = [c for c in DROP_COLS + ['peoplecount_value'] if c in df.columns]
    if drop:
        df = df.drop(columns=drop)

    if 'Hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df = df.drop(columns=['Hour'])
    if 'Day' in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df['Day'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['Day'] / 7)
        df = df.drop(columns=['Day'])

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"{room_name} missing features: {missing}")

    X = df[features].apply(pd.to_numeric, errors='coerce').ffill().bfill().values.astype(np.float32)
    y = pd.to_numeric(df['target'], errors='coerce').ffill().bfill().values.astype(np.float32)

    print(f"  Loaded {room_name}: {X.shape[0]} rows, {X.shape[1]} feat, mean={y.mean():.2f}")
    return X, y, df_full, features


##############################################################################
#  SPLITTING + SEQUENCES
##############################################################################

def split_by_scenario(X, y, scenario):
    n = len(X)
    if scenario == 'scenario_1':
        tr_end, va_end = int(0.7 * n), int(0.8 * n)
        tr_X, tr_y = X[:tr_end], y[:tr_end]
        va_X, va_y = X[tr_end:va_end], y[tr_end:va_end]
        te_X, te_y = X[va_end:], y[va_end:]
        info = {'scenario': 'scenario_1', 'train_start': 0, 'train_end': tr_end,
                'val_start': tr_end, 'val_end': va_end,
                'test_start': va_end, 'test_end': n}
    elif scenario == 'scenario_2':
        te_end, va_end = int(0.2 * n), int(0.3 * n)
        te_X, te_y = X[:te_end], y[:te_end]
        va_X, va_y = X[te_end:va_end], y[te_end:va_end]
        tr_X, tr_y = X[va_end:], y[va_end:]
        info = {'scenario': 'scenario_2', 'test_start': 0, 'test_end': te_end,
                'val_start': te_end, 'val_end': va_end,
                'train_start': va_end, 'train_end': n}
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    print(f"  Split ({scenario}): Train={len(tr_X)}, Val={len(va_X)}, Test={len(te_X)}")
    return tr_X, tr_y, va_X, va_y, te_X, te_y, info


def create_sequences(X, y, ws):
    Xs, ys = [], []
    for i in range(len(X) - ws + 1):
        Xs.append(X[i:i + ws])
        ys.append(y[i + ws - 1])
    return np.array(Xs), np.array(ys)


##############################################################################
#  DATA PREPARATION
##############################################################################

def _make_loader(X, y, bs, shuffle):
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32)),
        batch_size=bs, shuffle=shuffle,
        generator=make_generator(SEED) if shuffle else None,
        worker_init_fn=seed_worker,
    )


def prepare_target_only_data(tgt_tr_X, tgt_tr_y, tgt_va_X, tgt_va_y,
                              tgt_te_X, tgt_te_y, ws, bs=32):
    scaler = StandardScaler().fit(tgt_tr_X)
    X_tr, y_tr = create_sequences(scaler.transform(tgt_tr_X), tgt_tr_y, ws)
    X_va, y_va = create_sequences(scaler.transform(tgt_va_X), tgt_va_y, ws)
    X_te, y_te = create_sequences(scaler.transform(tgt_te_X), tgt_te_y, ws)

    for name, arr in [("Train", y_tr), ("Val", y_va), ("Test", y_te)]:
        if len(arr) == 0:
            raise ValueError(f"{name} has 0 sequences ws={ws}")

    return {'train_loader': _make_loader(X_tr, y_tr, bs, True),
            'val_loader':   _make_loader(X_va, y_va, bs, False),
            'test_loader':  _make_loader(X_te, y_te, bs, False),
            'input_size':   X_tr.shape[2],
            'scaler': scaler, 'has_weights': False,
            'weight_compute_time_sec': 0.0}


def prepare_transfer_data(src_X, src_y, tgt_tr_X, tgt_tr_y, tgt_va_X, tgt_va_y,
                          tgt_te_X, tgt_te_y, ws,
                          ib_method=None, apply_tsbw=False,
                          apply_tsw=False,              # ── NEW: TSW ──
                          bs=32):
    combined = np.vstack([src_X, tgt_tr_X])
    scaler = StandardScaler().fit(combined)
    src_scaled    = scaler.transform(src_X)
    tgt_tr_scaled = scaler.transform(tgt_tr_X)

    source_weights = None
    weight_compute_time_sec = 0.0

    if ib_method is not None:
        try:
            # ── NEW: TSW label in method name ────────────────────────────
            if apply_tsbw:
                method_label = f"{ib_method}+TSBW"
            elif apply_tsw:
                method_label = f"{ib_method}+TSW"
            else:
                method_label = ib_method
            # ── END NEW ──────────────────────────────────────────────────
            source_weights, _, weight_compute_time_sec = compute_adaptation_weights(
                method_label, src_scaled, tgt_tr_scaled, src_y, tgt_tr_y,
                apply_tsbw=apply_tsbw,
                apply_tsw=apply_tsw)                    # ── NEW: TSW ──
        except Exception as e:
            print(f"    {ib_method} (tsbw={apply_tsbw}, tsw={apply_tsw}) "  # ── NEW: TSW ──
                  f"failed: {e}. Falling back to uniform.")
            traceback.print_exc()                       # ── FIX: surface the real error ──
            weight_compute_time_sec = 0.0

    X_src, y_src = create_sequences(src_scaled, src_y, ws)
    X_tgt, y_tgt = create_sequences(tgt_tr_scaled, tgt_tr_y, ws)
    X_va,  y_va  = create_sequences(scaler.transform(tgt_va_X), tgt_va_y, ws)
    X_te,  y_te  = create_sequences(scaler.transform(tgt_te_X), tgt_te_y, ws)

    for name, arr in [("Src", y_src), ("Tgt", y_tgt), ("Val", y_va), ("Test", y_te)]:
        if len(arr) == 0:
            raise ValueError(f"{name} has 0 sequences ws={ws}")

    n_src, n_tgt = len(y_src), len(y_tgt)
    X_train = np.concatenate([X_src, X_tgt])
    y_train = np.concatenate([y_src, y_tgt])

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    has_weights = False
    if source_weights is not None:
        src_sw = np.array([source_weights[min(i + ws - 1, len(source_weights) - 1)]
                           for i in range(n_src)])
        seq_w = np.maximum(np.concatenate([src_sw, np.ones(n_tgt)]), 1e-8)
        train_ds = TensorDataset(X_t, y_t, torch.tensor(seq_w, dtype=torch.float32))
        has_weights = True
    else:
        train_ds = TensorDataset(X_t, y_t)

    return {
        'train_loader': DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            generator=make_generator(SEED), worker_init_fn=seed_worker),
        'val_loader':  _make_loader(X_va, y_va, bs, False),
        'test_loader': _make_loader(X_te, y_te, bs, False),
        'input_size': X_train.shape[2],
        'scaler': scaler,
        'has_weights': has_weights,
        'weight_compute_time_sec': weight_compute_time_sec,
    }


##############################################################################
#  MODEL + TRAIN + EVAL
##############################################################################

def build_model(model_type, input_size):
    if model_type == "RNN":
        return RNNRegressor(input_size=input_size, hidden_size=64, num_layers=2,
                            bidirectional_flag=False, aggregation_mode='one',
                            use_attention=False).to(device)
    elif model_type == "LSTM":
        return LSTMRegressor(input_size=input_size, hidden_size=64, num_layers=2,
                             bidirectional_flag=False, aggregation_mode='one',
                             use_attention=False).to(device)
    elif model_type == "Transformer":
        return TransformerRegressor(input_size=input_size, d_model=64, nhead=4,
                                    num_layers=2, dim_feedforward=128,
                                    dropout=0.1).to(device)
    raise ValueError(f"Unknown model: {model_type}")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def train_model(model, train_loader, val_loader, epochs, lr, has_weights=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_state, best_val = None, float('inf')
    t0 = time.perf_counter()

    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            if has_weights:
                bx, by, bw = [b.to(device) for b in batch]
            else:
                bx, by = batch[0].to(device), batch[1].to(device)
                bw = None

            optimizer.zero_grad()
            preds = model(bx)
            loss = ((bw * (preds - by) ** 2).sum() / bw.sum() if bw is not None
                    else nn.functional.mse_loss(preds, by))
            loss.backward()
            optimizer.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                vp.extend(model(bx.to(device)).cpu().numpy())
                vt.extend(by.numpy())
        vmae = mean_absolute_error(vt, vp)

        if (ep + 1) % max(1, epochs // 6) == 0:
            print(f"    Epoch [{ep + 1}/{epochs}] Val MAE: {vmae:.4f}")

        if vmae < best_val:
            best_val = vmae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.perf_counter() - t0
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"    Best Val MAE: {best_val:.4f}")
    print(f"    Target/train time: {train_time:.3f}s")
    return train_time


def evaluate_full(model, test_loader, label="Test"):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds.extend(model(bx.to(device)).cpu().numpy())
            trues.extend(by.numpy())
    preds, trues = np.array(preds), np.array(trues)

    m_all   = compute_all_metrics(trues, preds)
    m_occ   = compute_occupied_metrics(trues, preds)
    m_empty = compute_empty_metrics(trues, preds)
    m_pc    = compute_per_count_metrics(trues, preds)

    print(f"  {label}: MAE={m_all['mae']:.4f}, RMSE={m_all['rmse']:.4f}, "
          f"R²={m_all['r2']:.4f}, Acc±1={m_all['acc_pm1']:.1f}%")
    if m_occ:
        print(f"  {label} [Occ]: N={m_occ['sample_count']}, MAE={m_occ['mae']:.4f}, "
              f"Acc±1={m_occ['acc_pm1']:.1f}%")

    return {'all': m_all, 'occupied': m_occ, 'empty': m_empty, 'per_count': m_pc,
            'predictions': preds, 'targets': trues}


##############################################################################
#  SAVE HELPERS
##############################################################################

def save_outputs(model, metrics, tgt_df_full, tgt_split, ws, method_name,
                 model_type, out_dir):
    ensure_dir(out_dir)
    slug = slugify(method_name)
    torch.save(model.state_dict(), os.path.join(out_dir, f"weights_{slug}_ws{ws}.pth"))

    test_start = tgt_split['test_start'] + ws - 1
    df_test = tgt_df_full.iloc[test_start:tgt_split['test_end']].copy()
    p = metrics['predictions']
    min_len = min(len(df_test), len(p))
    df_test = df_test.iloc[:min_len]
    df_test['predicted_peoplecount'] = p[:min_len]
    df_test['predicted_rounded'] = np.round(np.clip(p[:min_len], 0, None)).astype(int)
    df_test.to_csv(os.path.join(out_dir, f"predicted_{slug}_ws{ws}.csv"), index=False)

    m = metrics['all']
    with open(os.path.join(out_dir, f"summary_{slug}_ws{ws}.txt"), "w") as f:
        f.write(f"Method: {method_name}\nModel: {model_type}\nWS: {ws}\n\n")
        f.write(f"=== ALL ===\nN={m['sample_count']}\nMAE={m['mae']:.4f}\n"
                f"RMSE={m['rmse']:.4f}\nR²={m['r2']:.4f}\nMedAE={m['medae']:.4f}\n"
                f"Acc±1={m['acc_pm1']:.2f}%\nAcc±2={m['acc_pm2']:.2f}%\n\n")
        if metrics['occupied']:
            mo = metrics['occupied']
            f.write(f"=== OCCUPIED ===\nN={mo['sample_count']}\nMAE={mo['mae']:.4f}\n"
                    f"Acc±1={mo['acc_pm1']:.2f}%\n"
                    f"Exact={mo['exact_match']} ({mo['exact_match_pct']:.2f}%)\n\n")
        if metrics['empty']:
            me = metrics['empty']
            f.write(f"=== EMPTY ===\nN={me['sample_count']}\nMAE={me['mae']:.4f}\n"
                    f"Acc±1={me['acc_pm1']:.2f}%\n")


##############################################################################
#  COMPARISON PLOTS
##############################################################################

def create_comparison_plots(results, ws, scenario_name, model_type, out_dir):
    results = [r for r in results if r is not None]
    if not results:
        return

    methods = [r['method'] for r in results]
    n = len(methods)
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(n)]

    metrics_list = [('mae', 'MAE'), ('rmse', 'RMSE'), ('r2', 'R²'),
                    ('medae', 'MedAE'), ('acc_pm1', 'Acc±1 (%)'), ('acc_pm2', 'Acc±2 (%)')]

    fig, axes = plt.subplots(2, 3, figsize=(24, 11))
    fig.suptitle(f'Instance-Based DA: {scenario_name} | {model_type} | ws={ws}',
                 fontsize=14, fontweight='bold')

    for idx, (key, ylabel) in enumerate(metrics_list):
        ax = axes[idx // 3][idx % 3]
        vals = [r['metrics']['all'][key] for r in results]
        bars = ax.bar(range(n), vals, color=colors, edgecolor='black', alpha=0.85)
        ax.set_xticks(range(n))
        ax.set_xticklabels(methods, rotation=40, ha='right', fontsize=6.5)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{v:.3f}',
                    ha='center', va='bottom', fontsize=5.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'comparison_{model_type}_ws{ws}.png'),
                dpi=200, bbox_inches='tight')
    plt.close()


##############################################################################
#  RESULT ROW
##############################################################################

def build_result_row(scenario_name, src_id, tgt_id, model_type, ws,
                     mr, src_cfg, tgt_cfg, variant):
    m = mr['metrics']['all']
    row = {
        'scenario': scenario_name, 'source': src_id, 'target': tgt_id,
        'method': mr['method'], 'weight_scheme': mr.get('weight_scheme', 'none'),
        'variant': variant, 'model': model_type, 'window_size': ws,
        'features': ', '.join(tgt_cfg['features']),
        'src_scenario': src_cfg['scenario'], 'tgt_scenario': tgt_cfg['scenario'],
        'initialization': 'random',

        'source_train_time_sec':     mr.get('source_train_time_sec', 0.0),
        'target_train_time_sec':     mr.get('target_train_time_sec', 0.0),
        'weight_compute_time_sec':   mr.get('weight_compute_time_sec', 0.0),
        'alignment_train_time_sec':  mr.get('alignment_train_time_sec', 0.0),
        'model_complexity_time_sec': mr.get('model_complexity_time_sec', 0.0),

        'total_params':     mr.get('total_params', np.nan),
        'trainable_params': mr.get('trainable_params', np.nan),
        'frozen_params':    mr.get('frozen_params', np.nan),

        'test_mae': m['mae'], 'test_rmse': m['rmse'], 'test_r2': m['r2'],
        'test_medae': m['medae'], 'test_acc_pm1': m['acc_pm1'],
        'test_acc_pm2': m['acc_pm2'], 'test_samples': m['sample_count'],
        'occ_mae': np.nan, 'occ_rmse': np.nan, 'occ_r2': np.nan, 'occ_medae': np.nan,
        'occ_acc_pm1': np.nan, 'occ_acc_pm2': np.nan, 'occ_exact_pct': np.nan,
        'occ_samples': np.nan,
        'empty_mae': np.nan, 'empty_acc_pm1': np.nan, 'empty_samples': np.nan,
    }
    if mr['metrics']['occupied']:
        mo = mr['metrics']['occupied']
        row.update({'occ_mae': mo['mae'], 'occ_rmse': mo['rmse'], 'occ_r2': mo['r2'],
                    'occ_medae': mo['medae'], 'occ_acc_pm1': mo['acc_pm1'],
                    'occ_acc_pm2': mo['acc_pm2'], 'occ_exact_pct': mo['exact_match_pct'],
                    'occ_samples': mo['sample_count']})
    if mr['metrics']['empty']:
        me = mr['metrics']['empty']
        row.update({'empty_mae': me['mae'], 'empty_acc_pm1': me['acc_pm1'],
                    'empty_samples': me['sample_count']})
    return row


##############################################################################
#  MAIN
##############################################################################

def main():
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 110)
    print("INSTANCE-BASED DOMAIN ADAPTATION — HOMOGENEOUS + HETEROGENEOUS")
    print(f"Methods ({len(ADAPTATION_METHODS)}): {[m[0] for m in ADAPTATION_METHODS]}")
    print(f"Models: {MODEL_TYPES} | WS: {WINDOW_SIZES} | Epochs: {EPOCHS} | LR: {LEARNING_RATE}")
    print(f"Variants: {FEATURE_VARIANTS}")
    print(f"TSBW: floor={TSBW_FLOOR}, absent={TSBW_ABSENT}, sqrt={TSBW_USE_SQRT}, "
          f"eps={FREQUENCY_EPS}, clip={WEIGHT_CLIP_MAX}")
    # ── NEW: TSW ──────────────────────────────────────────────────────────
    print(f"TSW:  absent={TSW_ABSENT}, linear ratio (no sqrt, no floor), "
          f"eps={FREQUENCY_EPS}, clip={WEIGHT_CLIP_MAX}")
    # ── END NEW ───────────────────────────────────────────────────────────
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"User: Azadshokrollahi | {timestamp} UTC")
    print("=" * 110)

    ensure_dir(OUTPUT_ROOT)
    results_by_variant = {v: [] for v in FEATURE_VARIANTS}

    for variant in FEATURE_VARIANTS:
        print(f"\n\n{'#' * 110}")
        print(f"#  FEATURE VARIANT: {variant.upper()}")
        print(f"{'#' * 110}")

        valid_scenarios = [(s, t) for (s, t) in TRANSFER_SCENARIOS
                           if variant_scenario_supported(s, t, variant)]

        runnable = []
        for s, t in valid_scenarios:
            sf = get_variant_features(s, variant)
            tf = get_variant_features(t, variant)
            if len(sf) == len(tf):
                runnable.append((s, t))
            else:
                print(f"  ⏭ Skip {s} → {t}: dim mismatch "
                      f"({len(sf)} vs {len(tf)}) — instance-based methods require equal dims")

        print(f"\n  Runnable {variant} scenarios: {len(runnable)}/{len(TRANSFER_SCENARIOS)}")
        for s, t in runnable:
            print(f"    {s} → {t}  "
                  f"(src={get_variant_features(s, variant)}, tgt={get_variant_features(t, variant)})")

        if not runnable:
            print(f"  ⚠ No runnable scenarios for {variant}; skipping.")
            continue

        total = len(runnable) * len(MODEL_TYPES) * len(WINDOW_SIZES)
        counter = 0

        for sc_idx, (src_id, tgt_id) in enumerate(runnable):
            scenario_name = f"{src_id} → {tgt_id}"
            src_cfg = build_room_cfg(src_id, variant)
            tgt_cfg = build_room_cfg(tgt_id, variant)

            print(f"\n{'=' * 110}")
            print(f"[{variant}] SCENARIO {sc_idx + 1}/{len(runnable)}: {scenario_name}")
            print(f"  src features: {src_cfg['features']}")
            print(f"  tgt features: {tgt_cfg['features']}")
            print(f"{'=' * 110}")

            if not os.path.exists(src_cfg['path']) or not os.path.exists(tgt_cfg['path']):
                print("  ❌ Data file missing"); continue

            src_X, src_y, _, _ = load_room_raw(
                src_cfg['path'], src_cfg['features'],
                src_cfg['name'], src_cfg['selected_rows'])
            tgt_X, tgt_y, tgt_df_full, _ = load_room_raw(
                tgt_cfg['path'], tgt_cfg['features'],
                tgt_cfg['name'], tgt_cfg['selected_rows'])

            src_tr_X, src_tr_y, _, _, _, _, _ = split_by_scenario(
                src_X, src_y, src_cfg['scenario'])

            tgt_tr_X, tgt_tr_y, tgt_va_X, tgt_va_y, tgt_te_X, tgt_te_y, tgt_split = \
                split_by_scenario(tgt_X, tgt_y, tgt_cfg['scenario'])

            for model_type in MODEL_TYPES:
                for ws in WINDOW_SIZES:
                    counter += 1
                    print(f"\n  ── [{variant}][{counter}/{total}] {model_type} | ws={ws} ──")

                    sc_folder = ensure_dir(os.path.join(
                        OUTPUT_ROOT, variant,
                        f"{src_id}_to_{tgt_id}", f"{model_type}_ws{ws}"))

                    log_file = os.path.join(sc_folder, "instance_based_log.txt")
                    for h in logging.root.handlers[:]:
                        logging.root.removeHandler(h)
                    logging.basicConfig(
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(log_file, mode='a'),
                                  logging.StreamHandler(sys.stdout)],
                        force=True)

                    method_results = []

                    for label, base_method, weight_scheme in ADAPTATION_METHODS:
                        print(f"\n    ── {label} ──")
                        try:
                            reset_all_seeds(SEED)

                            if base_method == 'target_only':
                                data = prepare_target_only_data(
                                    tgt_tr_X, tgt_tr_y, tgt_va_X, tgt_va_y,
                                    tgt_te_X, tgt_te_y, ws, BATCH_SIZE)
                            elif base_method == 'baseline':
                                data = prepare_transfer_data(
                                    src_tr_X, src_tr_y, tgt_tr_X, tgt_tr_y,
                                    tgt_va_X, tgt_va_y, tgt_te_X, tgt_te_y,
                                    ws, ib_method=None,
                                    apply_tsbw=False, apply_tsw=False,  # ── NEW: TSW ──
                                    bs=BATCH_SIZE)
                            else:
                                apply_tsbw = (weight_scheme == 'tsbw')
                                apply_tsw  = (weight_scheme == 'tsw')  # ── NEW: TSW ──
                                data = prepare_transfer_data(
                                    src_tr_X, src_tr_y, tgt_tr_X, tgt_tr_y,
                                    tgt_va_X, tgt_va_y, tgt_te_X, tgt_te_y,
                                    ws, ib_method=base_method,
                                    apply_tsbw=apply_tsbw,
                                    apply_tsw=apply_tsw,                # ── NEW: TSW ──
                                    bs=BATCH_SIZE)

                            model = build_model(model_type, data['input_size'])

                            target_train_time_sec = train_model(
                                model, data['train_loader'], data['val_loader'],
                                EPOCHS, LEARNING_RATE, data['has_weights'])

                            source_train_time_sec    = 0.0
                            weight_compute_time_sec  = data.get('weight_compute_time_sec', 0.0)
                            alignment_train_time_sec = 0.0
                            model_complexity_time_sec = (
                                source_train_time_sec + target_train_time_sec
                                + weight_compute_time_sec + alignment_train_time_sec)

                            total_p, train_p, frozen_p = count_parameters(model)

                            metrics = evaluate_full(model, data['test_loader'], label)
                            save_outputs(model, metrics, tgt_df_full, tgt_split, ws,
                                         label, model_type, sc_folder)
                            joblib.dump(data['scaler'],
                                        os.path.join(sc_folder,
                                                     f"scaler_{slugify(label)}_ws{ws}.pkl"))

                            method_results.append({
                                'method': label,
                                'weight_scheme': weight_scheme,
                                'metrics': metrics,
                                'source_train_time_sec':     source_train_time_sec,
                                'target_train_time_sec':     target_train_time_sec,
                                'weight_compute_time_sec':   weight_compute_time_sec,
                                'alignment_train_time_sec':  alignment_train_time_sec,
                                'model_complexity_time_sec': model_complexity_time_sec,
                                'total_params':     total_p,
                                'trainable_params': train_p,
                                'frozen_params':    frozen_p,
                            })

                        except Exception as e:
                            print(f"    ❌ FAILED {label}: {e}")
                            logging.error(f"FAILED {label}: {e}")
                            traceback.print_exc()

                    method_results = [r for r in method_results if r is not None]
                    create_comparison_plots(method_results, ws, scenario_name,
                                            model_type, sc_folder)

                    rows = []
                    for r in method_results:
                        row = build_result_row(scenario_name, src_id, tgt_id,
                                               model_type, ws, r,
                                               src_cfg, tgt_cfg, variant)
                        rows.append(row)
                        results_by_variant[variant].append(row)

                    if rows:
                        pd.DataFrame(rows).to_csv(
                            os.path.join(sc_folder,
                                         f"metrics_{model_type}_ws{ws}.csv"),
                            index=False)

                    print(f"\n    {'Method':<22} {'Scheme':<8} {'MAE':<8} {'RMSE':<8} "
                          f"{'R²':<8} {'Acc±1':<8} {'OccMAE':<8} {'Complex(s)':<12}")
                    print("    " + "-" * 100)
                    for r in method_results:
                        ma = r['metrics']['all']
                        om = r['metrics']['occupied']
                        occ = om['mae'] if om else float('nan')
                        print(f"    {r['method']:<22} {r['weight_scheme']:<8} "
                              f"{ma['mae']:<8.4f} {ma['rmse']:<8.4f} "
                              f"{ma['r2']:<8.4f} {ma['acc_pm1']:<8.2f} {occ:<8.4f} "
                              f"{r['model_complexity_time_sec']:<12.3f}")

                    print(f"\n  ✓ [{variant}] {scenario_name} | {model_type} | "
                          f"ws={ws} — {len(method_results)}/{len(ADAPTATION_METHODS)}")

    # ─────────────────────────────────────────────────────────────────────────
    # MASTER CSVs
    # ─────────────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 110)
    print("FINAL RESULTS — INSTANCE-BASED TRANSFER")
    print("=" * 110)

    COLUMN_ORDER = [
        'scenario', 'source', 'target', 'method', 'weight_scheme',
        'variant', 'model', 'window_size', 'features',
        'src_scenario', 'tgt_scenario', 'initialization',
        'source_train_time_sec', 'target_train_time_sec',
        'weight_compute_time_sec', 'alignment_train_time_sec',
        'model_complexity_time_sec',
        'total_params', 'trainable_params', 'frozen_params',
        'test_mae', 'test_rmse', 'test_r2', 'test_medae',
        'test_acc_pm1', 'test_acc_pm2', 'test_samples',
        'occ_mae', 'occ_rmse', 'occ_r2', 'occ_medae',
        'occ_acc_pm1', 'occ_acc_pm2', 'occ_exact_pct', 'occ_samples',
        'empty_mae', 'empty_acc_pm1', 'empty_samples',
    ]

    for variant, rows in results_by_variant.items():
        if not rows:
            print(f"\n  [{variant}] No results.")
            continue

        df = pd.DataFrame(rows)
        df = df[[c for c in COLUMN_ORDER if c in df.columns]]
        out_csv = os.path.join(OUTPUT_ROOT, f"all_instance_based_{variant}.csv")
        df.to_csv(out_csv, index=False)

        print(f"\n  [{variant}] Master CSV: {out_csv}")
        print(f"  [{variant}] Total rows: {len(df)}")

        print(f"\n  [{variant}] BEST METHOD PER SCENARIO × MODEL (by MAE):")
        print(f"  {'Scenario':<25} {'Model':<12} {'Best':<22} "
              f"{'Scheme':<8} {'MAE':<8} {'Acc±1':<8} {'Complex(s)':<11}")
        print("  " + "-" * 110)
        for (sc, mdl), grp in df.groupby(['scenario', 'model']):
            best = grp.loc[grp['test_mae'].idxmin()]
            print(f"  {sc:<25} {mdl:<12} {best['method']:<22} "
                  f"{best['weight_scheme']:<8} "
                  f"{best['test_mae']:<8.4f} {best['test_acc_pm1']:<8.2f} "
                  f"{best['model_complexity_time_sec']:<11.3f}")

        # IB vs IB+TSBW vs IB+TSW summary
        print(f"\n  [{variant}] IB vs IB+TSBW vs IB+TSW (avg MAE / avg complexity):")
        for base in ['KMM', 'ULSIF', 'RULSIF', 'IWC']:
            base_df = df[df['method'] == base]
            tsbw_df = df[df['method'] == f'{base}+TSBW']
            tsw_df  = df[df['method'] == f'{base}+TSW']   # ── NEW: TSW ──
            if len(base_df) > 0:
                line = (f"    {base:<8}: MAE={base_df['test_mae'].mean():.4f}, "
                        f"Time={base_df['model_complexity_time_sec'].mean():.3f}s")
                if len(tsbw_df) > 0:
                    line += (f"  |  +TSBW: MAE={tsbw_df['test_mae'].mean():.4f}, "
                             f"Time={tsbw_df['model_complexity_time_sec'].mean():.3f}s")
                # ── NEW: TSW ──────────────────────────────────────────────
                if len(tsw_df) > 0:
                    line += (f"  |  +TSW:  MAE={tsw_df['test_mae'].mean():.4f}, "
                             f"Time={tsw_df['model_complexity_time_sec'].mean():.3f}s")
                # ── END NEW ───────────────────────────────────────────────
                print(line)

    print("\n" + "=" * 110)
    print("✓ INSTANCE-BASED TRANSFER COMPLETED!")
    print(f"User: Azadshokrollahi | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 110)


if __name__ == "__main__":
    main()