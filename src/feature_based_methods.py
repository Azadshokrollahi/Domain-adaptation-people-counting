"""
Feature-Based Domain Adaptation for People Counting (Regression)

7 Methods × 3 weighting schemes (none, TSBW, TSW) = 21 runs per scenario × model × ws
ALL with RANDOM INITIALIZATION — no pretrained weights.

Reads:
  • configs/rooms.yaml   → room paths, scenarios, feature_variants
  • src/config.py        → seeds, epochs, hyperparams, paths

Outputs (per variant):
  • results/feature_based_transfer/all_feature_based_homogeneous.csv
  • results/feature_based_transfer/all_feature_based_heterogeneous.csv

Methods:
  1. CORAL        — Shallow covariance alignment on flattened features
  2. DeepCORAL    — Deep covariance alignment on encoder hidden states
  3. Deep MMD     — Maximum Mean Discrepancy on encoder hidden states
  4. PRED         — Source-only prediction appended as extra feature
  5. FA           — Feature Augmentation [shared, source-specific, target-specific]
  6. CCSA         — Label-aware contrastive semantic alignment (regression-adapted)
  7. WDGRL        — Wasserstein Distance Guided Representation Learning

Weight Schemes:
  none  — uniform weights
  TSBW  — sqrt frequency ratio + floor=0.3 (smoothed)
  TSW   — linear frequency ratio, no sqrt, no floor (stronger target emphasis)

Heterogeneous mode:
  • Each room uses its own feature list from rooms.yaml.
  • If a room has no `heterogeneous:`, fall back to its `homogeneous:` features.
  • Scenarios where src_dim ≠ tgt_dim are SKIPPED for feature-based methods
    (these methods require matching input dimensions).

Author: Azadshokrollahi
"""

import os, sys, ast, logging, traceback, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader
from scipy.linalg import sqrtm

# ─── Centralized config ─────────────────────────────────────────────────────
from config import (
    SEED, WINDOW_SIZES, NUM_EPOCHS,
    RESULTS_ROOT,
    load_rooms,
)
from main_regression_pretrain import (
    RNNRegressor, LSTMRegressor, TransformerRegressor,
    compute_all_metrics, compute_occupied_metrics, compute_empty_metrics,
    compute_per_count_metrics, reset_seed, device,
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
#  TSBW PARAMETERS (matches parameter_based_methods.py)
##############################################################################

WEIGHT_SCHEMES   = ['none', 'tsbw', 'tsw']              # ── NEW: TSW ──
FREQUENCY_EPS    = 1.0
WEIGHT_CLIP_MAX  = 10.0
TSBW_FLOOR       = 0.3
TSBW_ABSENT      = 0.05
TSBW_USE_SQRT    = True

# ── NEW: TSW ──────────────────────────────────────────────────────────────
TSW_ABSENT       = 0.05   # labels absent in target → weight 0 (hard exclusion)
# ── END NEW ───────────────────────────────────────────────────────────────


def compute_label_weight_table(y_ref, eps=FREQUENCY_EPS,
                               floor=TSBW_FLOOR, absent=TSBW_ABSENT,
                               use_sqrt=TSBW_USE_SQRT):
    y_int = np.clip(np.round(y_ref).astype(int), 0, None)
    max_y = int(y_int.max())
    counts = np.bincount(y_int, minlength=max_y + 1).astype(float)
    max_count = counts.max()

    table = {}
    for yv in range(max_y + 1):
        if counts[yv] > 0:
            ratio = (counts[yv] + eps) / (max_count + eps)
            w = np.sqrt(ratio) if use_sqrt else ratio
            table[int(yv)] = float(max(w, floor))
        else:
            table[int(yv)] = float(absent)
    table[-1] = float(absent)
    return table, max_y


# ── NEW: TSW ──────────────────────────────────────────────────────────────
def compute_tsw_label_weight_table(y_ref, eps=FREQUENCY_EPS,
                                    absent=TSW_ABSENT):
    """
    TSW: Target Sample Weighting.
    Linear proportional weighting — no sqrt, no floor.
    Labels absent in target → weight TSW_ABSENT (0.0).

    Difference from TSBW:
      TSBW: sqrt(ratio) + floor=0.3   → smoothed, prevents extreme down-weighting
      TSW:  raw ratio,  no floor      → stronger emphasis on target-frequent labels
    """
    y_int = np.clip(np.round(y_ref).astype(int), 0, None)
    max_y = int(y_int.max())
    counts = np.bincount(y_int, minlength=max_y + 1).astype(float)
    max_count = counts.max()

    table = {}
    for yv in range(max_y + 1):
        if counts[yv] > 0:
            ratio = (counts[yv] + eps) / (max_count + eps)
            table[int(yv)] = float(ratio)   # no sqrt, no floor
        else:
            table[int(yv)] = float(absent)
    table[-1] = float(absent)
    return table, max_y
# ── END NEW ───────────────────────────────────────────────────────────────


def apply_weight_table(y_samples, table, max_y_ref):
    absent_w = table.get(-1, TSBW_ABSENT)
    y_int = np.round(y_samples).astype(int)
    raw = np.empty(len(y_int), dtype=np.float64)
    for i, yv in enumerate(y_int):
        raw[i] = absent_w if (yv < 0 or yv > max_y_ref) else table.get(int(yv), absent_w)
    return raw


def normalize_and_clip(w, clip_max=WEIGHT_CLIP_MAX):
    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, 0)
    mean = max(w.mean(), 1e-8)
    w = np.clip(w / mean, 0, clip_max)
    return w.astype(np.float32)


def tsbw_source_weights(y_src, y_tgt_ref):
    table, max_y = compute_label_weight_table(y_tgt_ref)
    raw = apply_weight_table(y_src, table, max_y)
    return normalize_and_clip(raw)


# ── NEW: TSW ──────────────────────────────────────────────────────────────
def tsw_source_weights(y_src, y_tgt_ref):
    """TSW version of tsbw_source_weights — linear ratio, no sqrt, no floor."""
    table, max_y = compute_tsw_label_weight_table(y_tgt_ref)
    raw = apply_weight_table(y_src, table, max_y)
    return normalize_and_clip(raw)
# ── END NEW ───────────────────────────────────────────────────────────────


##############################################################################
#  CONFIGURATION
##############################################################################

MODEL_TYPES      = ['RNN', 'LSTM', 'Transformer']
EPOCHS           = NUM_EPOCHS
FEATURE_VARIANTS = ['homogeneous']

OUTPUT_ROOT = str(RESULTS_ROOT / "feature_based_transfer")
ROOMS       = load_rooms()

TRANSFER_SCENARIOS = [
    ("Room_A", "Room_B"), ("Room_B", "Room_C"), ("Room_C", "Room_A"),
    ("Room_A", "Room_P"), ("Room_B", "Room_P"), ("Room_C", "Room_P"),
    ("Room_P", "Room_A"), ("Room_P", "Room_B"), ("Room_P", "Room_C"),
]

ALL_METHODS = ["CORAL", "DeepCORAL", "DeepMMD", "PRED", "FA", "CCSA", "WDGRL"]


##############################################################################
#  VARIANT HELPERS
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

COLUMN_RENAMES = {'co2': 'co2_value', 'motion': 'motion_event_count'}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def build_model(model_type, input_size, hidden_size=64, num_layers=2):
    if model_type == 'RNN':
        return RNNRegressor(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional_flag=False, aggregation_mode='one',
                            use_attention=False).to(device)
    elif model_type == 'LSTM':
        return LSTMRegressor(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                             bidirectional_flag=False, aggregation_mode='one',
                             use_attention=False).to(device)
    elif model_type == 'Transformer':
        return TransformerRegressor(input_size=input_size, d_model=64, nhead=4,
                                    num_layers=num_layers, dim_feedforward=128,
                                    dropout=0.1).to(device)
    raise ValueError(f"Unknown model_type: {model_type}")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def add_complexity_info(result, model,
                        source_train_time_sec=0.0,
                        target_train_time_sec=0.0,
                        weight_compute_time_sec=0.0,
                        alignment_train_time_sec=0.0):
    total, trainable, frozen = count_parameters(model)
    result['source_train_time_sec']    = source_train_time_sec
    result['target_train_time_sec']    = target_train_time_sec
    result['weight_compute_time_sec']  = weight_compute_time_sec
    result['alignment_train_time_sec'] = alignment_train_time_sec
    result['model_complexity_time_sec'] = (source_train_time_sec + target_train_time_sec
                                           + weight_compute_time_sec + alignment_train_time_sec)
    result['total_params']     = total
    result['trainable_params'] = trainable
    result['frozen_params']    = frozen
    return result


def load_room_raw(csv_path, features, room_name, selected_rows=None):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={k: v for k, v in COLUMN_RENAMES.items() if k in df.columns})
    if 'peoplecount_value' in df.columns:
        def extract_target(val):
            try:
                return ast.literal_eval(val)[0] if isinstance(val, str) else val
            except Exception:
                return np.nan
        df['target'] = df['peoplecount_value'].apply(extract_target)
    if 'target' not in df.columns:
        raise ValueError(f"No 'target' column in {room_name}.")
    if selected_rows is not None:
        s, e = selected_rows
        df = df.iloc[s:e].reset_index(drop=True)
    df_full = df.copy()

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"{room_name} missing features: {missing}")

    X = df[features].apply(pd.to_numeric, errors='coerce').ffill().bfill().values.astype(np.float32)
    y = pd.to_numeric(df['target'], errors='coerce').ffill().bfill().values.astype(np.float32)
    print(f"    Loaded {room_name}: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"    Target: mean={y.mean():.2f}, std={y.std():.2f}, min={y.min():.0f}, max={y.max():.0f}")
    return X, y, df_full, features


def split_by_scenario(X, y, scenario):
    n = len(X)
    if scenario == "scenario_1":
        tr_end, va_end = int(n * 0.7), int(n * 0.8)
        tr_X, tr_y = X[:tr_end], y[:tr_end]
        va_X, va_y = X[tr_end:va_end], y[tr_end:va_end]
        te_X, te_y = X[va_end:], y[va_end:]
        info = {'scenario': 'scenario_1', 'train_start': 0, 'train_end': tr_end,
                'val_start': tr_end, 'val_end': va_end, 'test_start': va_end, 'test_end': n}
    elif scenario == "scenario_2":
        te_end, va_end = int(n * 0.2), int(n * 0.3)
        te_X, te_y = X[:te_end], y[:te_end]
        va_X, va_y = X[te_end:va_end], y[te_end:va_end]
        tr_X, tr_y = X[va_end:], y[va_end:]
        info = {'scenario': 'scenario_2', 'test_start': 0, 'test_end': te_end,
                'val_start': te_end, 'val_end': va_end, 'train_start': va_end, 'train_end': n}
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    print(f"    Split ({scenario}): Train={len(tr_X)}, Val={len(va_X)}, Test={len(te_X)}")
    return tr_X, tr_y, va_X, va_y, te_X, te_y, info


def create_sequences(X, y, ws):
    sx, sy = [], []
    for i in range(len(X) - ws + 1):
        sx.append(X[i:i + ws])
        sy.append(y[i + ws - 1])
    return np.array(sx), np.array(sy)


##############################################################################
#  DATA PREPARATION
##############################################################################

def _ld(X, y, bs, shuffle):
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32)),
        batch_size=bs, shuffle=shuffle,
        generator=make_generator(SEED) if shuffle else None,
        worker_init_fn=seed_worker,
    )


def _ld_w(X, y, w, bs, shuffle):
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32),
                      torch.tensor(w, dtype=torch.float32)),
        batch_size=bs, shuffle=shuffle,
        generator=make_generator(SEED) if shuffle else None,
        worker_init_fn=seed_worker,
    )


def prepare_source_target_data(src_cfg, tgt_cfg, ws, batch_size=32):
    src_X, src_y, _, _ = load_room_raw(src_cfg['path'], src_cfg['features'],
                                       src_cfg['name'], src_cfg['selected_rows'])
    tgt_X, tgt_y, tgt_df, _ = load_room_raw(tgt_cfg['path'], tgt_cfg['features'],
                                            tgt_cfg['name'], tgt_cfg['selected_rows'])

    if src_X.shape[1] != tgt_X.shape[1]:
        raise ValueError(f"Feature dim mismatch: src={src_X.shape[1]}, tgt={tgt_X.shape[1]}. "
                         f"Feature-based DA requires matching dims.")

    src_tr_X, src_tr_y, src_va_X, src_va_y, _, _, _ = split_by_scenario(src_X, src_y, src_cfg['scenario'])
    tgt_tr_X, tgt_tr_y, tgt_va_X, tgt_va_y, tgt_te_X, tgt_te_y, tgt_split = \
        split_by_scenario(tgt_X, tgt_y, tgt_cfg['scenario'])

    src_scaler = StandardScaler().fit(src_tr_X)
    tgt_scaler = StandardScaler().fit(tgt_tr_X)

    src_tr_seq_X, src_tr_seq_y = create_sequences(src_scaler.transform(src_tr_X), src_tr_y, ws)
    src_va_seq_X, src_va_seq_y = create_sequences(src_scaler.transform(src_va_X), src_va_y, ws)
    tgt_tr_seq_X, tgt_tr_seq_y = create_sequences(tgt_scaler.transform(tgt_tr_X), tgt_tr_y, ws)
    tgt_va_seq_X, tgt_va_seq_y = create_sequences(tgt_scaler.transform(tgt_va_X), tgt_va_y, ws)
    tgt_te_seq_X, tgt_te_seq_y = create_sequences(tgt_scaler.transform(tgt_te_X), tgt_te_y, ws)

    for name, arr in [("SrcTr", src_tr_seq_y), ("TgtTr", tgt_tr_seq_y),
                      ("TgtVal", tgt_va_seq_y), ("TgtTest", tgt_te_seq_y)]:
        if len(arr) == 0:
            raise ValueError(f"{name} has 0 sequences with ws={ws}")

    return {
        'src_train_loader': _ld(src_tr_seq_X, src_tr_seq_y, batch_size, True),
        'src_val_loader':   _ld(src_va_seq_X, src_va_seq_y, batch_size, False),
        'tgt_train_loader': _ld(tgt_tr_seq_X, tgt_tr_seq_y, batch_size, True),
        'tgt_val_loader':   _ld(tgt_va_seq_X, tgt_va_seq_y, batch_size, False),
        'tgt_test_loader':  _ld(tgt_te_seq_X, tgt_te_seq_y, batch_size, False),
        'X_tgt_test_t':     torch.tensor(tgt_te_seq_X, dtype=torch.float32),
        'input_size':       src_tr_seq_X.shape[2],
        'src_scaler': src_scaler, 'tgt_scaler': tgt_scaler,
        'tgt_df_full': tgt_df, 'tgt_split': tgt_split,
        'src_tr_X': src_tr_seq_X, 'src_tr_y': src_tr_seq_y,
        'src_va_X': src_va_seq_X, 'src_va_y': src_va_seq_y,
        'tgt_tr_X': tgt_tr_seq_X, 'tgt_tr_y': tgt_tr_seq_y,
        'tgt_va_X': tgt_va_seq_X, 'tgt_va_y': tgt_va_seq_y,
        'tgt_te_X': tgt_te_seq_X, 'tgt_te_y': tgt_te_seq_y,
    }


##############################################################################
#  EVAL + SAVE
##############################################################################

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
    print(f"    {label}: MAE={m_all['mae']:.4f}, RMSE={m_all['rmse']:.4f}, "
          f"R²={m_all['r2']:.4f}, Acc±1={m_all['acc_pm1']:.1f}%")
    if m_occ:
        print(f"    {label} [Occ]: N={m_occ['sample_count']}, MAE={m_occ['mae']:.4f}")
    return {'all': m_all, 'occupied': m_occ, 'empty': m_empty, 'per_count': m_pc,
            'predictions': preds, 'targets': trues}


def _write_summary_block(f, metrics, method_name, model_type, ws, scheme, extra=""):
    m = metrics['all']
    f.write(f"Method: {method_name}\nWeight scheme: {scheme}\nModel: {model_type}\n"
            f"WS: {ws}\nInit: Random\n{extra}\n\n")
    f.write(f"=== ALL ===\nN={m['sample_count']}\nMAE={m['mae']:.4f}\n"
            f"RMSE={m['rmse']:.4f}\nR²={m['r2']:.4f}\nMedAE={m['medae']:.4f}\n"
            f"Acc±1={m['acc_pm1']:.2f}%\nAcc±2={m['acc_pm2']:.2f}%\n\n")
    if metrics['occupied']:
        mo = metrics['occupied']
        f.write(f"=== OCCUPIED ===\nN={mo['sample_count']}\nMAE={mo['mae']:.4f}\n"
                f"Acc±1={mo['acc_pm1']:.2f}%\nExact={mo['exact_match_pct']:.2f}%\n\n")
    if metrics['empty']:
        me = metrics['empty']
        f.write(f"=== EMPTY ===\nN={me['sample_count']}\nMAE={me['mae']:.4f}\n"
                f"Acc±1={me['acc_pm1']:.2f}%\n")


def _build_prediction_df(data, preds, ws):
    test_start = data['tgt_split']['test_start'] + ws - 1
    test_end   = data['tgt_split']['test_end']
    df_test = data['tgt_df_full'].iloc[test_start:test_end].copy()
    min_len = min(len(df_test), len(preds))
    df_test = df_test.iloc[:min_len]
    preds = preds[:min_len]
    df_test['predicted_peoplecount'] = preds
    df_test['predicted_rounded'] = np.round(np.clip(preds, 0, None)).astype(int)
    return df_test


def save_method_outputs(model, metrics, data, ws, method_full, model_type, out_dir,
                        scheme, extra=""):
    tag = method_full.replace(' ', '_')
    torch.save(model.state_dict(), os.path.join(out_dir, f"weights_{tag}_ws{ws}.pth"))
    model.eval()
    with torch.no_grad():
        preds = model(data['X_tgt_test_t'].to(device)).cpu().numpy()
    df_test = _build_prediction_df(data, preds, ws)
    df_test.to_csv(os.path.join(out_dir, f"predicted_test_{tag}_ws{ws}.csv"), index=False)
    with open(os.path.join(out_dir, f"summary_{tag}_ws{ws}.txt"), "w") as f:
        _write_summary_block(f, metrics, method_full, model_type, ws, scheme, extra)


def save_augmented_outputs(model, metrics, data, preds, ws, method_full, model_type,
                           out_dir, scheme, extra=""):
    tag = method_full.replace(' ', '_')
    torch.save(model.state_dict(), os.path.join(out_dir, f"weights_{tag}_ws{ws}.pth"))
    df_test = _build_prediction_df(data, preds, ws)
    df_test.to_csv(os.path.join(out_dir, f"predicted_test_{tag}_ws{ws}.csv"), index=False)
    with open(os.path.join(out_dir, f"summary_{tag}_ws{ws}.txt"), "w") as f:
        _write_summary_block(f, metrics, method_full, model_type, ws, scheme, extra)


##############################################################################
#  ENCODER + CRITIC
##############################################################################

class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model_type = type(model).__name__

    def encode(self, x):
        if self.model_type == 'RNNRegressor':
            h0 = torch.zeros(self.model.num_layers, x.size(0), self.model.hidden_size).to(x.device)
            out, _ = self.model.rnn(x, h0)
            return out[:, -1, :]
        elif self.model_type == 'LSTMRegressor':
            h0 = torch.zeros(self.model.num_layers, x.size(0), self.model.hidden_size).to(x.device)
            c0 = torch.zeros(self.model.num_layers, x.size(0), self.model.hidden_size).to(x.device)
            out, _ = self.model.lstm(x, (h0, c0))
            return out[:, -1, :]
        elif self.model_type == 'TransformerRegressor':
            return self.model.transformer_encoder(self.model.input_proj(x))[:, -1, :]
        raise ValueError(f"Unknown: {self.model_type}")

    def predict(self, h):
        return self.model.fc(h).squeeze(-1)

    def forward(self, x):
        return self.model(x)

    @property
    def hidden_dim(self):
        if self.model_type in ('RNNRegressor', 'LSTMRegressor'):
            return self.model.hidden_size
        return self.model.input_proj.out_features


class DomainCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.net(x)


##############################################################################
#  LOSSES
##############################################################################

def coral_loss(h_src, h_tgt):
    d = h_src.size(1)
    ns, nt = h_src.size(0), h_tgt.size(0)
    sc, tc = h_src - h_src.mean(0, keepdim=True), h_tgt - h_tgt.mean(0, keepdim=True)
    Cs = (sc.t() @ sc) / max(ns - 1, 1)
    Ct = (tc.t() @ tc) / max(nt - 1, 1)
    return (Cs - Ct).pow(2).sum() / (4 * d * d)


def mmd_rbf_loss(h_src, h_tgt, bandwidths=None):
    if bandwidths is None:
        bandwidths = [0.1, 1.0, 10.0]
    def rbf(x, y, s):
        return torch.exp(-torch.cdist(x, y, p=2).pow(2) / (2 * s ** 2))
    loss = torch.tensor(0.0, device=h_src.device)
    for s in bandwidths:
        loss += rbf(h_src, h_src, s).mean() + rbf(h_tgt, h_tgt, s).mean() - 2 * rbf(h_src, h_tgt, s).mean()
    return loss / len(bandwidths)


def ccsa_loss(h_src, h_tgt, y_src, y_tgt, margin=1.0, threshold=1.0):
    dist = torch.cdist(h_src, h_tgt, p=2)
    ld = torch.abs(y_src.unsqueeze(1) - y_tgt.unsqueeze(0))
    pm = (ld <= threshold).float()
    nm = (ld > threshold).float()
    pl = (pm * dist.pow(2)).sum() / (pm.sum() + 1e-8)
    nl = (nm * torch.clamp(margin - dist, min=0).pow(2)).sum() / (nm.sum() + 1e-8)
    return pl + nl


def weighted_mse(pred, y, w):
    return ((pred - y) ** 2 * w).sum() / (w.sum() + 1e-8)


##############################################################################
#  TRAIN HELPERS
##############################################################################

def _validate(model, val_loader):
    model.eval()
    vp, vt = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            vp.extend(model(bx.to(device)).cpu().numpy())
            vt.extend(by.numpy())
    return mean_absolute_error(vt, vp)


def _get_src_batch(src_iter, src_loader):
    try:
        return next(src_iter), src_iter
    except StopIteration:
        src_iter = iter(src_loader)
        return next(src_iter), src_iter


def _simple_train_loop_weighted(model, train_ld, val_ld, epochs):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_val_mae, best_state = float('inf'), None
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        for batch in train_ld:
            if len(batch) == 3:
                bx, by, bw = [b.to(device) for b in batch]
                optimizer.zero_grad()
                weighted_mse(model(bx), by, bw).backward()
            else:
                bx, by = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                nn.functional.mse_loss(model(bx), by).backward()
            optimizer.step()
        vmae = _validate(model, val_ld)
        if vmae < best_val_mae:
            best_val_mae = vmae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"      Epoch [{ep + 1}/{epochs}] Val MAE: {vmae:.4f}")
    train_time = time.perf_counter() - t0
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"      Train time: {train_time:.3f}s")
    return model, train_time


def _deep_train_loop(encoder, data, epochs, align_fn, src_w_table_max, scheme):
    optimizer = optim.Adam(encoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    best_val_mae, best_state = float('inf'), None
    t0 = time.perf_counter()

    for ep in range(epochs):
        encoder.train()
        src_iter = iter(data['src_train_loader'])
        for tgt_bx, tgt_by in data['tgt_train_loader']:
            (src_bx, src_by), src_iter = _get_src_batch(src_iter, data['src_train_loader'])
            src_bx, src_by = src_bx.to(device), src_by.to(device)
            tgt_bx, tgt_by = tgt_bx.to(device), tgt_by.to(device)

            optimizer.zero_grad()
            h_src = encoder.encode(src_bx)
            h_tgt = encoder.encode(tgt_bx)
            pred_s = encoder.predict(h_src)
            pred_t = encoder.predict(h_tgt)

            if src_w_table_max is not None:
                table, max_y = src_w_table_max
                w_np = apply_weight_table(src_by.detach().cpu().numpy(), table, max_y)
                w_np = normalize_and_clip(w_np)
                w = torch.tensor(w_np, dtype=torch.float32, device=device)
                l_src = weighted_mse(pred_s, src_by, w)
            else:
                l_src = criterion(pred_s, src_by)

            l_tgt = criterion(pred_t, tgt_by)
            l_align = align_fn(h_src, h_tgt, src_by, tgt_by)
            (l_src + l_tgt + l_align).backward()
            optimizer.step()

        vmae = _validate(encoder.model, data['tgt_val_loader'])
        if vmae < best_val_mae:
            best_val_mae = vmae
            best_state = {k: v.cpu().clone() for k, v in encoder.model.state_dict().items()}
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"      Epoch [{ep + 1}/{epochs}] Val MAE: {vmae:.4f} ({scheme})")

    train_time = time.perf_counter() - t0
    if best_state:
        encoder.model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"      Deep train time: {train_time:.3f}s")
    return encoder, train_time


def _make_combined_weights(src_y, tgt_y, scheme):
    if scheme == 'tsbw':
        t0 = time.perf_counter()
        sw = tsbw_source_weights(src_y, tgt_y)
        w = np.concatenate([sw, np.ones(len(tgt_y), dtype=np.float32)])
        return w, time.perf_counter() - t0
    # ── NEW: TSW ──────────────────────────────────────────────────────────
    elif scheme == 'tsw':
        t0 = time.perf_counter()
        sw = tsw_source_weights(src_y, tgt_y)
        w = np.concatenate([sw, np.ones(len(tgt_y), dtype=np.float32)])
        print(f"      TSW weights: mean={sw.mean():.4f}, "
              f"min={sw.min():.4f}, max={sw.max():.4f}")
        return w, time.perf_counter() - t0
    # ── END NEW ───────────────────────────────────────────────────────────
    return None, 0.0


##############################################################################
#  METHOD 1: CORAL (shallow)
##############################################################################

def run_coral(data, model_type, ws, out_dir, epochs, scheme):
    name = f"CORAL_{scheme}"
    print(f"\n    ── 1. {name} ──")
    reset_all_seeds(SEED)

    t0_w = time.perf_counter()
    src_flat = data['src_tr_X'].reshape(data['src_tr_X'].shape[0], -1)
    tgt_flat = data['tgt_tr_X'].reshape(data['tgt_tr_X'].shape[0], -1)
    sm, tm = src_flat.mean(0), tgt_flat.mean(0)
    sc, tc = src_flat - sm, tgt_flat - tm
    Cs = (sc.T @ sc) / max(len(src_flat) - 1, 1)
    Ct = (tc.T @ tc) / max(len(tgt_flat) - 1, 1)
    reg = np.eye(Cs.shape[0]) * 1e-6
    try:
        Cs_si = np.linalg.inv(sqrtm(Cs + reg).real)
        src_aligned = sc @ (Cs_si @ sqrtm(Ct + reg).real) + tm
    except Exception as e:
        print(f"    ⚠ CORAL transform failed: {e}")
        src_aligned = src_flat
    coral_align_time = time.perf_counter() - t0_w

    d = data['input_size']
    combined_X = np.vstack([src_aligned, tgt_flat]).reshape(-1, ws, d)
    combined_y = np.concatenate([data['src_tr_y'], data['tgt_tr_y']])
    combined_w, w_time = _make_combined_weights(data['src_tr_y'], data['tgt_tr_y'], scheme)
    bs = min(32, max(4, len(combined_X) // 8))

    if combined_w is not None:
        train_ld = _ld_w(combined_X, combined_y, combined_w, bs, True)
    else:
        train_ld = _ld(combined_X, combined_y, bs, True)

    model, target_train_t = _simple_train_loop_weighted(
        build_model(model_type, d), train_ld, data['tgt_val_loader'], epochs)
    metrics = evaluate_full(model, data['tgt_test_loader'], name)
    save_method_outputs(model, metrics, data, ws, name, model_type, out_dir, scheme)
    result = {'method': name, 'weight_scheme': scheme, 'metrics': metrics}
    return add_complexity_info(
        result, model,
        target_train_time_sec=target_train_t,
        weight_compute_time_sec=w_time + coral_align_time,
    )


##############################################################################
#  METHOD 2: DeepCORAL
##############################################################################

def run_deep_coral(data, model_type, ws, out_dir, epochs, scheme, lambda_coral=1.0):
    name = f"DeepCORAL_{scheme}"
    print(f"\n    ── 2. {name} (λ={lambda_coral}) ──")
    reset_all_seeds(SEED)
    encoder = EncoderWrapper(build_model(model_type, data['input_size'])).to(device)

    if scheme == 'tsbw':
        t0_w = time.perf_counter()
        table_max = compute_label_weight_table(data['tgt_tr_y'])
        weight_compute_t = time.perf_counter() - t0_w
    # ── NEW: TSW ──────────────────────────────────────────────────────────
    elif scheme == 'tsw':
        t0_w = time.perf_counter()
        table_max = compute_tsw_label_weight_table(data['tgt_tr_y'])
        weight_compute_t = time.perf_counter() - t0_w
    # ── END NEW ───────────────────────────────────────────────────────────
    else:
        table_max, weight_compute_t = None, 0.0

    encoder, train_t = _deep_train_loop(
        encoder, data, epochs,
        lambda hs, ht, ys, yt: lambda_coral * coral_loss(hs, ht),
        table_max, scheme)
    metrics = evaluate_full(encoder.model, data['tgt_test_loader'], name)
    save_method_outputs(encoder.model, metrics, data, ws, name, model_type, out_dir, scheme)
    result = {'method': name, 'weight_scheme': scheme, 'metrics': metrics}
    return add_complexity_info(result, encoder.model,
                               target_train_time_sec=train_t,
                               weight_compute_time_sec=weight_compute_t)


##############################################################################
#  METHOD 3: DeepMMD
##############################################################################

def run_deep_mmd(data, model_type, ws, out_dir, epochs, scheme, lambda_mmd=1.0):
    name = f"DeepMMD_{scheme}"
    print(f"\n    ── 3. {name} (λ={lambda_mmd}) ──")
    reset_all_seeds(SEED)
    encoder = EncoderWrapper(build_model(model_type, data['input_size'])).to(device)

    if scheme == 'tsbw':
        t0_w = time.perf_counter()
        table_max = compute_label_weight_table(data['tgt_tr_y'])
        weight_compute_t = time.perf_counter() - t0_w
    # ── NEW: TSW ──────────────────────────────────────────────────────────
    elif scheme == 'tsw':
        t0_w = time.perf_counter()
        table_max = compute_tsw_label_weight_table(data['tgt_tr_y'])
        weight_compute_t = time.perf_counter() - t0_w
    # ── END NEW ───────────────────────────────────────────────────────────
    else:
        table_max, weight_compute_t = None, 0.0

    encoder, train_t = _deep_train_loop(
        encoder, data, epochs,
        lambda hs, ht, ys, yt: lambda_mmd * mmd_rbf_loss(hs, ht),
        table_max, scheme)
    metrics = evaluate_full(encoder.model, data['tgt_test_loader'], name)
    save_method_outputs(encoder.model, metrics, data, ws, name, model_type, out_dir, scheme)
    result = {'method': name, 'weight_scheme': scheme, 'metrics': metrics}
    return add_complexity_info(result, encoder.model,
                               target_train_time_sec=train_t,
                               weight_compute_time_sec=weight_compute_t)


##############################################################################
#  METHOD 4: PRED
##############################################################################

def run_pred(data, model_type, ws, out_dir, epochs, scheme):
    name = f"PRED_{scheme}"
    print(f"\n    ── 4. {name} ──")
    reset_all_seeds(SEED)
    d = data['input_size']

    src_model = build_model(model_type, d)
    optimizer = optim.Adam(src_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    t_src0 = time.perf_counter()
    for ep in range(epochs):
        src_model.train()
        for bx, by in data['src_train_loader']:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            criterion(src_model(bx), by).backward()
            optimizer.step()
    source_train_t = time.perf_counter() - t_src0
    print(f"      Source-only train time: {source_train_t:.3f}s")

    src_model.eval()
    def _pred(X):
        with torch.no_grad():
            return src_model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()

    sp = _pred(data['src_tr_X']); tp = _pred(data['tgt_tr_X'])
    vp = _pred(data['tgt_va_X']); te = _pred(data['tgt_te_X'])

    def _aug(X, p):
        f = np.broadcast_to(p[:, None, None], (X.shape[0], X.shape[1], 1))
        return np.concatenate([X, f], axis=2)

    sa, ta = _aug(data['src_tr_X'], sp), _aug(data['tgt_tr_X'], tp)
    va, tea = _aug(data['tgt_va_X'], vp), _aug(data['tgt_te_X'], te)

    combined_X = np.vstack([sa, ta]).astype(np.float32)
    combined_y = np.concatenate([data['src_tr_y'], data['tgt_tr_y']])
    combined_w, w_time = _make_combined_weights(data['src_tr_y'], data['tgt_tr_y'], scheme)
    bs = min(32, max(4, len(combined_X) // 8))

    if combined_w is not None:
        train_ld = _ld_w(combined_X, combined_y, combined_w, bs, True)
    else:
        train_ld = _ld(combined_X, combined_y, bs, True)

    val_ld  = _ld(va,  data['tgt_va_y'], 32, False)
    test_ld = _ld(tea, data['tgt_te_y'], 32, False)

    final, target_train_t = _simple_train_loop_weighted(
        build_model(model_type, d + 1), train_ld, val_ld, epochs)
    metrics = evaluate_full(final, test_ld, name)
    final.eval()
    with torch.no_grad():
        preds = final(torch.tensor(tea, dtype=torch.float32).to(device)).cpu().numpy()
    save_augmented_outputs(final, metrics, data, preds, ws, name, model_type, out_dir,
                           scheme, f"Input: d={d}+1={d + 1}")
    result = {'method': name, 'weight_scheme': scheme, 'metrics': metrics}
    return add_complexity_info(result, final,
                               source_train_time_sec=source_train_t,
                               target_train_time_sec=target_train_t,
                               weight_compute_time_sec=w_time)


##############################################################################
#  METHOD 5: FA
##############################################################################

def run_fa(data, model_type, ws, out_dir, epochs, scheme):
    name = f"FA_{scheme}"
    print(f"\n    ── 5. {name} ──")
    reset_all_seeds(SEED)
    d = data['input_size']

    def _aug_src(X): return np.concatenate([X, X, np.zeros_like(X)], axis=2)
    def _aug_tgt(X): return np.concatenate([X, np.zeros_like(X), X], axis=2)

    sa, ta = _aug_src(data['src_tr_X']), _aug_tgt(data['tgt_tr_X'])
    va, tea = _aug_tgt(data['tgt_va_X']), _aug_tgt(data['tgt_te_X'])

    combined_X = np.vstack([sa, ta]).astype(np.float32)
    combined_y = np.concatenate([data['src_tr_y'], data['tgt_tr_y']])
    combined_w, w_time = _make_combined_weights(data['src_tr_y'], data['tgt_tr_y'], scheme)
    bs = min(32, max(4, len(combined_X) // 8))

    if combined_w is not None:
        train_ld = _ld_w(combined_X, combined_y, combined_w, bs, True)
    else:
        train_ld = _ld(combined_X, combined_y, bs, True)

    val_ld  = _ld(va,  data['tgt_va_y'], 32, False)
    test_ld = _ld(tea, data['tgt_te_y'], 32, False)

    model, target_train_t = _simple_train_loop_weighted(
        build_model(model_type, 3 * d), train_ld, val_ld, epochs)
    metrics = evaluate_full(model, test_ld, name)
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(tea, dtype=torch.float32).to(device)).cpu().numpy()
    save_augmented_outputs(model, metrics, data, preds, ws, name, model_type, out_dir,
                           scheme, f"Input: 3×{d}={3 * d}")
    result = {'method': name, 'weight_scheme': scheme, 'metrics': metrics}
    return add_complexity_info(result, model,
                               target_train_time_sec=target_train_t,
                               weight_compute_time_sec=w_time)


##############################################################################
#  METHOD 6: CCSA
##############################################################################

def run_ccsa(data, model_type, ws, out_dir, epochs, scheme,
             lambda_ccsa=0.5, margin=1.0, threshold=1.0):
    name = f"CCSA_{scheme}"
    print(f"\n    ── 6. {name} (λ={lambda_ccsa}) ──")
    reset_all_seeds(SEED)
    encoder = EncoderWrapper(build_model(model_type, data['input_size'])).to(device)

    if scheme == 'tsbw':
        t0_w = time.perf_counter()
        table_max = compute_label_weight_table(data['tgt_tr_y'])
        weight_compute_t = time.perf_counter() - t0_w
    # ── NEW: TSW ──────────────────────────────────────────────────────────
    elif scheme == 'tsw':
        t0_w = time.perf_counter()
        table_max = compute_tsw_label_weight_table(data['tgt_tr_y'])
        weight_compute_t = time.perf_counter() - t0_w
    # ── END NEW ───────────────────────────────────────────────────────────
    else:
        table_max, weight_compute_t = None, 0.0

    encoder, train_t = _deep_train_loop(
        encoder, data, epochs,
        lambda hs, ht, ys, yt: lambda_ccsa * ccsa_loss(hs, ht, ys, yt, margin, threshold),
        table_max, scheme)
    metrics = evaluate_full(encoder.model, data['tgt_test_loader'], name)
    save_method_outputs(encoder.model, metrics, data, ws, name, model_type, out_dir, scheme)
    result = {'method': name, 'weight_scheme': scheme, 'metrics': metrics}
    return add_complexity_info(result, encoder.model,
                               target_train_time_sec=train_t,
                               weight_compute_time_sec=weight_compute_t)


##############################################################################
#  METHOD 7: WDGRL
##############################################################################

def run_wdgrl(data, model_type, ws, out_dir, epochs, scheme,
              lambda_wd=1.0, critic_iters=5, gp_lambda=10.0):
    name = f"WDGRL_{scheme}"
    print(f"\n    ── 7. {name} (λ={lambda_wd}) ──")
    reset_all_seeds(SEED)
    model = build_model(model_type, data['input_size'])
    encoder = EncoderWrapper(model).to(device)
    h_dim = encoder.hidden_dim
    critic = DomainCritic(h_dim, 64).to(device)
    opt_enc = optim.Adam(encoder.parameters(), lr=1e-3)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if scheme == 'tsbw':
        t0_w = time.perf_counter()
        table_max = compute_label_weight_table(data['tgt_tr_y'])
        weight_compute_t = time.perf_counter() - t0_w
    # ── NEW: TSW ──────────────────────────────────────────────────────────
    elif scheme == 'tsw':
        t0_w = time.perf_counter()
        table_max = compute_tsw_label_weight_table(data['tgt_tr_y'])
        weight_compute_t = time.perf_counter() - t0_w
    # ── END NEW ───────────────────────────────────────────────────────────
    else:
        table_max, weight_compute_t = None, 0.0

    def _gp(c, h_s, h_t):
        bs = min(h_s.size(0), h_t.size(0))
        a = torch.rand(bs, 1, device=device)
        i = (a * h_s[:bs] + (1 - a) * h_t[:bs]).requires_grad_(True)
        d_i = c(i)
        g = torch.autograd.grad(d_i, i, grad_outputs=torch.ones_like(d_i),
                                create_graph=True, retain_graph=True)[0]
        return ((g.view(bs, -1).norm(2, dim=1) - 1) ** 2).mean()

    best_val_mae, best_state = float('inf'), None
    t0 = time.perf_counter()
    for ep in range(epochs):
        encoder.train(); critic.train()
        src_iter = iter(data['src_train_loader'])
        for tgt_bx, tgt_by in data['tgt_train_loader']:
            (src_bx, src_by), src_iter = _get_src_batch(src_iter, data['src_train_loader'])
            src_bx, src_by = src_bx.to(device), src_by.to(device)
            tgt_bx, tgt_by = tgt_bx.to(device), tgt_by.to(device)

            for _ in range(critic_iters):
                with torch.no_grad():
                    h_s, h_t = encoder.encode(src_bx), encoder.encode(tgt_bx)
                opt_critic.zero_grad()
                wd = critic(h_t).mean() - critic(h_s).mean()
                (-wd + gp_lambda * _gp(critic, h_s.detach(), h_t.detach())).backward()
                opt_critic.step()

            opt_enc.zero_grad()
            h_s, h_t = encoder.encode(src_bx), encoder.encode(tgt_bx)
            ps, pt = encoder.predict(h_s), encoder.predict(h_t)

            if table_max is not None:
                table, max_y = table_max
                w_np = apply_weight_table(src_by.detach().cpu().numpy(), table, max_y)
                w_np = normalize_and_clip(w_np)
                w = torch.tensor(w_np, dtype=torch.float32, device=device)
                l_src = weighted_mse(ps, src_by, w)
            else:
                l_src = criterion(ps, src_by)

            l_tgt = criterion(pt, tgt_by)
            (l_src + l_tgt + lambda_wd * (critic(h_t).mean() - critic(h_s).mean())).backward()
            opt_enc.step()

        vmae = _validate(encoder.model, data['tgt_val_loader'])
        if vmae < best_val_mae:
            best_val_mae = vmae
            best_state = {k: v.cpu().clone() for k, v in encoder.model.state_dict().items()}
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"      Epoch [{ep + 1}/{epochs}] Val MAE: {vmae:.4f} ({scheme})")
    train_t = time.perf_counter() - t0

    if best_state:
        encoder.model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f"      WDGRL train time: {train_t:.3f}s")

    metrics = evaluate_full(encoder.model, data['tgt_test_loader'], name)
    save_method_outputs(encoder.model, metrics, data, ws, name, model_type, out_dir, scheme)
    result = {'method': name, 'weight_scheme': scheme, 'metrics': metrics}
    return add_complexity_info(result, encoder.model,
                               target_train_time_sec=train_t,
                               weight_compute_time_sec=weight_compute_t)


##############################################################################
#  PLOTS
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

    fig, axes = plt.subplots(2, 3, figsize=(26, 11))
    fig.suptitle(
        f'Feature-Based DA (none vs TSBW vs TSW): {scenario_name} | {model_type} | ws={ws}',  # ── NEW: TSW ──
        fontsize=13, fontweight='bold')
    for idx, (key, ylabel) in enumerate(metrics_list):
        ax = axes[idx // 3][idx % 3]
        vals = [r['metrics']['all'][key] for r in results]
        bars = ax.bar(range(n), vals, color=colors, edgecolor='black', alpha=0.85)
        ax.set_xticks(range(n))
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=6.5)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{v:.3f}',
                    ha='center', va='bottom', fontsize=5, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'comparison_all_{model_type}_ws{ws}.png'),
                dpi=180, bbox_inches='tight')
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
    print("=" * 100)
    print(f"FEATURE-BASED DA — HOMOGENEOUS + HETEROGENEOUS (RANDOM INIT)")
    print(f"7 Methods × {len(WEIGHT_SCHEMES)} schemes = {7 * len(WEIGHT_SCHEMES)} runs / config")
    print(f"Variants: {FEATURE_VARIANTS}")
    print(f"TSBW: floor={TSBW_FLOOR}, absent={TSBW_ABSENT}, sqrt={TSBW_USE_SQRT}")
    # ── NEW: TSW ──────────────────────────────────────────────────────────
    print(f"TSW:  absent={TSW_ABSENT}, linear ratio (no sqrt, no floor), "
          f"eps={FREQUENCY_EPS}, clip={WEIGHT_CLIP_MAX}")
    # ── END NEW ───────────────────────────────────────────────────────────
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"User: Azadshokrollahi | {timestamp} UTC")
    print("=" * 100)

    ensure_dir(OUTPUT_ROOT)
    results_by_variant = {v: [] for v in FEATURE_VARIANTS}

    for variant in FEATURE_VARIANTS:
        print(f"\n\n{'#' * 100}")
        print(f"#  FEATURE VARIANT: {variant.upper()}")
        print(f"{'#' * 100}")

        valid_scenarios = [(s, t) for (s, t) in TRANSFER_SCENARIOS
                           if variant_scenario_supported(s, t, variant)]

        runnable = []
        for s, t in valid_scenarios:
            s_feats = get_variant_features(s, variant)
            t_feats = get_variant_features(t, variant)
            if len(s_feats) == len(t_feats):
                runnable.append((s, t))
            else:
                print(f"  ⏭ Skip {s} → {t}: dim mismatch "
                      f"({len(s_feats)} vs {len(t_feats)}) — feature-based methods require equal dims")

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

            print(f"\n{'=' * 100}")
            print(f"[{variant}] SCENARIO {sc_idx + 1}/{len(runnable)}: {scenario_name}")
            print(f"  src features: {src_cfg['features']}")
            print(f"  tgt features: {tgt_cfg['features']}")
            print(f"{'=' * 100}")

            if not os.path.exists(src_cfg['path']) or not os.path.exists(tgt_cfg['path']):
                print("  ❌ Data file missing"); continue

            for model_type in MODEL_TYPES:
                for ws in WINDOW_SIZES:
                    counter += 1
                    print(f"\n  ── [{variant}][{counter}/{total}] {model_type} | ws={ws} ──")
                    sc_folder = ensure_dir(os.path.join(
                        OUTPUT_ROOT, variant,
                        f"{src_id}_to_{tgt_id}", f"{model_type}_ws{ws}"))
                    log_file = os.path.join(sc_folder, "feature_based_log.txt")
                    for h in logging.root.handlers[:]:
                        logging.root.removeHandler(h)
                    logging.basicConfig(
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(log_file, mode='a'),
                                  logging.StreamHandler(sys.stdout)],
                        force=True)

                    try:
                        data = prepare_source_target_data(src_cfg, tgt_cfg, ws)
                    except Exception as e:
                        print(f"    ❌ Data prep failed: {e}")
                        continue

                    joblib.dump(data['src_scaler'], os.path.join(sc_folder, "src_scaler.pkl"))
                    joblib.dump(data['tgt_scaler'], os.path.join(sc_folder, "tgt_scaler.pkl"))

                    method_results = []
                    for scheme in WEIGHT_SCHEMES:
                        runners = [
                            ("CORAL",     lambda s=scheme: run_coral(data, model_type, ws, sc_folder, EPOCHS, s)),
                            ("DeepCORAL", lambda s=scheme: run_deep_coral(data, model_type, ws, sc_folder, EPOCHS, s)),
                            ("DeepMMD",   lambda s=scheme: run_deep_mmd(data, model_type, ws, sc_folder, EPOCHS, s)),
                            ("PRED",      lambda s=scheme: run_pred(data, model_type, ws, sc_folder, EPOCHS, s)),
                            ("FA",        lambda s=scheme: run_fa(data, model_type, ws, sc_folder, EPOCHS, s)),
                            ("CCSA",      lambda s=scheme: run_ccsa(data, model_type, ws, sc_folder, EPOCHS, s)),
                            ("WDGRL",     lambda s=scheme: run_wdgrl(data, model_type, ws, sc_folder, EPOCHS, s)),
                        ]
                        for mname, runner in runners:
                            try:
                                method_results.append(runner())
                            except Exception as e:
                                print(f"    ❌ {mname}_{scheme} failed: {e}")
                                logging.error(f"{mname}_{scheme}: {e}")
                                traceback.print_exc()

                    method_results = [r for r in method_results if r is not None]
                    create_comparison_plots(method_results, ws, scenario_name, model_type, sc_folder)

                    rows = []
                    for r in method_results:
                        row = build_result_row(scenario_name, src_id, tgt_id, model_type, ws,
                                               r, src_cfg, tgt_cfg, variant)
                        rows.append(row)
                        results_by_variant[variant].append(row)

                    if rows:
                        pd.DataFrame(rows).to_csv(
                            os.path.join(sc_folder, f"metrics_{model_type}_ws{ws}.csv"),
                            index=False)

                    print(f"\n    {'Method':<20} {'Scheme':<8} {'MAE':<8} {'RMSE':<8} "
                          f"{'R²':<8} {'Acc±1':<8} {'Complex(s)':<11} {'Trainable':<12}")
                    print("    " + "-" * 100)
                    for r in method_results:
                        m = r['metrics']['all']
                        print(f"    {r['method']:<20} {r['weight_scheme']:<8} "
                              f"{m['mae']:<8.4f} {m['rmse']:<8.4f} {m['r2']:<8.4f} "
                              f"{m['acc_pm1']:<8.2f} "
                              f"{r.get('model_complexity_time_sec', 0.0):<11.3f} "
                              f"{int(r.get('trainable_params', 0)):<12}")
                    print(f"\n  ✓ [{variant}] {scenario_name} | {model_type} | ws={ws} — "
                          f"{len(method_results)}/{7 * len(WEIGHT_SCHEMES)}")

    # ─────────────────────────────────────────────────────────────────────────
    # MASTER CSVs
    # ─────────────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("FINAL RESULTS — FEATURE-BASED DA")
    print("=" * 100)

    COLUMN_ORDER = [
        'scenario', 'source', 'target', 'method', 'weight_scheme', 'variant', 'model',
        'window_size', 'features', 'src_scenario', 'tgt_scenario', 'initialization',
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
        out_csv = os.path.join(OUTPUT_ROOT, f"all_feature_based_{variant}.csv")
        df.to_csv(out_csv, index=False)

        print(f"\n  [{variant}] Master CSV: {out_csv}")
        print(f"  [{variant}] Total rows: {len(df)}")

        print(f"\n  [{variant}] BEST METHOD PER SCENARIO × MODEL (by MAE):")
        print(f"  {'Scenario':<25} {'Model':<12} {'Method':<22} "
              f"{'Scheme':<8} {'MAE':<8} {'Acc±1':<8}")
        print("  " + "-" * 90)
        for (sc, mdl), grp in df.groupby(['scenario', 'model']):
            best = grp.loc[grp['test_mae'].idxmin()]
            print(f"  {sc:<25} {mdl:<12} {best['method']:<22} "
                  f"{best['weight_scheme']:<8} "
                  f"{best['test_mae']:<8.4f} {best['test_acc_pm1']:<8.2f}")

        # ── NEW: TSW — TSBW vs TSW summary ────────────────────────────────
        print(f"\n  [{variant}] none vs TSBW vs TSW (avg MAE per method base):")
        for base in ALL_METHODS:
            none_df = df[df['method'] == f'{base}_none']
            tsbw_df = df[df['method'] == f'{base}_tsbw']
            tsw_df  = df[df['method'] == f'{base}_tsw']
            if len(none_df) > 0:
                line = f"    {base:<12}: none={none_df['test_mae'].mean():.4f}"
                if len(tsbw_df) > 0:
                    line += f"  TSBW={tsbw_df['test_mae'].mean():.4f}"
                if len(tsw_df) > 0:
                    line += f"  TSW={tsw_df['test_mae'].mean():.4f}"
                print(line)
        # ── END NEW ───────────────────────────────────────────────────────

    print("\n" + "=" * 100)
    print("✓ FEATURE-BASED DA COMPLETED!")
    print(f"User: Azadshokrollahi | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 100)


if __name__ == "__main__":
    main()