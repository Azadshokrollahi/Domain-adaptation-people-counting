"""
Parameter-Based Transfer Learning for People Counting — EXTENDED (26 setups)
Supports BOTH homogeneous and heterogeneous feature variants.

Reads:
  • configs/rooms.yaml   → room paths, scenarios, feature_variants
  • src/config.py        → seeds, epochs, hyperparams, paths

Fallback rules (heterogeneous mode):
  • If a room has no `heterogeneous` in feature_variants → use its `homogeneous` features.
  • If a room's `heterogeneous/` pretrain folder is missing → load weights from `homogeneous/`.

Outputs (per variant):
  • results/parameter_based_transfer/all_parameter_based_homogeneous.csv
  • results/parameter_based_transfer/all_parameter_based_heterogeneous.csv

Author: Azadshokrollahi
"""

import os, sys, ast, logging, joblib, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import TensorDataset, DataLoader

# ─── Centralized config ─────────────────────────────────────────────────────
from config import (
    SEED, WINDOW_SIZES, NUM_EPOCHS,
    RESULTS_ROOT, PRETRAIN_DIR_NAME,
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
#  CONFIGURATION
##############################################################################

MODEL_TYPES      = ['RNN', 'LSTM', 'Transformer']
EPOCHS           = NUM_EPOCHS
FEATURE_VARIANTS = ['homogeneous', 'heterogeneous']

# LinInt
LININT_ALPHAS = [0.00, 0.25, 0.50, 0.75, 1.00]
LININT_PROP   = 0.5

# Label weighting
WEIGHT_SCHEMES         = ['tsw', 'tsbw']
FREQUENCY_EPS          = 1.0
WEIGHT_CLIP_MAX        = 10.0
THRESHOLD_DROP         = 0.3
TARGET_PRIORITY_LAMBDA_T = 2.0
TARGET_PRIORITY_LAMBDA_S = 1.0
TSBW_FLOOR     = 0.3
TSBW_ABSENT    = 0.05
TSBW_USE_SQRT  = True

# Paths (relative to project root, via config.py)
PRETRAIN_DIR = str(RESULTS_ROOT / PRETRAIN_DIR_NAME)
TRANSFER_DIR = str(RESULTS_ROOT / "parameter_based_transfer")

# Rooms from configs/rooms.yaml
ROOMS = load_rooms()

TRANSFER_SCENARIOS = [
    ("Room_A", "Room_B"), ("Room_B", "Room_C"), ("Room_C", "Room_A"),
    ("Room_A", "Room_P"), ("Room_B", "Room_P"), ("Room_C", "Room_P"),
    ("Room_P", "Room_A"), ("Room_P", "Room_B"), ("Room_P", "Room_C"),
]


##############################################################################
#  VARIANT HELPERS  (with homogeneous fallback)
##############################################################################

def get_variant_features(room_id, variant):
    """
    Return feature list for (room_id, variant).
    Fallback: if heterogeneous is missing, use homogeneous.
    """
    fv = ROOMS[room_id].get("feature_variants", {})
    if variant in fv:
        return fv[variant]
    if variant == "heterogeneous" and "homogeneous" in fv:
        return fv["homogeneous"]
    return None


def variant_scenario_supported(src_id, tgt_id, variant):
    """True only if BOTH rooms have features for this variant (with fallback)."""
    return (get_variant_features(src_id, variant) is not None
            and get_variant_features(tgt_id, variant) is not None)


def build_room_cfg(room_id, variant):
    """Build a per-variant runtime config dict."""
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
    return (text.lower().replace(' ', '_').replace('±', 'pm').replace('(', '').replace(')', '')
            .replace('/', '_').replace('-', '_').replace('=', '').replace('α', 'alpha')
            .replace('→', 'to').replace('.', ''))


def build_model(model_type, input_size, hidden_size=64, num_layers=2):
    if model_type == "RNN":
        return RNNRegressor(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional_flag=False, aggregation_mode='one',
                            use_attention=False).to(device)
    elif model_type == "LSTM":
        return LSTMRegressor(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                             bidirectional_flag=False, aggregation_mode='one',
                             use_attention=False).to(device)
    elif model_type == "Transformer":
        return TransformerRegressor(input_size=input_size, d_model=64, nhead=4,
                                    num_layers=num_layers, dim_feedforward=128,
                                    dropout=0.1).to(device)
    raise ValueError(f"Unknown model: {model_type}")


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

    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  ⚠ Missing features in {room_name}: {missing}")
    X = df[available].apply(pd.to_numeric, errors='coerce').ffill().bfill().values.astype(np.float32)
    y = pd.to_numeric(df['target'], errors='coerce').ffill().bfill().values.astype(np.float32)

    print(f"  Loaded {room_name}: {X.shape[0]} rows, {X.shape[1]} features, target mean={y.mean():.2f}")
    return X, y, df_full, available


def split_by_scenario(X, y, scenario):
    n = len(X)
    if scenario == 'scenario_1':
        tr_end, va_end = int(0.7 * n), int(0.8 * n)
        tr_X, tr_y = X[:tr_end], y[:tr_end]
        va_X, va_y = X[tr_end:va_end], y[tr_end:va_end]
        te_X, te_y = X[va_end:], y[va_end:]
        info = {'scenario': 'scenario_1', 'train_start': 0, 'train_end': tr_end,
                'val_start': tr_end, 'val_end': va_end, 'test_start': va_end, 'test_end': n}
    else:
        te_end, va_end = int(0.2 * n), int(0.3 * n)
        te_X, te_y = X[:te_end], y[:te_end]
        va_X, va_y = X[te_end:va_end], y[te_end:va_end]
        tr_X, tr_y = X[va_end:], y[va_end:]
        info = {'scenario': 'scenario_2', 'test_start': 0, 'test_end': te_end,
                'val_start': te_end, 'val_end': va_end, 'train_start': va_end, 'train_end': n}
    print(f"  Split ({scenario}): Train={len(tr_X)}, Val={len(va_X)}, Test={len(te_X)}")
    return tr_X, tr_y, va_X, va_y, te_X, te_y, info


def create_sequences(X, y, ws):
    Xs, ys = [], []
    for i in range(len(X) - ws + 1):
        Xs.append(X[i:i + ws])
        ys.append(y[i + ws - 1])
    return np.array(Xs), np.array(ys)


##############################################################################
#  LABEL WEIGHTING (TSW / TSBW)
##############################################################################

def compute_label_weight_table(y_ref, scheme='tsw',
                                eps=FREQUENCY_EPS, floor=TSBW_FLOOR,
                                absent=TSBW_ABSENT, use_sqrt=TSBW_USE_SQRT, **kwargs):
    y_int = np.clip(np.round(y_ref).astype(int), 0, None)
    max_y = int(y_int.max())
    counts = np.bincount(y_int, minlength=max_y + 1).astype(float)
    max_count = counts.max()

    table = {}
    if scheme == 'tsw':
        for yv in range(max_y + 1):
            table[int(yv)] = (float((counts[yv] + eps) / (max_count + eps))
                              if counts[yv] > 0 else float(absent))
    elif scheme == 'tsbw':
        for yv in range(max_y + 1):
            if counts[yv] > 0:
                ratio = (counts[yv] + eps) / (max_count + eps)
                w = np.sqrt(ratio) if use_sqrt else ratio
                table[int(yv)] = float(max(w, floor))
            else:
                table[int(yv)] = float(absent)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    table[-1] = float(absent)
    return table, max_y


def apply_weight_table(y_samples, weight_table, max_y_ref, **kwargs):
    absent_w = weight_table.get(-1, TSBW_ABSENT)
    y_int = np.round(y_samples).astype(int)
    raw = np.empty(len(y_int), dtype=np.float64)
    for i, yv in enumerate(y_int):
        raw[i] = absent_w if (yv < 0 or yv > max_y_ref) else weight_table.get(int(yv), absent_w)
    return raw


def normalize_and_clip(w, clip_max=WEIGHT_CLIP_MAX):
    w = np.asarray(w, dtype=np.float64)
    w = np.maximum(w, 0)
    mean = max(w.mean(), 1e-8)
    w = np.clip(w / mean, 0, clip_max)
    return w.astype(np.float32)


def compute_sample_weights(y_samples, y_ref, scheme, **kwargs):
    table, max_y = compute_label_weight_table(y_ref, scheme=scheme)
    raw = apply_weight_table(y_samples, table, max_y)
    return normalize_and_clip(raw)


##############################################################################
#  DATA LOADERS
##############################################################################

def _make_loader(X, y, bs, shuffle):
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32)),
        batch_size=bs, shuffle=shuffle,
        generator=make_generator(SEED) if shuffle else None,
        worker_init_fn=seed_worker,
    )


def _make_weighted_loader(X, y, w, bs, shuffle):
    return DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32),
                      torch.tensor(y, dtype=torch.float32),
                      torch.tensor(w, dtype=torch.float32)),
        batch_size=bs, shuffle=shuffle,
        generator=make_generator(SEED) if shuffle else None,
        worker_init_fn=seed_worker,
    )


def prepare_target_dataloaders(tr_X, tr_y, va_X, va_y, te_X, te_y, ws, batch_size=32):
    scaler = StandardScaler().fit(tr_X)
    X_tr_seq, y_tr_seq = create_sequences(scaler.transform(tr_X), tr_y, ws)
    X_va_seq, y_va_seq = create_sequences(scaler.transform(va_X), va_y, ws)
    X_te_seq, y_te_seq = create_sequences(scaler.transform(te_X), te_y, ws)

    for name, arr in [("Train", y_tr_seq), ("Val", y_va_seq), ("Test", y_te_seq)]:
        if len(arr) == 0:
            raise ValueError(f"{name} has 0 sequences with ws={ws}")

    X_te_t = torch.tensor(X_te_seq, dtype=torch.float32)
    print(f"    Sequences: Train={X_tr_seq.shape}, Val={X_va_seq.shape}, Test={X_te_seq.shape}")

    return (_make_loader(X_tr_seq, y_tr_seq, batch_size, True),
            _make_loader(X_va_seq, y_va_seq, batch_size, False),
            _make_loader(X_te_seq, y_te_seq, batch_size, False),
            X_te_t, scaler, X_tr_seq.shape[2],
            X_tr_seq, y_tr_seq, X_va_seq, y_va_seq, X_te_seq, y_te_seq)


def prepare_source_data(src_cfg, ws, batch_size=32):
    X, y, _, _ = load_room_raw(src_cfg['path'], src_cfg['features'],
                               src_cfg['name'], src_cfg['selected_rows'])
    tr_X, tr_y, va_X, va_y, _, _, _ = split_by_scenario(X, y, src_cfg['scenario'])

    scaler = StandardScaler().fit(tr_X)
    X_tr_seq, y_tr_seq = create_sequences(scaler.transform(tr_X), tr_y, ws)
    X_va_seq, y_va_seq = create_sequences(scaler.transform(va_X), va_y, ws)

    return {
        'train_loader': _make_loader(X_tr_seq, y_tr_seq, batch_size, True),
        'val_loader':   _make_loader(X_va_seq, y_va_seq, batch_size, False),
        'input_size':   X_tr_seq.shape[2],
        'X_tr_seq': X_tr_seq, 'y_tr_seq': y_tr_seq,
        'X_va_seq': X_va_seq, 'y_va_seq': y_va_seq,
    }


##############################################################################
#  WEIGHT LOADING  (with homogeneous fallback)
##############################################################################

def load_pretrained_weights(model, weights_path, model_type, src_input, tgt_input):
    state = torch.load(weights_path, map_location=device)
    model_state = model.state_dict()

    if src_input != tgt_input:
        print(f"  ⚠ DIM MISMATCH: src={src_input} → tgt={tgt_input}")
        compat = {}
        for key, val in state.items():
            if model_type in ("RNN", "LSTM"):
                if "weight_ih_l0" in key or "bias_ih_l0" in key:
                    continue
            elif model_type == "Transformer":
                if "input_proj" in key:
                    continue
            if key in model_state and val.shape == model_state[key].shape:
                compat[key] = val
        model.load_state_dict(compat, strict=False)
        print(f"  Loaded {len(compat)}/{len(state)} compatible weights")
    else:
        model.load_state_dict(state)
        print(f"  Loaded all {len(state)} weights ✓")
    return model


def find_pretrained_weights(pretrain_dir, source_id, variant, model_type, ws):
    """
    Look in pretrain_dir/source_id/<variant>/<variant>_<model>_ws<ws>/.
    Fallback: if requested variant folder is missing, try 'homogeneous'.
    """
    def _find_in(v):
        config_name = f"{v}_{model_type}_ws{ws}"
        config_dir = os.path.join(pretrain_dir, source_id, v, config_name)

        for prefix in ['best_weights', 'best_model', 'last_weights', 'last_model']:
            p = os.path.join(config_dir, f"{prefix}_{config_name}.pth")
            if os.path.exists(p):
                return p

        if os.path.isdir(config_dir):
            for f in sorted(os.listdir(config_dir)):
                if 'best_weights' in f and f.endswith('.pth'):
                    return os.path.join(config_dir, f)
            for f in sorted(os.listdir(config_dir)):
                if f.endswith('.pth'):
                    return os.path.join(config_dir, f)
        return None

    # Try requested variant first
    path = _find_in(variant)
    if path is not None:
        return path

    # Fallback: try homogeneous
    if variant != "homogeneous":
        path = _find_in("homogeneous")
        if path is not None:
            print(f"  ℹ Fallback: using homogeneous weights for {source_id} "
                  f"(no {variant}/ folder found)")
            return path

    return None


##############################################################################
#  EVALUATION
##############################################################################

def evaluate_full(model, test_loader, label="Test"):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            bx, by = batch[0], batch[1]
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


def evaluate_from_arrays(trues, preds, label="Test"):
    m_all   = compute_all_metrics(trues, preds)
    m_occ   = compute_occupied_metrics(trues, preds)
    m_empty = compute_empty_metrics(trues, preds)
    m_pc    = compute_per_count_metrics(trues, preds)

    print(f"  {label}: MAE={m_all['mae']:.4f}, RMSE={m_all['rmse']:.4f}, "
          f"R²={m_all['r2']:.4f}, Acc±1={m_all['acc_pm1']:.1f}%")
    if m_occ:
        print(f"  {label} [Occ]: N={m_occ['sample_count']}, MAE={m_occ['mae']:.4f}")

    return {'all': m_all, 'occupied': m_occ, 'empty': m_empty, 'per_count': m_pc,
            'predictions': preds, 'targets': trues}


def predict_array(model, X_arr, bs=128):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_arr), bs):
            xb = torch.tensor(X_arr[i:i + bs], dtype=torch.float32).to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([])


##############################################################################
#  SAVE HELPERS
##############################################################################

def save_setup_outputs(model, metrics, tgt_df_full, tgt_split, ws, setup_name,
                       model_type, out_dir, extra_info=""):
    torch.save(model.state_dict(),
               os.path.join(out_dir, f"weights_{slugify(setup_name)}_ws{ws}.pth"))
    _save_pred_and_summary(metrics, tgt_df_full, tgt_split, ws, setup_name,
                           model_type, out_dir, extra_info)


def _save_pred_and_summary(metrics, tgt_df_full, tgt_split, ws, setup_name,
                           model_type, out_dir, extra_info=""):
    test_start = tgt_split['test_start'] + ws - 1
    df_test = tgt_df_full.iloc[test_start:tgt_split['test_end']].copy()
    if len(metrics['predictions']) > 0:
        p = metrics['predictions']
        min_len = min(len(df_test), len(p))
        df_test = df_test.iloc[:min_len]
        df_test['predicted_peoplecount'] = p[:min_len]
        df_test['predicted_rounded'] = np.round(np.clip(p[:min_len], 0, None)).astype(int)
        df_test.to_csv(os.path.join(out_dir, f"predicted_{slugify(setup_name)}_ws{ws}.csv"),
                       index=False)

    m = metrics['all']
    with open(os.path.join(out_dir, f"summary_{slugify(setup_name)}_ws{ws}.txt"), "w") as f:
        f.write(f"Setup: {setup_name}\nModel: {model_type}\nWS: {ws}\n{extra_info}\n\n")
        f.write(f"=== ALL SAMPLES ===\nSamples: {m['sample_count']}\n"
                f"MAE={m['mae']:.4f}\nRMSE={m['rmse']:.4f}\nR²={m['r2']:.4f}\n"
                f"MedAE={m['medae']:.4f}\nAcc±1={m['acc_pm1']:.2f}%\n"
                f"Acc±2={m['acc_pm2']:.2f}%\n\n")
        if metrics['occupied']:
            mo = metrics['occupied']
            f.write(f"=== OCCUPIED ===\nSamples: {mo['sample_count']}\n"
                    f"MAE={mo['mae']:.4f}\nRMSE={mo['rmse']:.4f}\nR²={mo['r2']:.4f}\n"
                    f"MedAE={mo['medae']:.4f}\nAcc±1={mo['acc_pm1']:.2f}%\n"
                    f"Exact={mo['exact_match']} ({mo['exact_match_pct']:.2f}%)\n\n")
        if metrics['empty']:
            me = metrics['empty']
            f.write(f"=== EMPTY ===\nSamples: {me['sample_count']}\n"
                    f"MAE={me['mae']:.4f}\nAcc±1={me['acc_pm1']:.2f}%\n\n")
        if metrics['per_count']:
            f.write(f"=== PER-COUNT ===\n")
            for c in sorted(metrics['per_count'].keys()):
                pc = metrics['per_count'][c]
                f.write(f"  Count={c}: N={pc['count']}, MAE={pc['mae']:.4f}, "
                        f"Acc±1={pc['acc_pm1']:.1f}%\n")


##############################################################################
#  TRAINING
##############################################################################

def train_transfer(model, train_ld, val_ld, epochs, optimizer):
    best_state, best_val = None, float('inf')
    t0 = time.perf_counter()

    for ep in range(epochs):
        model.train()
        for batch in train_ld:
            if len(batch) == 3:
                bx, by, bw = batch
                bx, by, bw = bx.to(device), by.to(device), bw.to(device)
                preds = model(bx)
                loss = ((preds - by) ** 2 * bw).sum() / (bw.sum() + 1e-8)
            else:
                bx, by = batch[0].to(device), batch[1].to(device)
                loss = nn.functional.mse_loss(model(bx), by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for batch in val_ld:
                bx, by = batch[0], batch[1]
                vp.extend(model(bx.to(device)).cpu().numpy())
                vt.extend(by.numpy())
        vmae = mean_absolute_error(vt, vp)

        if (ep + 1) % max(1, epochs // 6) == 0:
            print(f"    Epoch [{ep + 1}/{epochs}] Val MAE: {vmae:.4f}")

        if vmae < best_val:
            best_val = vmae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time_sec = time.perf_counter() - t0

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"    Best Val MAE: {best_val:.4f}")
    print(f"    Train time: {train_time_sec:.3f}s")
    return train_time_sec


##############################################################################
#  RUN: NO-WEIGHT SETUPS
##############################################################################

def run_direct_transfer(model_type, weights_path, test_ld, tgt_df_full, tgt_split,
                        tgt_input, src_input, ws, out_dir):
    print(f"\n  -- Direct-Transfer --")
    m = build_model(model_type, tgt_input)
    m = load_pretrained_weights(m, weights_path, model_type, src_input, tgt_input)
    for p in m.parameters(): p.requires_grad = False

    metrics = evaluate_full(m, test_ld, "Direct-Transfer")
    save_setup_outputs(m, metrics, tgt_df_full, tgt_split, ws, "Direct-Transfer",
                       model_type, out_dir, "Init: pretrained (frozen, no target training)")
    result = {'method': 'Direct-Transfer', 'weight_scheme': 'none',
              'metrics': metrics, 'initialization': 'pretrained'}
    return add_complexity_info(result, m)


def run_feature_extraction(model_type, weights_path, train_ld, val_ld, test_ld,
                           tgt_df_full, tgt_split, tgt_input, src_input, ws, out_dir, epochs):
    print(f"\n  -- Feature-Extraction --")
    m = build_model(model_type, tgt_input)
    m = load_pretrained_weights(m, weights_path, model_type, src_input, tgt_input)
    for p in m.parameters(): p.requires_grad = False
    for p in m.fc.parameters(): p.requires_grad = True
    opt = optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-4)

    t = train_transfer(m, train_ld, val_ld, epochs, opt)
    metrics = evaluate_full(m, test_ld, "Feature-Extraction")
    save_setup_outputs(m, metrics, tgt_df_full, tgt_split, ws, "Feature-Extraction",
                       model_type, out_dir, "Init: pretrained, fc only trainable")
    result = {'method': 'Feature-Extraction', 'weight_scheme': 'none',
              'metrics': metrics, 'initialization': 'pretrained'}
    return add_complexity_info(result, m, target_train_time_sec=t)


def run_partial_ft_last(model_type, weights_path, train_ld, val_ld, test_ld,
                        tgt_df_full, tgt_split, tgt_input, src_input, ws, out_dir, epochs):
    if model_type == "Transformer": return None
    print(f"\n  -- Partial-FT-Last --")
    m = build_model(model_type, tgt_input)
    m = load_pretrained_weights(m, weights_path, model_type, src_input, tgt_input)
    for p in m.parameters(): p.requires_grad = False
    for n, p in m.named_parameters():
        if ("rnn." in n or "lstm." in n) and "_l1" in n: p.requires_grad = True
    for p in m.fc.parameters(): p.requires_grad = True

    backbone = [p for n, p in m.named_parameters()
                if ("rnn." in n or "lstm." in n) and "_l1" in n and p.requires_grad]
    opt = optim.Adam([{'params': backbone, 'lr': 5e-5},
                      {'params': list(m.fc.parameters()), 'lr': 2e-4}])
    t = train_transfer(m, train_ld, val_ld, epochs, opt)
    metrics = evaluate_full(m, test_ld, "Partial-FT-Last")
    save_setup_outputs(m, metrics, tgt_df_full, tgt_split, ws, "Partial-FT-Last",
                       model_type, out_dir, "Init: pretrained, last layer + fc")
    result = {'method': 'Partial-FT-Last', 'weight_scheme': 'none',
              'metrics': metrics, 'initialization': 'pretrained'}
    return add_complexity_info(result, m, target_train_time_sec=t)


def run_partial_ft_first(model_type, weights_path, train_ld, val_ld, test_ld,
                         tgt_df_full, tgt_split, tgt_input, src_input, ws, out_dir, epochs):
    if model_type == "Transformer": return None
    print(f"\n  -- Partial-FT-First --")
    m = build_model(model_type, tgt_input)
    m = load_pretrained_weights(m, weights_path, model_type, src_input, tgt_input)
    for p in m.parameters(): p.requires_grad = False
    for n, p in m.named_parameters():
        if ("rnn." in n or "lstm." in n) and "_l0" in n: p.requires_grad = True
    for p in m.fc.parameters(): p.requires_grad = True

    backbone = [p for n, p in m.named_parameters()
                if ("rnn." in n or "lstm." in n) and "_l0" in n and p.requires_grad]
    opt = optim.Adam([{'params': backbone, 'lr': 5e-5},
                      {'params': list(m.fc.parameters()), 'lr': 2e-4}])
    t = train_transfer(m, train_ld, val_ld, epochs, opt)
    metrics = evaluate_full(m, test_ld, "Partial-FT-First")
    save_setup_outputs(m, metrics, tgt_df_full, tgt_split, ws, "Partial-FT-First",
                       model_type, out_dir, "Init: pretrained, first layer + fc")
    result = {'method': 'Partial-FT-First', 'weight_scheme': 'none',
              'metrics': metrics, 'initialization': 'pretrained'}
    return add_complexity_info(result, m, target_train_time_sec=t)


def run_full_fine_tuning(model_type, weights_path, train_ld, val_ld, test_ld,
                         tgt_df_full, tgt_split, tgt_input, src_input, ws, out_dir, epochs):
    print(f"\n  -- Full-Fine-Tuning --")
    m = build_model(model_type, tgt_input)
    m = load_pretrained_weights(m, weights_path, model_type, src_input, tgt_input)
    for p in m.parameters(): p.requires_grad = True
    opt = optim.Adam(m.parameters(), lr=1e-4)

    t = train_transfer(m, train_ld, val_ld, epochs, opt)
    metrics = evaluate_full(m, test_ld, "Full-Fine-Tuning")
    save_setup_outputs(m, metrics, tgt_df_full, tgt_split, ws, "Full-Fine-Tuning",
                       model_type, out_dir, "Init: pretrained, all unfrozen")
    result = {'method': 'Full-Fine-Tuning', 'weight_scheme': 'none',
              'metrics': metrics, 'initialization': 'pretrained'}
    return add_complexity_info(result, m, target_train_time_sec=t)


def run_joint_training(model_type, weights_path, src_data, tgt_tr_seq_X, tgt_tr_seq_y,
                       val_ld, test_ld, tgt_df_full, tgt_split, tgt_input, src_input,
                       ws, out_dir, epochs):
    print(f"\n  -- Joint-Training --")
    if src_data['input_size'] != tgt_input:
        print(f"  ⚠ Cannot joint-train: src_dim={src_data['input_size']} != tgt_dim={tgt_input}")
        return None

    combined_X = np.vstack([src_data['X_tr_seq'], tgt_tr_seq_X]).astype(np.float32)
    combined_y = np.concatenate([src_data['y_tr_seq'], tgt_tr_seq_y]).astype(np.float32)
    n_src, n_tgt = len(src_data['y_tr_seq']), len(tgt_tr_seq_y)
    print(f"    Combined: src={n_src} + tgt={n_tgt} = {len(combined_y)} samples")

    bs = min(32, max(4, len(combined_y) // 8))
    combined_ld = DataLoader(
        TensorDataset(torch.tensor(combined_X, dtype=torch.float32),
                      torch.tensor(combined_y, dtype=torch.float32)),
        batch_size=bs, shuffle=True,
        generator=make_generator(SEED), worker_init_fn=seed_worker)

    m = build_model(model_type, tgt_input)
    if weights_path:
        m = load_pretrained_weights(m, weights_path, model_type, src_input, tgt_input)
    for p in m.parameters(): p.requires_grad = True
    opt = optim.Adam(m.parameters(), lr=1e-4)

    t = train_transfer(m, combined_ld, val_ld, epochs, opt)
    metrics = evaluate_full(m, test_ld, "Joint-Training")
    save_setup_outputs(m, metrics, tgt_df_full, tgt_split, ws, "Joint-Training", model_type,
                       out_dir, f"Init: pretrained, trained on src({n_src})+tgt({n_tgt})")
    result = {'method': 'Joint-Training', 'weight_scheme': 'none',
              'metrics': metrics, 'initialization': 'pretrained'}
    return add_complexity_info(result, m, target_train_time_sec=t)


def run_target_only(model_type, train_ld, val_ld, test_ld, tgt_df_full, tgt_split,
                    input_size, ws, out_dir, epochs):
    print(f"\n  -- Target-Only --")
    m = build_model(model_type, input_size)
    opt = optim.Adam(m.parameters(), lr=1e-3)
    t = train_transfer(m, train_ld, val_ld, epochs, opt)
    metrics = evaluate_full(m, test_ld, "Target-Only")
    save_setup_outputs(m, metrics, tgt_df_full, tgt_split, ws, "Target-Only", model_type,
                       out_dir, "Init: random (no source knowledge)")
    result = {'method': 'Target-Only', 'weight_scheme': 'none',
              'metrics': metrics, 'initialization': 'random'}
    return add_complexity_info(result, m, target_train_time_sec=t)


##############################################################################
#  LinInt
##############################################################################

def save_linint_outputs(metrics, preds, tgt_df_full, tgt_split, ws, alpha, model_type, out_dir):
    tag = f"linint_alpha{alpha:.2f}"
    test_start = tgt_split['test_start'] + ws - 1
    df_test = tgt_df_full.iloc[test_start:tgt_split['test_end']].copy()
    min_len = min(len(df_test), len(preds))
    df_test = df_test.iloc[:min_len]
    df_test['predicted_peoplecount'] = preds[:min_len]
    df_test['predicted_rounded'] = np.round(np.clip(preds[:min_len], 0, None)).astype(int)
    df_test.to_csv(os.path.join(out_dir, f"predicted_{tag}_ws{ws}.csv"), index=False)

    m = metrics['all']
    with open(os.path.join(out_dir, f"summary_{tag}_ws{ws}.txt"), "w") as f:
        f.write(f"Setup: LinInt (alpha={alpha:.2f})\nModel: {model_type}\nWS: {ws}\n"
                f"Formula: y = {1 - alpha:.2f}*y_src + {alpha:.2f}*y_tgt\n\n")
        f.write(f"MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} R²={m['r2']:.4f} "
                f"Acc±1={m['acc_pm1']:.2f}%\n")


def run_linint_all_alphas(model_type, weights_path, src_cfg, src_data, train_ld, val_ld,
                          test_ld, tgt_df_full, tgt_split, tgt_input, src_input,
                          ws, out_dir, epochs, prop=0.5):
    print(f"\n  -- LinInt (alphas={LININT_ALPHAS}) --")
    src_model = build_model(model_type, src_data['input_size'])
    if weights_path:
        src_model = load_pretrained_weights(src_model, weights_path, model_type,
                                            src_input, src_data['input_size'])
    opt_src = optim.Adam(src_model.parameters(), lr=1e-4)
    src_t = train_transfer(src_model, src_data['train_loader'], src_data['val_loader'],
                           epochs, opt_src)

    # If source and target dims differ, we cannot use the src model directly on tgt sequences
    if src_data['input_size'] != tgt_input:
        print(f"  ⚠ LinInt: src_dim={src_data['input_size']} != tgt_dim={tgt_input}; "
              "transferring src weights to target-dim model")
        stage_path = os.path.join(out_dir, f"linint_src_stage_ws{ws}.pth")
        torch.save(src_model.state_dict(), stage_path)
        src_model = build_model(model_type, tgt_input)
        src_model = load_pretrained_weights(src_model, stage_path, model_type,
                                            src_data['input_size'], tgt_input)

    tgt_all_X, tgt_all_y = [], []
    for bx, by in train_ld:
        tgt_all_X.append(bx); tgt_all_y.append(by)
    tgt_all_X = torch.cat(tgt_all_X, dim=0); tgt_all_y = torch.cat(tgt_all_y, dim=0)
    n_tgt = len(tgt_all_y)
    perm = torch.randperm(n_tgt, generator=make_generator(SEED))
    cut = int(n_tgt * prop)
    tgt_fit_X = tgt_all_X[perm[:cut]]; tgt_fit_y = tgt_all_y[perm[:cut]]

    tgt_model = build_model(model_type, tgt_input)
    opt_tgt = optim.Adam(tgt_model.parameters(), lr=1e-3)
    bs = min(32, max(4, cut // 4))
    tgt_fit_ld = DataLoader(TensorDataset(tgt_fit_X, tgt_fit_y),
                            batch_size=bs, shuffle=True,
                            generator=make_generator(SEED), worker_init_fn=seed_worker)
    tgt_t = train_transfer(tgt_model, tgt_fit_ld, val_ld, epochs, opt_tgt)

    src_model.eval(); tgt_model.eval()
    ps, pt, tt = [], [], []
    with torch.no_grad():
        for bx, by in test_ld:
            ps.extend(src_model(bx.to(device)).cpu().numpy())
            pt.extend(tgt_model(bx.to(device)).cpu().numpy())
            tt.extend(by.numpy())
    ps, pt, tt = np.array(ps), np.array(pt), np.array(tt)

    results = []
    for alpha in LININT_ALPHAS:
        name = f"LinInt (alpha={alpha:.2f})"
        preds = (1 - alpha) * ps + alpha * pt
        metrics = evaluate_from_arrays(tt, preds, name)
        save_linint_outputs(metrics, preds, tgt_df_full, tgt_split, ws, alpha, model_type, out_dir)
        result = {'method': name, 'weight_scheme': 'none',
                  'metrics': metrics, 'initialization': 'pretrained+random'}
        results.append(add_complexity_info(result, tgt_model,
                                           source_train_time_sec=src_t,
                                           target_train_time_sec=tgt_t))

    torch.save(src_model.state_dict(), os.path.join(out_dir, f"weights_linint_srconly_ws{ws}.pth"))
    torch.save(tgt_model.state_dict(), os.path.join(out_dir, f"weights_linint_tgtonly_ws{ws}.pth"))
    return results


##############################################################################
#  RUN: LABEL-AWARE METHODS (TSW / TSBW)
##############################################################################

def _print_weight_summary(w, label):
    print(f"    [{label}] weights: mean={w.mean():.3f}, std={w.std():.3f}, "
          f"min={w.min():.3f}, max={w.max():.3f}")


def _src_tgt_dim_ok(src_data, tgt_input):
    """Joint-style methods require matching input dims (skip otherwise)."""
    if src_data['input_size'] != tgt_input:
        print(f"  ⚠ Skipping joint method: src_dim={src_data['input_size']} != tgt_dim={tgt_input}")
        return False
    return True


def run_label_aware_joint_ft(scheme, model_type, weights_path, src_data,
                              tgt_tr_seq_X, tgt_tr_seq_y,
                              val_ld, test_ld, tgt_df_full, tgt_split,
                              tgt_input, src_input, ws, out_dir, epochs):
    name = f"Label-Aware Joint FT ({scheme})"
    print(f"\n  -- {name} --")
    if not _src_tgt_dim_ok(src_data, tgt_input): return None

    combined_X = np.vstack([src_data['X_tr_seq'], tgt_tr_seq_X]).astype(np.float32)
    combined_y = np.concatenate([src_data['y_tr_seq'], tgt_tr_seq_y]).astype(np.float32)

    t0_w = time.perf_counter()
    w = compute_sample_weights(combined_y, tgt_tr_seq_y, scheme)
    w_t = time.perf_counter() - t0_w
    _print_weight_summary(w, scheme)

    bs = min(32, max(4, len(combined_y) // 8))
    ld = _make_weighted_loader(combined_X, combined_y, w, bs, True)

    m = build_model(model_type, tgt_input)
    m = load_pretrained_weights(m, weights_path, model_type, src_input, tgt_input)
    for p in m.parameters(): p.requires_grad = True
    opt = optim.Adam(m.parameters(), lr=1e-4)

    t = train_transfer(m, ld, val_ld, epochs, opt)
    metrics = evaluate_full(m, test_ld, name)
    save_setup_outputs(m, metrics, tgt_df_full, tgt_split, ws, name, model_type, out_dir,
                       f"Init: pretrained, joint FT with {scheme} label weights")
    result = {'method': name, 'weight_scheme': scheme,
              'metrics': metrics, 'initialization': 'pretrained'}
    return add_complexity_info(result, m, target_train_time_sec=t, weight_compute_time_sec=w_t)


def run_label_aware_pretrain_target_ft(scheme, model_type, weights_path, src_data,
                                        tgt_tr_seq_y, train_ld, val_ld, test_ld,
                                        tgt_df_full, tgt_split, tgt_input, src_input,
                                        ws, out_dir, epochs):
    name = f"Label-Aware Pretrain to Target FT ({scheme})"
    print(f"\n  -- {name} --")

    t0_w = time.perf_counter()
    w_src = compute_sample_weights(src_data['y_tr_seq'], tgt_tr_seq_y, scheme)
    w_t = time.perf_counter() - t0_w
    _print_weight_summary(w_src, f"{scheme}-src")

    bs_s = min(32, max(4, len(src_data['y_tr_seq']) // 8))
    src_ld = _make_weighted_loader(src_data['X_tr_seq'], src_data['y_tr_seq'], w_src, bs_s, True)

    # Stage 1: train on weighted source (at source dim)
    m_src = build_model(model_type, src_data['input_size'])
    m_src = load_pretrained_weights(m_src, weights_path, model_type,
                                    src_input, src_data['input_size'])
    for p in m_src.parameters(): p.requires_grad = True
    opt1 = optim.Adam(m_src.parameters(), lr=1e-4)
    s_t = train_transfer(m_src, src_ld, src_data['val_loader'], epochs, opt1)

    # Save stage-1 weights and reload into target-dim model
    stage1_path = os.path.join(out_dir, f"stage1_{slugify(name)}_ws{ws}.pth")
    torch.save(m_src.state_dict(), stage1_path)

    print(f"    Stage 2: target-only FT")
    m = build_model(model_type, tgt_input)
    m = load_pretrained_weights(m, stage1_path, model_type,
                                src_data['input_size'], tgt_input)
    for p in m.parameters(): p.requires_grad = True
    opt2 = optim.Adam(m.parameters(), lr=1e-4)
    tg_t = train_transfer(m, train_ld, val_ld, epochs, opt2)

    metrics = evaluate_full(m, test_ld, name)
    save_setup_outputs(m, metrics, tgt_df_full, tgt_split, ws, name, model_type, out_dir,
                       f"Stage1: weighted source ({scheme}); Stage2: target-only FT")
    result = {'method': name, 'weight_scheme': scheme,
              'metrics': metrics, 'initialization': 'pretrained'}
    return add_complexity_info(result, m,
                               source_train_time_sec=s_t,
                               target_train_time_sec=tg_t,
                               weight_compute_time_sec=w_t)


def run_target_priority_joint_ft(scheme, model_type, weights_path, src_data,
                                  tgt_tr_seq_X, tgt_tr_seq_y,
                                  val_ld, test_ld, tgt_df_full, tgt_split,
                                  tgt_input, src_input, ws, out_dir, epochs):
    name = f"Target-Priority Joint FT ({scheme})"
    print(f"\n  -- {name} --")
    if not _src_tgt_dim_ok(src_data, tgt_input): return None

    n_src = len(src_data['y_tr_seq'])
    n_tgt = len(tgt_tr_seq_y)
    combined_X = np.vstack([src_data['X_tr_seq'], tgt_tr_seq_X]).astype(np.float32)
    combined_y = np.concatenate([src_data['y_tr_seq'], tgt_tr_seq_y]).astype(np.float32)

    t0_w = time.perf_counter()
    w_label = compute_sample_weights(combined_y, tgt_tr_seq_y, scheme)
    w_domain = np.concatenate([
        np.full(n_src, TARGET_PRIORITY_LAMBDA_S, dtype=np.float32),
        np.full(n_tgt, TARGET_PRIORITY_LAMBDA_T, dtype=np.float32),
    ])
    w = normalize_and_clip(w_label * w_domain)
    w_t = time.perf_counter() - t0_w
    _print_weight_summary(w, f"{scheme}×domain")

    bs = min(32, max(4, len(combined_y) // 8))
    ld = _make_weighted_loader(combined_X, combined_y, w, bs, True)

    m = build_model(model_type, tgt_input)
    m = load_pretrained_weights(m, weights_path, model_type, src_input, tgt_input)
    for p in m.parameters(): p.requires_grad = True
    opt = optim.Adam(m.parameters(), lr=1e-4)

    t = train_transfer(m, ld, val_ld, epochs, opt)
    metrics = evaluate_full(m, test_ld, name)
    save_setup_outputs(m, metrics, tgt_df_full, tgt_split, ws, name, model_type, out_dir,
                       f"Label-weighted ({scheme}) × domain "
                       f"(λs={TARGET_PRIORITY_LAMBDA_S}, λt={TARGET_PRIORITY_LAMBDA_T})")
    result = {'method': name, 'weight_scheme': scheme,
              'metrics': metrics, 'initialization': 'pretrained'}
    return add_complexity_info(result, m, target_train_time_sec=t, weight_compute_time_sec=w_t)


def run_thresholded_label_aware_ft(scheme, model_type, weights_path, src_data,
                                    tgt_tr_seq_X, tgt_tr_seq_y,
                                    val_ld, test_ld, tgt_df_full, tgt_split,
                                    tgt_input, src_input, ws, out_dir, epochs):
    name = f"Thresholded Label-Aware FT ({scheme})"
    print(f"\n  -- {name} --")
    if not _src_tgt_dim_ok(src_data, tgt_input): return None

    n_src = len(src_data['y_tr_seq'])
    combined_X = np.vstack([src_data['X_tr_seq'], tgt_tr_seq_X]).astype(np.float32)
    combined_y = np.concatenate([src_data['y_tr_seq'], tgt_tr_seq_y]).astype(np.float32)

    t0_w = time.perf_counter()
    w = compute_sample_weights(combined_y, tgt_tr_seq_y, scheme)
    src_mask = w[:n_src] >= THRESHOLD_DROP
    keep_idx = np.concatenate([np.where(src_mask)[0], np.arange(n_src, len(combined_y))])
    X_f, y_f, w_f = combined_X[keep_idx], combined_y[keep_idx], w[keep_idx]
    w_t = time.perf_counter() - t0_w

    print(f"    Filtering: kept {src_mask.sum()}/{n_src} src rows (threshold={THRESHOLD_DROP})")
    _print_weight_summary(w_f, f"{scheme}-filtered")

    bs = min(32, max(4, len(y_f) // 8))
    ld = _make_weighted_loader(X_f, y_f, w_f, bs, True)

    m = build_model(model_type, tgt_input)
    m = load_pretrained_weights(m, weights_path, model_type, src_input, tgt_input)
    for p in m.parameters(): p.requires_grad = True
    opt = optim.Adam(m.parameters(), lr=1e-4)

    t = train_transfer(m, ld, val_ld, epochs, opt)
    metrics = evaluate_full(m, test_ld, name)
    save_setup_outputs(m, metrics, tgt_df_full, tgt_split, ws, name, model_type, out_dir,
                       f"Scheme: {scheme}, drop src if w<{THRESHOLD_DROP}, "
                       f"kept={src_mask.sum()}/{n_src}")
    result = {'method': name, 'weight_scheme': scheme,
              'metrics': metrics, 'initialization': 'pretrained'}
    return add_complexity_info(result, m, target_train_time_sec=t, weight_compute_time_sec=w_t)


def run_label_aware_source_training(scheme, model_type, weights_path, src_data,
                                     tgt_tr_seq_y, test_ld, tgt_df_full, tgt_split,
                                     tgt_input, src_input, ws, out_dir, epochs):
    name = f"Label-Aware Source Training ({scheme})"
    print(f"\n  -- {name} --")

    t0_w = time.perf_counter()
    w = compute_sample_weights(src_data['y_tr_seq'], tgt_tr_seq_y, scheme)
    w_t = time.perf_counter() - t0_w
    _print_weight_summary(w, scheme)

    bs = min(32, max(4, len(src_data['y_tr_seq']) // 8))
    ld = _make_weighted_loader(src_data['X_tr_seq'], src_data['y_tr_seq'], w, bs, True)

    # Train at source dim, then transfer
    m_src = build_model(model_type, src_data['input_size'])
    m_src = load_pretrained_weights(m_src, weights_path, model_type,
                                    src_input, src_data['input_size'])
    for p in m_src.parameters(): p.requires_grad = True
    opt = optim.Adam(m_src.parameters(), lr=1e-4)
    s_t = train_transfer(m_src, ld, src_data['val_loader'], epochs, opt)

    if src_data['input_size'] == tgt_input:
        m_eval = m_src
    else:
        stage_path = os.path.join(out_dir, f"stage_{slugify(name)}_ws{ws}.pth")
        torch.save(m_src.state_dict(), stage_path)
        m_eval = build_model(model_type, tgt_input)
        m_eval = load_pretrained_weights(m_eval, stage_path, model_type,
                                         src_data['input_size'], tgt_input)
        for p in m_eval.parameters(): p.requires_grad = False

    metrics = evaluate_full(m_eval, test_ld, name)
    save_setup_outputs(m_eval, metrics, tgt_df_full, tgt_split, ws, name, model_type, out_dir,
                       f"Scheme: {scheme}, weighted source FT, direct eval on target")
    result = {'method': name, 'weight_scheme': scheme,
              'metrics': metrics, 'initialization': 'pretrained'}
    return add_complexity_info(result, m_eval,
                               source_train_time_sec=s_t, weight_compute_time_sec=w_t)


def run_thresholded_label_aware_source_training(scheme, model_type, weights_path, src_data,
                                                 tgt_tr_seq_y, test_ld, tgt_df_full,
                                                 tgt_split, tgt_input, src_input,
                                                 ws, out_dir, epochs):
    name = f"Thresholded Label-Aware Source Training ({scheme})"
    print(f"\n  -- {name} --")

    t0_w = time.perf_counter()
    w = compute_sample_weights(src_data['y_tr_seq'], tgt_tr_seq_y, scheme)
    mask = w >= THRESHOLD_DROP
    X_f = src_data['X_tr_seq'][mask]; y_f = src_data['y_tr_seq'][mask]; w_f = w[mask]
    w_t = time.perf_counter() - t0_w

    print(f"    Filtering: kept {mask.sum()}/{len(w)} src rows")
    if len(y_f) < 10:
        print(f"    ⚠ Too few samples after filtering ({len(y_f)}), skipping")
        return None
    _print_weight_summary(w_f, f"{scheme}-filtered")

    bs = min(32, max(4, len(y_f) // 8))
    ld = _make_weighted_loader(X_f, y_f, w_f, bs, True)

    m_src = build_model(model_type, src_data['input_size'])
    m_src = load_pretrained_weights(m_src, weights_path, model_type,
                                    src_input, src_data['input_size'])
    for p in m_src.parameters(): p.requires_grad = True
    opt = optim.Adam(m_src.parameters(), lr=1e-4)
    s_t = train_transfer(m_src, ld, src_data['val_loader'], epochs, opt)

    if src_data['input_size'] == tgt_input:
        m_eval = m_src
    else:
        stage_path = os.path.join(out_dir, f"stage_{slugify(name)}_ws{ws}.pth")
        torch.save(m_src.state_dict(), stage_path)
        m_eval = build_model(model_type, tgt_input)
        m_eval = load_pretrained_weights(m_eval, stage_path, model_type,
                                         src_data['input_size'], tgt_input)
        for p in m_eval.parameters(): p.requires_grad = False

    metrics = evaluate_full(m_eval, test_ld, name)
    save_setup_outputs(m_eval, metrics, tgt_df_full, tgt_split, ws, name, model_type, out_dir,
                       f"Scheme: {scheme}, drop if w<{THRESHOLD_DROP}, "
                       f"kept={mask.sum()}/{len(w)}")
    result = {'method': name, 'weight_scheme': scheme,
              'metrics': metrics, 'initialization': 'pretrained'}
    return add_complexity_info(result, m_eval,
                               source_train_time_sec=s_t, weight_compute_time_sec=w_t)


##############################################################################
#  RUN: POST-HOC ADJUSTMENTS
##############################################################################

def _get_src_preds(weights_path, model_type, src_input, tgt_input, X_va_seq, X_te_seq):
    m = build_model(model_type, tgt_input)
    m = load_pretrained_weights(m, weights_path, model_type, src_input, tgt_input)
    for p in m.parameters(): p.requires_grad = False
    return m, predict_array(m, X_va_seq), predict_array(m, X_te_seq)


def run_target_calibration(model_type, weights_path, X_va_seq, y_va_seq, X_te_seq, y_te_seq,
                            tgt_df_full, tgt_split, tgt_input, src_input, ws, out_dir):
    name = "Source Model + Target Calibration"
    print(f"\n  -- {name} (isotonic) --")
    src_model, val_preds, test_preds = _get_src_preds(
        weights_path, model_type, src_input, tgt_input, X_va_seq, X_te_seq)

    try:
        cal = IsotonicRegression(out_of_bounds='clip').fit(val_preds, y_va_seq)
        test_cal = cal.predict(test_preds)
    except Exception as e:
        print(f"    Isotonic failed ({e}); fallback to identity")
        test_cal = test_preds

    metrics = evaluate_from_arrays(y_te_seq, test_cal, name)
    _save_pred_and_summary(metrics, tgt_df_full, tgt_split, ws, name, model_type, out_dir,
                           "Post-hoc: isotonic calibration on target val")
    torch.save(src_model.state_dict(),
               os.path.join(out_dir, f"weights_{slugify(name)}_ws{ws}.pth"))
    result = {'method': name, 'weight_scheme': 'none',
              'metrics': metrics, 'initialization': 'pretrained'}
    return add_complexity_info(result, src_model)


def run_linear_mapping(model_type, weights_path, X_va_seq, y_va_seq, X_te_seq, y_te_seq,
                        tgt_df_full, tgt_split, tgt_input, src_input, ws, out_dir):
    name = "Source-to-Target Linear Mapping"
    print(f"\n  -- {name} (y = a*y_pred + b) --")
    src_model, val_preds, test_preds = _get_src_preds(
        weights_path, model_type, src_input, tgt_input, X_va_seq, X_te_seq)

    try:
        a, b = np.polyfit(val_preds, y_va_seq, 1)
        print(f"    Fitted: a={a:.4f}, b={b:.4f}")
        test_mapped = a * test_preds + b
    except Exception as e:
        print(f"    polyfit failed ({e}); fallback to identity")
        a, b, test_mapped = 1.0, 0.0, test_preds

    metrics = evaluate_from_arrays(y_te_seq, test_mapped, name)
    _save_pred_and_summary(metrics, tgt_df_full, tgt_split, ws, name, model_type, out_dir,
                           f"Post-hoc: y = {a:.4f}*y_pred + {b:.4f}")
    torch.save(src_model.state_dict(),
               os.path.join(out_dir, f"weights_{slugify(name)}_ws{ws}.pth"))
    result = {'method': name, 'weight_scheme': 'none',
              'metrics': metrics, 'initialization': 'pretrained'}
    return add_complexity_info(result, src_model)


##############################################################################
#  PLOTS
##############################################################################

def create_comparison_plots(results, ws, scenario_name, model_type, out_dir):
    results = [r for r in results if r is not None]
    if not results: return

    methods = [r['method'] for r in results]
    n = len(methods)
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(n)]

    metrics_list = [('mae', 'MAE'), ('rmse', 'RMSE'), ('r2', 'R²'),
                    ('medae', 'MedAE'), ('acc_pm1', 'Acc±1 (%)'), ('acc_pm2', 'Acc±2 (%)')]

    fig, axes = plt.subplots(2, 3, figsize=(28, 13))
    fig.suptitle(f'Parameter-Based Transfer ({n} setups): {scenario_name} | '
                 f'{model_type} | ws={ws}', fontsize=13, fontweight='bold')

    for idx, (key, ylabel) in enumerate(metrics_list):
        ax = axes[idx // 3][idx % 3]
        vals = [r['metrics']['all'][key] for r in results]
        bars = ax.bar(range(n), vals, color=colors, edgecolor='black', alpha=0.85)
        ax.set_xticks(range(n))
        ax.set_xticklabels(methods, rotation=60, ha='right', fontsize=5.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{v:.3f}',
                    ha='center', va='bottom', fontsize=4.5, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'comparison_all_{model_type}_ws{ws}.png'),
                dpi=180, bbox_inches='tight')
    plt.close()


##############################################################################
#  RESULT ROW
##############################################################################

def build_result_row(scenario_name, src_id, tgt_id, model_type, ws,
                     method_result, src_cfg, tgt_cfg, variant):
    m = method_result['metrics']['all']
    row = {
        'scenario': scenario_name, 'source': src_id, 'target': tgt_id,
        'method': method_result['method'],
        'weight_scheme': method_result.get('weight_scheme', 'none'),
        'variant': variant,
        'model': model_type, 'window_size': ws,
        'features': ', '.join(tgt_cfg['features']),
        'src_scenario': src_cfg['scenario'], 'tgt_scenario': tgt_cfg['scenario'],
        'initialization': method_result.get('initialization', 'pretrained'),
        'source_train_time_sec':     method_result.get('source_train_time_sec', 0.0),
        'target_train_time_sec':     method_result.get('target_train_time_sec', 0.0),
        'weight_compute_time_sec':   method_result.get('weight_compute_time_sec', 0.0),
        'alignment_train_time_sec':  method_result.get('alignment_train_time_sec', 0.0),
        'model_complexity_time_sec': method_result.get('model_complexity_time_sec', 0.0),
        'total_params':     method_result.get('total_params', np.nan),
        'trainable_params': method_result.get('trainable_params', np.nan),
        'frozen_params':    method_result.get('frozen_params', np.nan),
        'test_mae': m['mae'], 'test_rmse': m['rmse'], 'test_r2': m['r2'],
        'test_medae': m['medae'], 'test_acc_pm1': m['acc_pm1'],
        'test_acc_pm2': m['acc_pm2'], 'test_samples': m['sample_count'],
        'occ_mae': np.nan, 'occ_rmse': np.nan, 'occ_r2': np.nan, 'occ_medae': np.nan,
        'occ_acc_pm1': np.nan, 'occ_acc_pm2': np.nan, 'occ_exact_pct': np.nan,
        'occ_samples': np.nan,
        'empty_mae': np.nan, 'empty_acc_pm1': np.nan, 'empty_samples': np.nan,
    }
    if method_result['metrics']['occupied']:
        mo = method_result['metrics']['occupied']
        row.update({'occ_mae': mo['mae'], 'occ_rmse': mo['rmse'], 'occ_r2': mo['r2'],
                    'occ_medae': mo['medae'], 'occ_acc_pm1': mo['acc_pm1'],
                    'occ_acc_pm2': mo['acc_pm2'], 'occ_exact_pct': mo['exact_match_pct'],
                    'occ_samples': mo['sample_count']})
    if method_result['metrics']['empty']:
        me = method_result['metrics']['empty']
        row.update({'empty_mae': me['mae'], 'empty_acc_pm1': me['acc_pm1'],
                    'empty_samples': me['sample_count']})
    return row


##############################################################################
#  SAFE WRAPPER
##############################################################################

def _safe(fn, label, *args, **kwargs):
    try:
        reset_all_seeds(SEED)
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"  ❌ {label} failed: {e}")
        logging.error(f"{label}: {e}", exc_info=True)
        return None


##############################################################################
#  MAIN
##############################################################################

def main():
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 120)
    print("PARAMETER-BASED TRANSFER LEARNING — HOMOGENEOUS + HETEROGENEOUS")
    print(f"Models: {MODEL_TYPES} | WS: {WINDOW_SIZES} | Epochs: {EPOCHS}")
    print(f"Variants: {FEATURE_VARIANTS}")
    print(f"Schemes: {WEIGHT_SCHEMES} | LinInt α: {LININT_ALPHAS}")
    print(f"PRETRAIN_DIR: {PRETRAIN_DIR}")
    print(f"TRANSFER_DIR: {TRANSFER_DIR}")
    print(f"User: Azadshokrollahi | {timestamp} UTC")
    print("=" * 120)

    ensure_dir(TRANSFER_DIR)
    results_by_variant = {v: [] for v in FEATURE_VARIANTS}

    for variant in FEATURE_VARIANTS:
        print(f"\n\n{'#' * 120}")
        print(f"#  FEATURE VARIANT: {variant.upper()}")
        print(f"{'#' * 120}")

        valid_scenarios = [(s, t) for (s, t) in TRANSFER_SCENARIOS
                           if variant_scenario_supported(s, t, variant)]

        print(f"  Valid scenarios for {variant}: {len(valid_scenarios)}/{len(TRANSFER_SCENARIOS)}")
        for s, t in valid_scenarios:
            print(f"    {s} -> {t}  "
                  f"(src={get_variant_features(s, variant)}, "
                  f"tgt={get_variant_features(t, variant)})")

        if not valid_scenarios:
            print(f"  ⚠ No valid scenarios for {variant}; skipping.")
            continue

        total = len(valid_scenarios) * len(MODEL_TYPES) * len(WINDOW_SIZES)
        counter = 0

        for sc_idx, (src_id, tgt_id) in enumerate(valid_scenarios):
            scenario_name = f"{src_id} -> {tgt_id}"
            src_cfg = build_room_cfg(src_id, variant)
            tgt_cfg = build_room_cfg(tgt_id, variant)

            print(f"\n{'=' * 120}")
            print(f"[{variant}] SCENARIO {sc_idx + 1}/{len(valid_scenarios)}: {scenario_name}")
            print(f"  src features: {src_cfg['features']}")
            print(f"  tgt features: {tgt_cfg['features']}")
            print(f"{'=' * 120}")

            if not os.path.exists(src_cfg['path']) or not os.path.exists(tgt_cfg['path']):
                print("  ❌ Data file missing"); continue

            tgt_X, tgt_y, tgt_df_full, _ = load_room_raw(
                tgt_cfg['path'], tgt_cfg['features'],
                tgt_cfg['name'], tgt_cfg['selected_rows'])
            tgt_tr_X, tgt_tr_y, tgt_va_X, tgt_va_y, tgt_te_X, tgt_te_y, tgt_split = \
                split_by_scenario(tgt_X, tgt_y, tgt_cfg['scenario'])

            for model_type in MODEL_TYPES:
                for ws in WINDOW_SIZES:
                    counter += 1
                    print(f"\n  ── [{variant}][{counter}/{total}] {model_type} | ws={ws} ──")

                    sc_folder = ensure_dir(os.path.join(
                        TRANSFER_DIR, variant,
                        f"{src_id}_to_{tgt_id}", f"{model_type}_ws{ws}"))

                    log_file = os.path.join(sc_folder, "transfer_log.txt")
                    for h in logging.root.handlers[:]:
                        logging.root.removeHandler(h)
                    logging.basicConfig(
                        level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(log_file, mode='a'),
                                  logging.StreamHandler(sys.stdout)], force=True)

                    weights_path = find_pretrained_weights(
                        PRETRAIN_DIR, src_id, variant, model_type, ws)
                    if weights_path is None:
                        print(f"  ⚠ No pretrained weights for "
                              f"{src_id}/{variant}/{model_type}_ws{ws} — skipping")
                        continue
                    print(f"  Pretrained: {weights_path}")

                    try:
                        (train_ld, val_ld, test_ld, X_te_t, scaler, actual_input,
                         tgt_tr_seq_X, tgt_tr_seq_y,
                         X_va_seq, y_va_seq, X_te_seq, y_te_seq
                         ) = prepare_target_dataloaders(
                            tgt_tr_X, tgt_tr_y, tgt_va_X, tgt_va_y,
                            tgt_te_X, tgt_te_y, ws)
                    except Exception as e:
                        print(f"  Data prep failed: {e}"); continue

                    joblib.dump(scaler, os.path.join(sc_folder, "tgt_scaler.pkl"))
                    tgt_input = actual_input

                    try:
                        src_data = prepare_source_data(src_cfg, ws)
                        src_input = src_data['input_size']
                    except Exception as e:
                        print(f"  Src data: {e}")
                        src_data, src_input = None, tgt_input

                    if src_data is not None and src_input != tgt_input:
                        print(f"  ℹ Heterogeneous: src_input={src_input}, "
                              f"tgt_input={tgt_input} — incompatible layers will be reinit")

                    setup_results = []

                    # --- FT setups ---
                    setup_results.append(_safe(
                        run_full_fine_tuning, "Full-FT", model_type, weights_path,
                        train_ld, val_ld, test_ld, tgt_df_full, tgt_split,
                        tgt_input, src_input, ws, sc_folder, EPOCHS))
                    setup_results.append(_safe(
                        run_partial_ft_first, "PFT-First", model_type, weights_path,
                        train_ld, val_ld, test_ld, tgt_df_full, tgt_split,
                        tgt_input, src_input, ws, sc_folder, EPOCHS))
                    setup_results.append(_safe(
                        run_partial_ft_last, "PFT-Last", model_type, weights_path,
                        train_ld, val_ld, test_ld, tgt_df_full, tgt_split,
                        tgt_input, src_input, ws, sc_folder, EPOCHS))
                    setup_results.append(_safe(
                        run_feature_extraction, "FeatExt", model_type, weights_path,
                        train_ld, val_ld, test_ld, tgt_df_full, tgt_split,
                        tgt_input, src_input, ws, sc_folder, EPOCHS))

                    # --- Label-aware FT (TSW + TSBW) ---
                    if src_data is not None:
                        for scheme in WEIGHT_SCHEMES:
                            setup_results.append(_safe(
                                run_label_aware_joint_ft, f"LA-JointFT-{scheme}",
                                scheme, model_type, weights_path, src_data,
                                tgt_tr_seq_X, tgt_tr_seq_y, val_ld, test_ld,
                                tgt_df_full, tgt_split, tgt_input, src_input,
                                ws, sc_folder, EPOCHS))
                            setup_results.append(_safe(
                                run_label_aware_pretrain_target_ft,
                                f"LA-PretrainTgtFT-{scheme}",
                                scheme, model_type, weights_path, src_data,
                                tgt_tr_seq_y, train_ld, val_ld, test_ld,
                                tgt_df_full, tgt_split, tgt_input, src_input,
                                ws, sc_folder, EPOCHS))
                            setup_results.append(_safe(
                                run_target_priority_joint_ft,
                                f"TargetPriority-{scheme}",
                                scheme, model_type, weights_path, src_data,
                                tgt_tr_seq_X, tgt_tr_seq_y, val_ld, test_ld,
                                tgt_df_full, tgt_split, tgt_input, src_input,
                                ws, sc_folder, EPOCHS))
                            setup_results.append(_safe(
                                run_thresholded_label_aware_ft,
                                f"ThreshLA-FT-{scheme}",
                                scheme, model_type, weights_path, src_data,
                                tgt_tr_seq_X, tgt_tr_seq_y, val_ld, test_ld,
                                tgt_df_full, tgt_split, tgt_input, src_input,
                                ws, sc_folder, EPOCHS))

                    # --- No-FT setups ---
                    setup_results.append(_safe(
                        run_direct_transfer, "DirectTransfer",
                        model_type, weights_path, test_ld, tgt_df_full, tgt_split,
                        tgt_input, src_input, ws, sc_folder))

                    if src_data is not None:
                        setup_results.append(_safe(
                            run_joint_training, "Joint",
                            model_type, weights_path, src_data,
                            tgt_tr_seq_X, tgt_tr_seq_y, val_ld, test_ld,
                            tgt_df_full, tgt_split, tgt_input, src_input,
                            ws, sc_folder, EPOCHS))
                        for scheme in WEIGHT_SCHEMES:
                            setup_results.append(_safe(
                                run_label_aware_source_training,
                                f"LA-SrcTrain-{scheme}",
                                scheme, model_type, weights_path, src_data,
                                tgt_tr_seq_y, test_ld, tgt_df_full, tgt_split,
                                tgt_input, src_input, ws, sc_folder, EPOCHS))
                            setup_results.append(_safe(
                                run_thresholded_label_aware_source_training,
                                f"ThreshLA-SrcTrain-{scheme}",
                                scheme, model_type, weights_path, src_data,
                                tgt_tr_seq_y, test_ld, tgt_df_full, tgt_split,
                                tgt_input, src_input, ws, sc_folder, EPOCHS))

                        try:
                            reset_all_seeds(SEED)
                            linint_results = run_linint_all_alphas(
                                model_type, weights_path, src_cfg, src_data,
                                train_ld, val_ld, test_ld, tgt_df_full, tgt_split,
                                tgt_input, src_input, ws, sc_folder, EPOCHS,
                                prop=LININT_PROP)
                            setup_results.extend(linint_results)
                        except Exception as e:
                            print(f"  LinInt: {e}")
                            logging.error(f"LinInt: {e}")

                    # --- Post-hoc ---
                    setup_results.append(_safe(
                        run_target_calibration, "Calibration",
                        model_type, weights_path,
                        X_va_seq, y_va_seq, X_te_seq, y_te_seq,
                        tgt_df_full, tgt_split, tgt_input, src_input,
                        ws, sc_folder))
                    setup_results.append(_safe(
                        run_linear_mapping, "LinearMap",
                        model_type, weights_path,
                        X_va_seq, y_va_seq, X_te_seq, y_te_seq,
                        tgt_df_full, tgt_split, tgt_input, src_input,
                        ws, sc_folder))

                    # --- Target-Only baseline ---
                    setup_results.append(_safe(
                        run_target_only, "TargetOnly",
                        model_type, train_ld, val_ld, test_ld,
                        tgt_df_full, tgt_split, tgt_input, ws, sc_folder, EPOCHS))

                    setup_results = [r for r in setup_results if r is not None]

                    create_comparison_plots(setup_results, ws, scenario_name,
                                            model_type, sc_folder)

                    rows = []
                    for r in setup_results:
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

                    print(f"\n  ✓ [{variant}] {scenario_name} | {model_type} | "
                          f"ws={ws} — {len(setup_results)} setups")

    # ─────────────────────────────────────────────────────────────────────────
    # MASTER CSVs (one per variant)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 120)
    print("FINAL RESULTS — PARAMETER-BASED TRANSFER")
    print("=" * 120)

    COLUMN_ORDER = [
        'scenario', 'source', 'target', 'method', 'weight_scheme', 'variant',
        'model', 'window_size', 'features', 'src_scenario', 'tgt_scenario',
        'initialization',
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
        out_csv = os.path.join(TRANSFER_DIR, f"all_parameter_based_{variant}.csv")
        df.to_csv(out_csv, index=False)

        print(f"\n  [{variant}] Master CSV: {out_csv}")
        print(f"  [{variant}] Total rows: {len(df)}")

        print(f"\n  [{variant}] BEST METHOD PER SCENARIO × MODEL (by MAE):")
        print(f"  {'Scenario':<25} {'Model':<12} {'Method':<45} "
              f"{'Scheme':<8} {'MAE':<8} {'Acc±1':<8}")
        print("  " + "-" * 110)
        for (sc, mdl), grp in df.groupby(['scenario', 'model']):
            best = grp.loc[grp['test_mae'].idxmin()]
            print(f"  {sc:<25} {mdl:<12} {best['method']:<45} "
                  f"{best['weight_scheme']:<8} "
                  f"{best['test_mae']:<8.4f} {best['test_acc_pm1']:<8.2f}")

    print("\n" + "=" * 120)
    print("✓ PARAMETER-BASED TRANSFER COMPLETED!")
    print(f"User: Azadshokrollahi | "
          f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 120)


if __name__ == "__main__":
    main()