"""
ALL TRANSFER COMBINATIONS — PT+IB, PT+FB, IB+FB, PT+IB+FB × {none, TSBW}

Homogeneous only (instance-based and feature-based methods require matching
input dimensions; heterogeneous mode is not applicable).

Reads:
  • configs/rooms.yaml   → room paths, scenarios, feature_variants, selected_rows
  • src/config.py        → seeds, epochs, batch size, LR, paths

Outputs:
  • results/all_transfer_combinations/all_transfer_combinations_homogeneous.csv
  • per-scenario folders with weights / predictions / summaries / plots

4 Approach Groups × 2 weighting schemes = 80 methods per (scenario × model × ws):

  ── GROUP 1: PT+IB (4 × 2 = 8) ──────────────────────────────────────────────
  Pretrained init + instance-weighted MSE
  L = Σ wᵢ(ŷᵢ - yᵢ)² / Σwᵢ,   weights: w_IB  or  w_IB × w_TSBW

  ── GROUP 2: PT+FB (4 × 2 = 8) ──────────────────────────────────────────────
  Pretrained init + feature alignment on encoder hidden states
  L = wMSE(src) + MSE(tgt) + λ·Align(h_s, h_t),   weights: 1  or  w_TSBW

  ── GROUP 3: IB+FB (16 × 2 = 32) ───────────────────────────────────────────
  Random init + instance-weighted alignment loss
  L = wMSE(src) + MSE(tgt) + λ·Align(h_s, h_t),   weights: w_IB  or  w_IB × w_TSBW

  ── GROUP 4: PT+IB+FB (16 × 2 = 32) ────────────────────────────────────────
  Pretrained init + instance-weighted alignment loss
  L = wMSE(src) + MSE(tgt) + λ·Align(h_s, h_t),   weights: w_IB  or  w_IB × w_TSBW

Complexity timing (per row):
  source_train_time_sec    = 0.0
  target_train_time_sec    = main training loop time
  weight_compute_time_sec  = IB + TSBW computation time
  alignment_train_time_sec = 0.0  (alignment is inside target training loop)
  model_complexity_time_sec = sum

Author: Azadshokrollahi
"""

import os, sys, logging, traceback, time, random
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
from matplotlib.patches import Patch

# ─── Centralized config ─────────────────────────────────────────────────────
from config import (
    SEED, WINDOW_SIZES, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    RESULTS_ROOT, PRETRAIN_DIR_NAME,
    load_rooms,
)

# ─── Reuse cleaned modules ──────────────────────────────────────────────────
from main_regression_pretrain import (
    compute_all_metrics, compute_occupied_metrics, compute_empty_metrics,
    compute_per_count_metrics, reset_seed, device,
)
from instance_based_methods import (
    load_room_raw, split_by_scenario, create_sequences,
    build_model, ensure_dir, slugify,
    compute_adaptation_weights,
    compute_tsbw_label_weights, normalize_and_clip,
    get_variant_features, build_room_cfg, count_parameters,
    WEIGHT_CLIP_MAX,
)
from parameter_based_methods import (
    find_pretrained_weights, load_pretrained_weights,
)
from feature_based_methods import (
    EncoderWrapper, DomainCritic, coral_loss, mmd_rbf_loss, ccsa_loss,
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
VARIANT          = 'homogeneous'    # this script is homogeneous-only

IB_METHODS       = ['KMM', 'ULSIF', 'RULSIF', 'IWC']
FB_METHODS       = ['DeepCORAL', 'DeepMMD', 'CCSA', 'WDGRL']
WEIGHT_SCHEMES   = ['none', 'tsbw']

# Alignment hyperparameters (match feature_based_methods.py)
LAMBDA_CORAL       = 1.0
LAMBDA_MMD         = 1.0
LAMBDA_CCSA        = 0.5
LAMBDA_WDGRL       = 1.0
CCSA_MARGIN        = 1.0
CCSA_THRESHOLD     = 1.0
WDGRL_GP_LAMBDA    = 10.0
WDGRL_CRITIC_ITERS = 5

OUTPUT_ROOT  = str(RESULTS_ROOT / "all_transfer_combinations")
PRETRAIN_DIR = str(RESULTS_ROOT / PRETRAIN_DIR_NAME)
ROOMS        = load_rooms()

TRANSFER_SCENARIOS = [
    ("Room_A", "Room_B"), ("Room_B", "Room_C"), ("Room_C", "Room_A"),
    ("Room_A", "Room_P"), ("Room_B", "Room_P"), ("Room_C", "Room_P"),
    ("Room_P", "Room_A"), ("Room_P", "Room_B"), ("Room_P", "Room_C"),
]


def get_approach(method_name):
    if method_name.startswith('PT+IB+FB'):
        return 'PT+IB+FB'
    if method_name.startswith('IB+FB'):
        return 'IB+FB'
    if method_name.startswith('PT+IB'):
        return 'PT+IB'
    if method_name.startswith('PT+FB'):
        return 'PT+FB'
    return 'unknown'


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


def _expand_weights_to_sequences(source_weights_raw, n_src_seq, ws):
    """Map per-row weights → per-sequence weights (use last-step weight)."""
    return np.array([
        source_weights_raw[min(i + ws - 1, len(source_weights_raw) - 1)]
        for i in range(n_src_seq)
    ], dtype=np.float32)


def _compute_source_weights(src_scaled, tgt_tr_scaled, src_y, tgt_tr_y,
                            ib_method, apply_tsbw):
    """
    Return (per-row source weights, weight_compute_time_sec).

    - ib_method=None,  apply_tsbw=False → None   (uniform)
    - ib_method=None,  apply_tsbw=True  → TSBW only
    - ib_method=X,     apply_tsbw=False → IB only
    - ib_method=X,     apply_tsbw=True  → IB × TSBW (already normalized inside)
    """
    if ib_method is None and not apply_tsbw:
        return None, 0.0

    t0 = time.perf_counter()

    if ib_method is not None:
        try:
            method_label = f"{ib_method}+TSBW" if apply_tsbw else ib_method
            w_ib, _, _ = compute_adaptation_weights(
                method_label, src_scaled, tgt_tr_scaled, src_y, tgt_tr_y,
                apply_tsbw=apply_tsbw)
            return w_ib, time.perf_counter() - t0
        except Exception as e:
            print(f"      IB ({ib_method}, tsbw={apply_tsbw}) failed: {e}; "
                  f"falling back to {'TSBW' if apply_tsbw else 'uniform'}")
            if not apply_tsbw:
                return None, time.perf_counter() - t0

    # TSBW-only branch (used by PT+FB + TSBW)
    w_label = compute_tsbw_label_weights(src_y, tgt_tr_y)
    w = normalize_and_clip(w_label, WEIGHT_CLIP_MAX)
    return w, time.perf_counter() - t0


def prepare_weighted_combined_data(src_tr_X, src_tr_y, tgt_tr_X, tgt_tr_y,
                                    tgt_va_X, tgt_va_y, tgt_te_X, tgt_te_y,
                                    ws, ib_method, apply_tsbw, bs=32):
    """For PT+IB: combined src+tgt loader with per-sample source weights."""
    combined = np.vstack([src_tr_X, tgt_tr_X])
    scaler = StandardScaler().fit(combined)
    src_scaled    = scaler.transform(src_tr_X)
    tgt_tr_scaled = scaler.transform(tgt_tr_X)

    source_weights, w_time = _compute_source_weights(
        src_scaled, tgt_tr_scaled, src_tr_y, tgt_tr_y, ib_method, apply_tsbw)

    X_src, y_src = create_sequences(src_scaled, src_tr_y, ws)
    X_tgt, y_tgt = create_sequences(tgt_tr_scaled, tgt_tr_y, ws)
    X_va,  y_va  = create_sequences(scaler.transform(tgt_va_X), tgt_va_y, ws)
    X_te,  y_te  = create_sequences(scaler.transform(tgt_te_X), tgt_te_y, ws)

    for name, arr in [("Src", y_src), ("Tgt", y_tgt), ("Val", y_va), ("Test", y_te)]:
        if len(arr) == 0:
            raise ValueError(f"{name} has 0 sequences ws={ws}")

    n_src, n_tgt = len(y_src), len(y_tgt)
    X_train = np.concatenate([X_src, X_tgt])
    y_train = np.concatenate([y_src, y_tgt])

    if source_weights is not None:
        src_sw = _expand_weights_to_sequences(source_weights, n_src, ws)
        seq_w  = np.maximum(np.concatenate([src_sw, np.ones(n_tgt)]), 1e-8)
        train_loader = _make_weighted_loader(X_train, y_train, seq_w, bs, True)
        has_weights = True
    else:
        train_loader = _make_loader(X_train, y_train, bs, True)
        has_weights = False

    return {
        'train_loader': train_loader,
        'val_loader':   _make_loader(X_va, y_va, bs, False),
        'test_loader':  _make_loader(X_te, y_te, bs, False),
        'input_size':   X_train.shape[2],
        'scaler': scaler,
        'has_weights': has_weights,
        'weight_compute_time_sec': w_time,
    }


def prepare_separate_loaders(src_tr_X, src_tr_y, tgt_tr_X, tgt_tr_y,
                              tgt_va_X, tgt_va_y, tgt_te_X, tgt_te_y,
                              ws, ib_method, apply_tsbw, bs=32):
    """For PT+FB / IB+FB / PT+IB+FB: separate src/tgt loaders, optional src weights."""
    combined = np.vstack([src_tr_X, tgt_tr_X])
    scaler = StandardScaler().fit(combined)
    src_scaled    = scaler.transform(src_tr_X)
    tgt_tr_scaled = scaler.transform(tgt_tr_X)

    source_weights, w_time = _compute_source_weights(
        src_scaled, tgt_tr_scaled, src_tr_y, tgt_tr_y, ib_method, apply_tsbw)

    X_src, y_src = create_sequences(src_scaled, src_tr_y, ws)
    X_tgt, y_tgt = create_sequences(tgt_tr_scaled, tgt_tr_y, ws)
    X_va,  y_va  = create_sequences(scaler.transform(tgt_va_X), tgt_va_y, ws)
    X_te,  y_te  = create_sequences(scaler.transform(tgt_te_X), tgt_te_y, ws)

    for name, arr in [("Src", y_src), ("Tgt", y_tgt), ("Val", y_va), ("Test", y_te)]:
        if len(arr) == 0:
            raise ValueError(f"{name} has 0 sequences ws={ws}")

    has_src_weights = False
    if source_weights is not None:
        src_sw = _expand_weights_to_sequences(source_weights, len(y_src), ws)
        src_sw = np.maximum(src_sw, 1e-8)
        src_loader = _make_weighted_loader(X_src, y_src, src_sw, bs, True)
        has_src_weights = True
    else:
        src_loader = _make_loader(X_src, y_src, bs, True)

    return {
        'src_loader':  src_loader,
        'tgt_loader':  _make_loader(X_tgt, y_tgt, bs, True),
        'val_loader':  _make_loader(X_va, y_va, bs, False),
        'test_loader': _make_loader(X_te, y_te, bs, False),
        'input_size':  X_src.shape[2],
        'scaler': scaler,
        'has_src_weights': has_src_weights,
        'weight_compute_time_sec': w_time,
    }


##############################################################################
#  TRAINING — GROUP 1 (PT+IB)
##############################################################################

def train_pt_ib(model, train_loader, val_loader, epochs, lr, has_weights):
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
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
            print(f"      Epoch [{ep + 1}/{epochs}] Val MAE: {vmae:.4f}")
        if vmae < best_val:
            best_val = vmae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.perf_counter() - t0
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"      Best Val MAE: {best_val:.4f}")
    print(f"      Target/train time: {train_time:.3f}s")
    return train_time


##############################################################################
#  TRAINING — GROUPS 2/3/4 (alignment)
##############################################################################

def _get_src_batch(src_iter, src_loader):
    try:
        return next(src_iter), src_iter
    except StopIteration:
        src_iter = iter(src_loader)
        return next(src_iter), src_iter


def _validate_encoder(encoder, val_loader):
    encoder.model.eval()
    vp, vt = [], []
    with torch.no_grad():
        for bx, by in val_loader:
            vp.extend(encoder.model(bx.to(device)).cpu().numpy())
            vt.extend(by.numpy())
    return mean_absolute_error(vt, vp)


def train_alignment(encoder, data, epochs, lr, fb_method, has_src_weights):
    """
    Unified alignment training for PT+FB, IB+FB, PT+IB+FB.
    Loss = wMSE(src) + MSE(tgt) + λ·Align(h_s, h_t)
    """
    criterion_pointwise = nn.MSELoss(reduction='none')
    src_loader = data['src_loader']
    tgt_loader = data['tgt_loader']
    val_loader = data['val_loader']

    critic, opt_critic = None, None
    if fb_method == 'WDGRL':
        h_dim = encoder.hidden_dim
        critic = DomainCritic(h_dim, 64).to(device)
        opt_critic = optim.Adam(critic.parameters(), lr=lr)

    optimizer = optim.Adam([p for p in encoder.parameters() if p.requires_grad], lr=lr)
    best_state, best_val = None, float('inf')

    def _grad_penalty(critic_net, h_s, h_t):
        bs = min(h_s.size(0), h_t.size(0))
        alpha = torch.rand(bs, 1, device=device)
        interp = (alpha * h_s[:bs] + (1 - alpha) * h_t[:bs]).requires_grad_(True)
        d_interp = critic_net(interp)
        grad = torch.autograd.grad(d_interp, interp, grad_outputs=torch.ones_like(d_interp),
                                   create_graph=True, retain_graph=True)[0]
        return ((grad.view(bs, -1).norm(2, dim=1) - 1) ** 2).mean()

    t0 = time.perf_counter()
    for ep in range(epochs):
        encoder.train()
        if critic is not None:
            critic.train()
        src_iter = iter(src_loader)

        for tgt_bx, tgt_by in tgt_loader:
            src_batch, src_iter = _get_src_batch(src_iter, src_loader)

            if has_src_weights:
                src_bx = src_batch[0].to(device)
                src_by = src_batch[1].to(device)
                src_bw = src_batch[2].to(device)
            else:
                src_bx = src_batch[0].to(device)
                src_by = src_batch[1].to(device)
                src_bw = None
            tgt_bx, tgt_by = tgt_bx.to(device), tgt_by.to(device)

            # WDGRL: train critic
            if fb_method == 'WDGRL':
                for _ in range(WDGRL_CRITIC_ITERS):
                    with torch.no_grad():
                        h_s, h_t = encoder.encode(src_bx), encoder.encode(tgt_bx)
                    opt_critic.zero_grad()
                    wd = critic(h_t).mean() - critic(h_s).mean()
                    (-wd + WDGRL_GP_LAMBDA *
                     _grad_penalty(critic, h_s.detach(), h_t.detach())).backward()
                    opt_critic.step()

            # Encoder step
            optimizer.zero_grad()
            h_s = encoder.encode(src_bx)
            h_t = encoder.encode(tgt_bx)
            pred_s = encoder.predict(h_s)
            pred_t = encoder.predict(h_t)

            src_losses = criterion_pointwise(pred_s, src_by)
            if src_bw is not None:
                loss_src = (src_bw * src_losses).sum() / src_bw.sum()
            else:
                loss_src = src_losses.mean()

            loss_tgt = criterion_pointwise(pred_t, tgt_by).mean()

            if fb_method == 'DeepCORAL':
                loss_align = LAMBDA_CORAL * coral_loss(h_s, h_t)
            elif fb_method == 'DeepMMD':
                loss_align = LAMBDA_MMD * mmd_rbf_loss(h_s, h_t)
            elif fb_method == 'CCSA':
                loss_align = LAMBDA_CCSA * ccsa_loss(h_s, h_t, src_by, tgt_by,
                                                      CCSA_MARGIN, CCSA_THRESHOLD)
            elif fb_method == 'WDGRL':
                loss_align = LAMBDA_WDGRL * (critic(h_t).mean() - critic(h_s).mean())
            else:
                raise ValueError(f"Unknown fb_method: {fb_method}")

            (loss_src + loss_tgt + loss_align).backward()
            optimizer.step()

        vmae = _validate_encoder(encoder, val_loader)
        if (ep + 1) % max(1, epochs // 6) == 0:
            print(f"      Epoch [{ep + 1}/{epochs}] Val MAE: {vmae:.4f}")
        if vmae < best_val:
            best_val = vmae
            best_state = {k: v.cpu().clone() for k, v in encoder.model.state_dict().items()}

    train_time = time.perf_counter() - t0
    if best_state:
        encoder.model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"      Best Val MAE: {best_val:.4f}")
    print(f"      Alignment train time: {train_time:.3f}s")
    return train_time


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


def save_outputs(model, metrics, tgt_df_full, tgt_split, ws, method_name,
                 model_type, out_dir):
    ensure_dir(out_dir)
    slug = slugify(method_name)
    torch.save(model.state_dict(),
               os.path.join(out_dir, f"weights_{slug}_ws{ws}.pth"))

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
#  RESULT ROW
##############################################################################

def build_result_row(scenario_name, src_id, tgt_id, model_type, ws,
                     mr, src_cfg, tgt_cfg, variant):
    m = mr['metrics']['all']
    method_name = mr['method']
    approach = get_approach(method_name)
    init = 'pretrained' if 'PT' in approach else 'random'

    row = {
        'scenario': scenario_name, 'source': src_id, 'target': tgt_id,
        'method': method_name, 'approach': approach,
        'weight_scheme': mr.get('weight_scheme', 'none'),
        'variant': variant, 'model': model_type, 'window_size': ws,
        'features': ', '.join(tgt_cfg['features']),
        'src_scenario': src_cfg['scenario'], 'tgt_scenario': tgt_cfg['scenario'],
        'initialization': init,

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
#  PLOTS
##############################################################################

def create_comparison_plots(results, ws, scenario_name, model_type, out_dir):
    results = [r for r in results if r is not None]
    if not results:
        return

    methods = [r['method'] for r in results]
    n = len(methods)

    def _color(m):
        if m.startswith('PT+IB+FB'):
            return '#9B59B6'
        elif m.startswith('IB+FB'):
            return '#E67E22'
        elif m.startswith('PT+IB'):
            return '#2980B9'
        elif m.startswith('PT+FB'):
            return '#E74C3C'
        return '#95A5A6'

    colors = [_color(m) for m in methods]

    metrics_list = [('mae', 'MAE'), ('rmse', 'RMSE'), ('r2', 'R²'),
                    ('medae', 'MedAE'), ('acc_pm1', 'Acc±1 (%)'), ('acc_pm2', 'Acc±2 (%)')]

    fig, axes = plt.subplots(2, 3, figsize=(36, 14))
    fig.suptitle(f'All Transfer Combinations ({n}): {scenario_name} | {model_type} | ws={ws}',
                 fontsize=14, fontweight='bold')
    for idx, (key, ylabel) in enumerate(metrics_list):
        ax = axes[idx // 3][idx % 3]
        vals = [r['metrics']['all'][key] for r in results]
        bars = ax.bar(range(n), vals, color=colors, edgecolor='black', alpha=0.85, width=0.8)
        ax.set_xticks(range(n))
        ax.set_xticklabels(methods, rotation=90, ha='center', fontsize=3.5)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

    legend_elements = [Patch(facecolor='#2980B9', label='PT+IB (8)'),
                       Patch(facecolor='#E74C3C', label='PT+FB (8)'),
                       Patch(facecolor='#E67E22', label='IB+FB (32)'),
                       Patch(facecolor='#9B59B6', label='PT+IB+FB (32)')]
    axes[0][0].legend(handles=legend_elements, loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'comparison_all_{model_type}_ws{ws}.png'),
                dpi=200, bbox_inches='tight')
    plt.close()

    # Best per group
    groups = {}
    for r in results:
        g = get_approach(r['method'])
        if g not in groups or r['metrics']['all']['mae'] < groups[g]['metrics']['all']['mae']:
            groups[g] = r

    if len(groups) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Best per Group: {scenario_name} | {model_type} | ws={ws}',
                     fontsize=13, fontweight='bold')
        group_order = ['PT+IB', 'PT+FB', 'IB+FB', 'PT+IB+FB']
        group_colors = ['#2980B9', '#E74C3C', '#E67E22', '#9B59B6']
        present = [(g, groups[g]) for g in group_order if g in groups]
        for idx, (key, ylabel) in enumerate([('mae', 'MAE'), ('r2', 'R²'),
                                              ('acc_pm1', 'Acc±1 (%)')]):
            labels = [f"{g}\n{r['method'][:25]}" for g, r in present]
            vals = [r['metrics']['all'][key] for _, r in present]
            gc = [group_colors[group_order.index(g)] for g, _ in present]
            bars = axes[idx].bar(range(len(present)), vals, color=gc,
                                  edgecolor='black', alpha=0.85)
            axes[idx].set_xticks(range(len(present)))
            axes[idx].set_xticklabels(labels, fontsize=6)
            axes[idx].set_ylabel(ylabel)
            axes[idx].set_title(ylabel, fontweight='bold')
            axes[idx].grid(True, axis='y', alpha=0.3)
            for b, v in zip(bars, vals):
                axes[idx].text(b.get_x() + b.get_width() / 2, b.get_height(), f'{v:.4f}',
                                ha='center', va='bottom', fontsize=7, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,
                                 f'comparison_best_per_group_{model_type}_ws{ws}.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()


##############################################################################
#  RUN ONE METHOD (returns method_result dict or None)
##############################################################################

def _finalize_result(method_name, weight_scheme, model, metrics, tgt_df_full,
                     tgt_split, ws, model_type, out_dir,
                     target_train_time, weight_time):
    save_outputs(model, metrics, tgt_df_full, tgt_split, ws,
                 method_name, model_type, out_dir)
    total_p, train_p, frozen_p = count_parameters(model)
    complexity = target_train_time + weight_time
    return {
        'method': method_name,
        'weight_scheme': weight_scheme,
        'metrics': metrics,
        'source_train_time_sec':     0.0,
        'target_train_time_sec':     target_train_time,
        'weight_compute_time_sec':   weight_time,
        'alignment_train_time_sec':  0.0,
        'model_complexity_time_sec': complexity,
        'total_params':     total_p,
        'trainable_params': train_p,
        'frozen_params':    frozen_p,
    }


##############################################################################
#  MAIN
##############################################################################

def main():
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    total_methods_per_config = (
        len(IB_METHODS) * len(WEIGHT_SCHEMES)                                  # PT+IB
        + len(FB_METHODS) * len(WEIGHT_SCHEMES)                                # PT+FB
        + len(IB_METHODS) * len(FB_METHODS) * len(WEIGHT_SCHEMES)              # IB+FB
        + len(IB_METHODS) * len(FB_METHODS) * len(WEIGHT_SCHEMES)              # PT+IB+FB
    )

    print("=" * 120)
    print("ALL TRANSFER COMBINATIONS — HOMOGENEOUS")
    print(f"  Methods per (scenario × model × ws): {total_methods_per_config} "
          f"({len(IB_METHODS)*len(WEIGHT_SCHEMES)} PT+IB + "
          f"{len(FB_METHODS)*len(WEIGHT_SCHEMES)} PT+FB + "
          f"{len(IB_METHODS)*len(FB_METHODS)*len(WEIGHT_SCHEMES)} IB+FB + "
          f"{len(IB_METHODS)*len(FB_METHODS)*len(WEIGHT_SCHEMES)} PT+IB+FB)")
    print(f"  Schemes: {WEIGHT_SCHEMES}")
    print(f"  Models: {MODEL_TYPES} | WS: {WINDOW_SIZES} | Epochs: {EPOCHS}")
    print(f"  Scenarios: {len(TRANSFER_SCENARIOS)}")
    print(f"  PRETRAIN_DIR: {PRETRAIN_DIR}")
    print(f"  OUTPUT_ROOT:  {OUTPUT_ROOT}")
    print(f"  User: Azadshokrollahi | {timestamp} UTC")
    print("=" * 120)

    ensure_dir(OUTPUT_ROOT)
    all_results = []

    total = len(TRANSFER_SCENARIOS) * len(MODEL_TYPES) * len(WINDOW_SIZES)
    counter = 0

    for sc_idx, (src_id, tgt_id) in enumerate(TRANSFER_SCENARIOS):
        scenario_name = f"{src_id} → {tgt_id}"
        src_cfg = build_room_cfg(src_id, VARIANT)
        tgt_cfg = build_room_cfg(tgt_id, VARIANT)

        print(f"\n{'=' * 120}")
        print(f"SCENARIO {sc_idx + 1}/{len(TRANSFER_SCENARIOS)}: {scenario_name}")
        print(f"  src features: {src_cfg['features']}")
        print(f"  tgt features: {tgt_cfg['features']}")
        print(f"{'=' * 120}")

        if not os.path.exists(src_cfg['path']) or not os.path.exists(tgt_cfg['path']):
            print("  ❌ Data file missing"); continue

        if len(src_cfg['features']) != len(tgt_cfg['features']):
            print(f"  ⏭ Skip: dim mismatch "
                  f"({len(src_cfg['features'])} vs {len(tgt_cfg['features'])})")
            continue

        src_X, src_y, _, _ = load_room_raw(
            src_cfg['path'], src_cfg['features'],
            src_cfg['name'], src_cfg['selected_rows'])
        tgt_X, tgt_y, tgt_df_full, _ = load_room_raw(
            tgt_cfg['path'], tgt_cfg['features'],
            tgt_cfg['name'], tgt_cfg['selected_rows'])

        # Source: split with its own scenario, use only TRAIN
        src_tr_X, src_tr_y, _, _, _, _, _ = split_by_scenario(
            src_X, src_y, src_cfg['scenario'])

        # Target: full split (train/val/test) with its own scenario
        tgt_tr_X, tgt_tr_y, tgt_va_X, tgt_va_y, tgt_te_X, tgt_te_y, tgt_split = \
            split_by_scenario(tgt_X, tgt_y, tgt_cfg['scenario'])

        for model_type in MODEL_TYPES:
            for ws in WINDOW_SIZES:
                counter += 1
                print(f"\n  ══ [{counter}/{total}] {model_type} | ws={ws} ══")

                sc_folder = ensure_dir(os.path.join(
                    OUTPUT_ROOT, VARIANT,
                    f"{src_id}_to_{tgt_id}", f"{model_type}_ws{ws}"))

                log_file = os.path.join(sc_folder, "transfer_log.txt")
                for h in logging.root.handlers[:]:
                    logging.root.removeHandler(h)
                logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler(log_file, mode='a'),
                              logging.StreamHandler(sys.stdout)],
                    force=True)

                weights_path = find_pretrained_weights(
                    PRETRAIN_DIR, src_id, VARIANT, model_type, ws)
                has_pretrained = weights_path is not None
                if has_pretrained:
                    print(f"    Pretrained: {weights_path}")
                else:
                    print(f"    ⚠ No pretrained weights — skipping PT+IB, PT+FB, PT+IB+FB")

                method_results = []
                method_counter = 0
                n_expected = total_methods_per_config if has_pretrained \
                             else len(IB_METHODS) * len(FB_METHODS) * len(WEIGHT_SCHEMES)

                # ═══════════════════════════════════════════════════════════
                # GROUP 1: PT+IB (4 IB × 2 schemes = 8)
                # ═══════════════════════════════════════════════════════════
                if has_pretrained:
                    for ib_method in IB_METHODS:
                        for scheme in WEIGHT_SCHEMES:
                            apply_tsbw = (scheme == 'tsbw')
                            method_name = f"PT+IB: {ib_method}"
                            if apply_tsbw:
                                method_name += "+TSBW"
                            method_counter += 1
                            print(f"\n    ── [{method_counter}/{n_expected}] {method_name} ──")
                            try:
                                reset_all_seeds(SEED)
                                data = prepare_weighted_combined_data(
                                    src_tr_X, src_tr_y, tgt_tr_X, tgt_tr_y,
                                    tgt_va_X, tgt_va_y, tgt_te_X, tgt_te_y,
                                    ws, ib_method, apply_tsbw, BATCH_SIZE)

                                model = build_model(model_type, data['input_size'])
                                model = load_pretrained_weights(
                                    model, weights_path, model_type,
                                    data['input_size'], data['input_size'])
                                for p in model.parameters():
                                    p.requires_grad = True

                                t_train = train_pt_ib(
                                    model, data['train_loader'], data['val_loader'],
                                    EPOCHS, LEARNING_RATE, data['has_weights'])
                                metrics = evaluate_full(model, data['test_loader'], method_name)

                                joblib.dump(data['scaler'],
                                            os.path.join(sc_folder,
                                                         f"scaler_{slugify(method_name)}_ws{ws}.pkl"))
                                method_results.append(_finalize_result(
                                    method_name, scheme, model, metrics, tgt_df_full,
                                    tgt_split, ws, model_type, sc_folder,
                                    t_train, data['weight_compute_time_sec']))
                            except Exception as e:
                                print(f"      ❌ FAILED: {e}")
                                logging.error(f"{method_name}: {e}")
                                traceback.print_exc()

                # ═══════════════════════════════════════════════════════════
                # GROUP 2: PT+FB (4 FB × 2 schemes = 8)
                # ═══════════════════════════════════════════════════════════
                if has_pretrained:
                    for fb_method in FB_METHODS:
                        for scheme in WEIGHT_SCHEMES:
                            apply_tsbw = (scheme == 'tsbw')
                            method_name = f"PT+FB: {fb_method}"
                            if apply_tsbw:
                                method_name += "+TSBW"
                            method_counter += 1
                            print(f"\n    ── [{method_counter}/{n_expected}] {method_name} ──")
                            try:
                                reset_all_seeds(SEED)
                                data = prepare_separate_loaders(
                                    src_tr_X, src_tr_y, tgt_tr_X, tgt_tr_y,
                                    tgt_va_X, tgt_va_y, tgt_te_X, tgt_te_y,
                                    ws, ib_method=None, apply_tsbw=apply_tsbw,
                                    bs=BATCH_SIZE)

                                model = build_model(model_type, data['input_size'])
                                model = load_pretrained_weights(
                                    model, weights_path, model_type,
                                    data['input_size'], data['input_size'])
                                for p in model.parameters():
                                    p.requires_grad = True
                                encoder = EncoderWrapper(model).to(device)

                                t_train = train_alignment(
                                    encoder, data, EPOCHS, LEARNING_RATE, fb_method,
                                    data['has_src_weights'])
                                metrics = evaluate_full(encoder.model,
                                                        data['test_loader'], method_name)

                                joblib.dump(data['scaler'],
                                            os.path.join(sc_folder,
                                                         f"scaler_{slugify(method_name)}_ws{ws}.pkl"))
                                method_results.append(_finalize_result(
                                    method_name, scheme, encoder.model, metrics,
                                    tgt_df_full, tgt_split, ws, model_type, sc_folder,
                                    t_train, data['weight_compute_time_sec']))
                            except Exception as e:
                                print(f"      ❌ FAILED: {e}")
                                logging.error(f"{method_name}: {e}")
                                traceback.print_exc()

                # ═══════════════════════════════════════════════════════════
                # GROUP 3: IB+FB (16 × 2 schemes = 32) — random init
                # ═══════════════════════════════════════════════════════════
                for ib_method in IB_METHODS:
                    for fb_method in FB_METHODS:
                        for scheme in WEIGHT_SCHEMES:
                            apply_tsbw = (scheme == 'tsbw')
                            method_name = f"IB+FB: {ib_method}+{fb_method}"
                            if apply_tsbw:
                                method_name += "+TSBW"
                            method_counter += 1
                            print(f"\n    ── [{method_counter}/{n_expected}] {method_name} ──")
                            try:
                                reset_all_seeds(SEED)
                                data = prepare_separate_loaders(
                                    src_tr_X, src_tr_y, tgt_tr_X, tgt_tr_y,
                                    tgt_va_X, tgt_va_y, tgt_te_X, tgt_te_y,
                                    ws, ib_method=ib_method, apply_tsbw=apply_tsbw,
                                    bs=BATCH_SIZE)

                                model = build_model(model_type, data['input_size'])
                                # random init (no pretrained load)
                                for p in model.parameters():
                                    p.requires_grad = True
                                encoder = EncoderWrapper(model).to(device)

                                t_train = train_alignment(
                                    encoder, data, EPOCHS, LEARNING_RATE, fb_method,
                                    data['has_src_weights'])
                                metrics = evaluate_full(encoder.model,
                                                        data['test_loader'], method_name)

                                joblib.dump(data['scaler'],
                                            os.path.join(sc_folder,
                                                         f"scaler_{slugify(method_name)}_ws{ws}.pkl"))
                                method_results.append(_finalize_result(
                                    method_name, scheme, encoder.model, metrics,
                                    tgt_df_full, tgt_split, ws, model_type, sc_folder,
                                    t_train, data['weight_compute_time_sec']))
                            except Exception as e:
                                print(f"      ❌ FAILED: {e}")
                                logging.error(f"{method_name}: {e}")
                                traceback.print_exc()

                # ═══════════════════════════════════════════════════════════
                # GROUP 4: PT+IB+FB (16 × 2 schemes = 32) — pretrained init
                # ═══════════════════════════════════════════════════════════
                if has_pretrained:
                    for ib_method in IB_METHODS:
                        for fb_method in FB_METHODS:
                            for scheme in WEIGHT_SCHEMES:
                                apply_tsbw = (scheme == 'tsbw')
                                method_name = f"PT+IB+FB: {ib_method}+{fb_method}"
                                if apply_tsbw:
                                    method_name += "+TSBW"
                                method_counter += 1
                                print(f"\n    ── [{method_counter}/{n_expected}] {method_name} ──")
                                try:
                                    reset_all_seeds(SEED)
                                    data = prepare_separate_loaders(
                                        src_tr_X, src_tr_y, tgt_tr_X, tgt_tr_y,
                                        tgt_va_X, tgt_va_y, tgt_te_X, tgt_te_y,
                                        ws, ib_method=ib_method, apply_tsbw=apply_tsbw,
                                        bs=BATCH_SIZE)

                                    model = build_model(model_type, data['input_size'])
                                    model = load_pretrained_weights(
                                        model, weights_path, model_type,
                                        data['input_size'], data['input_size'])
                                    for p in model.parameters():
                                        p.requires_grad = True
                                    encoder = EncoderWrapper(model).to(device)

                                    t_train = train_alignment(
                                        encoder, data, EPOCHS, LEARNING_RATE, fb_method,
                                        data['has_src_weights'])
                                    metrics = evaluate_full(encoder.model,
                                                            data['test_loader'], method_name)

                                    joblib.dump(data['scaler'],
                                                os.path.join(sc_folder,
                                                             f"scaler_{slugify(method_name)}_ws{ws}.pkl"))
                                    method_results.append(_finalize_result(
                                        method_name, scheme, encoder.model, metrics,
                                        tgt_df_full, tgt_split, ws, model_type, sc_folder,
                                        t_train, data['weight_compute_time_sec']))
                                except Exception as e:
                                    print(f"      ❌ FAILED: {e}")
                                    logging.error(f"{method_name}: {e}")
                                    traceback.print_exc()

                # ═══════════════════════════════════════════════════════════
                # RESULTS TABLE + PLOTS + CSV
                # ═══════════════════════════════════════════════════════════
                method_results = [r for r in method_results if r is not None]
                create_comparison_plots(method_results, ws, scenario_name,
                                        model_type, sc_folder)

                rows = []
                for r in method_results:
                    row = build_result_row(scenario_name, src_id, tgt_id, model_type,
                                           ws, r, src_cfg, tgt_cfg, VARIANT)
                    rows.append(row)
                    all_results.append(row)

                if rows:
                    pd.DataFrame(rows).to_csv(
                        os.path.join(sc_folder, f"metrics_{model_type}_ws{ws}.csv"),
                        index=False)

                print(f"\n    {'#':<4} {'Method':<40} {'Scheme':<6} {'MAE':<8} "
                      f"{'RMSE':<8} {'R²':<8} {'Acc±1':<8} {'Complex(s)':<11}")
                print("    " + "-" * 110)
                for i, r in enumerate(method_results):
                    ma = r['metrics']['all']
                    print(f"    {i:<4} {r['method']:<40} {r['weight_scheme']:<6} "
                          f"{ma['mae']:<8.4f} {ma['rmse']:<8.4f} {ma['r2']:<8.4f} "
                          f"{ma['acc_pm1']:<8.2f} "
                          f"{r['model_complexity_time_sec']:<11.3f}")

                print(f"\n  ✓ {scenario_name} | {model_type} | ws={ws} — "
                      f"{len(method_results)}/{n_expected}")

    # ═══════════════════════════════════════════════════════════════════════
    # MASTER CSV + GLOBAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 120)
    print("FINAL RESULTS — ALL TRANSFER COMBINATIONS")
    print("=" * 120)

    if not all_results:
        print("\n  No results.")
        return

    COLUMN_ORDER = [
        'scenario', 'source', 'target', 'method', 'approach', 'weight_scheme',
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

    df = pd.DataFrame(all_results)
    df = df[[c for c in COLUMN_ORDER if c in df.columns]]
    master_csv = os.path.join(OUTPUT_ROOT, f"all_transfer_combinations_{VARIANT}.csv")
    df.to_csv(master_csv, index=False)
    print(f"\nMaster CSV: {master_csv}")
    print(f"Total rows: {len(df)}")

    # Best per scenario × model
    print(f"\n{'=' * 120}\nBEST METHOD PER SCENARIO × MODEL (by MAE)\n{'=' * 120}")
    print(f"{'Scenario':<25} {'Model':<12} {'Best':<40} {'Approach':<12} "
          f"{'Scheme':<6} {'MAE':<8} {'Acc±1':<8}")
    print("-" * 120)
    for (sc, mdl), grp in df.groupby(['scenario', 'model']):
        best = grp.loc[grp['test_mae'].idxmin()]
        print(f"{sc:<25} {mdl:<12} {best['method']:<40} {best['approach']:<12} "
              f"{best['weight_scheme']:<6} "
              f"{best['test_mae']:<8.4f} {best['test_acc_pm1']:<8.2f}")

    # Best per approach group
    print(f"\n{'=' * 120}\nBEST METHOD PER APPROACH × SCENARIO × MODEL\n{'=' * 120}")
    for approach in ['PT+IB', 'PT+FB', 'IB+FB', 'PT+IB+FB']:
        adf = df[df['approach'] == approach]
        if len(adf) == 0:
            continue
        print(f"\n  ── {approach} ──")
        for (sc, mdl), grp in adf.groupby(['scenario', 'model']):
            best = grp.loc[grp['test_mae'].idxmin()]
            print(f"    {sc:<25} {mdl:<12} {best['method']:<40} "
                  f"scheme={best['weight_scheme']:<6} "
                  f"MAE={best['test_mae']:.4f} Acc±1={best['test_acc_pm1']:.2f}")

    # TSBW effect per approach
    print(f"\n{'=' * 120}\nTSBW EFFECT PER APPROACH (avg MAE, avg complexity)\n{'=' * 120}")
    print(f"{'Approach':<12} {'Scheme':<6} {'avg MAE':<10} {'avg R²':<10} "
          f"{'avg Acc±1':<10} {'avg Complex(s)':<14} {'N':<5}")
    print("-" * 80)
    for approach in ['PT+IB', 'PT+FB', 'IB+FB', 'PT+IB+FB']:
        for scheme in WEIGHT_SCHEMES:
            sub = df[(df['approach'] == approach) & (df['weight_scheme'] == scheme)]
            if len(sub) == 0:
                continue
            print(f"{approach:<12} {scheme:<6} "
                  f"{sub['test_mae'].mean():<10.4f} "
                  f"{sub['test_r2'].mean():<10.4f} "
                  f"{sub['test_acc_pm1'].mean():<10.2f} "
                  f"{sub['model_complexity_time_sec'].mean():<14.3f} "
                  f"{len(sub):<5}")

    # Win counts by approach
    print(f"\n{'=' * 60}\nAPPROACH WIN COUNT\n{'=' * 60}")
    approach_wins = {}
    for (sc, mdl, wv), grp in df.groupby(['scenario', 'model', 'window_size']):
        best = grp.loc[grp['test_mae'].idxmin()]
        a = best['approach']
        approach_wins[a] = approach_wins.get(a, 0) + 1
    tw = sum(approach_wins.values())
    print(f"\n{'Approach':<15} {'Wins':<8} {'Pct':<8}")
    print("-" * 35)
    for a in sorted(approach_wins, key=approach_wins.get, reverse=True):
        print(f"{a:<15} {approach_wins[a]:<8} {approach_wins[a] / tw * 100:<7.1f}%")

    # Win counts by scheme
    print(f"\n{'=' * 60}\nWEIGHT SCHEME WIN COUNT\n{'=' * 60}")
    scheme_wins = {}
    for (sc, mdl, wv), grp in df.groupby(['scenario', 'model', 'window_size']):
        best = grp.loc[grp['test_mae'].idxmin()]
        s = best['weight_scheme']
        scheme_wins[s] = scheme_wins.get(s, 0) + 1
    print(f"\n{'Scheme':<10} {'Wins':<8} {'Pct':<8}")
    print("-" * 30)
    for s in sorted(scheme_wins, key=scheme_wins.get, reverse=True):
        print(f"{s:<10} {scheme_wins[s]:<8} {scheme_wins[s] / tw * 100:<7.1f}%")

    # Global summary file
    sp = os.path.join(OUTPUT_ROOT, "GLOBAL_SUMMARY.txt")
    with open(sp, "w") as f:
        f.write("=" * 120 + "\n")
        f.write("ALL TRANSFER COMBINATIONS: GLOBAL SUMMARY\n")
        f.write(f"  PT+IB ({len(IB_METHODS)}×{len(WEIGHT_SCHEMES)}) + "
                f"PT+FB ({len(FB_METHODS)}×{len(WEIGHT_SCHEMES)}) + "
                f"IB+FB ({len(IB_METHODS)*len(FB_METHODS)}×{len(WEIGHT_SCHEMES)}) + "
                f"PT+IB+FB ({len(IB_METHODS)*len(FB_METHODS)}×{len(WEIGHT_SCHEMES)})"
                f" = {total_methods_per_config} methods\n")
        f.write("=" * 120 + "\n")
        f.write(f"Timestamp: {timestamp} UTC\nUser: Azadshokrollahi\n\n")

        f.write("APPROACH WIN COUNTS:\n")
        for a in sorted(approach_wins, key=approach_wins.get, reverse=True):
            f.write(f"  {a}: {approach_wins[a]} ({approach_wins[a] / tw * 100:.1f}%)\n")

        f.write("\nWEIGHT SCHEME WIN COUNTS:\n")
        for s in sorted(scheme_wins, key=scheme_wins.get, reverse=True):
            f.write(f"  {s}: {scheme_wins[s]} ({scheme_wins[s] / tw * 100:.1f}%)\n")

        f.write("\nAVERAGE METRICS PER APPROACH × SCHEME:\n")
        for approach in ['PT+IB', 'PT+FB', 'IB+FB', 'PT+IB+FB']:
            for scheme in WEIGHT_SCHEMES:
                sub = df[(df['approach'] == approach) & (df['weight_scheme'] == scheme)]
                if len(sub) > 0:
                    f.write(f"  {approach} ({scheme}): MAE={sub['test_mae'].mean():.4f}, "
                            f"R²={sub['test_r2'].mean():.4f}, "
                            f"Acc±1={sub['test_acc_pm1'].mean():.2f}%, "
                            f"Complex={sub['model_complexity_time_sec'].mean():.3f}s\n")

        f.write("\nBEST PER SCENARIO × MODEL:\n")
        for (sc, mdl), grp in df.groupby(['scenario', 'model']):
            best = grp.loc[grp['test_mae'].idxmin()]
            f.write(f"  {sc} | {mdl} → {best['method']} "
                    f"(scheme={best['weight_scheme']}) MAE={best['test_mae']:.4f}\n")
        f.write("=" * 120 + "\n")
    print(f"\nGlobal summary: {sp}")

    print("\n" + "=" * 120)
    print("✓ ALL TRANSFER COMBINATIONS COMPLETED!")
    print(f"User: Azadshokrollahi | "
          f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 120)


if __name__ == "__main__":
    main()