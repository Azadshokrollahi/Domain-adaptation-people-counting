"""
Progressive Data Efficiency Study — Feature Extraction & Full Fine-Tuning

How much target data do you need to get most of the transfer benefit?

For each transfer scenario × model × ws:
  • Run BOTH Feature Extraction (frozen backbone + trainable head)
                Full Fine-Tuning (pretrained init + everything trainable)
  • Progressively increase target training size: 1, 5, 10, 15, 20, ..., max
  • Include zero-shot baseline (n=0)
  • Compare against the no-target-data reference

Reads:
  • configs/rooms.yaml   → room paths, scenarios, feature_variants
  • src/config.py        → seeds, epochs, batch size, LR, paths

Outputs:
  • results/progressive_data_efficiency/all_progressive_data_efficiency_homogeneous.csv
  • per-scenario folders with weights / predictions / summaries / plots

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
    build_room_cfg, count_parameters,
)
from parameter_based_methods import (
    find_pretrained_weights, load_pretrained_weights,
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

MODEL_TYPES = ['RNN', 'LSTM', 'Transformer']
EPOCHS      = NUM_EPOCHS
VARIANT     = 'homogeneous'

METHODS = ['Feature-Extraction', 'Full-Fine-Tuning']

OUTPUT_ROOT  = str(RESULTS_ROOT / "progressive_data_efficiency")
PRETRAIN_DIR = str(RESULTS_ROOT / PRETRAIN_DIR_NAME)
ROOMS        = load_rooms()

TRANSFER_SCENARIOS = [
    ("Room_A", "Room_B"), ("Room_B", "Room_C"), ("Room_C", "Room_A"),
    ("Room_A", "Room_P"), ("Room_B", "Room_P"), ("Room_C", "Room_P"),
    ("Room_P", "Room_A"), ("Room_P", "Room_B"), ("Room_P", "Room_C"),
]


def generate_train_sizes(max_train_samples, ws):
    """
    Generate training sizes: 1, 5, 10, 15, 20, 25, ..., max (step=5).
    Skips sizes < ws (would produce 0 sequences).
    """
    sizes = [1]
    s = 5
    while s <= max_train_samples:
        sizes.append(s)
        s += 5
    if sizes[-1] != max_train_samples:
        sizes.append(max_train_samples)
    sizes = [s for s in sizes if s >= ws]
    if not sizes:
        sizes = [max_train_samples]
    return sizes


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


def prepare_target_full(tgt_X, tgt_y, scenario, ws, bs=32):
    """Split target with its scenario; fit scaler on FULL target train."""
    tr_X, tr_y, va_X, va_y, te_X, te_y, split = split_by_scenario(tgt_X, tgt_y, scenario)

    scaler = StandardScaler().fit(tr_X)

    va_seq_X, va_seq_y = create_sequences(scaler.transform(va_X), va_y, ws)
    te_seq_X, te_seq_y = create_sequences(scaler.transform(te_X), te_y, ws)

    if len(va_seq_y) == 0 or len(te_seq_y) == 0:
        raise ValueError(f"Val or Test has 0 sequences with ws={ws}")

    val_ld  = _make_loader(va_seq_X, va_seq_y, bs, False)
    test_ld = _make_loader(te_seq_X, te_seq_y, bs, False)

    return {
        'tr_X': tr_X, 'tr_y': tr_y,
        'val_loader': val_ld, 'test_loader': test_ld,
        'scaler': scaler, 'split': split,
        'input_size': te_seq_X.shape[2],
        'n_tr_total': len(tr_X),
    }


def make_train_subset_loader(scaler, tr_X, tr_y, n_train, ws, bs=32):
    """Build train loader from first n_train rows of target train data."""
    sub_X = scaler.transform(tr_X[:n_train])
    sub_y = tr_y[:n_train]
    seq_X, seq_y = create_sequences(sub_X, sub_y, ws)
    if len(seq_y) == 0:
        raise ValueError(f"Subset of {n_train} rows produced 0 sequences (ws={ws})")
    actual_bs = min(bs, max(4, len(seq_y) // 4))
    return _make_loader(seq_X, seq_y, actual_bs, True), len(seq_y)


##############################################################################
#  TRAINING & EVALUATION
##############################################################################

def train_model_with_method(model, method, train_loader, val_loader, epochs, lr):
    """Configure trainable params per method, then train with best-val tracking."""
    if method == 'Feature-Extraction':
        for p in model.parameters():
            p.requires_grad = False
        for p in model.fc.parameters():
            p.requires_grad = True
    elif method == 'Full-Fine-Tuning':
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown method: {method}")

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    best_state, best_val = None, float('inf')

    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                vp.extend(model(bx.to(device)).cpu().numpy())
                vt.extend(by.numpy())
        vmae = mean_absolute_error(vt, vp)
        if vmae < best_val:
            best_val = vmae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.perf_counter() - t0
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return train_time, best_val


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
    return {'all': m_all, 'occupied': m_occ, 'empty': m_empty, 'per_count': m_pc,
            'predictions': preds, 'targets': trues}


##############################################################################
#  RESULT ROW BUILDER (matches schema of other master CSVs)
##############################################################################

def build_result_row(scenario_name, src_id, tgt_id, model_type, ws,
                     method, train_size, train_pct, zs_mae,
                     metrics, train_time, params, src_cfg, tgt_cfg):
    m = metrics['all']
    mae = m['mae']
    improvement_pct = ((zs_mae - mae) / zs_mae * 100.0) if zs_mae > 0 else 0.0
    total_p, train_p, frozen_p = params

    row = {
        'scenario': scenario_name, 'source': src_id, 'target': tgt_id,
        'method': method, 'weight_scheme': 'none',
        'variant': VARIANT, 'model': model_type, 'window_size': ws,
        'features': ', '.join(tgt_cfg['features']),
        'src_scenario': src_cfg['scenario'], 'tgt_scenario': tgt_cfg['scenario'],
        'initialization': 'pretrained',

        'train_size': int(train_size),
        'train_percentage': float(train_pct),
        'zero_shot_mae': float(zs_mae),
        'improvement_pct': float(improvement_pct),

        'source_train_time_sec':     0.0,
        'target_train_time_sec':     float(train_time),
        'weight_compute_time_sec':   0.0,
        'alignment_train_time_sec':  0.0,
        'model_complexity_time_sec': float(train_time),

        'total_params':     int(total_p),
        'trainable_params': int(train_p),
        'frozen_params':    int(frozen_p),

        'test_mae': m['mae'], 'test_rmse': m['rmse'], 'test_r2': m['r2'],
        'test_medae': m['medae'], 'test_acc_pm1': m['acc_pm1'],
        'test_acc_pm2': m['acc_pm2'], 'test_samples': m['sample_count'],
        'occ_mae': np.nan, 'occ_rmse': np.nan, 'occ_r2': np.nan, 'occ_medae': np.nan,
        'occ_acc_pm1': np.nan, 'occ_acc_pm2': np.nan, 'occ_exact_pct': np.nan,
        'occ_samples': np.nan,
        'empty_mae': np.nan, 'empty_acc_pm1': np.nan, 'empty_samples': np.nan,
    }
    if metrics['occupied']:
        mo = metrics['occupied']
        row.update({'occ_mae': mo['mae'], 'occ_rmse': mo['rmse'], 'occ_r2': mo['r2'],
                    'occ_medae': mo['medae'], 'occ_acc_pm1': mo['acc_pm1'],
                    'occ_acc_pm2': mo['acc_pm2'], 'occ_exact_pct': mo['exact_match_pct'],
                    'occ_samples': mo['sample_count']})
    if metrics['empty']:
        me = metrics['empty']
        row.update({'empty_mae': me['mae'], 'empty_acc_pm1': me['acc_pm1'],
                    'empty_samples': me['sample_count']})
    return row


##############################################################################
#  PLOTTING — wide linear x-axis, all train sizes as ticks, annotated
##############################################################################

METHOD_STYLES = {
    'Feature-Extraction': {'color': '#3498DB', 'marker': 'o', 'ls': '-',  'label': 'Feature-Extraction'},
    'Full-Fine-Tuning':   {'color': '#E74C3C', 'marker': 's', 'ls': '-',  'label': 'Full-Fine-Tuning'},
}


def _make_x_labels(sizes):
    if len(sizes) <= 50:
        return [str(s) for s in sizes]
    labels = []
    for s in sizes:
        if s == 1 or s == sizes[-1] or s % 50 == 0:
            labels.append(str(s))
        elif s <= 20 or s % 25 == 0:
            labels.append(str(s))
        else:
            labels.append('')
    return labels


def _setup_linear_xaxis(ax, sizes):
    ax.set_xticks(sizes)
    ax.set_xticklabels(_make_x_labels(sizes), rotation=90, fontsize=7)
    ax.set_xlim([-5, sizes[-1] + 10])


def _fig_width(n_points):
    return max(24, n_points * 0.2)


def _annotate_key_points(ax, sizes, values, color):
    if len(values) == 0:
        return
    key_indices = set()
    key_indices.add(0)
    key_indices.add(len(values) - 1)
    key_indices.add(int(np.argmin(values)))
    key_indices.add(int(np.argmax(values)))
    for q in [0.25, 0.5, 0.75]:
        key_indices.add(int(q * (len(values) - 1)))

    for i in key_indices:
        if 0 <= i < len(values):
            fmt = f'{values[i]:.4f}' if abs(values[i]) < 100 else f'{values[i]:.1f}'
            ax.annotate(fmt, (sizes[i], values[i]),
                        textcoords="offset points", xytext=(0, 12),
                        ha='center', fontsize=7, fontweight='bold',
                        color=color,
                        bbox=dict(boxstyle='round,pad=0.25',
                                  facecolor='yellow', alpha=0.4),
                        zorder=10)


def create_two_method_metric_plot(df_by_method, ylabel, title, save_path,
                                  scenario_name, model_type, ws,
                                  y_hlines=None, y_lim=None, annotate=True):
    """Wide plot with linear x-axis showing two methods overlaid."""
    longest_sizes = max((s for s, v in df_by_method.values()), key=len, default=np.array([]))
    if len(longest_sizes) == 0:
        return

    fig, ax = plt.subplots(figsize=(_fig_width(len(longest_sizes)), 7))

    for method, (sizes, values) in df_by_method.items():
        style = METHOD_STYLES.get(method, {})
        ax.plot(sizes, values,
                marker=style.get('marker', 'o'),
                linestyle=style.get('ls', '-'),
                lw=2, ms=5, color=style.get('color', 'gray'),
                label=style.get('label', method), zorder=5)
        if annotate:
            _annotate_key_points(ax, sizes, values, style.get('color', 'gray'))

    _setup_linear_xaxis(ax, longest_sizes)

    ax.set_xlabel('Training Data Size', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title(f'{title}\n{scenario_name} | {model_type} | ws={ws}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')

    if y_hlines:
        for y_val, label, hcolor, style in y_hlines:
            ax.axhline(y=y_val, color=hcolor, linestyle=style, alpha=0.7,
                       lw=2, label=label)

    if y_lim:
        ax.set_ylim(y_lim)

    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Plot saved: {save_path}")


def create_accuracy_plot(df_by_method_w1, df_by_method_w2,
                          save_path, scenario_name, model_type, ws):
    """Wide accuracy plot: ±1 (solid) + ±2 (dashed), per method."""
    longest_sizes = max((s for s, v in df_by_method_w1.values()),
                        key=len, default=np.array([]))
    if len(longest_sizes) == 0:
        return

    fig, ax = plt.subplots(figsize=(_fig_width(len(longest_sizes)), 7))

    for method, (sizes, w1) in df_by_method_w1.items():
        style = METHOD_STYLES.get(method, {})
        ax.plot(sizes, w1, 'o-', lw=2, ms=5,
                color=style.get('color', 'gray'),
                label=f"{method} ±1", zorder=5)
        for idx in [0, len(w1) - 1]:
            ax.annotate(f'{w1[idx]:.1f}%', (sizes[idx], w1[idx]),
                        textcoords="offset points", xytext=(0, 12),
                        ha='center', fontsize=7, fontweight='bold',
                        color=style.get('color', 'gray'),
                        bbox=dict(boxstyle='round,pad=0.25',
                                  facecolor='yellow', alpha=0.4))

    for method, (sizes, w2) in df_by_method_w2.items():
        style = METHOD_STYLES.get(method, {})
        ax.plot(sizes, w2, 's--', lw=1.5, ms=4, alpha=0.6,
                color=style.get('color', 'gray'),
                label=f"{method} ±2", zorder=4)

    _setup_linear_xaxis(ax, longest_sizes)
    ax.set_xlabel('Training Data Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'Prediction Accuracy\n{scenario_name} | {model_type} | ws={ws}',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='lower right', ncol=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Plot saved: {save_path}")


def create_learning_curves(df, out_dir, model_type, ws, scenario_name, zs_mae):
    """Create one wide image per metric with both methods overlaid."""
    df_trained = df[df['train_size'] > 0].copy()
    if len(df_trained) == 0:
        return

    def _series(metric):
        out = {}
        for method in METHODS:
            sub = df_trained[df_trained['method'] == method].sort_values('train_size')
            if len(sub) > 0:
                out[method] = (sub['train_size'].values, sub[metric].values)
        return out

    # 1. MAE
    create_two_method_metric_plot(
        _series('test_mae'), 'MAE', 'Mean Absolute Error',
        os.path.join(out_dir, f'metric_MAE_{model_type}_ws{ws}.png'),
        scenario_name, model_type, ws,
        y_hlines=[(zs_mae, f'Zero-Shot (n=0): {zs_mae:.4f}', 'gray', '--')]
                  if not np.isnan(zs_mae) else None)

    # 2. RMSE
    create_two_method_metric_plot(
        _series('test_rmse'), 'RMSE', 'Root Mean Squared Error',
        os.path.join(out_dir, f'metric_RMSE_{model_type}_ws{ws}.png'),
        scenario_name, model_type, ws)

    # 3. R²
    r2_series = _series('test_r2')
    all_r2 = (np.concatenate([v for _, v in r2_series.values()])
              if r2_series else np.array([0]))
    create_two_method_metric_plot(
        r2_series, 'R²', 'R-Squared Score',
        os.path.join(out_dir, f'metric_R2_{model_type}_ws{ws}.png'),
        scenario_name, model_type, ws,
        y_lim=[max(-0.1, all_r2.min() - 0.05), 1.0])

    # 4. Accuracy (±1 + ±2, both methods)
    create_accuracy_plot(
        _series('test_acc_pm1'), _series('test_acc_pm2'),
        os.path.join(out_dir, f'metric_Accuracy_{model_type}_ws{ws}.png'),
        scenario_name, model_type, ws)

    # 5. Improvement %
    improv_series = _series('improvement_pct')
    y_hlines = [(0, 'Zero-Shot baseline', 'red', '--')]
    if improv_series:
        all_finals = [v[-1] for _, v in improv_series.values() if len(v) > 0]
        if all_finals:
            best_final = max(all_finals)
            if best_final > 0:
                y_hlines.append((0.9 * best_final, f'90% ({0.9*best_final:.1f}%)',
                                 'orange', '-.'))
                y_hlines.append((0.8 * best_final, f'80% ({0.8*best_final:.1f}%)',
                                 'cyan', ':'))
    create_two_method_metric_plot(
        improv_series, 'Improvement (%)', 'Improvement over Zero-Shot',
        os.path.join(out_dir, f'metric_Improvement_{model_type}_ws{ws}.png'),
        scenario_name, model_type, ws,
        y_hlines=y_hlines)

    # 6. Efficiency = Improvement% / Data%
    eff_series = {}
    for method in METHODS:
        sub = df_trained[df_trained['method'] == method].sort_values('train_size')
        if len(sub) == 0:
            continue
        pcts = sub['train_percentage'].values
        improv = sub['improvement_pct'].values
        valid = pcts > 0
        if valid.sum() > 1:
            eff = np.where(valid, improv / np.maximum(pcts, 0.01), 0)
            eff_series[method] = (sub['train_size'].values[valid], eff[valid])
    if eff_series:
        create_two_method_metric_plot(
            eff_series, 'Efficiency (Improvement% / Data%)', 'Training Efficiency',
            os.path.join(out_dir, f'metric_Efficiency_{model_type}_ws{ws}.png'),
            scenario_name, model_type, ws)


def create_cross_scenario_plot(master_df, out_dir, model_type, ws):
    """Per metric, all 9 scenarios as separate lines, faceted by method."""
    for metric, ylabel, fname in [
        ('test_mae', 'MAE', 'cross_MAE'),
        ('test_rmse', 'RMSE', 'cross_RMSE'),
        ('test_r2', 'R²', 'cross_R2'),
        ('improvement_pct', 'Improvement (%)', 'cross_Improvement'),
    ]:
        fig, axes = plt.subplots(1, len(METHODS),
                                  figsize=(14 * len(METHODS), 7),
                                  sharey=True)
        if len(METHODS) == 1:
            axes = [axes]

        scenarios = master_df['scenario'].unique()
        cmap = plt.get_cmap('tab10')
        scenario_colors = {sc: cmap(i % 10) for i, sc in enumerate(scenarios)}

        for ax, method in zip(axes, METHODS):
            sub_all = master_df[(master_df['model'] == model_type) &
                                (master_df['window_size'] == ws) &
                                (master_df['method'] == method) &
                                (master_df['train_size'] > 0)]
            if len(sub_all) == 0:
                continue
            max_size = int(sub_all['train_size'].max())

            for sc in scenarios:
                sub = sub_all[sub_all['scenario'] == sc].sort_values('train_size')
                if len(sub) == 0:
                    continue
                ax.plot(sub['train_size'].values, sub[metric].values,
                        'o-', lw=1.6, ms=3.5, color=scenario_colors[sc],
                        label=sc, alpha=0.85)

            ax.set_xlabel('Training Data Size', fontsize=11, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.set_title(f'{method} | {model_type} | ws={ws}',
                          fontsize=12, fontweight='bold')
            ax.set_xlim([-5, max_size + 10])
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=7, loc='best', ncol=2)
            if metric == 'improvement_pct':
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{fname}_{model_type}_ws{ws}.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()


##############################################################################
#  ONE SCENARIO × MODEL × WS
##############################################################################

def process_scenario(src_id, tgt_id, model_type, ws, sc_idx, total_idx):
    scenario_name = f"{src_id} → {tgt_id}"
    src_cfg = build_room_cfg(src_id, VARIANT)
    tgt_cfg = build_room_cfg(tgt_id, VARIANT)

    print(f"\n{'=' * 100}")
    print(f"[{sc_idx}/{total_idx}] {scenario_name} | {model_type} | ws={ws}")
    print(f"  src features: {src_cfg['features']}")
    print(f"  tgt features: {tgt_cfg['features']}")
    print(f"{'=' * 100}")

    if not os.path.exists(src_cfg['path']) or not os.path.exists(tgt_cfg['path']):
        print("  ❌ Data file missing")
        return None

    if len(src_cfg['features']) != len(tgt_cfg['features']):
        print(f"  ⏭ Skip: dim mismatch")
        return None

    sc_folder = ensure_dir(os.path.join(
        OUTPUT_ROOT, VARIANT, f"{src_id}_to_{tgt_id}", f"{model_type}_ws{ws}"))

    log_file = os.path.join(sc_folder, "progressive_log.txt")
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode='a'),
                  logging.StreamHandler(sys.stdout)],
        force=True)

    weights_path = find_pretrained_weights(PRETRAIN_DIR, src_id, VARIANT,
                                            model_type, ws)
    if weights_path is None:
        print(f"  ⚠ No pretrained weights for {src_id}/{model_type}/ws{ws}")
        return None
    print(f"  Pretrained: {weights_path}")

    tgt_X, tgt_y, tgt_df_full, _ = load_room_raw(
        tgt_cfg['path'], tgt_cfg['features'],
        tgt_cfg['name'], tgt_cfg['selected_rows'])

    try:
        data = prepare_target_full(tgt_X, tgt_y, tgt_cfg['scenario'], ws, BATCH_SIZE)
    except Exception as e:
        print(f"  ❌ Target data prep failed: {e}")
        return None

    joblib.dump(data['scaler'], os.path.join(sc_folder, "tgt_scaler.pkl"))

    n_max = data['n_tr_total']
    print(f"  Target train: {n_max}, val: {len(data['val_loader'].dataset)}, "
          f"test: {len(data['test_loader'].dataset)}")

    sizes = generate_train_sizes(n_max, ws)
    print(f"  Train sizes ({len(sizes)}): {sizes[0]}, {sizes[1] if len(sizes)>1 else ''}..., {sizes[-1]}")

    rows = []

    # ── 1. Zero-shot evaluation ──
    print(f"\n  ── Zero-Shot (n=0) ──")
    try:
        reset_all_seeds(SEED)
        model = build_model(model_type, data['input_size'])
        model = load_pretrained_weights(
            model, weights_path, model_type, data['input_size'], data['input_size'])
        for p in model.parameters():
            p.requires_grad = False
        metrics = evaluate_full(model, data['test_loader'], "Zero-Shot")
        zs_mae = metrics['all']['mae']
        print(f"    Zero-Shot MAE: {zs_mae:.4f}")

        params = count_parameters(model)
        for method in METHODS:
            rows.append(build_result_row(
                scenario_name, src_id, tgt_id, model_type, ws,
                method=method, train_size=0, train_pct=0.0, zs_mae=zs_mae,
                metrics=metrics, train_time=0.0, params=params,
                src_cfg=src_cfg, tgt_cfg=tgt_cfg))
        torch.save(model.state_dict(),
                   os.path.join(sc_folder, f"weights_zeroshot_ws{ws}.pth"))
    except Exception as e:
        print(f"  ❌ Zero-shot failed: {e}")
        traceback.print_exc()
        return None

    # ── 2. Progressive training (per method × per size) ──
    for method in METHODS:
        print(f"\n  ══ Method: {method} ══")
        for n_train in sizes:
            pct = n_train / n_max * 100
            try:
                reset_all_seeds(SEED)
                train_ld, n_seqs = make_train_subset_loader(
                    data['scaler'], data['tr_X'], data['tr_y'], n_train,
                    ws, BATCH_SIZE)

                model = build_model(model_type, data['input_size'])
                model = load_pretrained_weights(
                    model, weights_path, model_type,
                    data['input_size'], data['input_size'])

                t_train, best_val = train_model_with_method(
                    model, method, train_ld, data['val_loader'],
                    EPOCHS, LEARNING_RATE)
                metrics = evaluate_full(model, data['test_loader'], method)
                params = count_parameters(model)

                mae = metrics['all']['mae']
                improv = ((zs_mae - mae) / zs_mae * 100.0) if zs_mae > 0 else 0.0
                print(f"    [{method}] n={n_train:>5} ({pct:5.1f}%) | "
                      f"seqs={n_seqs:>4} | MAE={mae:.4f} | "
                      f"Improv={improv:+.1f}% | t={t_train:.1f}s")

                rows.append(build_result_row(
                    scenario_name, src_id, tgt_id, model_type, ws,
                    method=method, train_size=n_train, train_pct=pct,
                    zs_mae=zs_mae, metrics=metrics, train_time=t_train,
                    params=params, src_cfg=src_cfg, tgt_cfg=tgt_cfg))

                if n_train == sizes[-1]:
                    torch.save(model.state_dict(),
                               os.path.join(sc_folder,
                                            f"weights_{slugify(method)}_full_ws{ws}.pth"))
            except Exception as e:
                print(f"    ❌ {method} n={n_train} failed: {e}")
                logging.error(f"{method} n={n_train}: {e}")

    df = pd.DataFrame(rows)
    sc_csv = os.path.join(sc_folder, f"metrics_{model_type}_ws{ws}.csv")
    df.to_csv(sc_csv, index=False)
    print(f"\n  ✓ Saved: {sc_csv}")

    # Per-scenario plots
    create_learning_curves(df, sc_folder, model_type, ws, scenario_name, zs_mae)

    # 90% threshold per method
    print(f"\n  ── Key insights ──")
    for method in METHODS:
        sub = df[(df['method'] == method) & (df['train_size'] > 0)]
        if len(sub) == 0:
            continue
        full_imp = sub.iloc[-1]['improvement_pct']
        if full_imp > 0:
            for thr, lbl in [(0.9, "90%"), (0.8, "80%")]:
                target = thr * full_imp
                above = sub[sub['improvement_pct'] >= target]
                if len(above) > 0:
                    opt = above.iloc[0]
                    print(f"  [{method} {lbl}] size={int(opt['train_size'])} "
                          f"({opt['train_percentage']:.1f}%), "
                          f"MAE={opt['test_mae']:.4f}, "
                          f"savings={100 - opt['train_percentage']:.1f}%")

    return df


##############################################################################
#  MAIN
##############################################################################

def main():
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 100)
    print("PROGRESSIVE DATA EFFICIENCY — Feature Extraction & Full Fine-Tuning")
    print(f"Methods: {METHODS}")
    print(f"Models: {MODEL_TYPES} | WS: {WINDOW_SIZES} | Epochs: {EPOCHS} | LR: {LEARNING_RATE}")
    print(f"Variant: {VARIANT}")
    print(f"PRETRAIN_DIR: {PRETRAIN_DIR}")
    print(f"OUTPUT_ROOT:  {OUTPUT_ROOT}")
    print(f"User: Azadshokrollahi | {timestamp} UTC")
    print("=" * 100)

    ensure_dir(OUTPUT_ROOT)
    all_dfs = []
    total = len(TRANSFER_SCENARIOS) * len(MODEL_TYPES) * len(WINDOW_SIZES)
    counter = 0

    for src_id, tgt_id in TRANSFER_SCENARIOS:
        for model_type in MODEL_TYPES:
            for ws in WINDOW_SIZES:
                counter += 1
                df = process_scenario(src_id, tgt_id, model_type, ws, counter, total)
                if df is not None:
                    all_dfs.append(df)

    if not all_dfs:
        print("\n  No results collected.")
        return

    master_df = pd.concat(all_dfs, ignore_index=True)

    COLUMN_ORDER = [
        'scenario', 'source', 'target', 'method', 'weight_scheme', 'variant',
        'model', 'window_size', 'features',
        'src_scenario', 'tgt_scenario', 'initialization',
        'train_size', 'train_percentage', 'zero_shot_mae', 'improvement_pct',
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
    master_df = master_df[[c for c in COLUMN_ORDER if c in master_df.columns]]

    master_csv = os.path.join(OUTPUT_ROOT,
                              f"all_progressive_data_efficiency_{VARIANT}.csv")
    master_df.to_csv(master_csv, index=False)
    print(f"\n\n{'=' * 100}\nMaster CSV: {master_csv}\nTotal rows: {len(master_df)}")

    # Cross-scenario plots per (model, ws)
    for model_type in MODEL_TYPES:
        for ws in WINDOW_SIZES:
            create_cross_scenario_plot(master_df, OUTPUT_ROOT, model_type, ws)

    # Best per scenario × model × method
    print(f"\n{'=' * 100}\nBEST RESULT PER SCENARIO × MODEL × METHOD")
    print(f"{'=' * 100}")
    print(f"{'Scenario':<25} {'Model':<12} {'Method':<22} {'Best MAE':<10} "
          f"{'Size':<7} {'Pct%':<7} {'Improv%':<10}")
    print("-" * 100)
    for (sc, mdl, method), grp in master_df[master_df['train_size'] > 0].groupby(
            ['scenario', 'model', 'method']):
        best = grp.loc[grp['test_mae'].idxmin()]
        print(f"{sc:<25} {mdl:<12} {method:<22} "
              f"{best['test_mae']:<10.4f} "
              f"{int(best['train_size']):<7} "
              f"{best['train_percentage']:<7.1f} "
              f"{best['improvement_pct']:<+10.1f}")

    # 90% threshold table
    print(f"\n{'=' * 100}\n90% THRESHOLD: MIN SAMPLES FOR 90% OF MAX IMPROVEMENT")
    print(f"{'=' * 100}")
    print(f"{'Scenario':<25} {'Model':<12} {'Method':<22} {'90% Size':<10} "
          f"{'Pct%':<7} {'MAE':<10} {'Savings%':<10}")
    print("-" * 105)
    for (sc, mdl, method), grp in master_df[master_df['train_size'] > 0].groupby(
            ['scenario', 'model', 'method']):
        full_imp = grp.iloc[-1]['improvement_pct']
        if full_imp <= 0:
            continue
        target = 0.9 * full_imp
        above = grp[grp['improvement_pct'] >= target]
        if len(above) == 0:
            continue
        opt = above.iloc[0]
        print(f"{sc:<25} {mdl:<12} {method:<22} "
              f"{int(opt['train_size']):<10} "
              f"{opt['train_percentage']:<7.1f} "
              f"{opt['test_mae']:<10.4f} "
              f"{100 - opt['train_percentage']:<10.1f}")

    # FE vs Full-FT average comparison
    print(f"\n{'=' * 80}\nFE vs Full-FT — average at MAX size")
    print(f"{'=' * 80}")
    print(f"{'Method':<22} {'avg MAE':<10} {'avg Improv%':<14} {'avg Time(s)':<12}")
    print("-" * 60)
    for method in METHODS:
        sub = master_df[(master_df['method'] == method) & (master_df['train_size'] > 0)]
        if len(sub) == 0:
            continue
        max_rows = sub.loc[sub.groupby(['scenario', 'model', 'window_size'])
                              ['train_size'].idxmax()]
        print(f"{method:<22} {max_rows['test_mae'].mean():<10.4f} "
              f"{max_rows['improvement_pct'].mean():<14.2f} "
              f"{max_rows['target_train_time_sec'].mean():<12.3f}")

    # Global summary
    sp = os.path.join(OUTPUT_ROOT, "GLOBAL_SUMMARY.txt")
    with open(sp, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("PROGRESSIVE DATA EFFICIENCY — GLOBAL SUMMARY\n")
        f.write(f"Methods: {METHODS}\n")
        f.write(f"Models: {MODEL_TYPES} | WS: {WINDOW_SIZES} | Epochs: {EPOCHS}\n")
        f.write(f"Scenarios: {len(TRANSFER_SCENARIOS)}\n")
        f.write(f"Train sizes: 1, 5, 10, 15, ..., max (step 5)\n")
        f.write(f"Timestamp: {timestamp} UTC\nUser: Azadshokrollahi\n\n")
        f.write("=" * 100 + "\n90% THRESHOLD PER (SCENARIO × MODEL × METHOD)\n")
        f.write("=" * 100 + "\n")
        for (sc, mdl, method), grp in master_df[master_df['train_size'] > 0].groupby(
                ['scenario', 'model', 'method']):
            full_imp = grp.iloc[-1]['improvement_pct']
            if full_imp <= 0:
                continue
            target = 0.9 * full_imp
            above = grp[grp['improvement_pct'] >= target]
            if len(above) == 0:
                continue
            opt = above.iloc[0]
            f.write(f"  {sc} | {mdl} | {method}: 90% at size={int(opt['train_size'])} "
                    f"({opt['train_percentage']:.1f}%), MAE={opt['test_mae']:.4f}\n")
        f.write("=" * 100 + "\n")
    print(f"\nGlobal summary: {sp}")

    print("\n" + "=" * 100)
    print("✓ PROGRESSIVE DATA EFFICIENCY STUDY COMPLETED!")
    print(f"User: Azadshokrollahi | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("=" * 100)


if __name__ == "__main__":
    main()