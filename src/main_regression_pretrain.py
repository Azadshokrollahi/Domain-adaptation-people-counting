"""
RNN/LSTM/Transformer Independent Regression Training for People Counting
Selected rows only, no transfer learning.

Author: Azadshokrollahi

Reads:
  • configs/rooms.yaml  → room paths, features, scenarios, selected_rows
  • src/config.py       → seeds, epochs, batch size, model hyperparams, paths

Run:
  python src/main_regression_pretrain.py
"""

import os, sys, logging, ast, traceback
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
)

# ─── Centralized config ─────────────────────────────────────────────────────
from config import (
    SEED, WINDOW_SIZES, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    RESULTS_ROOT, PRETRAIN_DIR_NAME,
    RNN_HIDDEN_SIZE, RNN_NUM_LAYERS,
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS,
    TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD, TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_DIM_FEEDFORWARD, TRANSFORMER_DROPOUT,
    load_rooms,
)
# ────────────────────────────────────────────────────────────────────────────

# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def reset_seed(seed=SEED):
    """Reset all random states for fair per-experiment comparison."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


##############################################################################
#  MODEL DEFINITIONS
##############################################################################

class RNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 bidirectional_flag=False, aggregation_mode='one', use_attention=False):
        super().__init__()
        self.bidirectional_flag = bidirectional_flag
        self.aggregation_mode = aggregation_mode.lower()
        self.use_attention = use_attention
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                          batch_first=True, bidirectional=bidirectional_flag, nonlinearity='tanh')
        eff = hidden_size * 2 if bidirectional_flag else hidden_size
        if self.use_attention:
            self.attention_layer = nn.Linear(eff, 1)
        self.fc = nn.Linear(eff, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional_flag else 1),
                         x.size(0), self.hidden_size).to(x.device)
        outputs, _ = self.rnn(x, h0)
        if self.use_attention:
            attn = torch.softmax(self.attention_layer(outputs).squeeze(-1), dim=1)
            context = torch.bmm(attn.unsqueeze(1), outputs).squeeze(1)
            return self.fc(context).squeeze(-1)
        if self.aggregation_mode == 'one':
            return self.fc(outputs[:, -1, :]).squeeze(-1)
        if self.aggregation_mode == 'many':
            return self.fc(outputs).mean(dim=1).squeeze(-1)
        raise ValueError("aggregation_mode must be 'one' or 'many'")


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 bidirectional_flag=False, aggregation_mode='one', use_attention=False):
        super().__init__()
        self.bidirectional_flag = bidirectional_flag
        self.aggregation_mode = aggregation_mode.lower()
        self.use_attention = use_attention
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=bidirectional_flag)
        eff = hidden_size * 2 if bidirectional_flag else hidden_size
        if self.use_attention:
            self.attention_layer = nn.Linear(eff, 1)
        self.fc = nn.Linear(eff, 1)

    def forward(self, x):
        dirs = 2 if self.bidirectional_flag else 1
        h0 = torch.zeros(self.num_layers * dirs, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * dirs, x.size(0), self.hidden_size).to(x.device)
        outputs, _ = self.lstm(x, (h0, c0))
        if self.use_attention:
            attn = torch.softmax(self.attention_layer(outputs).squeeze(-1), dim=1)
            context = torch.bmm(attn.unsqueeze(1), outputs).squeeze(1)
            return self.fc(context).squeeze(-1)
        if self.aggregation_mode == 'one':
            return self.fc(outputs[:, -1, :]).squeeze(-1)
        if self.aggregation_mode == 'many':
            return self.fc(outputs).mean(dim=1).squeeze(-1)
        raise ValueError("aggregation_mode must be 'one' or 'many'")


class TransformerRegressor(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :]).squeeze(-1)


##############################################################################
#  MODEL FACTORY (uses config.py hyperparams)
##############################################################################

def build_model_configs():
    """Return RNN / LSTM / Transformer configs using values from config.py."""
    return {
        "RNN": {
            "class": RNNRegressor,
            "params": {
                "hidden_size":        RNN_HIDDEN_SIZE,
                "num_layers":         RNN_NUM_LAYERS,
                "bidirectional_flag": False,
                "aggregation_mode":   "one",
                "use_attention":      False,
            },
        },
        "LSTM": {
            "class": LSTMRegressor,
            "params": {
                "hidden_size":        LSTM_HIDDEN_SIZE,
                "num_layers":         LSTM_NUM_LAYERS,
                "bidirectional_flag": False,
                "aggregation_mode":   "one",
                "use_attention":      False,
            },
        },
        "Transformer": {
            "class": TransformerRegressor,
            "params": {
                "d_model":         TRANSFORMER_D_MODEL,
                "nhead":           TRANSFORMER_NHEAD,
                "num_layers":      TRANSFORMER_NUM_LAYERS,
                "dim_feedforward": TRANSFORMER_DIM_FEEDFORWARD,
                "dropout":         TRANSFORMER_DROPOUT,
            },
        },
    }


##############################################################################
#  METRICS
##############################################################################

def compute_all_metrics(all_true, all_preds):
    mae = mean_absolute_error(all_true, all_preds)
    rmse = np.sqrt(mean_squared_error(all_true, all_preds))
    r2 = r2_score(all_true, all_preds)
    medae = median_absolute_error(all_true, all_preds)
    preds_rounded = np.round(all_preds)
    abs_diff = np.abs(all_true - preds_rounded)
    n = len(all_true)
    acc_1 = float(np.sum(abs_diff <= 1)) / n * 100.0
    acc_2 = float(np.sum(abs_diff <= 2)) / n * 100.0
    return {
        'mae': mae, 'rmse': rmse, 'r2': r2, 'medae': medae,
        'acc_pm1': acc_1, 'acc_pm2': acc_2, 'sample_count': n,
    }


def compute_occupied_metrics(all_true, all_preds):
    mask = all_true > 0
    if mask.sum() == 0:
        return None
    true_occ = all_true[mask]
    pred_occ = all_preds[mask]
    metrics = compute_all_metrics(true_occ, pred_occ)
    preds_rounded = np.round(pred_occ)
    exact = int(np.sum(true_occ == preds_rounded))
    metrics['exact_match'] = exact
    metrics['exact_match_pct'] = exact / len(true_occ) * 100.0
    return metrics


def compute_empty_metrics(all_true, all_preds):
    mask = all_true == 0
    if mask.sum() == 0:
        return None
    return compute_all_metrics(all_true[mask], all_preds[mask])


def compute_per_count_metrics(all_true, all_preds, max_count=8):
    results = {}
    for c in range(max_count + 1):
        mask = all_true == c
        n = int(mask.sum())
        if n > 0:
            true_c = all_true[mask]
            pred_c = all_preds[mask]
            mae_c = mean_absolute_error(true_c, pred_c)
            rmse_c = np.sqrt(mean_squared_error(true_c, pred_c))
            preds_rounded = np.round(pred_c)
            acc1_c = float(np.sum(np.abs(true_c - preds_rounded) <= 1)) / n * 100.0
            results[c] = {'count': n, 'mae': mae_c, 'rmse': rmse_c, 'acc_pm1': acc1_c}
    return results


##############################################################################
#  DATA LOADING & PREPROCESSING
##############################################################################

COLUMN_RENAMES = {'co2': 'co2_value', 'motion': 'motion_event_count'}

DROP_COLS = [
    'Datetime', 'datetime', 't', 'passage_event_count',
    'acc_motion', 'acc_passage', 'sound_mean_voltage', 'sound_rms_voltage',
    'invites', 'acc_motion_per_min', 'acc_passage_per_min',
]


def split_indices_by_scenario(n_total, scenario):
    if scenario == 'scenario_1':
        train_end = int(0.7 * n_total)
        val_end = int(0.8 * n_total)
        return {'scenario': 'scenario_1',
                'train_start': 0, 'train_end': train_end,
                'val_start': train_end, 'val_end': val_end,
                'test_start': val_end, 'test_end': n_total}
    test_end = int(0.2 * n_total)
    val_end = int(0.3 * n_total)
    return {'scenario': 'scenario_2',
            'test_start': 0, 'test_end': test_end,
            'val_start': test_end, 'val_end': val_end,
            'train_start': val_end, 'train_end': n_total}


def load_and_preprocess_data(file_path, window_size, features_to_use,
                             selected_rows=None, split_scenario='scenario_1', room_name=""):
    df = pd.read_csv(file_path)
    logging.info("Loaded %s: %d rows from %s", room_name, len(df), file_path)

    df = df.rename(columns={k: v for k, v in COLUMN_RENAMES.items() if k in df.columns})

    if 'peoplecount_value' in df.columns:
        def extract_target(val):
            try:
                if isinstance(val, str):
                    return ast.literal_eval(val)[0]
                return val
            except Exception:
                return np.nan
        df['target'] = df['peoplecount_value'].apply(extract_target)

    if 'target' not in df.columns:
        raise ValueError(f"No 'target' column in {room_name}. Columns: {list(df.columns)}")

    if selected_rows is not None:
        row_start, row_end = selected_rows
        df = df.iloc[row_start:row_end].reset_index(drop=True)
        logging.info("Selected rows [%d:%d] → %d rows", row_start, row_end, len(df))

    df_full = df.copy()

    drop_cols = [c for c in DROP_COLS + ['peoplecount_value'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if 'Hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df = df.drop(columns=['Hour'])
    if 'Day' in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df['Day'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['Day'] / 7)
        df = df.drop(columns=['Day'])

    missing = [f for f in features_to_use if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in {room_name}: {missing}. Available: {list(df.columns)}")

    features = df[features_to_use].copy()
    target = df['target'].copy()
    features = features.apply(pd.to_numeric, errors='coerce').ffill().bfill()
    target = pd.to_numeric(target, errors='coerce').ffill().bfill()

    logging.info("Features: %s, Target: %s", features.shape, target.shape)
    logging.info("Target stats - Mean: %.2f, Std: %.2f, Min: %.2f, Max: %.2f",
                 target.mean(), target.std(), target.min(), target.max())

    n_total = len(features)
    split_info = split_indices_by_scenario(n_total, split_scenario)

    X_train_raw = features.iloc[split_info['train_start']:split_info['train_end']].values
    y_train_raw = target.iloc[split_info['train_start']:split_info['train_end']].values
    X_val_raw = features.iloc[split_info['val_start']:split_info['val_end']].values
    y_val_raw = target.iloc[split_info['val_start']:split_info['val_end']].values
    X_test_raw = features.iloc[split_info['test_start']:split_info['test_end']].values
    y_test_raw = target.iloc[split_info['test_start']:split_info['test_end']].values

    logging.info("Split: %s | Train: %s, Val: %s, Test: %s",
                 split_scenario, X_train_raw.shape, X_val_raw.shape, X_test_raw.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    def create_sequences(X, y, ws):
        Xs, ys = [], []
        for i in range(len(X) - ws + 1):
            Xs.append(X[i:i + ws])
            ys.append(y[i + ws - 1])
        return np.array(Xs), np.array(ys)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw, window_size)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_raw, window_size)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_raw, window_size)

    for name, arr in [("Train", y_train_seq), ("Val", y_val_seq), ("Test", y_test_seq)]:
        if len(arr) == 0:
            raise ValueError(f"{name} has 0 sequences. Reduce window_size={window_size}.")

    logging.info("Sequences - Train: %s, Val: %s, Test: %s",
                 X_train_seq.shape, X_val_seq.shape, X_test_seq.shape)

    if 'occupied' in df_full.columns:
        for split_name, s, e in [
            ("Train", split_info['train_start'], split_info['train_end']),
            ("Val",   split_info['val_start'],   split_info['val_end']),
            ("Test",  split_info['test_start'],  split_info['test_end']),
        ]:
            occ_start = s + window_size - 1
            occ_count = (df_full.iloc[occ_start:e]['occupied'] == 1).sum()
            logging.info("%s sequences with occupied==1: %d", split_name, int(occ_count))

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                      torch.tensor(y_train_seq, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_seq, dtype=torch.float32),
                      torch.tensor(y_val_seq, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False)
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
    test_loader = DataLoader(
        TensorDataset(X_test_t, torch.tensor(y_test_seq, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False)

    return (train_loader, val_loader, test_loader, X_test_t,
            X_train_seq.shape[2], df_full, split_info, window_size, scaler, features_to_use)


##############################################################################
#  TRAINING
##############################################################################

def train_model(model, train_loader, val_loader, num_epochs,
                lr=LEARNING_RATE, save_dir=None, config_name=""):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loss_hist, val_loss_hist, train_mae_hist, val_mae_hist = [], [], [], []
    best_val_mae, best_state = float('inf'), None

    for epoch in range(num_epochs):
        model.train()
        r_loss, r_mae, n_s = 0.0, 0.0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            mae = torch.abs(preds - by).mean()
            loss.backward()
            optimizer.step()
            r_loss += loss.item() * bx.size(0)
            r_mae += mae.item() * bx.size(0)
            n_s += bx.size(0)

        t_loss, t_mae = r_loss / n_s, r_mae / n_s
        train_loss_hist.append(t_loss)
        train_mae_hist.append(t_mae)

        model.eval()
        v_loss, v_mae, v_s = 0.0, 0.0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx)
                v_loss += criterion(preds, by).item() * bx.size(0)
                v_mae += torch.abs(preds - by).mean().item() * bx.size(0)
                v_s += bx.size(0)

        vl, vm = v_loss / v_s, v_mae / v_s
        val_loss_hist.append(vl)
        val_mae_hist.append(vm)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info("Epoch [%d/%d] TrLoss: %.4f, TrMAE: %.4f, VaLoss: %.4f, VaMAE: %.4f",
                         epoch + 1, num_epochs, t_loss, t_mae, vl, vm)

        if vm < best_val_mae:
            best_val_mae = vm
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_dir:
                torch.save(model, os.path.join(save_dir, f"best_model_{config_name}.pth"))
                torch.save(best_state, os.path.join(save_dir, f"best_weights_{config_name}.pth"))

    if save_dir:
        torch.save(model, os.path.join(save_dir, f"last_model_{config_name}.pth"))
        torch.save(model.state_dict(), os.path.join(save_dir, f"last_weights_{config_name}.pth"))

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        logging.info("✓ Best model loaded (Val MAE: %.4f)", best_val_mae)

    return train_loss_hist, train_mae_hist, val_loss_hist, val_mae_hist


##############################################################################
#  EVALUATION
##############################################################################

def evaluate_model(model, test_loader):
    model.eval()
    preds_list, true_list = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            preds_list.append(model(bx).cpu().numpy())
            true_list.append(by.cpu().numpy())
    all_preds = np.concatenate(preds_list)
    all_true = np.concatenate(true_list)

    metrics_all = compute_all_metrics(all_true, all_preds)
    metrics_occ = compute_occupied_metrics(all_true, all_preds)
    metrics_empty = compute_empty_metrics(all_true, all_preds)
    metrics_per_count = compute_per_count_metrics(all_true, all_preds)

    logging.info("=== TEST METRICS (ALL) ===")
    logging.info("  MAE: %.4f, RMSE: %.4f, R²: %.4f, MedAE: %.4f",
                 metrics_all['mae'], metrics_all['rmse'], metrics_all['r2'], metrics_all['medae'])
    logging.info("  Acc±1: %.2f%%, Acc±2: %.2f%%", metrics_all['acc_pm1'], metrics_all['acc_pm2'])

    if metrics_occ:
        logging.info("=== TEST METRICS (OCCUPIED, target>0) ===")
        logging.info("  N=%d, MAE: %.4f, RMSE: %.4f, MedAE: %.4f",
                     metrics_occ['sample_count'], metrics_occ['mae'],
                     metrics_occ['rmse'], metrics_occ['medae'])
        logging.info("  Acc±1: %.2f%%, Exact: %d (%.2f%%)",
                     metrics_occ['acc_pm1'], metrics_occ['exact_match'],
                     metrics_occ['exact_match_pct'])

    if metrics_empty:
        logging.info("=== TEST METRICS (EMPTY, target==0) ===")
        logging.info("  N=%d, MAE: %.4f, Acc±1: %.2f%%",
                     metrics_empty['sample_count'], metrics_empty['mae'], metrics_empty['acc_pm1'])

    print(f"  Test MAE: {metrics_all['mae']:.4f}, RMSE: {metrics_all['rmse']:.4f}, "
          f"R²: {metrics_all['r2']:.4f}, MedAE: {metrics_all['medae']:.4f}")
    print(f"  Acc±1: {metrics_all['acc_pm1']:.2f}%, Acc±2: {metrics_all['acc_pm2']:.2f}%")
    if metrics_occ:
        print(f"  [Occupied] MAE: {metrics_occ['mae']:.4f}, "
              f"Acc±1: {metrics_occ['acc_pm1']:.2f}%, Exact: {metrics_occ['exact_match_pct']:.1f}%")

    return {
        'all': metrics_all, 'occupied': metrics_occ,
        'empty': metrics_empty, 'per_count': metrics_per_count,
    }, all_true, all_preds


##############################################################################
#  PLOTTING
##############################################################################

def plot_regression_results(all_true, all_preds, save_path, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=14)
    axes[0].scatter(all_true, all_preds, alpha=0.5, s=10)
    mn, mx = min(all_true.min(), all_preds.min()), max(all_true.max(), all_preds.max())
    axes[0].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect')
    axes[0].set_xlabel('True'); axes[0].set_ylabel('Predicted')
    axes[0].set_title('Predictions vs True'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    residuals = all_true - all_preds
    axes[1].hist(residuals, bins=50, edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Residual'); axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Residuals (Mean: {residuals.mean():.3f})'); axes[1].grid(True, alpha=0.3)
    axes[2].scatter(all_preds, residuals, alpha=0.5, s=10)
    axes[2].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('Residual')
    axes[2].set_title('Residuals vs Predicted'); axes[2].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    logging.info("Plot saved: %s", save_path)


def plot_training_curves(train_loss, val_loss, train_mae, val_mae, save_path, title=""):
    epochs = range(1, len(train_loss) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14)
    axes[0].plot(epochs, train_loss, 'r-', label='Train')
    axes[0].plot(epochs, val_loss, 'b-', label='Val')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epochs, train_mae, 'r-', label='Train')
    axes[1].plot(epochs, val_mae, 'b-', label='Val')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MAE'); axes[1].legend(); axes[1].grid(True)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


##############################################################################
#  PREDICTION SAVING & ANALYSIS
##############################################################################

def predict_and_save(model, X_test_tensor, df_full, split_info, window_size, output_file):
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor.to(device)).cpu().numpy()

    test_start = split_info['test_start'] + window_size - 1
    test_end = split_info['test_end']
    df_test = df_full.iloc[test_start:test_end].copy()

    if len(df_test) != len(preds):
        min_len = min(len(df_test), len(preds))
        df_test = df_test.iloc[:min_len]
        preds = preds[:min_len]

    df_test['predicted_peoplecount'] = preds
    df_test['predicted_rounded'] = np.round(preds)
    df_test.to_csv(output_file, index=False)
    logging.info("Predictions saved: %s (%d rows)", output_file, len(df_test))


def analyze_prediction_results(file_path):
    df = pd.read_csv(file_path)
    if 'target' not in df.columns or 'predicted_peoplecount' not in df.columns:
        return "Missing target or predicted_peoplecount column; analysis skipped."

    true = df['target'].values
    pred = df['predicted_peoplecount'].values
    m_all = compute_all_metrics(true, pred)

    parts = [
        "=" * 60, "PREDICTION ANALYSIS", "=" * 60,
        f"Total samples: {m_all['sample_count']}", "",
        "--- ALL SAMPLES ---",
        f"  MAE:   {m_all['mae']:.4f}",   f"  RMSE:  {m_all['rmse']:.4f}",
        f"  R²:    {m_all['r2']:.4f}",    f"  MedAE: {m_all['medae']:.4f}",
        f"  Acc±1: {m_all['acc_pm1']:.2f}%", f"  Acc±2: {m_all['acc_pm2']:.2f}%",
    ]

    if 'occupied' in df.columns:
        occ = df[df['occupied'] == 1]
        if len(occ) > 0:
            t_o = occ['target'].values
            p_o = occ['predicted_peoplecount'].values
            m_o = compute_all_metrics(t_o, p_o)
            exact = int(np.sum(t_o == np.round(p_o)))
            parts += [
                "", "--- OCCUPIED SAMPLES (target > 0) ---",
                f"  Total: {len(occ)}",
                f"  MAE:   {m_o['mae']:.4f}",   f"  RMSE:  {m_o['rmse']:.4f}",
                f"  R²:    {m_o['r2']:.4f}",    f"  MedAE: {m_o['medae']:.4f}",
                f"  Acc±1: {m_o['acc_pm1']:.2f}%", f"  Acc±2: {m_o['acc_pm2']:.2f}%",
                f"  Exact match: {exact} ({exact / len(occ) * 100:.2f}%)",
            ]
        emp = df[df['occupied'] == 0]
        if len(emp) > 0:
            m_e = compute_all_metrics(emp['target'].values, emp['predicted_peoplecount'].values)
            parts += [
                "", "--- EMPTY SAMPLES (target == 0) ---",
                f"  Total: {len(emp)}", f"  MAE:   {m_e['mae']:.4f}",
                f"  Acc±1: {m_e['acc_pm1']:.2f}%",
            ]
    else:
        m_o = compute_occupied_metrics(true, pred)
        if m_o:
            parts += [
                "", "--- OCCUPIED (target > 0) ---",
                f"  Total: {m_o['sample_count']}",
                f"  MAE:   {m_o['mae']:.4f}", f"  RMSE:  {m_o['rmse']:.4f}",
                f"  MedAE: {m_o['medae']:.4f}", f"  Acc±1: {m_o['acc_pm1']:.2f}%",
            ]

    per_count = compute_per_count_metrics(true, pred)
    if per_count:
        parts += ["", "--- PER-COUNT BREAKDOWN ---"]
        parts.append(f"  {'Count':<7} {'N':<7} {'MAE':<10} {'RMSE':<10} {'Acc±1':<10}")
        parts.append("  " + "-" * 45)
        for c in sorted(per_count.keys()):
            m = per_count[c]
            parts.append(
                f"  {c:<7} {m['count']:<7} {m['mae']:<10.4f} {m['rmse']:<10.4f} {m['acc_pm1']:<10.1f}%")

    parts.append("=" * 60)
    summary = "\n".join(parts)
    logging.info(summary)
    print(f"    {summary}")
    return summary


def write_summary_file(folder, ws, metrics_result, analysis_text, feature_variant):
    sf = os.path.join(folder, f"summary_ws{ws}.txt")
    m = metrics_result['all']
    with open(sf, "w") as f:
        f.write(f"Variant: {feature_variant}\n")
        f.write(f"Window Size: {ws}\n")
        f.write(f"Samples: {m['sample_count']}\n\n")
        f.write("=== ALL SAMPLES ===\n")
        f.write(f"MAE:   {m['mae']:.4f}\nRMSE:  {m['rmse']:.4f}\n")
        f.write(f"R²:    {m['r2']:.4f}\nMedAE: {m['medae']:.4f}\n")
        f.write(f"Acc±1: {m['acc_pm1']:.2f}%\nAcc±2: {m['acc_pm2']:.2f}%\n\n")
        if metrics_result['occupied']:
            mo = metrics_result['occupied']
            f.write("=== OCCUPIED (target > 0) ===\n")
            f.write(f"Samples: {mo['sample_count']}\n")
            f.write(f"MAE:   {mo['mae']:.4f}\nRMSE:  {mo['rmse']:.4f}\n")
            f.write(f"R²:    {mo['r2']:.4f}\nMedAE: {mo['medae']:.4f}\n")
            f.write(f"Acc±1: {mo['acc_pm1']:.2f}%\nAcc±2: {mo['acc_pm2']:.2f}%\n")
            f.write(f"Exact: {mo['exact_match']} ({mo['exact_match_pct']:.2f}%)\n\n")
        if metrics_result['empty']:
            me = metrics_result['empty']
            f.write("=== EMPTY (target == 0) ===\n")
            f.write(f"Samples: {me['sample_count']}\nMAE:   {me['mae']:.4f}\n")
            f.write(f"Acc±1: {me['acc_pm1']:.2f}%\n\n")
        if metrics_result['per_count']:
            f.write("=== PER-COUNT ===\n")
            f.write(f"{'Count':<7} {'N':<7} {'MAE':<10} {'RMSE':<10} {'Acc±1':<10}\n")
            f.write("-" * 45 + "\n")
            for c in sorted(metrics_result['per_count'].keys()):
                pc = metrics_result['per_count'][c]
                f.write(f"{c:<7} {pc['count']:<7} {pc['mae']:<10.4f} "
                        f"{pc['rmse']:<10.4f} {pc['acc_pm1']:<10.1f}%\n")
            f.write("\n")
        f.write("\n" + analysis_text)
    logging.info("Summary saved: %s", sf)


##############################################################################
#  MAIN
##############################################################################

def main():
    print("\n" + "=" * 100)
    print("INDEPENDENT TRAINING — SELECTED ROWS")
    print(f"Device: {device}  |  Seed: {SEED}  |  Epochs: {NUM_EPOCHS}  |  LR: {LEARNING_RATE}")
    print("Author: Azadshokrollahi")
    print("=" * 100)

    # ─── Load configuration from rooms.yaml + config.py ─────────────────────
    rooms = load_rooms()
    model_types = build_model_configs()
    base_root = RESULTS_ROOT / PRETRAIN_DIR_NAME
    base_root.mkdir(parents=True, exist_ok=True)
    # ────────────────────────────────────────────────────────────────────────

    all_results = []

    for room_id, rc in rooms.items():
        for variant_name, features in rc['feature_variants'].items():

            result_folder = base_root / room_id / variant_name
            result_folder.mkdir(parents=True, exist_ok=True)

            print(f"\n{'=' * 100}")
            print(f"{rc['name']} | {variant_name} | features={features}")
            print(f"{'=' * 100}")

            if not os.path.exists(rc['path']):
                print(f"  ❌ Not found: {rc['path']}")
                continue

            log_file = result_folder / "training_log.txt"
            for h in logging.root.handlers[:]:
                logging.root.removeHandler(h)
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[logging.FileHandler(log_file, mode='a'),
                          logging.StreamHandler(sys.stdout)],
                force=True)

            for model_name, model_info in model_types.items():
                for ws in WINDOW_SIZES:
                    config_name = f"{variant_name}_{model_name}_ws{ws}"
                    config_folder = result_folder / config_name
                    config_folder.mkdir(parents=True, exist_ok=True)

                    print(f"\n  --- {config_name} ---")
                    logging.info("=== %s | %s | %s | ws=%d ===",
                                 rc['name'], variant_name, model_name, ws)

                    try:
                        (train_loader, val_loader, test_loader, X_test_t,
                         input_size, df_full, split_info, _,
                         scaler, resolved_features) = load_and_preprocess_data(
                            file_path=rc['path'], window_size=ws,
                            features_to_use=features,
                            selected_rows=rc['selected_rows'],
                            split_scenario=rc['scenario'],
                            room_name=f"{rc['name']} | {variant_name}")

                        joblib.dump(scaler, config_folder / f"scaler_{config_name}.pkl")

                        reset_seed(SEED)
                        model = model_info['class'](input_size=input_size,
                                                    **model_info['params']).to(device)

                        tl, tm, vl, vm = train_model(
                            model, train_loader, val_loader,
                            num_epochs=NUM_EPOCHS,
                            lr=LEARNING_RATE,
                            save_dir=str(config_folder),
                            config_name=config_name)

                        metrics_result, all_true, all_preds = evaluate_model(model, test_loader)

                        plot_regression_results(
                            all_true, all_preds,
                            str(config_folder / f"regression_plot_ws{ws}.png"),
                            title=f"{rc['name']} | {config_name}")

                        plot_training_curves(
                            tl, vl, tm, vm,
                            str(config_folder / f"training_curves_ws{ws}.png"),
                            title=f"{rc['name']} | {config_name}")

                        out_f = config_folder / f"predicted_test_data_ws{ws}.csv"
                        predict_and_save(model, X_test_t, df_full, split_info, ws, str(out_f))
                        analysis = analyze_prediction_results(str(out_f))
                        write_summary_file(str(config_folder), ws, metrics_result, analysis, variant_name)

                        m_all = metrics_result['all']
                        row = {
                            'room': room_id, 'variant': variant_name, 'model': model_name,
                            'window_size': ws, 'features': ', '.join(resolved_features),
                            'scenario': rc['scenario'],
                            'test_mae': m_all['mae'], 'test_rmse': m_all['rmse'],
                            'test_r2': m_all['r2'], 'test_medae': m_all['medae'],
                            'test_acc_pm1': m_all['acc_pm1'], 'test_acc_pm2': m_all['acc_pm2'],
                            'test_samples': m_all['sample_count'],
                        }
                        if metrics_result['occupied']:
                            mo = metrics_result['occupied']
                            row.update({
                                'occ_mae': mo['mae'], 'occ_rmse': mo['rmse'],
                                'occ_r2': mo['r2'], 'occ_medae': mo['medae'],
                                'occ_acc_pm1': mo['acc_pm1'], 'occ_acc_pm2': mo['acc_pm2'],
                                'occ_exact_pct': mo['exact_match_pct'],
                                'occ_samples': mo['sample_count'],
                            })
                        if metrics_result['empty']:
                            me = metrics_result['empty']
                            row.update({
                                'empty_mae': me['mae'], 'empty_acc_pm1': me['acc_pm1'],
                                'empty_samples': me['sample_count'],
                            })
                        all_results.append(row)

                        logging.info("✓ %s completed: MAE=%.4f, Acc±1=%.2f%%",
                                     config_name, m_all['mae'], m_all['acc_pm1'])

                    except Exception as e:
                        print(f"  ✗ FAILED {config_name}: {e}")
                        logging.error("✗ FAILED %s: %s", config_name, str(e))
                        traceback.print_exc()

    # ───────── FINAL SUMMARY ─────────
    print("\n\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_csv = base_root / "all_independent_selected_rows_results.csv"
        results_df.to_csv(results_csv, index=False)
        print(f"\nResults CSV: {results_csv}")

        print(f"\n{'Room':<10} {'Variant':<15} {'Model':<12} {'WS':<5} "
              f"{'MAE':<8} {'RMSE':<8} {'R²':<8} {'MedAE':<8} "
              f"{'Acc±1':<8} {'Acc±2':<8} {'OccMAE':<8} {'OccAcc±1':<10}")
        print("-" * 110)
        for _, r in results_df.iterrows():
            occ_mae = r.get('occ_mae', float('nan'))
            occ_acc1 = r.get('occ_acc_pm1', float('nan'))
            print(f"{r['room']:<10} {r['variant']:<15} {r['model']:<12} "
                  f"{r['window_size']:<5} "
                  f"{r['test_mae']:<8.4f} {r['test_rmse']:<8.4f} "
                  f"{r['test_r2']:<8.4f} {r['test_medae']:<8.4f} "
                  f"{r['test_acc_pm1']:<8.2f} {r['test_acc_pm2']:<8.2f} "
                  f"{occ_mae:<8.4f} {occ_acc1:<10.2f}")

        print(f"\n{'Room':<10} {'Best Config':<35} {'MAE':<8} {'Acc±1':<8} {'OccMAE':<8}")
        print("-" * 75)
        for room_id, grp in results_df.groupby('room'):
            best = grp.loc[grp['test_mae'].idxmin()]
            cfg = f"{best['variant']}_{best['model']}_ws{best['window_size']}"
            occ_mae = best.get('occ_mae', float('nan'))
            print(f"{room_id:<10} {cfg:<35} {best['test_mae']:<8.4f} "
                  f"{best['test_acc_pm1']:<8.2f} {occ_mae:<8.4f}")

        print(f"\n{'Room':<10} {'Best Config (by Occ MAE)':<35} "
              f"{'OccMAE':<8} {'OccAcc±1':<10} {'AllMAE':<8}")
        print("-" * 75)
        for room_id, grp in results_df.groupby('room'):
            if 'occ_mae' in grp.columns and grp['occ_mae'].notna().any():
                best = grp.loc[grp['occ_mae'].idxmin()]
                cfg = f"{best['variant']}_{best['model']}_ws{best['window_size']}"
                print(f"{room_id:<10} {cfg:<35} {best['occ_mae']:<8.4f} "
                      f"{best.get('occ_acc_pm1', float('nan')):<10.2f} "
                      f"{best['test_mae']:<8.4f}")

    print("\n" + "=" * 100)
    print("✓ ALL INDEPENDENT TRAINING COMPLETED!")
    print("User: Azadshokrollahi")
    print("=" * 100)


if __name__ == "__main__":
    main()