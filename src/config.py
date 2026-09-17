"""
Central configuration for hyperparameters and paths.
Room/dataset info is in configs/rooms.yaml (loaded via load_rooms()).
"""
import os
from pathlib import Path

# =========================================================
# PATHS
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = Path(os.getenv("DATA_ROOT",    PROJECT_ROOT / "data"))
RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", PROJECT_ROOT / "results"))
ROOMS_YAML   = PROJECT_ROOT / "configs" / "rooms.yaml"

# Sub-folder name for pretraining results
PRETRAIN_DIR_NAME = "pretrain_selected_rows_independent"

# =========================================================
# REPRODUCIBILITY
# =========================================================
SEED = 42

# =========================================================
# TRAINING HYPERPARAMETERS
# =========================================================
WINDOW_SIZES   = [12]
NUM_EPOCHS     = 50
LEARNING_RATE  = 0.001
BATCH_SIZE     = 32

# =========================================================
# MODEL SETTINGS
# =========================================================
RNN_HIDDEN_SIZE  = 64
RNN_NUM_LAYERS   = 2

LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS  = 2

TRANSFORMER_D_MODEL         = 64
TRANSFORMER_NHEAD           = 4
TRANSFORMER_NUM_LAYERS      = 2
TRANSFORMER_DIM_FEEDFORWARD = 128
TRANSFORMER_DROPOUT         = 0.1


# =========================================================
# Helpers
# =========================================================
def load_rooms(yaml_path: Path = ROOMS_YAML) -> dict:
    """Load room definitions from configs/rooms.yaml."""
    import yaml
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    rooms = cfg["rooms"]
    # Resolve paths relative to project root
    for rid, rc in rooms.items():
        rc["path"] = str(PROJECT_ROOT / rc["path"])
        # normalize selected_rows to tuple or None
        sr = rc.get("selected_rows")
        rc["selected_rows"] = tuple(sr) if sr else None
    return rooms


def model_configs() -> dict:
    """Return the 3 model definitions using config.py hyperparams."""
    # Import here to avoid circular imports at module load
    from main_regression_pretrain import RNNRegressor, LSTMRegressor, TransformerRegressor
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
