from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

input_path = PROJECT_ROOT / "data" / "public_data" / "Occupancy_Estimation.csv"
output_path = PROJECT_ROOT / "data" / "public_data" / "public_room_5min.csv"


# =========================================================
# Config
# =========================================================

ASSUMED_IMPEDANCE = 1.0


# =========================================================
# Read dataset
# =========================================================

df = pd.read_csv(
    input_path,
    usecols=[
        "Date", "Time",
        "S5_CO2", "S6_PIR", "S7_PIR", "Room_Occupancy_Count",
        "S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
        "S1_Light", "S2_Light", "S3_Light", "S4_Light",
        "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
    ],
)

df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
df = df.set_index("datetime").sort_index()


# =========================================================
# Align start time
# =========================================================

first_time = df.index[0]
aligned_start = first_time.replace(minute=50, second=0, microsecond=0)

if aligned_start < first_time:
    aligned_start += pd.Timedelta(minutes=5)

df = df[df.index >= aligned_start]


# =========================================================
# Sound preparation
# =========================================================

sound_cols = ["S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound"]

for col in sound_cols:
    df[f"{col}_sq"] = df[col] ** 2

for col in sound_cols:
    power_watts = (df[col] ** 2) / ASSUMED_IMPEDANCE
    power_mw = power_watts * 1000
    df[f"{col}_dbm"] = 10 * np.log10(np.clip(power_mw, 1e-12, None))


# =========================================================
# Resample to 5 minutes
# =========================================================

df_5min = df.resample("5min", origin=aligned_start).agg(
    {
        "S5_CO2": "mean",
        "S6_PIR": "sum",
        "S7_PIR": "sum",

        "S1_Temp": "mean",
        "S2_Temp": "mean",
        "S3_Temp": "mean",
        "S4_Temp": "mean",

        "S1_Light": "mean",
        "S2_Light": "mean",
        "S3_Light": "mean",
        "S4_Light": "mean",

        "S1_Sound": "mean",
        "S2_Sound": "mean",
        "S3_Sound": "mean",
        "S4_Sound": "mean",

        "S1_Sound_sq": "mean",
        "S2_Sound_sq": "mean",
        "S3_Sound_sq": "mean",
        "S4_Sound_sq": "mean",

        "S1_Sound_dbm": "mean",
        "S2_Sound_dbm": "mean",
        "S3_Sound_dbm": "mean",
        "S4_Sound_dbm": "mean",

        "Room_Occupancy_Count": lambda x: sp_stats.mode(
            x,
            keepdims=True
        )[0][0],
    }
)


# =========================================================
# Feature engineering
# =========================================================

df_5min["motion"] = df_5min["S6_PIR"] + df_5min["S7_PIR"]

df_5min["temperature"] = df_5min[
    ["S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp"]
].mean(axis=1)

df_5min["light"] = df_5min[
    ["S1_Light", "S2_Light", "S3_Light", "S4_Light"]
].mean(axis=1)

df_5min["sound_mean_voltage"] = df_5min[
    ["S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound"]
].mean(axis=1)

df_5min["S1_Sound_rms"] = np.sqrt(df_5min["S1_Sound_sq"])
df_5min["S2_Sound_rms"] = np.sqrt(df_5min["S2_Sound_sq"])
df_5min["S3_Sound_rms"] = np.sqrt(df_5min["S3_Sound_sq"])
df_5min["S4_Sound_rms"] = np.sqrt(df_5min["S4_Sound_sq"])

df_5min["sound_rms_voltage"] = df_5min[
    ["S1_Sound_rms", "S2_Sound_rms", "S3_Sound_rms", "S4_Sound_rms"]
].mean(axis=1)

df_5min["sound_relative_dbm"] = df_5min[
    ["S1_Sound_dbm", "S2_Sound_dbm", "S3_Sound_dbm", "S4_Sound_dbm"]
].mean(axis=1)


# =========================================================
# Final dataset
# =========================================================

df_final = df_5min[
    [
        "S5_CO2",
        "motion",
        "temperature",
        "light",
        "sound_mean_voltage",
        "sound_rms_voltage",
        "sound_relative_dbm",
        "Room_Occupancy_Count",
    ]
]

df_final = df_final.rename(
    columns={
        "S5_CO2": "co2",
        "Room_Occupancy_Count": "target",
    }
)

df_final = df_final.dropna()


# =========================================================
# Save
# =========================================================

df_final.to_csv(output_path)

print("Saved file to:")
print(output_path)
print("Included sound_mean_voltage, sound_rms_voltage, and sound_relative_dbm")
