"""
KMM verification — proves the float64 cast fixes the
"TypeError: buffer format not supported" error from cvxopt.

Run:  python src/check.py
"""

import sys
import traceback
import numpy as np

SEED = 42


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ─── 1. Environment ─────────────────────────────────────────────────────────
section("1. ENVIRONMENT")
try:
    import adapt
    try:
        from importlib.metadata import version
        print(f"  adapt version : {version('adapt')}")
    except Exception:
        print("  adapt imported (version unknown)")
except Exception as e:
    print(f"  ❌ adapt not importable: {e}")
    sys.exit(1)

try:
    import cvxopt
    print(f"  cvxopt version: {cvxopt.__version__}")
except Exception:
    print("  ❌ cvxopt NOT installed")

from adapt.instance_based import KMM


# ─── 2. Synthetic data (float32, like your real pipeline) ───────────────────
section("2. SYNTHETIC DATA")
rng = np.random.RandomState(SEED)
X_src = rng.randn(500, 8).astype(np.float32)          # float32 → triggers the bug
X_tgt = rng.randn(300, 8).astype(np.float32) + 0.5
print(f"  X_src: {X_src.shape}, dtype={X_src.dtype}")
print(f"  X_tgt: {X_tgt.shape}, dtype={X_tgt.dtype}")


def run_kmm(Xs, Xt, label):
    """Run the exact KMM call from compute_kmm_weights and report."""
    print(f"\n  ── {label} (src dtype={Xs.dtype}) ──")
    n_s = len(Xs)
    try:
        kmm = KMM(estimator=None, Xt=Xt, kernel="rbf", B=10,
                  eps=(np.sqrt(n_s) - 1) / np.sqrt(n_s),
                  max_size=2000, verbose=0, random_state=SEED)
        w = np.maximum(kmm.fit_weights(Xs, Xt), 0)
        uniform = np.allclose(w, 1.0)
        print(f"    ✅ SUCCESS — mean={w.mean():.4f}, std={w.std():.4f}, "
              f"min={w.min():.4f}, max={w.max():.4f}")
        print("    ⚠ weights are uniform (≈1.0)" if uniform
              else "    → non-uniform weights = KMM genuinely working")
        return True
    except Exception as e:
        print(f"    ❌ FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


# ─── 3. BEFORE: current code (float32) — expected to FAIL ───────────────────
section("3. BEFORE FIX  (float32 — current behavior)")
ok_before = run_kmm(X_src, X_tgt, "float32 call")


# ─── 4. AFTER: float64 cast (the fix) — expected to SUCCEED ──────────────────
section("4. AFTER FIX  (float64 cast)")
Xs64 = np.ascontiguousarray(X_src, dtype=np.float64)
Xt64 = np.ascontiguousarray(X_tgt, dtype=np.float64)
ok_after = run_kmm(Xs64, Xt64, "float64 call")


# ─── 5. Verdict ─────────────────────────────────────────────────────────────
section("5. VERDICT")
if not ok_before and ok_after:
    print("  ✅ CONFIRMED: float32 fails, float64 works.")
    print("     → Cast X_src/X_tgt to float64 inside compute_kmm_weights().")
elif ok_before and ok_after:
    print("  ℹ Both worked — float32 isn't the problem on this ADAPT build.")
elif not ok_after:
    print("  ❌ float64 also failed — different root cause; read the traceback above.")