"""
Generisanje 30 zasumljenjih varijanti test skupa (5 vrsta suma x 6 SNR nivoa).
Sprema NPZ fajlove u data/noisy/ sa deterministickim seedovima.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekg_mi.noise.injection import add_noise

CONFIG_PATH = Path("configs/config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

PROCESSED_PATH = Path(cfg["paths"]["processed"])
NOISY_PATH     = Path(cfg["paths"]["noisy"])
NOISY_PATH.mkdir(parents=True, exist_ok=True)

NOISE_TYPES = cfg["noise"]["types"]
SNR_LEVELS  = cfg["noise"]["snr_db"]

print("Ucitavanje cistog test skupa ...")
test_data = np.load(PROCESSED_PATH / "test_clean.npz")
X_clean   = test_data["signals"]   # (N, 12, 1000)
labels    = test_data["labels"]
pids      = test_data["patient_ids"]
N = len(X_clean)
print(f"  Test: {X_clean.shape}  (N={N})")

t0 = time.time()
n_combos = len(NOISE_TYPES) * len(SNR_LEVELS)
print(f"\nGenerisanje {n_combos} kombinacija ...")

combo_index = 0
for noise_type in NOISE_TYPES:
    for snr_db in SNR_LEVELS:
        fname    = f"test_{noise_type}_snr{snr_db:+03d}.npz"
        out_path = NOISY_PATH / fname

        if out_path.exists():
            print(f"  Vec postoji: {fname}")
            combo_index += 1
            continue

        np.random.seed(42 + combo_index)

        t_c = time.time()
        X_noisy = np.empty_like(X_clean)
        for i in range(N):
            X_noisy[i] = add_noise(X_clean[i], noise_type=noise_type, snr_db=snr_db)
        X_noisy = X_noisy.astype(np.float32)

        np.savez(out_path, signals=X_noisy, labels=labels, patient_ids=pids)
        elapsed_c = time.time() - t_c
        print(f"  {fname}: {X_noisy.shape}  ({elapsed_c:.1f}s)")
        combo_index += 1

#  Sanity check: mjeri postignuti SNR na 3 nasumicne kombinacije 
print("\nSanity check SNR-a (3 kombinacije) ...")
all_combos = [(nt, snr) for nt in NOISE_TYPES for snr in SNR_LEVELS]
rng = np.random.default_rng(0)
check_indices = rng.choice(len(all_combos), size=3, replace=False)

for idx in sorted(check_indices):
    noise_type, snr_db = all_combos[idx]
    fname = f"test_{noise_type}_snr{snr_db:+03d}.npz"
    noisy_data = np.load(NOISY_PATH / fname)
    X_noisy = noisy_data["signals"].astype(np.float64)
    X_ref   = X_clean.astype(np.float64)

    noise_comp = X_noisy - X_ref                            # (N, 12, 1000)
    p_signal = np.mean(X_ref    ** 2, axis=(0, 2))         # (12,)
    p_noise  = np.mean(noise_comp ** 2, axis=(0, 2))       # (12,)
    valid = p_noise > 0
    measured = 10.0 * np.log10(p_signal[valid] / p_noise[valid])
    mean_snr = float(np.mean(measured))
    deviation = abs(mean_snr - snr_db)

    status = "OK" if deviation <= 0.5 else "UPOZORENJE"
    print(f"  [{status}] {noise_type:20s} @ {snr_db:+3d} dB: "
          f"izmjeren={mean_snr:.2f} dB, odstupanje={deviation:.3f} dB")

elapsed = time.time() - t0
print(f"\nZavrseno za {elapsed:.1f}s")
print(f"NPZ fajlovi u: {NOISY_PATH}")
