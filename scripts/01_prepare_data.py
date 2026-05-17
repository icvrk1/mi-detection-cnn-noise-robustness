"""
PTB-XL priprema podataka:
  - Ekstrakcija arhive
  - Ucitavanje metapodataka i SCP taksonomije
  - Mapiranje dijagnosticki kodova na binarne labele (0=NORM, 1=MI, -1=iskljuceno)
  - Ucitavanje i predobrada signala za svaki split posebno (stedljivo memorijsko)
  - Podjela po strat_fold: trening/validacija/test
  - Cuvanje NPZ fajlova u obliku (N, 12, 1000)
  - Provjera curenja pacijenata
  - Izvjestaj o distribuciji
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ekg_mi.data.loader import (
    ensure_extracted,
    load_ptbxl_metadata,
    load_scp_statements,
    load_signals,
    map_to_binary_label,
)
from ekg_mi.data.preprocessing import preprocess_dataset

CONFIG_PATH = Path("configs/config.yaml")
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

RAW_PATH       = Path(cfg["paths"]["raw"])
PROCESSED_PATH = Path(cfg["paths"]["processed"])
REPORTS_PATH   = Path(cfg["paths"]["reports"])
FS             = cfg["preprocessing"]["sampling_rate"]

TRAIN_FOLDS = list(range(1, 9))
VAL_FOLD    = 9
TEST_FOLD   = 10

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
REPORTS_PATH.mkdir(parents=True, exist_ok=True)

t0 = time.time()

#  Korak 1: Ekstrakcija 
print("=" * 60)
print("Korak 1: Ekstrakcija PTB-XL arhive ...")
ptbxl_path = ensure_extracted(RAW_PATH)
print(f"  Korijen: {ptbxl_path}")

#  Korak 2: Metapodaci 
print("\nKorak 2: Ucitavanje metapodataka ...")
metadata  = load_ptbxl_metadata(ptbxl_path)
scp_stmts = load_scp_statements(ptbxl_path)
print(f"  Ukupno zapisa: {len(metadata)}")

#  Korak 3: Binarne labele 
print("\nKorak 3: Racunanje binarnih labela ...")
metadata["label"] = metadata["scp_codes"].apply(
    lambda codes: map_to_binary_label(codes, scp_stmts)
)
n_total    = len(metadata)
n_mi       = int((metadata["label"] == 1).sum())
n_norm     = int((metadata["label"] == 0).sum())
n_excluded = int((metadata["label"] == -1).sum())
print(f"  Ukupno   : {n_total}")
print(f"  MI (1)   : {n_mi}")
print(f"  NORM (0) : {n_norm}")
print(f"  Iskljuceno (-1): {n_excluded}")

#  Korak 4: Filtriranje 
valid = metadata[metadata["label"] != -1].copy().reset_index(drop=True)
print(f"\n  Validnih zapisa: {len(valid)}")

labels      = valid["label"].to_numpy(dtype=np.int64)
patient_ids = valid["patient_id"].to_numpy()

train_mask = valid["strat_fold"].isin(TRAIN_FOLDS).to_numpy()
val_mask   = (valid["strat_fold"] == VAL_FOLD).to_numpy()
test_mask  = (valid["strat_fold"] == TEST_FOLD).to_numpy()

splits = {
    "train":      (train_mask, PROCESSED_PATH / "train.npz"),
    "val":        (val_mask,   PROCESSED_PATH / "val.npz"),
    "test_clean": (test_mask,  PROCESSED_PATH / "test_clean.npz"),
}

# Korak 5: Ucitavanje i predobrada po splitovima 
# Svaki split se obradjuje zasebno radi stednje memorije
print("\nKorak 5: Ucitavanje i predobrada po splitovima ...")
for split_name, (mask, out_path) in splits.items():
    if out_path.exists():
        print(f"  {split_name:12s}: vec postoji, preskacemo -> {out_path}")
        continue

    sub        = valid[mask].reset_index(drop=True)
    sub_labels = labels[mask]
    sub_pids   = patient_ids[mask]

    print(f"\n  [{split_name}] Ucitavanje {len(sub)} signala ...")
    t_start = time.time()
    sigs = load_signals(sub, ptbxl_path, sampling_rate=FS)
    print(f"    Ucitano: {sigs.shape}  ({time.time()-t_start:.1f}s)")

    print(f"  [{split_name}] Predobrada ...")
    t_pp = time.time()
    sigs = preprocess_dataset(sigs, fs=FS)
    print(f"    Predobradjeno: {sigs.shape}  ({time.time()-t_pp:.1f}s)")

    # wfdb vraca (N, 1000, 12) -> transpozicija u (N, 12, 1000)
    sigs = np.transpose(sigs, (0, 2, 1))

    np.savez(out_path, signals=sigs, labels=sub_labels, patient_ids=sub_pids)

    size_mb = out_path.stat().st_size / 1024 ** 2
    n_mi_s  = int(sub_labels.sum())
    n_no_s  = int((sub_labels == 0).sum())
    print(
        f"  {split_name:12s}: {len(sub):5d} zapisa "
        f"(MI={n_mi_s}, NORM={n_no_s})  "
        f"oblik={sigs.shape}  {size_mb:.1f} MB  -> {out_path}"
    )

    # Oslobadjanje memorije prije sljedeceg splita
    del sigs

#  Korak 6: Provjera curenja pacijenata 
print("\nKorak 6: Provjera curenja pacijenata ...")
train_pids = set(patient_ids[train_mask])
val_pids   = set(patient_ids[val_mask])
test_pids  = set(patient_ids[test_mask])

assert len(train_pids & val_pids)  == 0, f"Curenje train/val: {train_pids & val_pids}"
assert len(train_pids & test_pids) == 0, f"Curenje train/test: {train_pids & test_pids}"
assert len(val_pids   & test_pids) == 0, f"Curenje val/test: {val_pids & test_pids}"
print("  Nema curenja pacijenata izmedju splitova.")

#  Korak 7: Izvjestaj 
report = {
    "total_raw"  : n_total,
    "mi"         : n_mi,
    "norm"        : n_norm,
    "excluded"   : n_excluded,
    "valid_total": int(len(valid)),
    "splits": {
        sn: {
            "n"   : int(mask.sum()),
            "mi"  : int(labels[mask].sum()),
            "norm": int((labels[mask] == 0).sum()),
        }
        for sn, (mask, _) in splits.items()
    },
}
report_path = REPORTS_PATH / "data_distribution.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  Izvjestaj spremen -> {report_path}")

#  Rezime 
total_time = time.time() - t0
print("\n" + "=" * 60)
print("REZIME")
print("=" * 60)
for split_name, (mask, out_path) in splits.items():
    if out_path.exists():
        size_mb = out_path.stat().st_size / 1024 ** 2
        n_mi_s  = int(labels[mask].sum())
        n_no_s  = int((labels[mask] == 0).sum())
        balance = n_mi_s / mask.sum() * 100
        print(
            f"  {split_name:12s}: N={mask.sum():5d}  MI={n_mi_s}  NORM={n_no_s}  "
            f"MI%={balance:.1f}%  {size_mb:.1f} MB"
        )
print(f"\n  Ukupno vrijeme: {total_time:.1f}s")
print("=" * 60)
print("Gotovo. Podaci su u data/processed/")
