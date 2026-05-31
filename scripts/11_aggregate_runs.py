"""
Agregacija multi-seed runova za V1 i V3.

Cita:
  outputs/runs/{v1,v3}/seed_*/eval_clean_{v1,v3}.json
  outputs/runs/{v1,v3}/seed_*/eval_noisy_{v1,v3}.json
  outputs/runs/{v1,v3}/seed_*/training_history_{v1,v3}.json

Pise:
  outputs/reports/aggregate_v1.json
  outputs/reports/aggregate_v3.json
  outputs/reports/aggregate_comparison.json   # V1 vs V3 sa p-vrijednostima

Format aggregate_{v1,v3}.json:
{
  "n_runs": 10,
  "seeds":  [42, 43, ...],
  "training": {
      "n_epochs":      {"mean": .., "std": .., "min": .., "max": .., "values": [..]},
      "best_epoch":    {...},
      "best_val_loss": {...},
      "best_val_f1":   {...}
  },
  "clean": {
      "accuracy":    {"mean": .., "std": .., "min": .., "max": .., "values": [..]},
      "precision":   {...}, "recall": {...}, "specificity": {...},
      "f1": {...}, "auc_roc": {...}, "auc_pr": {...},
      "tp": {...}, "tn": {...}, "fp": {...}, "fn": {...}
  },
  "noisy": {
      "gaussian": {
          "snr_-6": {"f1": {"mean": .., "std": .., ...}, "auc_roc": {...}, ...},
          ...
      },
      ...
  }
}

aggregate_comparison.json:
{
  "clean": {
      "f1":      {"v1_mean": .., "v1_std": .., "v3_mean": .., "v3_std": .., "delta_mean": .., "t_p": .., "wilcoxon_p": .., "v1_values": [..], "v3_values": [..]},
      ...
  },
  "noisy": {
      "gaussian": {
          "snr_-6": {"f1": {...}, "auc_roc": {...}}, ...
      },
      ...
  }
}
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

ROOT       = Path(__file__).resolve().parent.parent
RUNS_DIR   = ROOT / "outputs" / "runs"
REPORTS    = ROOT / "outputs" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

CLEAN_METRICS = ["accuracy", "precision", "recall", "specificity",
                 "f1", "auc_roc", "auc_pr"]
CM_FIELDS     = ["tp", "tn", "fp", "fn"]


def list_seed_dirs(variant: str) -> list[Path]:
    base = RUNS_DIR / variant
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir()
                   if p.is_dir() and p.name.startswith("seed_")],
                  key=lambda p: int(p.name.split("_")[1]))


def summarize(values: Iterable[float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "values": []}
    return {
        "mean":   float(np.mean(arr)),
        "std":    float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "values": arr.tolist(),
    }


def aggregate_variant(variant: str) -> dict | None:
    seed_dirs = list_seed_dirs(variant)
    if not seed_dirs:
        print(f"[{variant}] nema runova u {RUNS_DIR / variant}")
        return None

    tag = variant
    seeds: list[int] = []

    # Training history
    training_keys = ["n_epochs_run", "best_epoch", "best_val_loss"]
    training_vals: dict[str, list[float]] = {k: [] for k in training_keys}
    training_vals["best_val_f1"] = []

    # Clean
    clean_vals: dict[str, list[float]] = {m: [] for m in CLEAN_METRICS}
    for f in CM_FIELDS:
        clean_vals[f] = []

    # Noisy: {noise_type: {snr_key: {metric: [values]}}}
    noisy_vals: dict[str, dict[str, dict[str, list[float]]]] = {}

    for sd in seed_dirs:
        seed = int(sd.name.split("_")[1])
        hist = sd / f"training_history_{tag}.json"
        clean_p = sd / f"eval_clean_{tag}.json"
        noisy_p = sd / f"eval_noisy_{tag}.json"

        if not clean_p.exists():
            print(f"[{variant} seed={seed}] preskacem - nema {clean_p.name}")
            continue
        seeds.append(seed)

        # training
        if hist.exists():
            with open(hist) as fh:
                h = json.load(fh)
            n_epochs = h.get("n_epochs_run", len(h.get("train_loss", [])))
            training_vals["n_epochs_run"].append(n_epochs)
            training_vals["best_epoch"].append(h.get("best_epoch", float("nan")))
            training_vals["best_val_loss"].append(h.get("best_val_loss", float("nan")))
            vf1_list = h.get("val_f1", [])
            training_vals["best_val_f1"].append(max(vf1_list) if vf1_list else float("nan"))

        # clean
        with open(clean_p) as fh:
            c = json.load(fh)
        for m in CLEAN_METRICS:
            clean_vals[m].append(float(c["metrics"][m]))
        for f in CM_FIELDS:
            clean_vals[f].append(float(c["confusion_matrix"][f]))

        # noisy
        if noisy_p.exists():
            with open(noisy_p) as fh:
                n = json.load(fh)
            for noise_type, by_snr in n.items():
                noisy_vals.setdefault(noise_type, {})
                for snr_key, entry in by_snr.items():
                    bucket = noisy_vals[noise_type].setdefault(snr_key, {})
                    for m in CLEAN_METRICS:
                        bucket.setdefault(m, []).append(float(entry[m]))
                    cm = entry.get("confusion_matrix")
                    if cm:
                        bucket.setdefault("tn", []).append(float(cm[0][0]))
                        bucket.setdefault("fp", []).append(float(cm[0][1]))
                        bucket.setdefault("fn", []).append(float(cm[1][0]))
                        bucket.setdefault("tp", []).append(float(cm[1][1]))

    out = {
        "variant": variant,
        "n_runs":  len(seeds),
        "seeds":   seeds,
        "training": {k: summarize(v) for k, v in training_vals.items()},
        "clean":    {m: summarize(v) for m, v in clean_vals.items()},
        "noisy":    {
            nt: {snr: {m: summarize(v) for m, v in metrics.items()}
                 for snr, metrics in by_snr.items()}
            for nt, by_snr in noisy_vals.items()
        },
    }
    return out


def paired_compare(v1: list[float], v3: list[float]) -> dict:
    """Upareni t-test + Wilcoxon. Vraca NaN-ove ako uzorak premali ili invariant."""
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v3, dtype=float)
    n = min(a.size, b.size)
    if n == 0:
        return {"n": 0, "delta_mean": None, "t_stat": None, "t_p": None,
                "wilcoxon_stat": None, "wilcoxon_p": None}
    a = a[:n]; b = b[:n]
    delta = b - a
    result = {
        "n":          n,
        "delta_mean": float(np.mean(delta)),
        "delta_std":  float(np.std(delta, ddof=1)) if n > 1 else 0.0,
    }
    if HAS_SCIPY and n >= 2:
        try:
            ts, tp = stats.ttest_rel(b, a)
            result["t_stat"] = float(ts) if not math.isnan(ts) else None
            result["t_p"]    = float(tp) if not math.isnan(tp) else None
        except Exception:
            result["t_stat"] = None; result["t_p"] = None
        try:
            if np.all(delta == 0):
                result["wilcoxon_stat"] = None
                result["wilcoxon_p"]    = 1.0
            else:
                ws, wp = stats.wilcoxon(b, a, zero_method="wilcox")
                result["wilcoxon_stat"] = float(ws)
                result["wilcoxon_p"]    = float(wp)
        except Exception:
            result["wilcoxon_stat"] = None
            result["wilcoxon_p"]    = None
    else:
        result["t_stat"] = None; result["t_p"] = None
        result["wilcoxon_stat"] = None; result["wilcoxon_p"] = None
    return result


def build_comparison(agg_v1: dict, agg_v3: dict) -> dict:
    cmp_out: dict = {"n_v1": agg_v1["n_runs"], "n_v3": agg_v3["n_runs"], "clean": {}, "noisy": {}}

    # Clean: spojiti po metrici
    for m in CLEAN_METRICS:
        v1 = agg_v1["clean"][m]["values"]
        v3 = agg_v3["clean"][m]["values"]
        pc = paired_compare(v1, v3)
        cmp_out["clean"][m] = {
            "v1_mean": agg_v1["clean"][m]["mean"],
            "v1_std":  agg_v1["clean"][m]["std"],
            "v3_mean": agg_v3["clean"][m]["mean"],
            "v3_std":  agg_v3["clean"][m]["std"],
            **pc,
            "v1_values": v1, "v3_values": v3,
        }

    # Noisy: za svaki (noise_type, snr_key, metric)
    common_noise = sorted(set(agg_v1["noisy"]) & set(agg_v3["noisy"]))
    for nt in common_noise:
        cmp_out["noisy"][nt] = {}
        common_snr = sorted(set(agg_v1["noisy"][nt]) & set(agg_v3["noisy"][nt]),
                            key=lambda k: int(k.split("_")[1]))
        for snr in common_snr:
            cmp_out["noisy"][nt][snr] = {}
            common_metrics = sorted(set(agg_v1["noisy"][nt][snr]) &
                                    set(agg_v3["noisy"][nt][snr]))
            for m in common_metrics:
                if m in CM_FIELDS:
                    continue  # CM polja agregiramo kao broj, ne testiramo statisticki
                v1 = agg_v1["noisy"][nt][snr][m]["values"]
                v3 = agg_v3["noisy"][nt][snr][m]["values"]
                pc = paired_compare(v1, v3)
                cmp_out["noisy"][nt][snr][m] = {
                    "v1_mean": agg_v1["noisy"][nt][snr][m]["mean"],
                    "v1_std":  agg_v1["noisy"][nt][snr][m]["std"],
                    "v3_mean": agg_v3["noisy"][nt][snr][m]["mean"],
                    "v3_std":  agg_v3["noisy"][nt][snr][m]["std"],
                    **pc,
                }
    return cmp_out


def fmt(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "  -   "
    return f"{x:.{digits}f}"


def print_summary(agg: dict) -> None:
    v = agg["variant"]
    print(f"\n[{v}] n_runs={agg['n_runs']}  seedovi={agg['seeds']}")
    t = agg["training"]
    print(f"  Trening epoha: {fmt(t['n_epochs_run']['mean'], 1)} +/- {fmt(t['n_epochs_run']['std'], 1)} "
          f"(min={fmt(t['n_epochs_run']['min'], 1)}, max={fmt(t['n_epochs_run']['max'], 1)})")
    print(f"  Best epoch:    {fmt(t['best_epoch']['mean'], 1)} +/- {fmt(t['best_epoch']['std'], 1)}")
    print(f"  Best val loss: {fmt(t['best_val_loss']['mean'])} +/- {fmt(t['best_val_loss']['std'])}")
    print(f"  Best val F1:   {fmt(t['best_val_f1']['mean'])} +/- {fmt(t['best_val_f1']['std'])}")
    c = agg["clean"]
    print(f"  CLEAN  F1   = {fmt(c['f1']['mean'])} +/- {fmt(c['f1']['std'])}")
    print(f"         AUC  = {fmt(c['auc_roc']['mean'])} +/- {fmt(c['auc_roc']['std'])}")
    print(f"         Acc  = {fmt(c['accuracy']['mean'])} +/- {fmt(c['accuracy']['std'])}")


def main() -> None:
    if not HAS_SCIPY:
        print("UPOZORENJE: scipy nije dostupan, p-vrijednosti se nece racunati.")

    agg_v1 = aggregate_variant("v1")
    agg_v3 = aggregate_variant("v3")

    if agg_v1 is None and agg_v3 is None:
        print("Nema runova - pokreni prvo scripts/10_run_multi_seed.py")
        return

    if agg_v1 is not None:
        out = REPORTS / "aggregate_v1.json"
        with open(out, "w") as fh:
            json.dump(agg_v1, fh, indent=2)
        print(f"Spremljeno: {out}")
        print_summary(agg_v1)

    if agg_v3 is not None:
        out = REPORTS / "aggregate_v3.json"
        with open(out, "w") as fh:
            json.dump(agg_v3, fh, indent=2)
        print(f"Spremljeno: {out}")
        print_summary(agg_v3)

    if agg_v1 is not None and agg_v3 is not None:
        cmp_out = build_comparison(agg_v1, agg_v3)
        out = REPORTS / "aggregate_comparison.json"
        with open(out, "w") as fh:
            json.dump(cmp_out, fh, indent=2)
        print(f"\nSpremljeno: {out}")
        # Kratki pregled clean comparison-a
        print("\n[CLEAN] V1 vs V3 (p-vrijednosti uparenog t-testa):")
        for m in CLEAN_METRICS:
            row = cmp_out["clean"][m]
            print(f"  {m:12s} V1={fmt(row['v1_mean'])}+/-{fmt(row['v1_std'])}  "
                  f"V3={fmt(row['v3_mean'])}+/-{fmt(row['v3_std'])}  "
                  f"d={fmt(row['delta_mean'])}  p={fmt(row.get('t_p'))}")


if __name__ == "__main__":
    main()
