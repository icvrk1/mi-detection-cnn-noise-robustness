"""
Multi-seed sweep za V1 i V3 modele.

Za svaki seed u opsegu [start_seed, start_seed+num_seeds) izvrsi se:
  - V1 trening (02_train.py)              -> outputs/runs/v1/seed_<S>/best_model_v1.pt
  - V1 clean evaluacija (03_evaluate_clean.py)
  - V1 noisy evaluacija (05_evaluate_noisy.py)
  - V3 trening (02_train_v3.py)           -> outputs/runs/v3/seed_<S>/best_model_v3.pt
  - V3 clean evaluacija (03_evaluate_clean_v3.py)
  - V3 noisy evaluacija (05_evaluate_noisy_v3.py)

Sve evaluacije koriste prag 0.5 (deterministicki, bez naknadnog tuninga praga,
kako bi rezultati bili uporedivi izmedju runova).

Primjer:
  python scripts/10_run_multi_seed.py --num-seeds 2                # smoke test
  python scripts/10_run_multi_seed.py --num-seeds 10 --start-seed 42
  python scripts/10_run_multi_seed.py --num-seeds 10 --variants v1 # samo V1
  python scripts/10_run_multi_seed.py --num-seeds 10 --skip-noisy  # bez noisy eval
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR    = SCRIPTS_DIR.parent
RUNS_DIR    = ROOT_DIR / "outputs" / "runs"

PY = sys.executable

VARIANTS = {
    "v1": {
        "train":      SCRIPTS_DIR / "02_train.py",
        "eval_clean": SCRIPTS_DIR / "03_evaluate_clean.py",
        "eval_noisy": SCRIPTS_DIR / "05_evaluate_noisy.py",
        "tag":        "v1",
    },
    "v3": {
        "train":      SCRIPTS_DIR / "02_train_v3.py",
        "eval_clean": SCRIPTS_DIR / "03_evaluate_clean_v3.py",
        "eval_noisy": SCRIPTS_DIR / "05_evaluate_noisy_v3.py",
        "tag":        "v3",
    },
}


def run(cmd: list[str], log_path: Path | None = None) -> None:
    """Izvrsi komandu, propusti stdout, pri gresci stampa puni log."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as lf:
            res = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT_DIR))
    else:
        res = subprocess.run(cmd, cwd=str(ROOT_DIR))
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"  ! GRESKA (exit={res.returncode}, {dt:.1f}s)")
        if log_path is not None and log_path.exists():
            print(f"  log: {log_path}")
            print(log_path.read_text(encoding="utf-8", errors="replace")[-4000:])
        raise SystemExit(res.returncode)
    print(f"  ok ({dt:.1f}s)")


def run_seed(variant: str, seed: int, skip_noisy: bool, threshold: float | None) -> None:
    spec = VARIANTS[variant]
    out_dir = RUNS_DIR / variant / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag       = spec["tag"]
    model_pt  = out_dir / f"best_model_{tag}.pt"
    clean_js  = out_dir / f"eval_clean_{tag}.json"
    noisy_js  = out_dir / f"eval_noisy_{tag}.json"
    noisy_csv = out_dir / f"results_noisy_{tag}.csv"

    print(f"\n[{variant} seed={seed}]")
    print(f"  out: {out_dir}")

    # 1) Trening
    if model_pt.exists():
        print(f"  (preskacem trening - {model_pt.name} vec postoji)")
    else:
        train_cmd = [
            PY, str(spec["train"]),
            "--seed", str(seed),
            "--output-dir", str(out_dir),
            "--tag", tag,
        ]
        run(train_cmd, out_dir / f"train_{tag}.log")

    # 2) Clean eval (uvijek prag 0.5 da budu uporedivi izmedju runova)
    if clean_js.exists():
        print(f"  (preskacem clean eval - {clean_js.name} vec postoji)")
    else:
        eval_cmd = [
            PY, str(spec["eval_clean"]),
            "--model-path", str(model_pt),
            "--output-json", str(clean_js),
            "--no-figures",
        ]
        if variant == "v3":
            t = 0.5 if threshold is None else threshold
            eval_cmd += ["--threshold", str(t)]
        run(eval_cmd, out_dir / f"eval_clean_{tag}.log")

    # 3) Noisy eval
    if skip_noisy:
        print("  (preskacem noisy eval)")
        return
    if noisy_js.exists():
        print(f"  (preskacem noisy eval - {noisy_js.name} vec postoji)")
        return
    eval_cmd = [
        PY, str(spec["eval_noisy"]),
        "--model-path", str(model_pt),
        "--output-json", str(noisy_js),
        "--output-csv",  str(noisy_csv),
    ]
    if variant == "v3":
        t = 0.5 if threshold is None else threshold
        eval_cmd += ["--threshold", str(t)]
    run(eval_cmd, out_dir / f"eval_noisy_{tag}.log")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed sweep za V1 i V3.")
    parser.add_argument("--num-seeds",  type=int, default=10)
    parser.add_argument("--start-seed", type=int, default=42)
    parser.add_argument("--variants",   nargs="+", default=["v1", "v3"],
                        choices=["v1", "v3"])
    parser.add_argument("--skip-noisy", action="store_true",
                        help="Preskoci noisy evaluaciju (samo trening + clean).")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Prag odluke za eval. Default 0.5 (uporedivo).")
    parser.add_argument("--seeds",      type=int, nargs="+", default=None,
                        help="Eksplicitna lista seedova; nadjacava --num-seeds/--start-seed.")
    args = parser.parse_args()

    if args.seeds is not None:
        seeds = list(args.seeds)
    else:
        seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))

    print("=" * 78)
    print(f"Multi-seed sweep")
    print(f"  varijante: {args.variants}")
    print(f"  seedovi:   {seeds}")
    print(f"  threshold: {args.threshold}")
    print(f"  skip noisy: {args.skip_noisy}")
    print(f"  izlaz:     {RUNS_DIR}")
    print("=" * 78)

    t0 = time.time()
    for seed in seeds:
        for variant in args.variants:
            run_seed(variant, seed, args.skip_noisy, args.threshold)

    dt = time.time() - t0
    print("\n" + "=" * 78)
    print(f"Sweep zavrsen za {dt/60:.1f} min ({dt/3600:.2f}h)")
    print("Sljedeći korak: python scripts/11_aggregate_runs.py")


if __name__ == "__main__":
    main()
