"""
Kopiranje figura i tabela u thesis/ folder i opcionalno u LaTeX projekat.

Cilj 1 (uvijek):   bachelorPublish/thesis/{figures,tables}/
Cilj 2 (opciono):  LaTeX projekat (npr. ETF template) ako je dat --latex-dir

Po defaultu kopira sve PDF i PNG figure iz outputs/figures/thesis/ i
sve .tex tabele iz outputs/tables/thesis/. Sa --only-agg kopira samo
artefakte koji u imenu sadrze "_agg" (regenerisane mean+/-std verzije).

Primjer:
  python scripts/09_sync_thesis_assets.py
  python scripts/09_sync_thesis_assets.py --latex-dir "D:/ETF_latex_template_BoEE (1)/ETF_latex_template_BoEE ver1"
  python scripts/09_sync_thesis_assets.py --only-agg \
      --latex-dir "D:/ETF_latex_template_BoEE (1)/ETF_latex_template_BoEE ver1"
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT         = Path(__file__).resolve().parent.parent
FIGURES_SRC  = ROOT / "outputs" / "figures" / "thesis"
TABLES_SRC   = ROOT / "outputs" / "tables"  / "thesis"
THESIS_FIG   = ROOT / "thesis"  / "figures"
THESIS_TAB   = ROOT / "thesis"  / "tables"


def copy_file(src: Path, dst_dir: Path) -> bool:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync thesis assets.")
    parser.add_argument("--latex-dir", type=str, default=None,
                        help="Putanja do LaTeX projekta (cilj ima poddirektorij 'figures').")
    parser.add_argument("--only-agg", action="store_true",
                        help="Kopiraj samo artefakte ciji naziv sadrzi '_agg' (multi-seed verzije).")
    parser.add_argument("--include-png", action="store_true",
                        help="Kopiraj i PNG verzije figura (uz PDF). Default: samo PDF u thesis/, "
                             "ali u LaTeX projekat se po defaultu kopiraju i PNG.")
    args = parser.parse_args()

    fig_glob = "*_agg*.pdf" if args.only_agg else "*.pdf"
    tab_glob = "*_agg.tex"  if args.only_agg else "*.tex"
    png_glob = "*_agg*.png" if args.only_agg else "*.png"

    # agg_macros.tex je centralni fajl sa LaTeX macros - uvijek sinhroniziraj uz tabele
    extra_tab_files = ["agg_macros.tex"]

    copied = 0
    print(f"Izvor figura: {FIGURES_SRC}")
    print(f"Izvor tabela: {TABLES_SRC}")
    print(f"Cilj 1: {THESIS_FIG}, {THESIS_TAB}")

    # 1) thesis/ folder
    for src in sorted(FIGURES_SRC.glob(fig_glob)):
        copy_file(src, THESIS_FIG); copied += 1
        print(f"  fig PDF -> thesis/figures/{src.name}")
    if args.include_png:
        for src in sorted(FIGURES_SRC.glob(png_glob)):
            copy_file(src, THESIS_FIG); copied += 1
            print(f"  fig PNG -> thesis/figures/{src.name}")
    for src in sorted(TABLES_SRC.glob(tab_glob)):
        copy_file(src, THESIS_TAB); copied += 1
        print(f"  tab     -> thesis/tables/{src.name}")
    for fname in extra_tab_files:
        src = TABLES_SRC / fname
        if src.exists():
            copy_file(src, THESIS_TAB); copied += 1
            print(f"  tab     -> thesis/tables/{fname}")

    # 2) LaTeX projekat (po izboru)
    if args.latex_dir:
        latex_dir  = Path(args.latex_dir)
        if not latex_dir.exists():
            print(f"  ! LaTeX dir ne postoji: {latex_dir}")
        else:
            print(f"Cilj 2: {latex_dir}")
            latex_fig = latex_dir / "figures"
            for src in sorted(FIGURES_SRC.glob(fig_glob)):
                copy_file(src, latex_fig); copied += 1
                print(f"  fig PDF -> {latex_fig.name}/{src.name}")
            for src in sorted(FIGURES_SRC.glob(png_glob)):
                copy_file(src, latex_fig); copied += 1
                print(f"  fig PNG -> {latex_fig.name}/{src.name}")
            # tabele - opciono, neki LaTeX projekti drze tabele inline; svejedno kopiramo u tables/
            latex_tab = latex_dir / "tables"
            for src in sorted(TABLES_SRC.glob(tab_glob)):
                copy_file(src, latex_tab); copied += 1
                print(f"  tab     -> {latex_tab.name}/{src.name}")
            for fname in extra_tab_files:
                src = TABLES_SRC / fname
                if src.exists():
                    copy_file(src, latex_tab); copied += 1
                    print(f"  tab     -> {latex_tab.name}/{fname}")

    print(f"\nUkupno kopirano: {copied} fajlova")


if __name__ == "__main__":
    main()
