"""
Kopiranje figura i tabela u thesis/ folder.
Koristi nove putanje: outputs/figures/thesis/ i outputs/tables/thesis/
"""
from __future__ import annotations

import shutil
from pathlib import Path

FIGURES_SRC = Path("outputs/figures/thesis")
TABLES_SRC  = Path("outputs/tables/thesis")
THESIS_FIG  = Path("thesis/figures")
THESIS_TAB  = Path("thesis/tables")

THESIS_FIG.mkdir(parents=True, exist_ok=True)
THESIS_TAB.mkdir(parents=True, exist_ok=True)

copied = 0
for src in sorted(FIGURES_SRC.glob("*.pdf")):
    dst = THESIS_FIG / src.name
    shutil.copy2(src, dst)
    print(f"  figura: {src.name} -> {dst}")
    copied += 1

for src in sorted(TABLES_SRC.glob("*.tex")):
    dst = THESIS_TAB / src.name
    shutil.copy2(src, dst)
    print(f"  tabela: {src.name} -> {dst}")
    copied += 1

print(f"\nUkupno kopirano: {copied} fajlova")
print(f"  Figures -> {THESIS_FIG}")
print(f"  Tables  -> {THESIS_TAB}")
