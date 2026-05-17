"""
PTB-XL data loader: extraction, metadata loading, label mapping, signal loading.
"""
from __future__ import annotations

import ast
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

_ZIP_NAME = (
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
)
_EXTRACTED_DIR = (
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)


def ensure_extracted(raw_path: Path) -> Path:
    """
    Extract the PTB-XL zip archive into raw_path if not already done.
    Returns the path to the extracted dataset root directory.
    """
    target = raw_path / _EXTRACTED_DIR
    if target.exists():
        return target

    zip_path = raw_path / _ZIP_NAME
    if not zip_path.exists():
        raise FileNotFoundError(
            f"PTB-XL archive not found at {zip_path}. "
            "Place the zip file in the data/raw/ directory."
        )

    print(f"Extracting {zip_path.name} to {raw_path} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw_path)
    print(f"Extraction complete -> {target}")
    return target


def load_ptbxl_metadata(ptbxl_path: Path) -> pd.DataFrame:
    """
    Load ptbxl_database.csv and parse the scp_codes column from string to dict.
    """
    csv_path = ptbxl_path / "ptbxl_database.csv"
    df = pd.read_csv(csv_path)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def load_scp_statements(ptbxl_path: Path) -> pd.DataFrame:
    """
    Load scp_statements.csv, which maps SCP codes to diagnostic classes.
    Index is the SCP code string.
    """
    return pd.read_csv(ptbxl_path / "scp_statements.csv", index_col=0)


def map_to_binary_label(
    scp_codes: dict,
    scp_statements: pd.DataFrame,
) -> int:
    """
    Map an scp_codes dict to a binary label.

    Returns
    -------
    1   pure MI record (only MI diagnostic class present)
    0   pure normal record (only NORM diagnostic class present)
    -1  ambiguous - excluded from training
    """
    # keep only codes that are diagnostic
    diag = scp_statements[scp_statements["diagnostic"] == 1.0]

    classes: set[str] = set()
    for code in scp_codes:
        if code in diag.index:
            cls = diag.loc[code, "diagnostic_class"]
            if pd.notna(cls):
                classes.add(cls)

    # MI present without any NORM code -> label as MI
    if "MI" in classes and "NORM" not in classes:
        return 1
    # only NORM diagnostic class present -> label as normal
    if classes == {"NORM"}:
        return 0
    return -1


def load_signals(
    metadata: pd.DataFrame,
    ptbxl_path: Path,
    sampling_rate: int = 100,
) -> np.ndarray:
    """
    Load ECG signals for all records in metadata using wfdb.

    Parameters
    ----------
    metadata      : DataFrame with filename_lr / filename_hr columns
    ptbxl_path    : root directory of the extracted dataset
    sampling_rate : 100 for LR version, 500 for HR version

    Returns
    -------
    np.ndarray of shape (N, 1000, 12) for 100 Hz 10-second records, float32
    """
    col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    n = len(metadata)

    # Pre-alokacija sprjecava dupliranje memorije iz liste u np.array
    out = np.empty((n, 1000, 12), dtype=np.float32)
    for i, (_, row) in enumerate(tqdm(metadata.iterrows(), total=n, desc="Loading signals")):
        signal, _ = wfdb.rdsamp(str(ptbxl_path / row[col]))
        out[i] = signal

    return out
