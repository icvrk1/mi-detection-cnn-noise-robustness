import zipfile
from pathlib import Path

import numpy as np
import scipy.signal
import wfdb

_RECORD_MAP: dict[str, str] = {
    "baseline_wander": "bw",
    "muscle_artifact": "ma",
    "electrode_motion": "em",
}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ZIP_PATH = _PROJECT_ROOT / "data" / "raw" / "mit-bih-noise-stress-test-database-1.0.0.zip"
_NSTDB_DIR = _PROJECT_ROOT / "data" / "raw" / "nstdb"
_NATIVE_FS = 360
_TARGET_FS = 100

_CACHE: dict[str, np.ndarray] = {}


def _extract_nstdb() -> None:
    """Extract bw/ma/em records from the ZIP into data/raw/nstdb/."""
    _NSTDB_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(_ZIP_PATH) as zf:
        for entry in zf.namelist():
            stem = Path(entry).stem    # e.g. "bw"  from "dir/bw.dat"
            name = Path(entry).name    # e.g. "bw.dat"
            if stem in ("bw", "ma", "em") and name:  # skip bare directory entries
                dest = _NSTDB_DIR / name
                if not dest.exists():
                    dest.write_bytes(zf.read(entry))


def load_nstdb_noise(noise_type: str) -> np.ndarray:
    """
    Return channel-0 of the requested NSTDB record, resampled to 100 Hz.

    Parameters
    ----------
    noise_type : 'baseline_wander' | 'muscle_artifact' | 'electrode_motion'

    Returns
    -------
    1D float64 numpy array (~650 000 samples at 100 Hz).
    """
    if noise_type not in _RECORD_MAP:
        raise ValueError(
            f"Unknown NSTDB noise type {noise_type!r}. "
            f"Valid options: {sorted(_RECORD_MAP)}"
        )

    if noise_type in _CACHE:
        return _CACHE[noise_type]

    if not _NSTDB_DIR.exists() or not any(_NSTDB_DIR.iterdir()):
        _extract_nstdb()

    record_name = _RECORD_MAP[noise_type]
    record = wfdb.rdrecord(str(_NSTDB_DIR / record_name))
    channel0 = record.p_signal[:, 0].astype(np.float64)

    n_out = int(round(len(channel0) * _TARGET_FS / _NATIVE_FS))
    resampled = scipy.signal.resample(channel0, n_out)

    _CACHE[noise_type] = resampled
    return resampled
