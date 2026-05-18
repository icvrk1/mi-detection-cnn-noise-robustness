# dashboard/utils/download_assets.py

from pathlib import Path
import urllib.request
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = _ROOT / "data" / "processed"
NOISY_DIR = _ROOT / "data" / "noisy"
MODEL_DIR = _ROOT / "outputs" / "models"
LOG_DIR = _ROOT / "outputs" / "logs"
REPORT_DIR = _ROOT / "outputs" / "reports"

BASE_URL = (
    "https://github.com/icvrk1/mi-detection-cnn-noise-robustness"
    "/releases/download/v1.0-assets"
)

NOISY_BASE_URL = (
    "https://github.com/icvrk1/mi-detection-cnn-noise-robustness"
    "/releases/download/v1.0-noisy-data"
)

NOISE_TYPES = [
    "gaussian",
    "baseline_wander",
    "muscle_artifact",
    "electrode_motion",
    "powerline",
]

SNR_VALUES = [-6, 0, 6, 12, 18, 24]


def _snr_to_tag(snr_db: int | float | str) -> str:
    snr = int(float(snr_db))
    if snr < 0:
        return f"-{abs(snr):02d}"
    return f"+{snr:02d}"


def _noisy_filename(noise_type: str, snr_db: int | float | str) -> str:
    return f"test_{noise_type}_snr{_snr_to_tag(snr_db)}.npz"


ASSETS = {
    # Data
    "train": {
        "path": DATA_DIR / "train.npz",
        "url": f"{BASE_URL}/train.npz",
    },
    "val": {
        "path": DATA_DIR / "val.npz",
        "url": f"{BASE_URL}/val.npz",
    },
    "test": {
        "path": DATA_DIR / "test_clean.npz",
        "url": f"{BASE_URL}/test_clean.npz",
    },

    # Best modeli
    "model_v1": {
        "path": MODEL_DIR / "best_model.pt",
        "url": f"{BASE_URL}/best_model.pt",
    },
    "model_v3": {
        "path": MODEL_DIR / "best_model_v3.pt",
        "url": f"{BASE_URL}/best_model_v3.pt",
    },

    # Training history
    "training_history_v1": {
        "path": LOG_DIR / "training_history.json",
        "url": f"{BASE_URL}/training_history.json",
    },
    "training_history_v3": {
        "path": LOG_DIR / "training_history_v3.json",
        "url": f"{BASE_URL}/training_history_v3.json",
    },

    # Clean evaluacija
    "eval_clean_v1": {
        "path": REPORT_DIR / "eval_clean.json",
        "url": f"{BASE_URL}/eval_clean.json",
    },
    "eval_clean_v3": {
        "path": REPORT_DIR / "eval_clean_v3.json",
        "url": f"{BASE_URL}/eval_clean_v3.json",
    },

    # Noisy evaluacija, samo metrike
    "eval_noisy_v1": {
        "path": REPORT_DIR / "eval_noisy.json",
        "url": f"{BASE_URL}/eval_noisy.json",
    },
    "eval_noisy_v3": {
        "path": REPORT_DIR / "eval_noisy_v3.json",
        "url": f"{BASE_URL}/eval_noisy_v3.json",
    },
}


# Dodavanje 30 noisy .npz fajlova u ASSETS
for noise_type in NOISE_TYPES:
    for snr in SNR_VALUES:
        filename = _noisy_filename(noise_type, snr)
        key = f"noisy_{noise_type}_snr{_snr_to_tag(snr)}"
        ASSETS[key] = {
            "path": NOISY_DIR / filename,
            "url": f"{NOISY_BASE_URL}/{filename}",
        }


def _download_file(url: str, path: Path) -> Path:
    """
    Preuzima fajl sa GitHub Release-a ako ne postoji lokalno.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        return path

    temp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        urllib.request.urlretrieve(url, temp_path)
        temp_path.replace(path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Neuspjelo preuzimanje fajla: {url}") from e

    return path


@st.cache_data(show_spinner=False)
def ensure_asset(asset_name: str) -> str:
    """
    Preuzima jedan asset ako ne postoji lokalno i vraća putanju kao string.
    """
    if asset_name not in ASSETS:
        raise ValueError(f"Nepoznat asset: {asset_name}")

    asset = ASSETS[asset_name]
    path = _download_file(asset["url"], asset["path"])
    return str(path)


def ensure_test_data() -> str:
    with st.spinner("Preuzimanje test skupa..."):
        return ensure_asset("test")


def ensure_train_data() -> str:
    with st.spinner("Preuzimanje trening skupa..."):
        return ensure_asset("train")


def ensure_val_data() -> str:
    with st.spinner("Preuzimanje validacijskog skupa..."):
        return ensure_asset("val")


def ensure_model_v1() -> str:
    with st.spinner("Preuzimanje V1 best modela..."):
        return ensure_asset("model_v1")


def ensure_model_v3() -> str:
    with st.spinner("Preuzimanje V3 best modela..."):
        return ensure_asset("model_v3")


def ensure_noisy_test_data(noise_type: str, snr_db: int | float | str) -> str:
    """
    Preuzima tačno jednu noisy test varijantu, npr.
    test_gaussian_snr+06.npz.
    """
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"Nepoznat tip šuma: {noise_type}")

    snr = int(float(snr_db))
    if snr not in SNR_VALUES:
        raise ValueError(f"Nepoznata SNR vrijednost: {snr_db}")

    key = f"noisy_{noise_type}_snr{_snr_to_tag(snr)}"

    with st.spinner(f"Preuzimanje noisy test skupa: {noise_type}, SNR={snr} dB..."):
        return ensure_asset(key)


def ensure_live_prediction_assets() -> dict:
    """
    Preuzima fajlove potrebne za Live Prediction stranicu.
    Ne preuzima noisy .npz jer Live Prediction može dodati šum dinamički.
    """
    return {
        "test": ensure_test_data(),
        "model_v1": ensure_model_v1(),
        "model_v3": ensure_model_v3(),
    }


def ensure_training_results() -> dict:
    """
    Preuzima fajlove potrebne za Training Results stranicu.
    """
    with st.spinner("Preuzimanje rezultata treninga..."):
        return {
            "training_history_v1": ensure_asset("training_history_v1"),
            "training_history_v3": ensure_asset("training_history_v3"),
            "eval_clean_v1": ensure_asset("eval_clean_v1"),
            "eval_clean_v3": ensure_asset("eval_clean_v3"),
        }


def ensure_noisy_results() -> dict:
    """
    Preuzima JSON rezultate evaluacije robustnosti.
    Ovo ne skida 30 noisy .npz fajlova.
    """
    with st.spinner("Preuzimanje rezultata evaluacije robustnosti..."):
        return {
            "eval_noisy_v1": ensure_asset("eval_noisy_v1"),
            "eval_noisy_v3": ensure_asset("eval_noisy_v3"),
            "eval_clean_v1": ensure_asset("eval_clean_v1"),
            "eval_clean_v3": ensure_asset("eval_clean_v3"),
        }


def ensure_dashboard_assets() -> dict:
    """
    Preuzima najvažnije fajlove za dashboard.
    Ne preuzima train.npz ni svih 30 noisy .npz fajlova jer su veliki.
    """
    return {
        **ensure_live_prediction_assets(),
        **ensure_training_results(),
        **ensure_noisy_results(),
    }