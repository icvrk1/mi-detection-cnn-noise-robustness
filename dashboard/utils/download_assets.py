from pathlib import Path
import urllib.request
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR   = _ROOT / "data"    / "processed"
MODEL_DIR  = _ROOT / "outputs" / "models"
LOG_DIR    = _ROOT / "outputs" / "logs"
REPORT_DIR = _ROOT / "outputs" / "reports"

BASE_URL = (
    "https://github.com/icvrk1/mi-detection-cnn-noise-robustness"
    "/releases/download/v1.0-assets"
)

ASSETS = {
    "train": {
        "path": DATA_DIR / "train.npz",
        "url":  f"{BASE_URL}/train.npz",
    },
    "val": {
        "path": DATA_DIR / "val.npz",
        "url":  f"{BASE_URL}/val.npz",
    },
    "test": {
        "path": DATA_DIR / "test_clean.npz",
        "url":  f"{BASE_URL}/test_clean.npz",
    },
    "model_v1": {
        "path": MODEL_DIR / "best_model.pt",
        "url":  f"{BASE_URL}/best_model.pt",
    },
    "model_v3": {
        "path": MODEL_DIR / "best_model_v3.pt",
        "url":  f"{BASE_URL}/best_model_v3.pt",
    },
    "training_history_v1": {
        "path": LOG_DIR / "training_history.json",
        "url":  f"{BASE_URL}/training_history.json",
    },
    "training_history_v3": {
        "path": LOG_DIR / "training_history_v3.json",
        "url":  f"{BASE_URL}/training_history_v3.json",
    },
    "eval_clean_v1": {
        "path": REPORT_DIR / "eval_clean.json",
        "url":  f"{BASE_URL}/eval_clean.json",
    },
    "eval_clean_v3": {
        "path": REPORT_DIR / "eval_clean_v3.json",
        "url":  f"{BASE_URL}/eval_clean_v3.json",
    },
    "eval_noisy_v1": {
        "path": REPORT_DIR / "eval_noisy.json",
        "url":  f"{BASE_URL}/eval_noisy.json",
    },
    "eval_noisy_v3": {
        "path": REPORT_DIR / "eval_noisy_v3.json",
        "url":  f"{BASE_URL}/eval_noisy_v3.json",
    },
}


def _download_file(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, temp)
        temp.replace(path)
    except Exception as e:
        if temp.exists():
            temp.unlink()
        raise RuntimeError(f"Neuspjelo preuzimanje: {url}") from e
    return path


@st.cache_data(show_spinner=False)
def ensure_asset(asset_name: str) -> str:
    if asset_name not in ASSETS:
        raise ValueError(f"Nepoznat asset: {asset_name}")
    asset = ASSETS[asset_name]
    path = _download_file(asset["url"], asset["path"])
    return str(path)


def ensure_test_data() -> str:
    with st.spinner("Preuzimanje test skupa..."):
        return ensure_asset("test")


def ensure_model_v1() -> str:
    with st.spinner("Preuzimanje V1 modela..."):
        return ensure_asset("model_v1")


def ensure_model_v3() -> str:
    with st.spinner("Preuzimanje V3 modela..."):
        return ensure_asset("model_v3")


def ensure_training_results() -> dict:
    with st.spinner("Preuzimanje rezultata treninga..."):
        return {
            "training_history_v1": ensure_asset("training_history_v1"),
            "training_history_v3": ensure_asset("training_history_v3"),
            "eval_clean_v1":       ensure_asset("eval_clean_v1"),
            "eval_clean_v3":       ensure_asset("eval_clean_v3"),
        }


def ensure_noisy_results() -> dict:
    with st.spinner("Preuzimanje rezultata evaluacije robustnosti..."):
        return {
            "eval_noisy_v1": ensure_asset("eval_noisy_v1"),
            "eval_noisy_v3": ensure_asset("eval_noisy_v3"),
            "eval_clean_v1": ensure_asset("eval_clean_v1"),
            "eval_clean_v3": ensure_asset("eval_clean_v3"),
        }


def ensure_dashboard_assets() -> dict:
    return {
        "test":    ensure_test_data(),
        "model_v1": ensure_model_v1(),
        "model_v3": ensure_model_v3(),
        **ensure_training_results(),
        **ensure_noisy_results(),
    }
