import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Task 1 — synthetic.py
# ---------------------------------------------------------------------------

def test_generate_powerline_shape_and_frequency():
    from ekg_mi.noise.synthetic import generate_powerline
    # Use fs=360 for frequency-peak test: 50 Hz at exactly fs/2 (100 Hz) maps all
    # samples to sin(πk)=0 (Nyquist degenerate case).  360 Hz avoids this.
    sig360 = generate_powerline(length=3600, fs=360.0, frequency=50.0)
    assert sig360.shape == (3600,)
    freqs = np.fft.rfftfreq(3600, d=1 / 360.0)
    mag = np.abs(np.fft.rfft(sig360))
    peak_freq = freqs[np.argmax(mag)]
    assert abs(peak_freq - 50.0) < 1.0

    # At the project's own fs=100 the default phase offset must give non-zero power.
    sig100 = generate_powerline(length=1000, fs=100.0, frequency=50.0)
    assert sig100.shape == (1000,)
    assert np.mean(sig100 ** 2) > 0.1, "powerline at 100 Hz must have non-zero power"


def test_generate_gaussian_shape_and_distribution():
    from ekg_mi.noise.synthetic import generate_gaussian
    sig = generate_gaussian(length=10000)
    assert sig.shape == (10000,)
    assert abs(np.mean(sig)) < 0.05
    assert abs(np.std(sig) - 1.0) < 0.05


# ---------------------------------------------------------------------------
# Task 2 — nstdb.py
# ---------------------------------------------------------------------------

def test_load_nstdb_noise_shape_and_type():
    from ekg_mi.noise.nstdb import load_nstdb_noise
    for noise_type in ("baseline_wander", "muscle_artifact", "electrode_motion"):
        arr = load_nstdb_noise(noise_type)
        assert arr.ndim == 1, f"{noise_type}: expected 1D array"
        assert arr.dtype == np.float64, f"{noise_type}: expected float64"
        assert len(arr) > 100_000, f"{noise_type}: expected >100k samples after resample"


def test_load_nstdb_noise_invalid():
    from ekg_mi.noise.nstdb import load_nstdb_noise
    with pytest.raises(ValueError):
        load_nstdb_noise("unknown_type")


def test_load_nstdb_noise_caching():
    from ekg_mi.noise.nstdb import load_nstdb_noise, _CACHE
    _CACHE.clear()
    a = load_nstdb_noise("baseline_wander")
    b = load_nstdb_noise("baseline_wander")
    assert a is b


# ---------------------------------------------------------------------------
# Task 3 — injection.py
# ---------------------------------------------------------------------------

import neurokit2 as nk

NOISE_TYPES = [
    "baseline_wander",
    "muscle_artifact",
    "electrode_motion",
    "powerline",
    "gaussian",
]
SNR_VALUES = [-6, 0, 6, 12, 18, 24]
TOLERANCE_DB = 0.5


@pytest.fixture(scope="module")
def ecg_1d():
    return nk.ecg_simulate(duration=10, sampling_rate=100, heart_rate=70)


@pytest.mark.parametrize("noise_type", NOISE_TYPES)
@pytest.mark.parametrize("snr_db", SNR_VALUES)
def test_snr_accuracy(ecg_1d, noise_type, snr_db):
    from ekg_mi.noise.injection import add_noise
    noisy = add_noise(ecg_1d, noise_type, snr_db)
    noise_actual = noisy - ecg_1d
    P_signal = np.mean(ecg_1d ** 2)
    P_noise = np.mean(noise_actual ** 2)
    snr_measured = 10 * np.log10(P_signal / P_noise)
    assert abs(snr_measured - snr_db) < TOLERANCE_DB, (
        f"{noise_type} @ {snr_db} dB: measured {snr_measured:.4f} dB "
        f"(error {snr_measured - snr_db:+.4f} dB)"
    )


@pytest.mark.parametrize("noise_type", NOISE_TYPES)
def test_output_shape_1d(ecg_1d, noise_type):
    from ekg_mi.noise.injection import add_noise
    noisy = add_noise(ecg_1d, noise_type, snr_db=0)
    assert noisy.shape == ecg_1d.shape


def test_output_shape_multichannel():
    from ekg_mi.noise.injection import add_noise
    mc = np.random.randn(3, 1000)
    noisy = add_noise(mc, "gaussian", snr_db=6)
    assert noisy.shape == (3, 1000)


def test_invalid_noise_type(ecg_1d):
    from ekg_mi.noise.injection import add_noise
    with pytest.raises(ValueError, match="Unknown noise type"):
        add_noise(ecg_1d, "bad_type", snr_db=0)


def test_zero_signal_returns_unchanged():
    from ekg_mi.noise.injection import add_noise
    silent = np.zeros(1000)
    result = add_noise(silent, "gaussian", snr_db=0)
    np.testing.assert_array_equal(result, silent)
