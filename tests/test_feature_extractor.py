import numpy as np
import pytest

from faster_whisper.feature_extractor import FeatureExtractor


def test_stft_requires_return_complex():
    fe = FeatureExtractor()
    x = np.random.randn(100).astype(np.float32)
    with pytest.raises(ValueError):
        fe.stft(x, n_fft=64)


def test_stft_complex_input_infers_return_complex():
    fe = FeatureExtractor()
    x = (np.random.randn(100) + 1j * np.random.randn(100)).astype(np.complex64)
    out = fe.stft(x, n_fft=64)
    assert np.iscomplexobj(out)


def test_stft_handles_window_none():
    fe = FeatureExtractor()
    x = np.random.randn(128).astype(np.float32)
    out = fe.stft(x, n_fft=64, hop_length=32, window=None, return_complex=True)
    assert out.shape == (33, 5)


def test_cached_hann_window_reused(monkeypatch):
    fe = FeatureExtractor()
    captured_windows = []

    original_stft = fe.stft

    def wrapped(input_array, *args, **kwargs):
        captured_windows.append(kwargs.get("window"))
        return original_stft(input_array, *args, **kwargs)

    monkeypatch.setattr(fe, "stft", wrapped)

    x = np.random.randn(fe.n_fft).astype(np.float32)
    fe(x)
    fe(x)

    assert captured_windows[0] is fe._hann_window
    assert captured_windows[1] is fe._hann_window
