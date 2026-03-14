"""
EEG signal processing pipeline for Neuro-Visual Assistant.

This module provides a practical baseline for P300/SSVEP workflows:
- Band-pass filtering
- Notch filtering (power line noise removal)
- Epoch extraction from continuous EEG
- Feature extraction (time + frequency domain)
- Baseline ML training/inference with scikit-learn
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class EpochConfig:
    """Window configuration for event-locked EEG segments."""

    tmin: float = 0.0
    tmax: float = 0.8


class EEGProcessor:
    """Signal cleaning, segmentation, and feature extraction utilities."""

    def __init__(self, sampling_rate: int = 250):
        self.sampling_rate = sampling_rate

    def bandpass_filter(
        self,
        signal: np.ndarray,
        low_freq: float = 0.5,
        high_freq: float = 40.0,
        order: int = 4,
    ) -> np.ndarray:
        """Apply zero-phase Butterworth band-pass filter.

        Args:
            signal: EEG array with shape (n_channels, n_samples) or (n_samples,).
        """
        nyquist = 0.5 * self.sampling_rate
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, signal, axis=-1)

    def notch_filter(
        self,
        signal: np.ndarray,
        freq: float = 50.0,
        quality_factor: float = 30.0,
    ) -> np.ndarray:
        """Remove powerline artifact using notch filter."""
        w0 = freq / (0.5 * self.sampling_rate)
        b, a = iirnotch(w0=w0, Q=quality_factor)
        return filtfilt(b, a, signal, axis=-1)

    def preprocess(
        self,
        signal: np.ndarray,
        low_freq: float = 0.5,
        high_freq: float = 40.0,
        notch_freq: Optional[float] = 50.0,
    ) -> np.ndarray:
        """Apply notch then band-pass filters in sequence."""
        cleaned = signal
        if notch_freq is not None:
            cleaned = self.notch_filter(cleaned, freq=notch_freq)
        cleaned = self.bandpass_filter(cleaned, low_freq=low_freq, high_freq=high_freq)
        return cleaned

    def epoch_signal(
        self,
        signal: np.ndarray,
        event_samples: Sequence[int],
        config: EpochConfig = EpochConfig(),
    ) -> np.ndarray:
        """Extract fixed-size epochs around event sample indices.

        Returns:
            epochs with shape (n_epochs, n_channels, n_times)
        """
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]

        n_channels, n_samples = signal.shape
        start_offset = int(config.tmin * self.sampling_rate)
        end_offset = int(config.tmax * self.sampling_rate)

        epochs: List[np.ndarray] = []
        for event in event_samples:
            start = event + start_offset
            end = event + end_offset
            if start < 0 or end > n_samples:
                continue
            epochs.append(signal[:, start:end])

        if not epochs:
            return np.empty((0, n_channels, end_offset - start_offset))

        return np.stack(epochs, axis=0)

    def extract_features(self, epoch: np.ndarray) -> np.ndarray:
        """Extract compact features from a single epoch.

        Features per channel:
        - mean, std, max, min, peak-to-peak
        - relative band powers: theta, alpha, beta
        """
        if epoch.ndim == 1:
            epoch = epoch[np.newaxis, :]

        features: List[float] = []
        for ch_data in epoch:
            mean_val = float(np.mean(ch_data))
            std_val = float(np.std(ch_data))
            max_val = float(np.max(ch_data))
            min_val = float(np.min(ch_data))
            ptp_val = float(np.ptp(ch_data))

            freqs, psd = welch(ch_data, fs=self.sampling_rate, nperseg=min(256, len(ch_data)))
            total_power = float(np.sum(psd) + 1e-12)

            theta_power = self._band_power(freqs, psd, 4.0, 8.0) / total_power
            alpha_power = self._band_power(freqs, psd, 8.0, 13.0) / total_power
            beta_power = self._band_power(freqs, psd, 13.0, 30.0) / total_power

            features.extend(
                [
                    mean_val,
                    std_val,
                    max_val,
                    min_val,
                    ptp_val,
                    theta_power,
                    alpha_power,
                    beta_power,
                ]
            )

        return np.asarray(features, dtype=np.float32)

    def build_feature_matrix(self, epochs: np.ndarray) -> np.ndarray:
        """Convert (n_epochs, n_channels, n_times) into ML-ready matrix."""
        if len(epochs) == 0:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([self.extract_features(ep) for ep in epochs], axis=0)

    @staticmethod
    def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return 0.0
        return float(np.trapz(psd[mask], freqs[mask]))


class EEGClassifier:
    """Baseline classifier wrapper for quick P300/SSVEP experiments."""

    def __init__(self):
        self.model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError("X must be 2D feature matrix")
        if len(X) != len(y):
            raise ValueError("X and y length mismatch")
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted")
        return self.model.predict(X)

    def score_cv(self, X: np.ndarray, y: np.ndarray, folds: int = 5) -> float:
        if len(X) < folds:
            folds = max(2, len(X))
        if folds < 2:
            return 0.0
        scores = cross_val_score(self.model, X, y, cv=folds)
        return float(np.mean(scores))


if __name__ == "__main__":
    from bci_simulator import BCIDataGenerator

    np.random.seed(42)

    print("EEG Processor quick test")
    print("-" * 40)

    gen = BCIDataGenerator(sampling_rate=250, n_channels=8)
    processor = EEGProcessor(sampling_rate=250)

    n_trials = 40
    trial_len_sec = 1.0
    trial_samples = int(trial_len_sec * 250)

    epochs = []
    labels = []

    for i in range(n_trials):
        noise = gen.generate_noise(duration=trial_len_sec)
        if i % 2 == 0:
            signal = noise + gen.generate_p300_event(event_time=0.32, amplitude=6.0)
            labels.append(1)
        else:
            signal = noise
            labels.append(0)

        cleaned = processor.preprocess(signal, low_freq=0.5, high_freq=20.0, notch_freq=50.0)
        event_sample = int(0.0 * 250)
        ep = processor.epoch_signal(cleaned, [event_sample], EpochConfig(tmin=0.0, tmax=0.8))
        if len(ep) > 0:
            epochs.append(ep[0])

    epochs_np = np.asarray(epochs)
    y = np.asarray(labels[: len(epochs_np)])

    X = processor.build_feature_matrix(epochs_np)
    clf = EEGClassifier()
    cv_acc = clf.score_cv(X, y, folds=5)
    clf.fit(X, y)
    preds = clf.predict(X[:5])

    print(f"Epochs shape: {epochs_np.shape}")
    print(f"Feature matrix: {X.shape}")
    print(f"Cross-val accuracy: {cv_acc:.3f}")
    print(f"Sample preds: {preds.tolist()}")
