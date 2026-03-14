"""EEG processing module with simulation and real-file workflow support.

Supported input modes:
- Simulated P300/noise trials (default demo)
- CSV trial tables (one row = one sample, grouped by trial id)
- EDF recordings (optional, requires mne)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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
        return self._safe_filtfilt(b, a, signal)

    def notch_filter(
        self,
        signal: np.ndarray,
        freq: float = 50.0,
        quality_factor: float = 30.0,
    ) -> np.ndarray:
        """Remove powerline artifact using notch filter."""
        w0 = freq / (0.5 * self.sampling_rate)
        b, a = iirnotch(w0=w0, Q=quality_factor)
        return self._safe_filtfilt(b, a, signal)

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

    @staticmethod
    def _safe_filtfilt(b: np.ndarray, a: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """Apply filtfilt only when signal length is long enough.

        For very short windows, return signal unchanged to avoid scipy pad errors.
        """
        if signal.ndim == 1:
            n_samples = signal.shape[0]
        else:
            n_samples = signal.shape[-1]

        padlen = 3 * max(len(a), len(b))
        if n_samples <= padlen:
            return signal

        return filtfilt(b, a, signal, axis=-1)


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
        n_classes = len(np.unique(y))
        if len(X) <= n_classes:
            raise ValueError("Number of samples must be greater than number of classes")
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted")
        return self.model.predict(X)

    def score_cv(self, X: np.ndarray, y: np.ndarray, folds: int = 5) -> float:
        if len(X) < 2:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        if len(counts) < 2:
            return 0.0

        max_valid_folds = int(min(np.min(counts), len(X)))
        folds = min(folds, max_valid_folds)
        if folds < 2:
            return 0.0

        try:
            scores = cross_val_score(self.model, X, y, cv=folds)
            return float(np.mean(scores))
        except ValueError:
            return 0.0


def load_csv_trials(
    csv_path: str,
    channel_cols: Optional[List[str]] = None,
    trial_col: str = "trial",
    label_col: str = "label",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load trial-wise EEG data from CSV.

    Expected CSV format (recommended):
    - Columns: trial,label,ch1,ch2,...,chN
    - Each row is one time sample.
    - `trial` groups samples of a trial; label is constant within each trial.

    Returns:
        epochs: (n_trials, n_channels, n_samples_per_trial)
        labels: (n_trials,) or None
    """
    df = pd.read_csv(csv_path)

    if channel_cols is None:
        excluded = {trial_col, label_col, "time", "timestamp", "sample"}
        channel_cols = [c for c in df.columns if c not in excluded]

    if not channel_cols:
        raise ValueError("No channel columns found in CSV")

    has_trial = trial_col in df.columns
    has_label = label_col in df.columns

    epochs: List[np.ndarray] = []
    labels: List[int] = []

    if has_trial:
        grouped = df.groupby(trial_col, sort=True)
        for _, group in grouped:
            epoch = group[channel_cols].to_numpy(dtype=np.float32).T
            epochs.append(epoch)
            if has_label:
                labels.append(int(group[label_col].iloc[0]))
    else:
        epoch = df[channel_cols].to_numpy(dtype=np.float32).T
        epochs.append(epoch)
        if has_label:
            labels.append(int(df[label_col].iloc[0]))

    epochs_np = np.asarray(epochs, dtype=np.float32)
    labels_np = np.asarray(labels, dtype=np.int64) if labels else None
    return epochs_np, labels_np


def load_edf_signal(edf_path: str) -> Tuple[np.ndarray, int]:
    """Load EDF as continuous signal (n_channels, n_samples).

    Requires `mne`. Install with: `pip install mne`.
    """
    try:
        import mne
    except ImportError as exc:
        raise ImportError("EDF support requires mne. Install with `pip install mne`.") from exc

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    signal = raw.get_data().astype(np.float32)
    sampling_rate = int(raw.info["sfreq"])
    return signal, sampling_rate


def make_fixed_windows(
    signal: np.ndarray,
    sampling_rate: int,
    window_sec: float,
    step_sec: Optional[float] = None,
) -> np.ndarray:
    """Create fixed windows from continuous EEG signal.

    Returns windows as (n_windows, n_channels, n_window_samples).
    """
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    if step_sec is None:
        step_sec = window_sec

    n_channels, n_samples = signal.shape
    win = max(1, int(window_sec * sampling_rate))
    step = max(1, int(step_sec * sampling_rate))

    windows = []
    for start in range(0, n_samples - win + 1, step):
        windows.append(signal[:, start : start + win])

    if not windows:
        return np.empty((0, n_channels, win), dtype=np.float32)
    return np.asarray(windows, dtype=np.float32)


def run_simulation_demo(sampling_rate: int = 250) -> None:
    """Run the synthetic P300 workflow to sanity-check pipeline."""
    from bci_simulator import BCIDataGenerator

    np.random.seed(42)
    print("EEG Processor quick test (simulated)")
    print("-" * 40)

    gen = BCIDataGenerator(sampling_rate=sampling_rate, n_channels=8)
    processor = EEGProcessor(sampling_rate=sampling_rate)

    n_trials = 40
    epochs = []
    labels = []

    for i in range(n_trials):
        noise = gen.generate_noise(duration=1.0)
        if i % 2 == 0:
            signal = noise + gen.generate_p300_event(event_time=0.32, amplitude=6.0)
            labels.append(1)
        else:
            signal = noise
            labels.append(0)

        cleaned = processor.preprocess(signal, low_freq=0.5, high_freq=20.0, notch_freq=50.0)
        ep = processor.epoch_signal(cleaned, [0], EpochConfig(tmin=0.0, tmax=0.8))
        if len(ep) > 0:
            epochs.append(ep[0])

    epochs_np = np.asarray(epochs, dtype=np.float32)
    y = np.asarray(labels[: len(epochs_np)], dtype=np.int64)

    X = processor.build_feature_matrix(epochs_np)
    clf = EEGClassifier()
    cv_acc = clf.score_cv(X, y, folds=5)
    clf.fit(X, y)
    preds = clf.predict(X[:5])

    print(f"Epochs shape: {epochs_np.shape}")
    print(f"Feature matrix: {X.shape}")
    print(f"Cross-val accuracy: {cv_acc:.3f}")
    print(f"Sample preds: {preds.tolist()}")


def run_file_workflow(
    input_path: str,
    input_format: str,
    sampling_rate: int,
    trial_col: str,
    label_col: str,
    channel_cols: Optional[List[str]],
    window_sec: float,
    step_sec: Optional[float],
) -> None:
    """Load EEG from file, preprocess, extract features, and optionally classify."""
    processor = EEGProcessor(sampling_rate=sampling_rate)
    path = Path(input_path)

    if input_format == "auto":
        suffix = path.suffix.lower()
        if suffix == ".csv":
            input_format = "csv"
        elif suffix == ".edf":
            input_format = "edf"
        else:
            raise ValueError("Unsupported file extension for auto mode")

    labels = None
    if input_format == "csv":
        epochs, labels = load_csv_trials(
            str(path),
            channel_cols=channel_cols,
            trial_col=trial_col,
            label_col=label_col,
        )
    elif input_format == "edf":
        signal, detected_fs = load_edf_signal(str(path))
        processor = EEGProcessor(sampling_rate=detected_fs)
        epochs = make_fixed_windows(signal, detected_fs, window_sec=window_sec, step_sec=step_sec)
    else:
        raise ValueError("input_format must be one of: auto,csv,edf")

    if len(epochs) == 0:
        raise RuntimeError("No epochs found in input data")

    cleaned_epochs = []
    for ep in epochs:
        cleaned = processor.preprocess(ep, low_freq=0.5, high_freq=40.0, notch_freq=50.0)
        cleaned_epochs.append(cleaned)

    cleaned_np = np.asarray(cleaned_epochs, dtype=np.float32)
    X = processor.build_feature_matrix(cleaned_np)

    print("EEG file workflow")
    print("-" * 40)
    print(f"Input format: {input_format}")
    print(f"Epochs: {cleaned_np.shape}")
    print(f"Feature matrix: {X.shape}")

    if labels is not None and len(labels) == len(X) and len(np.unique(labels)) >= 2:
        clf = EEGClassifier()
        cv_folds = min(5, len(labels))
        cv_acc = clf.score_cv(X, labels, folds=cv_folds)
        n_classes = len(np.unique(labels))
        pred_preview = []
        if len(labels) > n_classes:
            clf.fit(X, labels)
            pred_preview = clf.predict(X[: min(5, len(X))]).tolist()
        print(f"Labels shape: {labels.shape}")
        print(f"Classes: {np.unique(labels).tolist()}")
        print(f"Cross-val accuracy: {cv_acc:.3f}")
        print(f"Preview preds: {pred_preview}")
    else:
        print("Labels not suitable for classification. Feature extraction completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEG processing for Neuro-Visual Assistant")
    parser.add_argument("--input", type=str, default=None, help="Path to CSV/EDF EEG file")
    parser.add_argument(
        "--input-format",
        type=str,
        default="auto",
        choices=["auto", "csv", "edf"],
        help="File format selector",
    )
    parser.add_argument("--sampling-rate", type=int, default=250, help="Sampling rate for CSV mode")
    parser.add_argument("--trial-col", type=str, default="trial", help="CSV trial id column")
    parser.add_argument("--label-col", type=str, default="label", help="CSV label column")
    parser.add_argument(
        "--channel-cols",
        type=str,
        default=None,
        help="Comma-separated channel columns for CSV (optional)",
    )
    parser.add_argument("--window-sec", type=float, default=0.8, help="Window size for EDF segmentation")
    parser.add_argument("--step-sec", type=float, default=None, help="Step size for EDF segmentation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    channel_cols = args.channel_cols.split(",") if args.channel_cols else None

    if args.input:
        run_file_workflow(
            input_path=args.input,
            input_format=args.input_format,
            sampling_rate=args.sampling_rate,
            trial_col=args.trial_col,
            label_col=args.label_col,
            channel_cols=channel_cols,
            window_sec=args.window_sec,
            step_sec=args.step_sec,
        )
    else:
        run_simulation_demo(sampling_rate=args.sampling_rate)
