"""
Dataset helper for EEG visual-stimulus studies.

Purpose:
- Keep useful open EEG dataset links in one place.
- Optionally download a small file to validate network + data folder flow.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.request import urlretrieve


DATASETS = [
    {
        "name": "PhysioNet EEG Motor Movement/Imagery",
        "type": "EEG",
        "url": "https://physionet.org/content/eegmmidb/1.0.0/",
        "notes": "Baseline EEG benchmark dataset with multiple tasks.",
    },
    {
        "name": "BCI Competition (official archive)",
        "type": "EEG/BCI",
        "url": "http://www.bbci.de/competition/",
        "notes": "Classic BCI challenge datasets, useful for model baselines.",
    },
    {
        "name": "OpenNeuro",
        "type": "EEG/MEG/fMRI",
        "url": "https://openneuro.org/",
        "notes": "Large open repository; many EEG studies include event markers.",
    },
]


def export_catalog(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(DATASETS, indent=2), encoding="utf-8")
    print(f"Catalog saved: {output_path}")


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    urlretrieve(url, str(destination))
    print(f"Saved to: {destination}")


def print_catalog() -> None:
    print("Available EEG resources")
    print("-" * 60)
    for idx, item in enumerate(DATASETS, start=1):
        print(f"{idx}. {item['name']}")
        print(f"   URL   : {item['url']}")
        print(f"   Notes : {item['notes']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEG dataset discovery helper")
    parser.add_argument("--list", action="store_true", help="Print dataset catalog")
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Export catalog as JSON (example: ../data/eeg_dataset_catalog.json)",
    )
    parser.add_argument(
        "--download-url",
        type=str,
        default=None,
        help="Optional direct file URL for quick connectivity/download test",
    )
    parser.add_argument(
        "--download-to",
        type=str,
        default=None,
        help="Destination path for --download-url",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list or (not args.export_json and not args.download_url):
        print_catalog()

    if args.export_json:
        export_catalog(Path(args.export_json))

    if args.download_url:
        if not args.download_to:
            raise ValueError("--download-to is required when using --download-url")
        download_file(args.download_url, Path(args.download_to))


if __name__ == "__main__":
    main()
