"""Download the NASA C-MAPSS FD001 files and write them as the CSVs this project uses.

Usage (from the repository root):

    python data/download_data.py

Creates data/PM_train.csv, data/PM_test.csv and data/PM_truth.csv.

Source: NASA Prognostics Center of Excellence, Turbofan Engine Degradation
Simulation Data Set (C-MAPSS), subset FD001. The original text files are fetched
from a public mirror of the NASA archive, then converted to the project's CSV
layout (header row, comma separated). Please cite:
    A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation
    Data Set", NASA Ames Prognostics Data Repository, NASA Ames Research Center.
"""

import csv
import os
import sys
import urllib.request

MIRROR_BASE = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master"

COLUMNS = (
    ["id", "cycle", "setting1", "setting2", "setting3"]
    + [f"s{i}" for i in range(1, 22)]
)

EXPECTED_ROWS = {"PM_train.csv": 20631, "PM_test.csv": 13096, "PM_truth.csv": 100}


def download(url, dest):
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)


def convert_series(src, dest):
    """Space-separated NASA text file -> CSV with a header row."""
    with open(src) as fin, open(dest, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(COLUMNS)
        for line in fin:
            values = line.split()
            if values:
                writer.writerow(values)


def convert_truth(src, dest):
    """RUL_FD001.txt (one RUL per line) -> CSV with id,cycle columns."""
    with open(src) as fin, open(dest, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["id", "cycle"])
        for engine_id, line in enumerate(fin, start=1):
            value = line.strip()
            if value:
                writer.writerow([engine_id, value])


def count_rows(path):
    with open(path) as f:
        return sum(1 for _ in f) - 1  # minus the header


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    jobs = [
        ("train_FD001.txt", "PM_train.csv", convert_series),
        ("test_FD001.txt", "PM_test.csv", convert_series),
        ("RUL_FD001.txt", "PM_truth.csv", convert_truth),
    ]

    for raw_name, csv_name, convert in jobs:
        raw_path = os.path.join(raw_dir, raw_name)
        csv_path = os.path.join(data_dir, csv_name)
        if not os.path.exists(raw_path):
            download(f"{MIRROR_BASE}/{raw_name}", raw_path)
        convert(raw_path, csv_path)
        rows = count_rows(csv_path)
        if rows != EXPECTED_ROWS[csv_name]:
            raise RuntimeError(
                f"{csv_name}: expected {EXPECTED_ROWS[csv_name]} rows, got {rows}. "
                "The downloaded file may be corrupt - delete data/raw and try again."
            )
        print(f"Wrote {csv_path} ({rows} rows)")

    print("Done. Raw text files are kept in data/raw/ for reference.")


if __name__ == "__main__":
    sys.exit(main())
