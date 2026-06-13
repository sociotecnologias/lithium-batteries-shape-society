"""
Run the full pipeline: preparation -> analysis figures.

Usage:
    python main.py
"""
from __future__ import annotations

import sys

from preparation import prepare
from analysis import run_all
from config import SCOPUS_CSV, FIG_DIR


def main():
    if not SCOPUS_CSV.exists():
        print(f"ERROR: Scopus CSV not found at {SCOPUS_CSV}", file=sys.stderr)
        print("Place the file 'scopus_20_8_25.csv' inside the 0_raw/ folder.",
              file=sys.stderr)
        sys.exit(1)

    print("[1/2] Preparing data...")
    procdata, procdata_long = prepare(write_outputs=True)
    print(f"  procdata_final:      {procdata.shape}")
    print(f"  procdata_long_final: {procdata_long.shape}")

    print("[2/2] Generating figures...")
    run_all(procdata, procdata_long)
    print(f"Done. Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
