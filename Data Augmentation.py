"""
Data Augmentation for TC Dataset
==================================
Reads train split files (already with phase_score computed).
For each row with phase_score == 2 (multiphase FCC, disordered), generates:

  - derived_fcc:   composition of the disordered FCC phase(s), taken from
                   x(El@FCC_L12#N) columns where fcc_type@FCC_L12#N == 'disordered'

  - interpolated:  point between derived_fcc and original composition
                   at each interpolation distance

Special cases handled:
  - One ordered + one disordered FCC: only disordered is used as derived_fcc
  - Spinodal decomposition (two+ disordered FCC): one derived_fcc per phase,
    using the composition of each disordered instance separately

Author: Caroline Binde Stoco
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_DIR  = Path("tc_splits")
OUTPUT_DIR = Path("tc_augmented")

SUFFIXES = ["ter0", "ter10", "ter20", "ter30", "ter40", "ter50", "ter60", "ter70", "ter80", "ter90"]

ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Mn", "Mo", "Ni", "Ti", "V"]

INTERP_PERCENTS = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]

FINAL_COLUMNS = [
    "task_id",
    "phase_name",
    "phase_score",
    "sample_origin",
    "interp_percent",
    *ELEMENTS,
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_disordered_fcc_phases(row: pd.Series) -> list:
    """
    Return a list of FCC_L12#N phase names that are disordered in this row.
    Uses fcc_type@FCC_L12#N columns.
    Falls back to all FCC_L12#N in phase_name if fcc_type columns are absent.
    """
    phase_name    = row.get("phase_name", "")
    fcc_type_cols = [c for c in row.index if c.startswith("fcc_type@FCC_L12")]

    if fcc_type_cols:
        return [
            c.replace("fcc_type@", "")
            for c in fcc_type_cols
            if row[c] == "disordered" and
               c.replace("fcc_type@", "") in phase_name
        ]
    else:
        # Fallback: use all FCC_L12#N present in phase_name
        return [
            term.strip()
            for term in phase_name.split("+")
            if "FCC_L12" in term
        ]


def make_original_row(row: pd.Series) -> dict:
    """Convert a row into a baseline record (sample_origin = original)."""
    return {
        "task_id":       row["task_id"],
        "phase_name":    row["phase_name"],
        "phase_score":   row["phase_score"],
        "sample_origin": "original",
        "interp_percent": 0,
        **{el: row[el] for el in ELEMENTS},
    }


def augment(df: pd.DataFrame, interp_percent: int) -> pd.DataFrame:
    """
    Build augmented dataset for a given interpolation distance.

    Only rows with phase_score == 2 are augmented.
    For each disordered FCC phase found, one derived_fcc and one interpolated
    point are generated.
    """
    interp_frac = interp_percent / 100.0
    records     = [make_original_row(row) for _, row in df.iterrows()]

    for _, row in df.iterrows():
        if row["phase_score"] != 2:
            continue

        mother_vec        = row[ELEMENTS].values.astype(float)
        disordered_phases = get_disordered_fcc_phases(row)

        if not disordered_phases:
            continue

        for fcc_phase in disordered_phases:
            fcc_cols = [f"x({el}@{fcc_phase})" for el in ELEMENTS]

            # Skip if composition columns are missing or all zero
            if not all(col in row.index for col in fcc_cols):
                continue
            fcc_vec = np.array([row.get(col, 0.0) for col in fcc_cols], dtype=float)
            if fcc_vec.sum() == 0:
                continue

            # derived_fcc — pure FCC endpoint
            records.append({
                "task_id":        row["task_id"],
                "phase_name":     fcc_phase,
                "phase_score":    1,
                "sample_origin":  "derived_fcc",
                "interp_percent": 0,
                **{el: val for el, val in zip(ELEMENTS, fcc_vec)},
            })

            # interpolated — between FCC endpoint and original
            interp_vec = fcc_vec + interp_frac * (mother_vec - fcc_vec)
            records.append({
                "task_id":        row["task_id"],
                "phase_name":     row["phase_name"],
                "phase_score":    row["phase_score"],
                "sample_origin":  "interpolated",
                "interp_percent": interp_percent,
                **{el: val for el, val in zip(ELEMENTS, interp_vec)},
            })

    return pd.DataFrame(records)[FINAL_COLUMNS]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for suffix in SUFFIXES:
        input_path = INPUT_DIR / f"train_original_{suffix}.csv"
        print(f"\n── Processing {input_path.name} ──")

        try:
            df = pd.read_csv(input_path)
        except FileNotFoundError:
            print(f"  File not found, skipping: {input_path}")
            continue

        n_score2 = (df["phase_score"] == 2).sum()
        print(f"  Total rows      : {len(df)}")
        print(f"  Phase score dist: {df['phase_score'].value_counts().sort_index().to_dict()}")
        print(f"  Score 2 (augmentable): {n_score2}")

        # Baseline — originals only, no augmentation
        baseline      = pd.DataFrame(
            [make_original_row(row) for _, row in df.iterrows()]
        )[FINAL_COLUMNS]
        baseline_path = OUTPUT_DIR / f"tc_augmented_{suffix}_dist0.csv"
        baseline.to_csv(baseline_path, index=False)
        print(f"  Saved baseline  : {baseline_path.name} ({len(baseline)} rows)")

        # Augmented — originals + derived_fcc + interpolated
        for x in INTERP_PERCENTS:
            aug_df   = augment(df, interp_percent=x)
            out_path = OUTPUT_DIR / f"tc_augmented_{suffix}_dist{x}.csv"
            aug_df.to_csv(out_path, index=False)

            n_orig    = (aug_df["sample_origin"] == "original").sum()
            n_derived = (aug_df["sample_origin"] == "derived_fcc").sum()
            n_interp  = (aug_df["sample_origin"] == "interpolated").sum()
            print(f"  Saved dist{x:2d}    : {out_path.name} "
                  f"({n_orig} orig + {n_derived} derived + {n_interp} interp)")


if __name__ == "__main__":
    main()