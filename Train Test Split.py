"""
Train/Test Split for Filtered TC Dataset
==========================================
Loads the TC-filtered composition files, assigns a phase score target,
and produces stratified train/test splits for each ternary threshold.

Phase score:
    1 — exactly one FCC_L12#N term, monophasic (e.g. "FCC_L12#1")
    2 — contains FCC_L12 alongside other phases (e.g. "BCC_B2#1+FCC_L12#1")
    3 — everything else (no FCC_L12 present)

Note: FCC_L12 numbering (#1, #2, ...) is dynamic in TC — the phase score
only checks for the presence of "FCC_L12" in the name, not the specific number.

Author: Caroline Binde Stoco
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_DIR  = Path("output")
OUTPUT_DIR = Path("tc_splits")

TEST_SIZE    = 0.20
RANDOM_STATE = 42

SUFFIXES = ["ter0", "ter10", "ter20", "ter30", "ter40", "ter50", "ter60", "ter70", "ter80", "ter90"]

ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Mn", "Mo", "Ni", "Ti", "V"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def assign_phase_score(row) -> int:
    """
    Classify into one of three scores:
        1 — FCC monophasic AND disordered (fcc_type@FCC_L12#N == 'disordered')
        2 — FCC multiphase AND at least one disordered FCC_L12 present
        3 — everything else (no FCC, ordered L12 only, or missing)

    Uses fcc_type@FCC_L12#N columns to distinguish ordered vs disordered.
    Falls back to phase_name only if no fcc_type columns are present.
    """
    phase_name = row.get("phase_name", "")
    if not isinstance(phase_name, str):
        return 3
    if "FCC_L12" not in phase_name:
        return 3

    # Check ordering via fcc_type columns — any #N variant
    fcc_type_cols = [c for c in row.index if c.startswith("fcc_type@FCC_L12")]
    if fcc_type_cols:
        has_disordered = any(row[c] == "disordered" for c in fcc_type_cols)
        if not has_disordered:
            return 3   # only ordered L12 present → class 3
    # else: no fcc_type columns available — fall back to phase_name only

    if "+" not in phase_name:
        return 1
    return 2


def get_fcc_columns(df: pd.DataFrame) -> list:
    """
    Return all FCC_L12-related columns present in the DataFrame.
    Covers any #N variant: f(@FCC_L12#N), x(El@FCC_L12#N),
    order_param@FCC_L12#N, fcc_type@FCC_L12#N, tracer@FCC_L12#N.
    """
    return [
        c for c in df.columns
        if "FCC_L12" in c and any(c.startswith(p) for p in [
            "f(@FCC_L12",
            "x(",
            "order_param@FCC_L12",
            "fcc_type@FCC_L12",
            "tracer@FCC_L12",
        ])
    ]


def class_distribution(train: pd.Series, test: pd.Series) -> pd.DataFrame:
    """Return a comparison of class fractions between train and test sets."""
    train_dist  = train.value_counts(normalize=True).sort_index()
    test_dist   = test.value_counts(normalize=True).sort_index()
    all_classes = sorted(set(train_dist.index) | set(test_dist.index))

    return pd.DataFrame({
        "phase_score":    all_classes,
        "train_fraction": [train_dist.get(c, 0.0) for c in all_classes],
        "test_fraction":  [test_dist.get(c,  0.0) for c in all_classes],
    })


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for suffix in SUFFIXES:
        input_path = INPUT_DIR / f"tc_filtered_{suffix}.csv"
        print(f"\n── Processing {input_path.name} ──")

        try:
            df = pd.read_csv(input_path)
        except FileNotFoundError:
            print(f"  File not found, skipping: {input_path}")
            continue

        df["phase_score"] = df.apply(assign_phase_score, axis=1)

        print(f"  Total rows  : {len(df)}")
        print(f"  Score dist  : {df['phase_score'].value_counts().sort_index().to_dict()}")

        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=df["phase_score"],
        )

        # Build test columns: fixed + FCC_L12 dynamic columns
        fcc_cols = get_fcc_columns(df)
        test_columns = ["task_id", "phase_name", "phase_score", "FAT",
                        "sample_origin", "interp_percent",
                        *ELEMENTS, *fcc_cols]
        test_columns = [c for c in test_columns if c in df.columns]

        # Save splits — full train, reduced test
        train_path = OUTPUT_DIR / f"train_original_{suffix}.csv"
        test_path  = OUTPUT_DIR / f"test_original_{suffix}.csv"

        train_df.to_csv(train_path, index=False)
        test_df[test_columns].to_csv(test_path, index=False)

        # Save split summary
        pd.DataFrame([{
            "suffix":       suffix,
            "n_total":      len(df),
            "n_train":      len(train_df),
            "n_test":       len(test_df),
            "test_size":    TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_score_1":    (df["phase_score"] == 1).sum(),
            "n_score_2":    (df["phase_score"] == 2).sum(),
            "n_score_3":    (df["phase_score"] == 3).sum(),
        }]).to_csv(OUTPUT_DIR / f"split_summary_{suffix}.csv", index=False)

        # Save class distribution
        class_distribution(
            train_df["phase_score"],
            test_df["phase_score"],
        ).to_csv(OUTPUT_DIR / f"class_distribution_{suffix}.csv", index=False)

        print(f"  Train: {train_path} ({len(train_df)} rows)")
        print(f"  Test:  {test_path}  ({len(test_df)} rows)")


if __name__ == "__main__":
    main()