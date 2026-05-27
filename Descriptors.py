"""
Descriptor Generation for TC Phase Dataset
============================================
Computes physics-based descriptors for TC train/test/augmented files.

Author: Caroline Binde Stoco
"""

import pandas as pd
import numpy as np
import itertools
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

# Input directories to scan
INPUT_DIRS = [
    Path("tc_augmented"),   # augmented train files
    Path("tc_splits"),      # original train/test files
]

OUTPUT_DIR = Path("tc_descriptors")

ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Mn", "Mo", "Ni", "Ti", "V"]

PROPERTIES_FILE = "Elemental_properties.csv"
HMIX_FILE       = "Hmix_RK_parameters.csv"
EXCP_FILE       = "exCp_RK_parameters.csv"

# ── File selection filters ────────────────────────────────────────────────────
# Set to None to process all available options.
# Example: SUFFIXES = ["ter50"] or DISTANCES = [0, 2, 5, 10]

SUFFIXES   = ["ter0", "ter40"]    # e.g. ["ter0", "ter40", "ter50", "ter60"] or None for all
DISTANCES  = ["dist0", "dist10", "dist30", "dist50", "dist70", "dist90"]    # e.g. ["dist0", "dist2", "dist5", "dist10"] or None for all


# ── Load reference data ───────────────────────────────────────────────────────

properties     = pd.read_csv(PROPERTIES_FILE, delimiter=";")
hmix_params_df = pd.read_csv(HMIX_FILE, index_col=0)
excp_params_df = pd.read_csv(EXCP_FILE, index_col=0)


# ── File filter ───────────────────────────────────────────────────────────────

def should_process(file_path: Path) -> bool:
    """
    Returns True if the file should be processed.

    Rules:
    - From tc_augmented: all tc_augmented_* files (augmented train data)
    - From tc_splits: only test_original_* files
      (train_original not needed — ML trains on tc_augmented)
    - Suffix and distance filters applied on top
    """
    name   = file_path.name
    parent = file_path.parent.name  # immediate parent directory name

    if parent == "tc_splits":
        if not name.startswith("test_original_"):
            return False
    elif parent == "tc_augmented":
        if not name.startswith("tc_augmented_"):
            return False
    else:
        return False

    name_lower = name.lower()

    # Suffix filter
    if SUFFIXES is not None:
        if not any(s in name_lower for s in SUFFIXES):
            return False

    # Distance filter — only applies to tc_augmented files
    if DISTANCES is not None and parent == "tc_augmented":
        if not any(d in name_lower for d in DISTANCES):
            return False

    return True


# ── Vectorized descriptor functions ───────────────────────────────────────────

def weighted_mean(prop: np.ndarray, comp: np.ndarray) -> np.ndarray:
    return comp @ prop


def weighted_std(prop: np.ndarray, comp: np.ndarray) -> np.ndarray:
    avg    = weighted_mean(prop, comp).reshape(-1, 1)
    diff_sq = (prop - avg) ** 2
    return np.sqrt((comp * diff_sq).sum(axis=1))


def atomic_size_mismatch(atomic_radius: np.ndarray, comp: np.ndarray) -> np.ndarray:
    avg_radius = weighted_mean(atomic_radius, comp).reshape(-1, 1)
    diff_sq    = (1 - atomic_radius / avg_radius) ** 2
    return 100 * np.sqrt((comp * diff_sq).sum(axis=1))


def mixing_entropy(comp: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        log_comp = np.where(comp > 0, np.log(comp), 0.0)
    return -(comp * log_comp).sum(axis=1)


def redlich_kister_integral(comp: np.ndarray, params_df: pd.DataFrame) -> np.ndarray:
    result = np.zeros(comp.shape[0])
    for el_a, el_b in itertools.combinations(ELEMENTS, 2):
        binary_key = f"{el_a}-{el_b}"
        if binary_key not in params_df.index:
            continue
        a0, a1, a2, a3 = params_df.loc[binary_key, ["a0", "a1", "a2", "a3"]]
        idx_a = ELEMENTS.index(el_a)
        idx_b = ELEMENTS.index(el_b)
        x_a   = comp[:, idx_a]
        x_b   = comp[:, idx_b]
        d     = x_a - x_b
        result += x_a * x_b * (a0 + a1 * d + a2 * d**2 + a3 * d**3)
    return np.round(result, 3)


# ── Descriptor generation ─────────────────────────────────────────────────────

def generate_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    raw_comp = df[ELEMENTS].values.astype(float)
    row_sums = raw_comp.sum(axis=1, keepdims=True)
    comp     = raw_comp / row_sums

    prop = {
        name: properties[name].values
        for name in ["Hfus", "Sfus", "Electronegativity",
                     "NValence", "AtomicRadius", "AtomicWeight",
                     "MeltT", "BoilingT"]
    }

    descriptor_cols = {}
    prop_labels = {
        "Hfus":              "Hfus",
        "Sfus":              "Sfus",
        "Electronegativity": "Electronegativity",
        "NValence":          "VEC",
        "AtomicRadius":      "AtomicRadius",
        "AtomicWeight":      "AtomicWeight",
        "MeltT":             "MeltT",
        "BoilingT":          "BoilingT",
    }
    for prop_key, label in prop_labels.items():
        p = prop[prop_key]
        descriptor_cols[f"{label}_avgs_w"]     = weighted_mean(p, comp)
        descriptor_cols[f"{label}_std_devs_w"] = weighted_std(p, comp)

    descriptor_cols["AtomicSizeMismatch"] = atomic_size_mismatch(prop["AtomicRadius"], comp)
    descriptor_cols["Sid"]                = mixing_entropy(comp)
    descriptor_cols["Hmix"]               = redlich_kister_integral(comp, hmix_params_df) / 1000
    descriptor_cols["exCp"]               = redlich_kister_integral(comp, excp_params_df)

    metadata = pd.DataFrame({
        "task_id":        df["task_id"].values,
        "phase_name":     df["phase_name"].values,
        "phase_score":    df["phase_score"].values,
        "sample_origin":  df["sample_origin"].values  if "sample_origin"  in df.columns else "original",
        "interp_percent": df["interp_percent"].values if "interp_percent" in df.columns else 0,
        **{el: df[el].values for el in ELEMENTS},
    })

    return pd.concat([metadata, pd.DataFrame(descriptor_cols, index=df.index)], axis=1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Report active filters
    print("── Descriptor generation ──")
    print(f"  Suffixes  : {SUFFIXES  if SUFFIXES  is not None else 'all'}")
    print(f"  Distances : {DISTANCES if DISTANCES is not None else 'all'}")

    processed = 0
    for input_dir in INPUT_DIRS:
        for file_path in sorted(input_dir.glob("*.csv")):
            if not should_process(file_path):
                continue

            print(f"\n  Processing {file_path.name}...")
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"    [SKIP] Could not read: {e}")
                continue

            result_df   = generate_descriptors(df)
            output_path = OUTPUT_DIR / file_path.name.replace(".csv", "_descriptors.csv")
            result_df.to_csv(output_path, index=False)
            print(f"    Rows: {len(result_df)} | Saved: {output_path.name}")
            processed += 1

    print(f"\n── Done. {processed} file(s) processed. ──")


if __name__ == "__main__":
    main()