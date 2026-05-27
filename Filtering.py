"""
Database Filtering - TCHEA5 Coverage
======================================
Filters the TC grid results based on TCHEA5 ternary coverage (FAT).

Binary coverage is not computed — all 45 binaries from our 10 elements
are assessed in TCHEA5, so FAB = 1.0 for every composition by definition.

The output retains only the essential columns (composition, phase info,
FCC_L12 fractions, compositions and ordering) plus FAT for reference.

Note: FCC_L12 columns use dynamic numbering (#1, #2, #3...) — all variants
are included automatically.

Author: Caroline Binde Stoco
"""

import pandas as pd
from itertools import combinations
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_FILE  = "tc_grid_results.csv"
DELIMITER   = ","
OUTPUT_DIR  = Path("output")

ELEMENTS = ["Al", "Co", "Cr", "Cu", "Fe", "Mn", "Mo", "Ni", "Ti", "V"]

MIN_TERNARY_COVERAGE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ── TCHEA5 critically assessed ternaries — complete list (192 total) ──────────

TERNARY_TCHEA5 = set(
    tuple(sorted(t)) for t in [
        # Al-X-Y
        ("Al","B","Ti"),   ("Al","C","Co"),   ("Al","C","Fe"),   ("Al","C","Ti"),
        ("Al","Co","Ni"),  ("Al","Co","Ti"),   ("Al","Co","W"),   ("Al","Co","Zr"),
        ("Al","Cr","Ni"),  ("Al","Cr","Sn"),   ("Al","Cr","Ti"),  ("Al","Cu","Fe"),
        ("Al","Cu","Mn"),  ("Al","Cu","Ni"),   ("Al","Cu","Si"),  ("Al","Cu","Sn"),
        ("Al","Cu","Zn"),  ("Al","Fe","Mn"),   ("Al","Fe","N"),   ("Al","Fe","Si"),
        ("Al","Fe","Ti"),  ("Al","Mn","Ni"),   ("Al","Mn","Si"),  ("Al","Mn","Ti"),
        ("Al","Mn","Zn"),  ("Al","Mo","Ni"),   ("Al","Mo","Ti"),  ("Al","Nb","Ni"),
        ("Al","Nb","Ti"),  ("Al","Ni","Ru"),   ("Al","Ni","Si"),  ("Al","Ni","Ta"),
        ("Al","Ni","Ti"),  ("Al","Ni","W"),    ("Al","Ni","Zn"),  ("Al","N","Ti"),
        ("Al","Ru","Ti"),  ("Al","Si","Sn"),   ("Al","Si","Ti"),  ("Al","Si","Zn"),
        ("Al","Sn","Ti"),  ("Al","Sn","Zn"),   ("Al","Ta","Ti"),  ("Al","Ti","V"),
        ("Al","Ti","W"),   ("Al","Ti","Y"),    ("Al","Ti","Zr"),
        # C-X-Y
        ("C","Co","Cr"),   ("C","Co","Fe"),    ("C","Co","Mo"),   ("C","Co","Nb"),
        ("C","Co","Ni"),   ("C","Co","Ta"),    ("C","Co","Ti"),   ("C","Co","W"),
        ("C","Cr","Fe"),   ("C","Cr","Hf"),    ("C","Cr","Si"),   ("C","Cr","Ti"),
        ("C","Cr","V"),    ("C","Cr","Zr"),    ("C","Cu","Fe"),   ("C","Fe","Mn"),
        ("C","Fe","Mo"),   ("C","Fe","N"),     ("C","Fe","Ni"),   ("C","Fe","Si"),
        ("C","Fe","Ti"),   ("C","Fe","V"),     ("C","Fe","W"),    ("C","Hf","Mo"),
        ("C","Hf","Ni"),   ("C","Mn","Si"),    ("C","Mo","Ni"),   ("C","Mo","Ta"),
        ("C","Mo","Ti"),   ("C","Mo","V"),     ("C","Mo","W"),    ("C","Mo","Zr"),
        ("C","Nb","Ni"),   ("C","Nb","Re"),    ("C","Nb","Ti"),   ("C","Nb","W"),
        ("C","Ni","Ta"),   ("C","Ni","Ti"),    ("C","Ni","W"),    ("C","Ni","Zr"),
        ("C","Ta","W"),    ("C","Ti","W"),
        # Co-X-Y
        ("Co","Cr","Cu"),  ("Co","Cr","Fe"),   ("Co","Cr","Ni"),  ("Co","Cr","Ti"),
        ("Co","Cr","W"),   ("Co","Cu","Fe"),   ("Co","Cu","Mn"),  ("Co","Cu","Nb"),
        ("Co","Cu","Ni"),  ("Co","Fe","Mo"),   ("Co","Fe","N"),   ("Co","Fe","Ni"),
        ("Co","Fe","W"),   ("Co","Mo","Ti"),   ("Co","Ni","Si"),  ("Co","Ni","V"),
        ("Co","Ni","W"),   ("Co","Sn","Ti"),   ("Co","Ta","Ti"),  ("Co","W","Zr"),
        # Cr-X-Y
        ("Cr","Cu","Mo"),  ("Cr","Cu","Nb"),   ("Cr","Cu","Si"),  ("Cr","Cu","Sn"),
        ("Cr","Cu","Zr"),  ("Cr","Fe","Mn"),   ("Cr","Fe","Mo"),  ("Cr","Fe","N"),
        ("Cr","Fe","Ni"),  ("Cr","Fe","Si"),   ("Cr","Fe","V"),   ("Cr","Mn","N"),
        ("Cr","Mn","Ti"),  ("Cr","Mo","Ni"),   ("Cr","Nb","Ni"),  ("Cr","Ni","Re"),
        ("Cr","Ni","Si"),  ("Cr","Ni","Ta"),   ("Cr","Ni","Ti"),  ("Cr","Ni","W"),
        ("Cr","Ni","Zr"),  ("Cr","N","Ni"),    ("Cr","Si","Ti"),  ("Cr","Ti","Zr"),
        # Cu-X-Y
        ("Cu","Fe","Mn"),  ("Cu","Fe","N"),    ("Cu","Fe","Ni"),  ("Cu","Fe","Si"),
        ("Cu","Fe","Sn"),  ("Cu","Fe","Ti"),   ("Cu","Fe","V"),   ("Cu","Mn","Ni"),
        ("Cu","Mn","Si"),  ("Cu","Mn","Sn"),   ("Cu","Mn","Zn"),  ("Cu","Mo","Ni"),
        ("Cu","Ni","Ti"),  ("Cu","Ni","Zn"),   ("Cu","Ti","Zr"),
        # Fe-X-Y
        ("Fe","Mn","N"),   ("Fe","Mn","Si"),   ("Fe","Mo","Ni"),  ("Fe","Nb","Ni"),
        ("Fe","Ni","Ru"),  ("Fe","Ni","Si"),   ("Fe","Ni","Ti"),  ("Fe","Ni","W"),
        ("Fe","N","Nb"),   ("Fe","N","Ni"),    ("Fe","N","Ti"),   ("Fe","N","V"),
        ("Fe","Ti","V"),
        # Hf-X-Y
        ("Hf","Nb","Si"),  ("Hf","Ni","Ti"),
        # Ir-X-Y
        ("Ir","Rh","Ru"),
        # Mn-X-Y
        ("Mn","Si","Zn"),
        # Mo-X-Y
        ("Mo","Nb","Ti"),  ("Mo","Ni","Ta"),   ("Mo","N","Ni"),   ("Mo","Ta","Ti"),
        ("Mo","Ti","V"),   ("Mo","Ti","W"),    ("Mo","Ti","Zr"),
        # Nb-X-Y
        ("Nb","Ni","Ti"),  ("Nb","Sn","Ti"),   ("Nb","Ta","Ti"),  ("Nb","Ti","V"),
        ("Nb","Ti","W"),   ("Nb","Ti","Zr"),
        # Ni-X-Y
        ("Ni","Si","Ti"),  ("Ni","Ta","Ti"),   ("Ni","Ta","W"),   ("Ni","Ti","W"),
        ("Ni","Ti","Zr"),  ("N","Ni","Ti"),
        # Re-X-Y
        ("Re","Ta","W"),
        # Si-X-Y
        ("Si","Ti","W"),
        # Ta-X-Y
        ("Ta","Ti","V"),   ("Ta","Ti","W"),    ("Ta","Ti","Zr"),
        # Ti-X-Y
        ("Ti","V","W"),    ("Ti","V","Zr"),    ("Ti","W","Zr"),
    ]
)


# ── Coverage function ──────────────────────────────────────────────────────────

def get_present_elements(row):
    """Return elements with non-zero composition in this row."""
    return [el for el in ELEMENTS if row.get(el, 0) > 0]


def ternary_coverage(present_els):
    """Fraction of ternary subsystems covered by TCHEA5."""
    ters = [tuple(sorted(t)) for t in combinations(present_els, 3)]
    if not ters:
        return 1.0
    covered = sum(1 for t in ters if t in TERNARY_TCHEA5)
    return covered / len(ters)


# ── Column selection ───────────────────────────────────────────────────────────

def select_columns(df):
    """
    Keep only essential columns:
      - task_id, elements, phase_name, n_phases, FAT
      - f(@FCC_L12#N) for any N
      - x(El@FCC_L12#N) for any N and any element
      - order_param@FCC_L12#N, fcc_type@FCC_L12#N, tracer@FCC_L12#N for any N
    """
    fixed = ["task_id"] + ELEMENTS + ["phase_name", "n_phases", "FAT"]

    fcc_cols = [
        c for c in df.columns
        if "FCC_L12" in c and any(c.startswith(p) for p in [
            "f(@FCC_L12",
            "order_param@FCC_L12",
            "fcc_type@FCC_L12",
            "tracer@FCC_L12",
        ]) or (c.startswith("x(") and "FCC_L12" in c)
    ]

    keep = [c for c in fixed if c in df.columns] + \
           [c for c in fcc_cols if c not in fixed]

    return df[keep]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, delimiter=DELIMITER)
    print(f"  Total rows: {len(df)}")

    # Remove errors
    df = df[~df["phase_name"].isin(["ERROR", "FATAL_ERROR"])].copy()
    print(f"  Valid rows after removing errors: {len(df)}")

    # Compute FAT
    print("Computing ternary coverage (FAT)...")
    df["present_els"] = df.apply(get_present_elements, axis=1)
    df["FAT"]         = df["present_els"].apply(ternary_coverage)
    df.drop(columns=["present_els"], inplace=True)

    # Summary
    print(f"\n── FAT summary ──")
    print(f"  FAT mean  : {df['FAT'].mean():.3f}")
    for thr in MIN_TERNARY_COVERAGE:
        n = (df["FAT"] >= thr).sum()
        print(f"  FAT ≥ {thr:.1f} : {n} ({n/len(df)*100:.1f}%)")

    print(f"\n── TCHEA5 ternary coverage (our 10 elements) ──")
    all_ters  = set(tuple(sorted(t)) for t in combinations(ELEMENTS, 3))
    our_ters  = TERNARY_TCHEA5 & all_ters
    missing   = all_ters - our_ters
    print(f"  Possible ternaries : {len(all_ters)}")
    print(f"  Assessed in TCHEA5 : {len(our_ters)}")
    print(f"  Missing            : {len(missing)}")

    # Filter, reduce columns, save
    print()
    for fat_thresh in MIN_TERNARY_COVERAGE:
        df_filt  = df[df["FAT"] >= fat_thresh].copy()
        df_filt  = select_columns(df_filt)
        suffix   = f"ter{int(fat_thresh*100)}"
        out_path = OUTPUT_DIR / f"tc_filtered_{suffix}.csv"
        df_filt.to_csv(out_path, index=False)
        print(f"  FAT ≥ {fat_thresh:.1f} ({suffix}): {len(df_filt)} rows, "
              f"{len(df_filt.columns)} columns → {out_path}")

    # ── Analysis 1: % data lost per FAT threshold ──────────────────────────────
    print(f"\n── Analysis 1: % data lost per FAT threshold ──")
    print(f"  {'FAT':>6}  {'Kept':>8}  {'Removed':>8}  {'% Removed':>10}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*10}")
    total = len(df)
    for thr in MIN_TERNARY_COVERAGE:
        kept    = (df["FAT"] >= thr).sum()
        removed = total - kept
        print(f"  {thr:>6.2f}  {kept:>8}  {removed:>8}  {removed/total*100:>9.1f}%")

    # ── Analysis 2: missing ternaries per element pair ─────────────────────────
    # For each pair of elements, how many ternaries with a third element
    # are NOT covered in TCHEA5 — reveals which pairs drive removals
    print(f"\n── Analysis 2: missing TCHEA5 ternaries per element pair ──")
    print(f"  (number of ternaries X-Y-Z not assessed, for each pair X-Y)")
    all_ters_our = set(tuple(sorted(t)) for t in combinations(ELEMENTS, 3))
    missing_ters = all_ters_our - TERNARY_TCHEA5

    pair_missing = {}
    for ter in missing_ters:
        for pair in combinations(ter, 2):
            pair = tuple(sorted(pair))
            pair_missing[pair] = pair_missing.get(pair, 0) + 1

    # Sort by most missing
    pair_missing_sorted = sorted(pair_missing.items(), key=lambda x: -x[1])
    print(f"  {'Pair':>12}  {'Missing ternaries':>18}  {'Out of':>8}")
    print(f"  {'-'*12}  {'-'*18}  {'-'*8}")
    for pair, n_miss in pair_missing_sorted:
        # total possible ternaries for this pair
        third_els  = [e for e in ELEMENTS if e not in pair]
        n_possible = len(third_els)
        print(f"  {pair[0]+'-'+pair[1]:>12}  {n_miss:>18}  {n_possible:>8}")

    # ── Analysis 3: % removal per system order ────────────────────────────────
    print(f"\n── Analysis 3: % alloys removed per system order ──")
    df["n_elements"] = df[ELEMENTS].apply(lambda r: (r > 0).sum(), axis=1)
    header = f"  {'N elements':>12}  {'N total':>8}" + \
             "".join(f"  {'FAT≥'+str(t):>10}" for t in MIN_TERNARY_COVERAGE)
    print(header)
    print("  " + "-" * (22 + 12 * len(MIN_TERNARY_COVERAGE)))
    for n in sorted(df["n_elements"].unique()):
        df_n    = df[df["n_elements"] == n]
        n_tot   = len(df_n)
        row_str = f"  {n:>12}  {n_tot:>8}"
        for thr in MIN_TERNARY_COVERAGE:
            kept    = (df_n["FAT"] >= thr).sum()
            removed = n_tot - kept
            row_str += f"  {removed/n_tot*100:>9.1f}%"
        print(row_str)

    # ── Analysis 4: FAT distribution (histogram) ──────────────────────────────
    print(f"\n── Analysis 4: FAT distribution ──")
    bins   = [round(i * 0.1, 1) for i in range(11)]
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    df["fat_bin"] = pd.cut(df["FAT"], bins=bins, labels=labels, include_lowest=True)
    dist = df["fat_bin"].value_counts().sort_index()
    print(f"  {'FAT range':>12}  {'Count':>8}  {'%':>8}  Bar")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*20}")
    for label, count in dist.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {label:>12}  {count:>8}  {pct:>7.1f}%  {bar}")

    # ── Analysis 6: % alloys removed per element per FAT threshold ───────────
    print(f"\n── Analysis 6: % alloys removed per element ──")
    print(f"  (of all alloys containing that element, how many % are removed)")

    header = f"  {'Element':>8}  {'Present':>8}" + \
             "".join(f"  {'FAT≥'+str(round(t,1)):>10}" for t in MIN_TERNARY_COVERAGE if t > 0)
    print(header)
    print("  " + "-" * (18 + 12 * (len(MIN_TERNARY_COVERAGE) - 1)))

    elem_removal_results = []
    for el in ELEMENTS:
        df_el   = df[df[el] > 0]
        n_el    = len(df_el)
        row_str = f"  {el:>8}  {n_el:>8}"
        row_data = {"element": el, "n_present": n_el}
        for thr in MIN_TERNARY_COVERAGE:
            if thr == 0.0:
                continue
            kept    = (df_el["FAT"] >= thr).sum()
            removed = n_el - kept
            pct     = removed / n_el * 100
            row_str += f"  {pct:>9.1f}%"
            row_data[f"pct_removed_fat{int(thr*10)}"] = round(pct, 1)
        print(row_str)
        elem_removal_results.append(row_data)

    pd.DataFrame(elem_removal_results).to_csv(
        OUTPUT_DIR / "element_removal_by_fat.csv", index=False
    )

    # ── Analysis 5: composition bias — removed vs kept ────────────────────────
    print(f"\n── Analysis 5: mean element composition (at.%) — removed vs kept ──")
    for thr in MIN_TERNARY_COVERAGE:
        if thr == 0.0:
            continue   # nothing removed at FAT=0
        kept_mask   = df["FAT"] >= thr
        df_kept     = df[kept_mask]
        df_removed  = df[~kept_mask]
        if len(df_removed) == 0:
            continue
        print(f"\n  FAT ≥ {thr:.1f}  "
              f"(kept={len(df_kept)}, removed={len(df_removed)})")
        print(f"  {'Element':>8}  {'Kept mean':>10}  {'Removed mean':>13}  {'Diff':>8}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*13}  {'-'*8}")
        for el in ELEMENTS:
            m_kept    = df_kept[el].mean()
            m_removed = df_removed[el].mean()
            diff      = m_removed - m_kept
            flag      = " ←" if abs(diff) > 3 else ""
            print(f"  {el:>8}  {m_kept:>10.2f}  {m_removed:>13.2f}  {diff:>+8.2f}{flag}")


if __name__ == "__main__":
    main()