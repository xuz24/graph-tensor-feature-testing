"""Build the reverse cross-study fold table: train on Jaaks, test on DrugCombDB.

The forward Jaaks table already contains a leakage-audited, exact-overlap-free union of the two
sources. This tool changes only the source roles: Jaaks rows become the internal train/validation
universe and DrugCombDB rows receive the EXT test-only marker ``fold=-2``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dpsyn import config as C


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forward", type=Path, default=C.FOLD_CSVS["jaaks_full"])
    p.add_argument("--out", type=Path,
                   default=C.DATA_ROOT / "unified" / "canonical_folds_jaaks_to_extmap.csv")
    p.add_argument("--audit", type=Path,
                   default=C.DATA_ROOT / "external" / "jaaks2022" /
                   "jaaks_to_extmap_audit.json")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    d = pd.read_csv(a.forward)
    required = {"triplet_id", "drug1_dbid", "drug2_dbid", "cell_name", "label", "fold", "source"}
    missing = sorted(required - set(d.columns))
    if missing:
        raise RuntimeError(f"forward table is missing columns: {missing}")
    expected = {"DrugCombDB", "Jaaks2022_validation"}
    found = set(d.source.dropna().astype(str))
    if found != expected:
        raise RuntimeError(f"expected sources {sorted(expected)}, found {sorted(found)}")
    if not set(d.label.dropna().astype(int).unique()).issubset({0, 1}):
        raise RuntimeError("labels are not binary")

    jaaks = d.source.eq("Jaaks2022_validation")
    drugcomb = d.source.eq("DrugCombDB")
    if not int(jaaks.sum()) or not int(drugcomb.sum()):
        raise RuntimeError("both sources must be non-empty")

    # The forward builder removed exact source overlaps. Recheck using the actual model keys.
    def keys(x: pd.DataFrame) -> set[tuple[str, str, str]]:
        a1 = x.drug1_dbid.astype(str).to_numpy()
        a2 = x.drug2_dbid.astype(str).to_numpy()
        lo = np.minimum(a1, a2)
        hi = np.maximum(a1, a2)
        return set(zip(lo, hi, x.cell_name.astype(str)))

    overlap = keys(d.loc[jaaks]) & keys(d.loc[drugcomb])
    if overlap:
        raise RuntimeError(f"found {len(overlap)} exact cross-source triplet overlaps")

    out = d.copy()
    out.loc[jaaks, "fold"] = 0
    out.loc[drugcomb, "fold"] = -2
    out["fold"] = out.fold.astype(int)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    audit = {
        "purpose": "reverse cross-study validation: Jaaks train/validation, DrugCombDB test-only",
        "input": str(a.forward),
        "input_sha256": sha256(a.forward),
        "output": str(a.out),
        "output_sha256": sha256(a.out),
        "counts": {
            "jaaks_train_validation_universe": int(jaaks.sum()),
            "jaaks_positive": int(d.loc[jaaks, "label"].sum()),
            "drugcomb_test": int(drugcomb.sum()),
            "drugcomb_positive": int(d.loc[drugcomb, "label"].sum()),
            "exact_cross_source_triplet_overlap": 0,
        },
        "safety": ("Under setup EXT, fold=-2 rows are test-only. DrugCombDB labels therefore do "
                   "not enter training, validation, feature transforms, early stopping, or the "
                   "training-only DD graph."),
    }
    a.audit.parent.mkdir(parents=True, exist_ok=True)
    a.audit.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
