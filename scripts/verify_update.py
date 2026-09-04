"""Verify this delta bundle and, when supplied, its combination with the earlier bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external" / "jaaks2022"


def check(ok: bool, message: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {message}")
    if not ok:
        raise SystemExit(1)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def triplet_keys(rs: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    return {
        (min(r["drug1_dbid"], r["drug2_dbid"]),
         max(r["drug1_dbid"], r["drug2_dbid"]), r["cell_name"])
        for r in rs
    }


def verify_delta() -> None:
    required = [
        ROOT / "scripts" / "splits.py",
        ROOT / "scripts" / "metrics.py",
        ROOT / "scripts" / "generate_all_splits.py",
        ROOT / "scripts" / "pipeline_tools" / "build_jaaks_external.py",
        ROOT / "scripts" / "pipeline_tools" / "build_jaaks_reverse_external.py",
        ROOT / "data" / "drug_features" / "drug_sim_topk30_atompair_full.pt",
        ROOT / "data" / "monotherapy_v15.csv",
        EXT / "canonical_folds_extmap_plus_jaaks_full.csv",
        EXT / "canonical_folds_jaaks_to_extmap.csv",
        EXT / "cell_line_expression_plus_jaaks.npz",
        EXT / "cell_line_ccle_map_plus_jaaks.csv",
        EXT / "jaaks_validation_audit.json",
        EXT / "jaaks_to_extmap_audit.json",
    ]
    check(all(p.is_file() for p in required), "all required delta files are present")
    split_root = ROOT / "data" / "splits"
    print(len(list((split_root / "internal").glob("*.npz"))))
    check(len(list((split_root / "internal").glob("*.npz"))) == 96,
          "96 pre-generated internal split files")
    check(len(list((split_root / "ext_drugcomb_to_jaaks").glob("*.npz"))) == 12,
          "12 pre-generated forward EXT split files")
    check(len(list((split_root / "ext_jaaks_to_drugcomb").glob("*.npz"))) == 12,
          "12 pre-generated reverse EXT split files")

    manifest = ROOT / "CHECKSUMS.sha256"
    for line in manifest.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        p = ROOT / rel
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        check(got == digest, f"SHA-256 {rel}")

    forward = rows(EXT / "canonical_folds_extmap_plus_jaaks_full.csv")
    reverse = rows(EXT / "canonical_folds_jaaks_to_extmap.csv")
    check(len(forward) == len(reverse) == 44_738, "both cross-study tables have 44,738 rows")
    check(sum(r["fold"] == "-2" for r in forward) == 1_864,
          "forward table has 1,864 Jaaks test-only rows")
    check(sum(r["fold"] == "-2" for r in reverse) == 42_874,
          "reverse table has 42,874 DrugCombDB test-only rows")
    for name, table in (("forward", forward), ("reverse", reverse)):
        a = [r for r in table if r["source"] == "DrugCombDB"]
        b = [r for r in table if r["source"] == "Jaaks2022_validation"]
        check(len(a) == 42_874 and len(b) == 1_864, f"{name} source counts")
        check(not (triplet_keys(a) & triplet_keys(b)), f"{name} has no exact cross-source overlap")

    z = np.load(EXT / "cell_line_expression_plus_jaaks.npz", allow_pickle=True)
    check(tuple(z["expression"].shape) == (219, 19_193), "extended cell-expression shape")
    check(int(z["matched_mask"].sum()) == 183, "extended cell-expression matched rows")

    audit = json.loads((EXT / "jaaks_validation_audit.json").read_text())
    check(audit["counts"]["full_external_test_rows"] == 1_864,
          "forward audit agrees with the fold table")
    audit = json.loads((EXT / "jaaks_to_extmap_audit.json").read_text())
    check(audit["counts"]["drugcomb_test"] == 42_874,
          "reverse audit agrees with the fold table")

    with (ROOT / "data" / "monotherapy_v15.csv").open() as fh:
        check(sum(1 for _ in fh) - 1 == 245_194, "correct extmap monotherapy table")


def verify_with_base(data_root: Path) -> None:
    check((data_root / "canonical_folds_extmap.csv").is_file(), "base extmap fold table found")
    check((data_root / "graphs" / "dp_drugs" / "drug_id_map.csv").is_file(),
          "base graph drug map found")

    script = ROOT / "scripts" / "generate_all_splits.py"
    spec = importlib.util.spec_from_file_location("bundle_splits", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    data = module.load(data_root=data_root)
    expected_test_means = {
        "B1": 10_718.5,
        "LCOW": 10_542.75,
        "B2": 10_718.5,
        "B2P": 1_270.3333333333333,
        "B3": 18_697.25,
        "B3A": 8_123.75,
        "B4": 4_681.916666666667,
    }
    for setup, expected in expected_test_means.items():
        sizes = []
        for fold in range(4):
            for seed in range(3):
                tr, va, te = module.one(setup, data, 4, fold, seed)
                report = module.describe_split(setup, data, tr, va, te)
                check(report["flags"] == "-", f"{setup} fold {fold} seed {seed} leakage audit")
                sizes.append(len(te))
        check(abs(float(np.mean(sizes)) - expected) < 1e-9,
              f"{setup} 4-fold x 3-seed test-size mean")

    for name, expected_test in (
        ("canonical_folds_extmap_plus_jaaks_full.csv", 1_864),
        ("canonical_folds_jaaks_to_extmap.csv", 42_874),
    ):
        data = module.load(EXT / name, data_root=data_root)
        tr, va, te = module.one("EXT", data, 4, 0, 0)
        check(len(te) == expected_test, f"EXT split for {name}")
        check(not (set(tr) & set(te)) and not (set(va) & set(te)),
              f"EXT isolation for {name}")


def main() -> int:
    print(ROOT)
    print(EXT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", type=Path,
                        help="data/ directory from the earlier bundle; enables split verification")
    args = parser.parse_args()
    verify_delta()
    if args.base_data:
        verify_with_base(args.base_data.resolve())
    else:
        print("SKIP  combined split checks (pass --base-data /path/to/old_bundle/data)")
    print("\nBundle update verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
