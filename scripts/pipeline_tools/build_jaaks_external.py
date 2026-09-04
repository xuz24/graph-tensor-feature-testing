"""Build a leakage-safe Jaaks et al. 2022 external-validation fold table.

The public validation screen has 23,505 fitted rows.  Following the paper exactly, replicate rows
are majority-called within an (anchored combination, cell, anchor concentration), then the
combination-cell pair is positive if either anchor concentration is synergistic.  The records whose
library is itself ``Afatinib | Trametinib`` are a three-drug condition and are excluded.  The current
public file then contains 4,898 two-drug anchored combination-cell pairs, 17 more than the paper's
reported 4,881; that publication/file-version discrepancy is retained in the audit rather than
silently deleting records to force the count.

The present dpsyn cell block contains only the 106 DrugComb cell lines.  This first external setting
therefore uses the conservative intersection: both drugs must already be graph nodes and the Jaaks
SIDM must map to one of those 106 cells.  Exact (drug, drug, cell) overlaps with internal DrugComb
are removed from external test.  Pair or cell overlap alone is reported, not removed; independent
validation means an independent source/measurement, not necessarily unseen entities.

Outputs
-------
``external/jaaks2022/jaaks_validation_mapped.csv``
    One row per mapped unordered pair/cell before exact-overlap removal, with audit flags.
``unified/canonical_folds_extmap_plus_jaaks_overlap.csv``
    Internal extmap rows followed by external-only rows.  External rows have ``fold=-2``; the EXT
    splitter is the only code allowed to interpret that marker.
``unified/canonical_folds_extmap_plus_jaaks_full.csv``
    The preferred external table: adds the 86/97 Jaaks cells with real local DepMap expression.
``external/jaaks2022/jaaks_validation_audit.json``
    Counts, paths, hashes and label-rule provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dpsyn import config as C


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def norm(s) -> str:
    return str(s).strip().lower()


def truth(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def parse_args() -> argparse.Namespace:
    ext = C.DATA_ROOT / "external" / "jaaks2022"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path,
                   default=ext / "Validation screen_All tissues_fitted data.csv")
    p.add_argument("--internal", type=Path, default=C.FOLD_CSVS["extmap"])
    p.add_argument("--mapping", type=Path,
                   default=C.DATA_ROOT / "drugcomb" / "drugcomb_name_to_dbid_mapping_extended.csv")
    p.add_argument("--graph-drugs", type=Path,
                   default=C.GRAPHS["dp_drugs"] / "drug_id_map.csv")
    p.add_argument("--cell-map", type=Path, default=C.CELL_CCLE_MAP)
    p.add_argument("--model", type=Path, default=C.DEPMAP_MODEL)
    p.add_argument("--depmap-expression", type=Path,
                   default=C.DATA_ROOT / "ccle" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    p.add_argument("--mapped-out", type=Path, default=ext / "jaaks_validation_mapped.csv")
    p.add_argument("--combined-out", type=Path,
                   default=C.DATA_ROOT / "unified" / "canonical_folds_extmap_plus_jaaks_overlap.csv")
    p.add_argument("--mapped-full-out", type=Path, default=ext / "jaaks_validation_mapped_full.csv")
    p.add_argument("--combined-full-out", type=Path,
                   default=C.DATA_ROOT / "unified" / "canonical_folds_extmap_plus_jaaks_full.csv")
    p.add_argument("--cell-expression-out", type=Path,
                   default=ext / "cell_line_expression_plus_jaaks.npz")
    p.add_argument("--cell-map-out", type=Path,
                   default=ext / "cell_line_ccle_map_plus_jaaks.csv")
    p.add_argument("--audit-out", type=Path, default=ext / "jaaks_validation_audit.json")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    raw = pd.read_csv(a.raw, dtype=str, low_memory=False)
    n_raw = len(raw)

    # The public file also contains conditions in which the "library" is already a two-drug
    # cocktail. A pair model cannot represent those three-drug conditions.
    is_three_drug = raw.LIBRARY_NAME.fillna("").str.contains(r"\|", regex=True)
    raw = raw.loc[~is_three_drug].copy()
    raw["synergy_bool"] = truth(raw.Synergy)
    if raw.synergy_bool.isna().any():
        raise ValueError(f"unparseable Synergy values: {raw.loc[raw.synergy_bool.isna(), 'Synergy'].unique()}")

    mp = pd.read_csv(a.mapping, dtype=str).dropna(subset=["drugbank_id"])
    name2db = {}
    for r in mp.itertuples(index=False):
        name2db[norm(r.name)] = str(r.drugbank_id).strip()
        name2db[norm(r.name_norm)] = str(r.drugbank_id).strip()
    raw["anchor_dbid"] = raw.ANCHOR_NAME.map(lambda x: name2db.get(norm(x)))
    raw["library_dbid"] = raw.LIBRARY_NAME.map(lambda x: name2db.get(norm(x)))

    model = pd.read_csv(a.model, dtype=str)
    sidm2model = dict(zip(model.SangerModelID, model.ModelID))
    raw["model_id"] = raw.SIDM.map(sidm2model)

    # Build a cell source that can represent genuinely external cells. Preserve the base block
    # byte-for-byte, append one SIDM-keyed alias per Jaaks cell, and pull expression from the local
    # DepMap matrix. Reading in chunks keeps the 19,193-gene CSV below a few hundred MB of RAM.
    base_cell = np.load(C.CCLE_EXPR, allow_pickle=True)
    base_gene = [str(x) for x in base_cell["gene_ids"]]
    need_models = set(raw.model_id.dropna())
    expr_by_model: dict[str, np.ndarray] = {}
    raw_gene = None
    for chunk in pd.read_csv(a.depmap_expression, chunksize=128):
        id_col = chunk.columns[0]
        if raw_gene is None:
            raw_gene = [str(x).rsplit("(", 1)[-1].rstrip(")") for x in chunk.columns[1:]]
            if raw_gene != base_gene:
                raise RuntimeError("DepMap expression gene order differs from the current cell block")
        hit = chunk[id_col].astype(str).isin(need_models)
        for _, r in chunk.loc[hit].iterrows():
            expr_by_model[str(r[id_col])] = r.iloc[1:].to_numpy(dtype=np.float32)
    raw["has_external_expression"] = raw.model_id.isin(expr_by_model)

    sidm_rows = (raw[["SIDM", "model_id"]].drop_duplicates().sort_values("SIDM"))
    alias_names, alias_expr, alias_mask = [], [], []
    for r in sidm_rows.itertuples(index=False):
        alias_names.append(f"JAAKS-{r.SIDM}")
        x = expr_by_model.get(str(r.model_id))
        alias_mask.append(x is not None)
        alias_expr.append(x if x is not None else np.zeros(len(base_gene), np.float32))
    np.savez_compressed(
        a.cell_expression_out,
        cell_names=np.concatenate([base_cell["cell_names"].astype(str), np.asarray(alias_names)]),
        expression=np.concatenate([base_cell["expression"].astype(np.float32),
                                   np.stack(alias_expr).astype(np.float32)]),
        gene_ids=base_cell["gene_ids"].astype(str),
        matched_mask=np.concatenate([base_cell["matched_mask"].astype(bool),
                                     np.asarray(alias_mask, bool)]),
    )
    base_cmap = pd.read_csv(a.cell_map, dtype=str)
    alias_map = pd.DataFrame({
        "drugcomb_name": alias_names,
        "ccle_modelid": sidm_rows.model_id.to_numpy(),
        "match_type": "jaaks_sidm",
    })
    pd.concat([base_cmap, alias_map], ignore_index=True).to_csv(a.cell_map_out, index=False)

    internal = pd.read_csv(a.internal, dtype=str)
    cmap = pd.read_csv(a.cell_map, dtype=str).dropna(subset=["ccle_modelid"])
    name2model = {str(r.drugcomb_name).strip().upper(): str(r.ccle_modelid).strip()
                  for r in cmap.itertuples(index=False)}
    internal["model_id"] = internal.cell_name.str.strip().str.upper().map(name2model)
    # Prefer the spelling actually used most often in the current fold table when aliases share a
    # ModelID (for example A-375/A375).  That keeps the combined table inside the frozen cell vocab.
    ach2cell = (internal.dropna(subset=["model_id"])
                       .groupby(["model_id", "cell_name"]).size().rename("n").reset_index()
                       .sort_values(["model_id", "n", "cell_name"], ascending=[True, False, True])
                       .drop_duplicates("model_id").set_index("model_id").cell_name.to_dict())
    raw["cell_name_current"] = raw.model_id.map(ach2cell)
    raw["cell_name_full"] = raw.cell_name_current.where(
        raw.cell_name_current.notna(), "JAAKS-" + raw.SIDM.astype(str))

    graph_ids = set(pd.read_csv(a.graph_drugs, dtype=str).drugbank_id)
    raw["both_drugs_in_graph"] = raw.anchor_dbid.isin(graph_ids) & raw.library_dbid.isin(graph_ids)
    raw["cell_in_current_pipeline"] = raw.cell_name_current.notna()

    # Paper label rule, stage 1: majority across replicates at one anchor concentration.  Keep
    # COMBI_ID so reverse anchored orientations remain distinct through the authors' aggregation.
    dose_key = ["COMBI_ID", "SIDM", "ANCHOR_CONC"]
    dose = (raw.groupby(dose_key, dropna=False)
               .agg(replicate_synergy_rate=("synergy_bool", "mean"), n_replicates=("synergy_bool", "size"),
                    anchor_dbid=("anchor_dbid", "first"), library_dbid=("library_dbid", "first"),
                    model_id=("model_id", "first"), cell_name=("cell_name_current", "first"),
                    cell_name_full=("cell_name_full", "first"),
                    has_external_expression=("has_external_expression", "first"),
                    tissue=("Tissue", "first"), anchor_name=("ANCHOR_NAME", "first"),
                    library_name=("LIBRARY_NAME", "first"),
                    delta_emax=("SYNERGY_DELTA_EMAX", lambda x: pd.to_numeric(x, errors="coerce").mean()),
                    delta_ic50=("SYNERGY_DELTA_XMID", lambda x: pd.to_numeric(x, errors="coerce").mean()))
               .reset_index())
    dose["dose_synergy"] = dose.replicate_synergy_rate >= 0.5

    # Paper label rule, stage 2: either anchor concentration.  The public file currently yields
    # 4,898 representable two-drug records, against 4,881 reported in the publication.
    anchored_key = ["COMBI_ID", "SIDM"]
    anchored = (dose.groupby(anchored_key, dropna=False)
                    .agg(label=("dose_synergy", "max"), n_anchor_concentrations=("ANCHOR_CONC", "nunique"),
                         anchor_dbid=("anchor_dbid", "first"), library_dbid=("library_dbid", "first"),
                         model_id=("model_id", "first"), cell_name=("cell_name", "first"),
                         cell_name_full=("cell_name_full", "first"),
                         has_external_expression=("has_external_expression", "first"),
                         tissue=("tissue", "first"), anchor_name=("anchor_name", "first"),
                         library_name=("library_name", "first"),
                         mean_delta_emax=("delta_emax", "mean"), mean_delta_ic50=("delta_ic50", "mean"))
                    .reset_index())

    anchored["d1"] = np.minimum(anchored.anchor_dbid.fillna(""), anchored.library_dbid.fillna(""))
    anchored["d2"] = np.maximum(anchored.anchor_dbid.fillna(""), anchored.library_dbid.fillna(""))
    anchored["both_drugs_in_graph"] = anchored.d1.isin(graph_ids) & anchored.d2.isin(graph_ids)
    anchored["cell_in_current_pipeline"] = anchored.cell_name.notna()

    # The model is unordered, so combine reverse anchored orientations.  "Any" retains the paper's
    # operational definition: the pair is useful if synergy appears at any validated anchor regime.
    mapped = anchored.loc[anchored.both_drugs_in_graph & anchored.cell_in_current_pipeline].copy()
    ext = (mapped.groupby(["d1", "d2", "model_id", "cell_name"], dropna=False)
                 .agg(label=("label", "max"), anchored_positive_rate=("label", "mean"),
                      n_anchored_assays=("label", "size"), tissue=("tissue", "first"),
                      mean_delta_emax=("mean_delta_emax", "mean"),
                      mean_delta_ic50=("mean_delta_ic50", "mean"))
                 .reset_index())

    internal_pairs = set(zip(np.minimum(internal.drug1_dbid, internal.drug2_dbid),
                             np.maximum(internal.drug1_dbid, internal.drug2_dbid)))
    internal_triplets = set(zip(np.minimum(internal.drug1_dbid, internal.drug2_dbid),
                                np.maximum(internal.drug1_dbid, internal.drug2_dbid),
                                internal.model_id))
    ext["pair_in_internal"] = [(x, y) in internal_pairs for x, y in zip(ext.d1, ext.d2)]
    ext["exact_triplet_in_internal"] = [
        (x, y, c) in internal_triplets for x, y, c in zip(ext.d1, ext.d2, ext.model_id)]
    ext["source"] = "Jaaks2022_validation"
    a.mapped_out.parent.mkdir(parents=True, exist_ok=True)
    ext.to_csv(a.mapped_out, index=False)

    external_test = ext.loc[~ext.exact_triplet_in_internal].copy()
    base = pd.read_csv(a.internal)
    base["source"] = "DrugCombDB"
    add = pd.DataFrame({
        "triplet_id": np.arange(len(base), len(base) + len(external_test)),
        "drug1_dbid": external_test.d1.to_numpy(),
        "drug2_dbid": external_test.d2.to_numpy(),
        "cell_name": external_test.cell_name.to_numpy(),
        "label": external_test.label.astype(int).to_numpy(),
        "mean_zip": np.nan,
        "fold": -2,
        "source": "Jaaks2022_validation",
    })
    combined = pd.concat([base, add], ignore_index=True, sort=False)
    a.combined_out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(a.combined_out, index=False)

    # Preferred full-cell evaluation: all graph-representable pairs in the 86 Jaaks cells with
    # real local DepMap expression. Shared ModelIDs use the existing cell name; unseen ModelIDs use
    # the appended JAAKS-SIDM alias and are genuinely cold cells under EXT.
    mapped_full = anchored.loc[anchored.both_drugs_in_graph & anchored.has_external_expression].copy()
    ext_full = (mapped_full.groupby(["d1", "d2", "model_id", "cell_name_full"], dropna=False)
                           .agg(label=("label", "max"), anchored_positive_rate=("label", "mean"),
                                n_anchored_assays=("label", "size"), tissue=("tissue", "first"),
                                mean_delta_emax=("mean_delta_emax", "mean"),
                                mean_delta_ic50=("mean_delta_ic50", "mean"))
                           .reset_index().rename(columns={"cell_name_full": "cell_name"}))
    ext_full["pair_in_internal"] = [(x, y) in internal_pairs for x, y in zip(ext_full.d1, ext_full.d2)]
    ext_full["exact_triplet_in_internal"] = [
        (x, y, c) in internal_triplets for x, y, c in zip(ext_full.d1, ext_full.d2, ext_full.model_id)]
    ext_full["source"] = "Jaaks2022_validation"
    ext_full.to_csv(a.mapped_full_out, index=False)
    external_full = ext_full.loc[~ext_full.exact_triplet_in_internal].copy()
    add_full = pd.DataFrame({
        "triplet_id": np.arange(len(base), len(base) + len(external_full)),
        "drug1_dbid": external_full.d1.to_numpy(), "drug2_dbid": external_full.d2.to_numpy(),
        "cell_name": external_full.cell_name.to_numpy(),
        "label": external_full.label.astype(int).to_numpy(), "mean_zip": np.nan,
        "fold": -2, "source": "Jaaks2022_validation",
    })
    pd.concat([base, add_full], ignore_index=True, sort=False).to_csv(a.combined_full_out, index=False)

    audit = {
        "source": {
            "paper": "Jaaks et al., Nature 2022, DOI 10.1038/s41586-022-04437-2",
            "figshare": "10.6084/m9.figshare.16843600.v1",
            "raw": str(a.raw), "raw_md5": md5(a.raw),
        },
        "label_rule": ("majority of replicate Synergy calls per anchored-combination/cell/anchor-"
                       "concentration, then positive if either anchor concentration is positive"),
        "counts": {
            "raw_fitted_rows": int(n_raw),
            "three_drug_rows_excluded": int(is_three_drug.sum()),
            "three_drug_anchored_combination_cell_pairs_excluded":
                int(pd.read_csv(a.raw, dtype=str, low_memory=False).loc[is_three_drug]
                       .groupby(["COMBI_ID", "SIDM"], dropna=False).ngroups),
            "two_drug_anchored_combination_cell_pairs": int(len(anchored)),
            "publication_reported_combination_cell_pairs": 4881,
            "public_file_minus_publication_after_three_drug_exclusion": int(len(anchored) - 4881),
            "unique_drug_names": int(len(set(raw.ANCHOR_NAME) | set(raw.LIBRARY_NAME))),
            "unique_drug_names_mapped": int(sum(name2db.get(norm(n)) is not None
                                                 for n in set(raw.ANCHOR_NAME) | set(raw.LIBRARY_NAME))),
            "cell_lines": int(raw.SIDM.nunique()),
            "cell_lines_in_current_pipeline": int(raw.loc[raw.cell_in_current_pipeline, "SIDM"].nunique()),
            "anchored_pairs_both_drugs_in_graph": int(anchored.both_drugs_in_graph.sum()),
            "anchored_pairs_in_current_cells_and_graph": int(len(mapped)),
            "unordered_pair_cell_rows_before_overlap_removal": int(len(ext)),
            "exact_internal_triplets_removed": int(ext.exact_triplet_in_internal.sum()),
            "external_test_rows": int(len(external_test)),
            "external_test_positive": int(external_test.label.sum()),
            "external_test_prevalence": float(external_test.label.mean()),
            "external_test_pairs_seen_in_internal": int(external_test.pair_in_internal.sum()),
            "external_test_unique_cells": int(external_test.model_id.nunique()),
            "external_test_unique_drugs": int(len(set(external_test.d1) | set(external_test.d2))),
            "external_cells_with_depmap_expression": int(len(expr_by_model)),
            "full_unordered_pair_cell_rows_before_overlap_removal": int(len(ext_full)),
            "full_exact_internal_triplets_removed": int(ext_full.exact_triplet_in_internal.sum()),
            "full_external_test_rows": int(len(external_full)),
            "full_external_test_positive": int(external_full.label.sum()),
            "full_external_test_prevalence": float(external_full.label.mean()),
            "full_external_test_unique_cells": int(external_full.model_id.nunique()),
            "full_external_test_cold_cells":
                int(len(set(external_full.model_id) - set(ach2cell))),
            "full_external_test_unique_drugs": int(len(set(external_full.d1) | set(external_full.d2))),
        },
        "outputs": {"mapped_overlap": str(a.mapped_out), "combined_overlap": str(a.combined_out),
                    "mapped_full": str(a.mapped_full_out), "combined_full": str(a.combined_full_out),
                    "cell_expression": str(a.cell_expression_out), "cell_map": str(a.cell_map_out)},
        "safety": ("Rows with fold=-2 are external test only. dpsyn.split_EXT builds train/val from "
                   "fold!=-2 and never exposes external labels to training or early stopping."),
    }
    a.audit_out.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit["counts"], indent=2))
    print(f"wrote {a.mapped_out}")
    print(f"wrote {a.combined_out}")
    print(f"wrote {a.combined_full_out}")
    print(f"wrote {a.audit_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
