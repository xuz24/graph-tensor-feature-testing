"""Document-exact evaluation splits for drug-synergy OOD validation.

PORTED VERBATIM from `src/data/eval_splits.py` so `dpsyn` is self-contained and so its numbers
remain directly comparable to every result produced by the old pipeline. The split functions below
are byte-identical to that file; the only additions are at the bottom
(`describe`, `feature_identical_leak`, `split_summary`). If you change a splitter here, change it
there too -- or delete the old one.

Implements the setups in `Full_Evaluation_Setup.pdf` (Sriram, 2026-05-26):

  B1  ID setting           — stratified triplet K-fold (drugs+cells shared)
  B2  Cell-line OOD        — K-fold over cell lines (cell never in train)
  B3  Drug OOD             — K-fold over drugs, OR rule (≥1 test drug ⇒ test)
  B4  Joint Drug+Cell OOD  — K-fold over both; test = (≥1 test drug) AND test cell
  C2  Cell-line LOO        — one held-out cell at a time
  C3  Drug LOO             — one held-out drug at a time (OR rule)
  C4  Leave-one-triplet-out — one triplet at a time (see caveat below)
  LOSO Leave-one-study-out — hold out an entire source study (needs `study`)

All functions take the parallel triplet arrays (cid1, cid2, cell, y) — drug
node indices, drug node indices, cell index, label — and return
(train_idx, val_idx, test_idx) as int64 numpy arrays into those triplet
rows. They guarantee the document's no-leak constraint: no test entity
appears in any train/val label triplet.

NOTE on B3/C3 vs the legacy `drug_split` in train_het_channels: the legacy
splitter used the AND rule (BOTH drugs held out, cross-set triplets
dropped). The document's B3/C3 use the OR rule (ANY held-out drug ⇒ test),
which keeps the "one new drug" triplets in test. This module follows the
document.

NOTE on C4: the document's C4 val definition (D_val = all drugs in V*,
then exclude all D_val drugs from train) empties the training set, because
V* contains essentially every drug. We treat that as a spec bug and DO NOT
implement it as written — see `split_C4_caveat()`. A sensible per-triplet
strict-OOD variant is provided that mirrors B4's intent at single-triplet
granularity.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

IntArr = np.ndarray


def _entity_folds(n_entities: int, num_folds: int, seed: int) -> list[np.ndarray]:
    """KFold over [0, n_entities); returns the held-out index array per fold."""
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    return [test for _, test in kf.split(np.arange(n_entities))]


def _isin(arr: np.ndarray, values: set) -> np.ndarray:
    if not values:
        return np.zeros(len(arr), dtype=bool)
    return np.isin(arr, np.fromiter(values, dtype=arr.dtype, count=len(values)))


# ── B1: ID setting — stratified triplet K-fold ────────────────────────────────
def split_B1(y: np.ndarray, num_folds: int, fold_index: int, seed: int):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = [test for _, test in skf.split(np.arange(len(y)), y)]
    test_idx = folds[fold_index]
    val_idx = folds[(fold_index + 1) % num_folds]
    held = np.concatenate([test_idx, val_idx])
    train_idx = np.setdiff1d(np.arange(len(y)), held, assume_unique=False)
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


# ── B2: Cell-line OOD ─────────────────────────────────────────────────────────
def split_B2(cell: np.ndarray, num_folds: int, fold_index: int, seed: int):
    cells = np.unique(cell)
    folds = _entity_folds(len(cells), num_folds, seed)
    C_test = set(cells[folds[fold_index]].tolist())
    C_val = set(cells[folds[(fold_index + 1) % num_folds]].tolist())
    c_test = _isin(cell, C_test)
    c_val = _isin(cell, C_val)
    test_idx = np.where(c_test)[0]
    val_idx = np.where(c_val)[0]
    train_idx = np.where(~c_test & ~c_val)[0]
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


# ── B3: Drug OOD (OR rule) ────────────────────────────────────────────────────
def split_B3(cid1: np.ndarray, cid2: np.ndarray, num_folds: int, fold_index: int, seed: int):
    drugs = np.unique(np.concatenate([cid1, cid2]))
    folds = _entity_folds(len(drugs), num_folds, seed)
    D_test = set(drugs[folds[fold_index]].tolist())
    D_val = set(drugs[folds[(fold_index + 1) % num_folds]].tolist())
    d1_test = _isin(cid1, D_test); d2_test = _isin(cid2, D_test)
    d1_val = _isin(cid1, D_val); d2_val = _isin(cid2, D_val)
    test_mask = d1_test | d2_test
    val_mask = (d1_val | d2_val) & ~d1_test & ~d2_test
    train_mask = ~d1_test & ~d2_test & ~d1_val & ~d2_val  # both drugs in D_train
    return (np.where(train_mask)[0].astype(np.int64),
            np.where(val_mask)[0].astype(np.int64),
            np.where(test_mask)[0].astype(np.int64))


# ── B3A: Drug OOD under the AND rule (BOTH drugs held out) ────────────────────
def split_B3A(cid1: np.ndarray, cid2: np.ndarray, num_folds: int, fold_index: int, seed: int):
    """B3 with the AND rule on TEST: a test row needs BOTH of its drugs cold.

    Why this exists
    ---------------
    The five reported settings are not a difficulty ladder, and B3 is the reason. Under B3's OR
    rule a test row qualifies if EITHER drug is held out, so 56.0% of B3 test rows still carry one
    drug the model has seen -- and `wq_mrr` ranks candidate PARTNERS, so a seen partner still has a
    learnable marginal rate. B3 therefore measures a mixture. Ordered by what is actually removed
    the settings run B1 (nothing) -> LCO (the pair) -> B3 (one or both drugs) -> B4 (drugs and
    cells), and there is no rung on which BOTH drugs are guaranteed cold.

    B3A is that rung. Neither drug has ever appeared in training, so no per-drug main effect is
    available for either side of the pair. It is the setting at which the identity floor should
    finally collapse toward the all-tied floor -- `partner_marginal` gives every candidate exactly
    the prior, and one-hot has never updated either drug's row. If it does NOT collapse, something
    is reaching the model that should not be.

    Same train, same val, same model
    --------------------------------
    `split_B3`'s train and val masks are functions of the fold assignment alone -- train requires
    both drugs in D_train, val requires a D_val drug and no D_test drug -- so restricting TEST does
    not touch either. This function returns B3's train and val UNCHANGED and only filters test, so
    at a matched (fold, seed) the trained model is bit-identical to the B3 run and the B3 -> B3A
    contrast is a pure test-set restriction with no retraining confound. Do not "fix" this by also
    applying the AND rule to val: that would change early stopping and forfeit the property.

    Cold is defined against the TRAIN ROWS, not against D_test. A drug in D_val is absent from
    training too, so a (D_test, D_val) pair is genuinely both-unseen and belongs in this setting;
    keying on `d1_test & d2_test` would silently drop those rows.
    """
    tr, va, te = split_B3(cid1, cid2, num_folds, fold_index, seed)
    train_drugs = set(cid1[tr].tolist()) | set(cid2[tr].tolist())
    both_cold = ~_isin(cid1[te], train_drugs) & ~_isin(cid2[te], train_drugs)
    return tr, va, te[both_cold]

# ── B4: Joint Drug + Cell-line OOD ────────────────────────────────────────────
def split_B4(cid1: np.ndarray, cid2: np.ndarray, cell: np.ndarray,
             num_folds: int, fold_index: int, seed: int,
             cell_seed: Optional[int] = None):
    """Spec B4 step 1: "Reuse the drug folds AND cell-line folds from Setups 2 and 3."

    So B4's cell folds must be B2's cell folds at the same seed, and its drug folds must be B3's.
    Until 2026-08-12 the cell folds used `seed + 1`, which satisfied the drug half (verified: B4's
    test drugs are a subset of B3's at every fold) but NOT the cell half -- B4 and B2 shared only
    3-8 of 26 test cells, so B4's difficulty could not be decomposed into its B2 and B3 components
    fold-by-fold.

    Measured before changing it, in case the offset was guarding against a degenerate split: it was
    not. Over 12 folds x 3 seeds on extmap, spec-exact gives mean n_train 5,454 vs 5,312 and
    identical degeneracy flags. `LOW_TRAIN_FRAC` and `DROPPED_ROWS` fire on every B4 fold either
    way -- that is inherent to B4 (it discards every row outside a coherent drug x cell quadrant),
    not an artifact of the seeding.

    `cell_seed` is NOT reachable from `make_split`, `RunConfig` or the CLI -- it exists so a
    script can reproduce pre-2026-08-12 numbers by calling this function directly with
    `cell_seed=seed + 1`. Do not add a flag for it; the deviation is not a supported mode.
    """
    drugs = np.unique(np.concatenate([cid1, cid2]))
    cells = np.unique(cell)
    dfolds = _entity_folds(len(drugs), num_folds, seed)
    cfolds = _entity_folds(len(cells), num_folds, cell_seed if cell_seed is not None else seed)
    D_test = set(drugs[dfolds[fold_index]].tolist())
    D_val = set(drugs[dfolds[(fold_index + 1) % num_folds]].tolist())
    C_test = set(cells[cfolds[fold_index]].tolist())
    C_val = set(cells[cfolds[(fold_index + 1) % num_folds]].tolist())
    d1_test = _isin(cid1, D_test); d2_test = _isin(cid2, D_test)
    d1_val = _isin(cid1, D_val); d2_val = _isin(cid2, D_val)
    c_test = _isin(cell, C_test); c_val = _isin(cell, C_val)
    any_test_drug = d1_test | d2_test
    any_val_drug = d1_val | d2_val
    test_mask = any_test_drug & c_test
    val_mask = any_val_drug & c_val & ~d1_test & ~d2_test & ~c_test
    train_mask = (~d1_test & ~d2_test & ~d1_val & ~d2_val  # both drugs in D_train
                  & ~c_test & ~c_val)                       # cell in C_train
    return (np.where(train_mask)[0].astype(np.int64),
            np.where(val_mask)[0].astype(np.int64),
            np.where(test_mask)[0].astype(np.int64))


# ── LCO: Combination OOD (leave-combination-out) ──────────────────────────────
def split_LCO(cid1: np.ndarray, cid2: np.ndarray, num_folds: int, fold_index: int,
              seed: int, strict: bool = True):
    """K-fold over distinct unordered drug PAIRS. Both drugs stay in train via other partners.

    Why this setting exists
    -----------------------
    B1 holds out nothing. The same (drug1, drug2) pair sits in train and test in different cells,
    so `pair_memorizer` -- the train-only mean label of the SAME pair in OTHER cells -- reaches
    0.7851 at B1 and 0.8917 at B2 without ever reading the cell line, and a RANDOM per-drug vector
    matches a real fingerprint because drug identity is already a sufficient statistic. B1 measures
    memorisation capacity; it cannot discriminate features.

    B3 removes the shortcut but overshoots: the held-out drug is absent from training entirely, so
    it tests extrapolation to a molecule the model has never seen in any context.

    LCO removes only the PAIR. Both drugs remain in training with OTHER partners, so per-drug main
    effects are still learnable and identity is still useful -- but the specific combination is
    unseen, so `pair_memorizer` MUST collapse to the prior. This is the drug-combination-discovery
    question ("two well-characterised drugs, will THIS combination work?") and the
    leave-combination-out sub-challenge of the AstraZeneca-Sanger DREAM consortium.

    `strict` drops test/val rows containing a drug that does not appear in train at all. Those rows
    are B3, not LCO, and they come from the singleton tail (51.6% of drugs occur in exactly one
    triplet); keeping them would silently blend the two settings. It is ALWAYS ON in the pipeline --
    `make_split` does not pass it and there is no flag, deliberately, because a non-strict LCO is
    not a setting we report. The parameter exists only so this function can be called directly to
    measure the contamination (at num_folds 3/4/5 it admits 143-186 cold drugs).
    `split_summary` reports the resulting `n_dropped` and flags any cold drug that survives.
    """
    lo = np.minimum(cid1, cid2)
    hi = np.maximum(cid1, cid2)
    uniq, inv = np.unique(np.stack([lo, hi], 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    folds = _entity_folds(len(uniq), num_folds, seed)
    p_test = np.zeros(len(uniq), bool); p_test[folds[fold_index]] = True
    p_val = np.zeros(len(uniq), bool); p_val[folds[(fold_index + 1) % num_folds]] = True
    test_mask = p_test[inv]
    val_mask = p_val[inv]
    train_mask = ~test_mask & ~val_mask
    if strict:
        tr_d = set(cid1[train_mask].tolist()) | set(cid2[train_mask].tolist())
        seen = _isin(cid1, tr_d) & _isin(cid2, tr_d)
        test_mask &= seen
        val_mask &= seen
    return (np.where(train_mask)[0].astype(np.int64),
            np.where(val_mask)[0].astype(np.int64),
            np.where(test_mask)[0].astype(np.int64))


# ── LCOW: Combination OOD with every entity individually warm ───────────────────────────────
def split_LCOW(cid1: np.ndarray, cid2: np.ndarray, cell: np.ndarray,
               num_folds: int, fold_index: int, seed: int):
    """LCO restricted to warm cells as well as warm individual drugs.

    The exact unordered drug pair is absent from training, but both drugs and the cell line of
    every validation/test row must each appear somewhere in training. This answers the narrow
    recombination question without mixing in either drug-cold or cell-cold rows.
    """
    tr, va, te = split_LCO(cid1, cid2, num_folds, fold_index, seed, strict=True)
    train_cells = set(np.asarray(cell)[tr].tolist())
    va_warm = _isin(np.asarray(cell)[va], train_cells)
    te_warm = _isin(np.asarray(cell)[te], train_cells)
    return tr, va[va_warm], te[te_warm]


# ── B2P: Cell-line OOD with the drug PAIR also held out ───────────────────────
def split_B2P(cid1: np.ndarray, cid2: np.ndarray, cell: np.ndarray,
              num_folds: int, fold_index: int, seed: int, strict: bool = True):
    """Cell-line OOD that actually measures cell-line generalization.

    The problem this fixes, measured on this data
    ---------------------------------------------
    B2 holds out cell lines but NOT drug pairs, and 62.6% of its test rows reuse a pair seen in
    training -- HIGHER than B1's 39.0%, because removing whole cell lines leaves the surviving
    pairs densely measured elsewhere. The consequence is that at B2:

        pair_memorizer  0.8892     <- ignores the cell line entirely
        one-hot         0.8856     <- no features, no graph
        best of 8 model arms  0.8706

    Both trivial references beat every model at the setting whose stated purpose is generalizing
    to unseen cell lines. A predictor that never reads the held-out factor should not win, so B2 as
    specified is measuring pair memorisation, not cell-line transfer.

    B2P holds out cells AND pairs jointly: the test cell line is unseen, and the drug pair in a test
    row was never scored in ANY training cell. `pair_memorizer` then has no same-pair history and
    collapses to the prior, exactly as it does at LCO, so whatever remains is cell-line transfer.

    `strict` additionally requires both drugs to appear somewhere in training, keeping this a
    cell+combination setting rather than sliding into drug-OOD.
    """
    cells = np.unique(cell)
    lo, hi = np.minimum(cid1, cid2), np.maximum(cid1, cid2)
    uniq, pinv = np.unique(np.stack([lo, hi], 1), axis=0, return_inverse=True)
    pinv = pinv.ravel()
    cf = _entity_folds(len(cells), num_folds, seed)
    pf = _entity_folds(len(uniq), num_folds, seed)
    c_te = _isin(cell, set(cells[cf[fold_index]].tolist()))
    c_va = _isin(cell, set(cells[cf[(fold_index + 1) % num_folds]].tolist()))
    p_te = np.zeros(len(uniq), bool); p_te[pf[fold_index]] = True
    p_va = np.zeros(len(uniq), bool); p_va[pf[(fold_index + 1) % num_folds]] = True
    pte, pva = p_te[pinv], p_va[pinv]
    # train: cell in C_train AND pair in P_train -- so a test pair is unseen in EVERY training cell
    train = ~c_te & ~c_va & ~pte & ~pva
    test = c_te & pte
    val = c_va & pva & ~c_te & ~pte
    if strict:
        trd = set(cid1[train].tolist()) | set(cid2[train].tolist())
        seen = _isin(cid1, trd) & _isin(cid2, trd)
        test &= seen
        val &= seen
    return (np.where(train)[0].astype(np.int64), np.where(val)[0].astype(np.int64),
            np.where(test)[0].astype(np.int64))


# ── C2: Cell-line LOO (one cell at a time) ────────────────────────────────────
def split_C2(cell: np.ndarray, loo_index: int):
    cells = np.unique(cell)  # sorted
    c_star = cells[loo_index]
    c_val = cells[(loo_index + 1) % len(cells)]
    test_idx = np.where(cell == c_star)[0]
    val_idx = np.where(cell == c_val)[0]
    train_idx = np.where((cell != c_star) & (cell != c_val))[0]
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


# ── C3: Drug LOO (one drug at a time, OR rule) ────────────────────────────────
def split_C3(cid1: np.ndarray, cid2: np.ndarray, loo_index: int):
    drugs = np.unique(np.concatenate([cid1, cid2]))  # sorted
    d_star = drugs[loo_index]
    d_val = drugs[(loo_index + 1) % len(drugs)]
    test_mask = (cid1 == d_star) | (cid2 == d_star)
    val_mask = ((cid1 == d_val) | (cid2 == d_val)) & (cid1 != d_star) & (cid2 != d_star)
    train_mask = ((cid1 != d_star) & (cid2 != d_star)
                  & (cid1 != d_val) & (cid2 != d_val))
    return (np.where(train_mask)[0].astype(np.int64),
            np.where(val_mask)[0].astype(np.int64),
            np.where(test_mask)[0].astype(np.int64))


# ── C4: Leave-one-triplet-out (per-triplet strict OOD, corrected) ─────────────
def split_C4(cid1: np.ndarray, cid2: np.ndarray, cell: np.ndarray, loo_index: int,
             val_frac: float = 0.1, seed: int = 0):
    """Per-triplet strict OOD. The document's literal C4 val definition empties
    the train set (D_val = every drug in V*); we instead hold out the target
    triplet's two drugs and cell from train, and carve a random val slice from
    the surviving train pool. This preserves the intent ("neither drug nor the
    cell of the test triplet appears in training") without the degenerate val.
    """
    d1s = int(cid1[loo_index]); d2s = int(cid2[loo_index]); cs = int(cell[loo_index])
    test_idx = np.array([loo_index], dtype=np.int64)
    eligible = np.where((cid1 != d1s) & (cid2 != d1s)
                        & (cid1 != d2s) & (cid2 != d2s)
                        & (cell != cs))[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(eligible)
    n_val = int(round(len(perm) * val_frac))
    val_idx = perm[:n_val].astype(np.int64)
    train_idx = perm[n_val:].astype(np.int64)
    return train_idx, val_idx, test_idx


def split_C4_caveat() -> str:
    return (
        "Document C4 spec sets D_val = all drugs in V* and excludes them from "
        "train, which empties S_train. split_C4() implements the intended "
        "strict per-triplet OOD instead (test triplet's drugs+cell excluded "
        "from train; val is a random slice of the surviving pool)."
    )


# ── LOSO: Leave-one-study-out ─────────────────────────────────────────────────
def split_LOSO(study: np.ndarray, loo_index: int, val_frac: float = 0.1, seed: int = 0):
    """Hold out one entire source study (ALMANAC / ONEIL / CLOUD / ...).

    This is the strongest OOD split available to us, and arguably the most honest:
    cross-study is exactly where synergy labels stop reproducing. On overlapping
    ALMANAC<->O'Neil triplets the scores agree at only Pearson r=0.09 (ZIP), 0.25
    (Loewe), 0.342 (CSS) -- Zhang et al., Commun Biol 6:397 (2023). B2/B3/B4 hold out
    entities but keep the assay protocol fixed; LOSO changes lab, protocol and
    readout at once, which is the transfer that actually matters.

    Requires a `study` label per triplet (from DrugComb v1.5 `study_name`).
    Test = the held-out study. Val is also out-of-study (so hyperparameters are never
    tuned on the test protocol), but is assembled from the SMALLEST remaining studies
    until it reaches `val_frac` of the non-test rows. Taking "the next study by size"
    instead is pathological here: holding out CLOUD would hand ALMANAC's 12,505 rows to
    val and leave only 2,629 for train. If no combination of small studies can cover
    val_frac (e.g. only two studies exist), val falls back to a deterministic random
    slice of the non-test rows -- in-study, but used only for early stopping.
    """
    study = np.asarray(study)
    uniq, counts = np.unique(study[study != ""], return_counts=True)
    order = np.argsort(-counts)                 # deterministic: largest study first
    uniq, counts = uniq[order], counts[order]
    if not (0 <= loo_index < len(uniq)):
        raise ValueError(f"loo_index {loo_index} out of range for {len(uniq)} studies: "
                         f"{list(uniq)}")
    held = uniq[loo_index]
    test_mask = study == held
    n_rest = int((~test_mask & (study != "")).sum())
    target = val_frac * n_rest

    # Greedily take the smallest non-test studies until we reach the val budget, but NEVER admit a
    # study large enough to gut the training set: without this guard, holding out ONEIL sweeps
    # ALMANAC's 12,505 rows into val (target was 3,605) and leaves train = CLOUD alone, turning a
    # protocol-transfer test into an extreme prevalence-shift test.
    cand = [(c, u) for u, c in zip(uniq, counts) if u != held]
    cand.sort()                                  # ascending by count
    picked, acc = [], 0
    for c, u in cand:
        if acc >= target:
            break
        # Size guard applies to EVERY candidate. The previous `and picked` clause exempted the
        # first one, so the smallest study was admitted however large it was -- which is exactly
        # the ONEIL/ALMANAC pathology this guard exists to prevent (restricted to
        # {ALMANAC, CLOUD, ONEIL}, holding out ONEIL put ALMANAC's 14,331 rows in val and left
        # train = CLOUD alone). Today's data escapes only because 8 micro-studies get picked first.
        if c > 2 * target:                       # too big for val; leave it in train
            continue
        picked.append(u); acc += c
    val_mask = _isin(study, set(picked)) if picked else np.zeros(len(study), bool)

    # If the small studies could not fund the val budget, top up with a random slice of the
    # remaining train rows rather than swallowing a whole large study.
    if acc < 0.5 * target:
        rng = np.random.RandomState(seed)
        pool = np.where(~test_mask & ~val_mask & (study != ""))[0]
        need = min(len(pool) - 1, int(round(target - acc)))
        if need > 0:
            extra = pool[rng.permutation(len(pool))[:need]]
            val_mask = val_mask.copy(); val_mask[extra] = True

    # guard: never let val swallow most of the data, and never leave train empty
    if acc > 0.5 * n_rest or (~test_mask & ~val_mask & (study != "")).sum() == 0:
        rng = np.random.RandomState(seed)
        rest = np.where(~test_mask & (study != ""))[0]
        pick = rng.permutation(len(rest))[: max(1, int(round(val_frac * len(rest))))]
        val_mask = np.zeros(len(study), bool)
        val_mask[rest[pick]] = True

    train_mask = ~test_mask & ~val_mask & (study != "")
    if train_mask.sum() == 0:
        raise ValueError(
            f"split_LOSO(loo_index={loo_index}) leaves an EMPTY training set: study {held!r} "
            f"covers every usable row. With a single study (or one study plus unjoined rows) "
            f"leave-one-study-out is not defined. Check `loso_studies(study)` first.")
    return (np.where(train_mask)[0].astype(np.int64),
            np.where(val_mask)[0].astype(np.int64),
            np.where(test_mask)[0].astype(np.int64))


def loo_candidates(setup: str, *, cid1, cid2, cell, y, study=None, limit=None) -> list:
    """Which leave-one-out indices can actually produce a number, best first.

    Taking the FIRST N indices is close to worthless for C3. Drug indices are sorted graph node ids
    and 51.6% of drugs occur in exactly one triplet, so `--loo-count 20` on C3 selected
    n_test = [198, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 189, 1, 1, 1, 181, 193, 1, 319, 2] -- 5 of 20
    usable, and `run` refuses the rest with SINGLE_CLASS_TEST. Across all 1,108 C3 indices, 733 have
    an empty or single-class test set. LOSO is the same story: indices 7/8/9 (WILSON, YOHE, DYALL)
    have prevalence 1.000 / 1.000 / 0.000, and the sweep default of 8 always included index 7.

    This returns only indices whose test set is non-empty AND two-class, ordered by test size
    descending, so `--loo-count N` spends the whole budget on runs that yield a metric. The ordering
    is deterministic (size, then index) so a sweep is reproducible.

    NOTE C4 is excluded by construction: its test set is a single triplet, so it is never two-class.
    Leave-one-triplet-out needs predictions POOLED across indices before scoring, which the
    per-run metric path does not do -- see `split_C4_caveat`.
    """
    y = np.asarray(y)
    s = setup.upper()
    if s == "C2":
        n = len(np.unique(np.asarray(cell)))
    elif s == "C3":
        n = len(np.unique(np.concatenate([np.asarray(cid1), np.asarray(cid2)])))
    elif s == "LOSO":
        n = len(loso_studies(study)) if study is not None else 0
    elif s == "C4":
        return []
    else:
        raise ValueError(f"{setup} is not a leave-one-out setup")
    out = []
    for i in range(n):
        try:
            tr, va, te = make_split(setup, cid1=cid1, cid2=cid2, cell=cell, y=y,
                                    loo_index=i, study=study)
        except Exception:
            continue
        if len(te) and len(tr) and len(np.unique(y[te])) >= 2:
            out.append((len(te), i))
    out.sort(key=lambda t: (-t[0], t[1]))
    idx = [i for _, i in out]
    return idx[:limit] if limit else idx


def loso_studies(study: np.ndarray) -> list:
    """Study names in the same order `split_LOSO` indexes them (largest first)."""
    study = np.asarray(study)
    uniq, counts = np.unique(study[study != ""], return_counts=True)
    return list(uniq[np.argsort(-counts)])


# ── Dispatcher ────────────────────────────────────────────────────────────────
def split_EXT(y: np.ndarray, fold_col: np.ndarray, num_folds: int, fold_index: int, seed: int):
    """Independent-source validation: internal rows train/val, fold=-2 rows test only.

    The external label is never consulted by the splitter.  A stratified fold of INTERNAL labels is
    used for early stopping and every other internal row trains the model.  External rows never
    enter training, validation, feature transforms or the train-only DD graph.
    """
    y, marker = np.asarray(y), np.asarray(fold_col)
    ext = np.where(marker == -2)[0]
    internal = np.where(marker != -2)[0]
    if not len(ext):
        raise ValueError("EXT needs rows marked fold=-2 in its fold table")
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = [internal[test] for _, test in skf.split(internal, y[internal])]
    val = folds[fold_index]
    train = np.setdiff1d(internal, val, assume_unique=False)
    return train.astype(np.int64), val.astype(np.int64), ext.astype(np.int64)


def make_split(setup: str, *, cid1, cid2, cell, y,
               num_folds: int = 4, fold_index: int = 0, seed: int = 0,
               loo_index: Optional[int] = None,
               study: Optional[np.ndarray] = None,
               fold_col: Optional[np.ndarray] = None) -> Tuple[IntArr, IntArr, IntArr]:
    cid1 = np.asarray(cid1); cid2 = np.asarray(cid2)
    cell = np.asarray(cell); y = np.asarray(y)
    s = setup.upper()
    if s == "EXT":
        if fold_col is None:
            raise ValueError("EXT needs fold_col with external rows marked -2")
        return split_EXT(y, fold_col, num_folds, fold_index, seed)
    if s == "LOSO":
        if study is None:
            raise ValueError("LOSO needs `study` (per-triplet study_name).")
        if loo_index is None:
            raise ValueError("LOSO needs --loo-index (which study to hold out).")
        return split_LOSO(study, loo_index, seed=seed)
    if s == "B1":
        return split_B1(y, num_folds, fold_index, seed)
    if s == "B2":
        return split_B2(cell, num_folds, fold_index, seed)
    if s == "B3":
        return split_B3(cid1, cid2, num_folds, fold_index, seed)
    if s == "B3A":
        return split_B3A(cid1, cid2, num_folds, fold_index, seed)
    if s == "B4":
        return split_B4(cid1, cid2, cell, num_folds, fold_index, seed)
    if s == "LCO":
        return split_LCO(cid1, cid2, num_folds, fold_index, seed)
    if s == "LCOW":
        return split_LCOW(cid1, cid2, cell, num_folds, fold_index, seed)
    if s == "B2P":
        return split_B2P(cid1, cid2, cell, num_folds, fold_index, seed)
    if s == "C2":
        if loo_index is None:
            raise ValueError("C2 needs --loo-index (which cell to hold out).")
        return split_C2(cell, loo_index)
    if s == "C3":
        if loo_index is None:
            raise ValueError("C3 needs --loo-index (which drug to hold out).")
        return split_C3(cid1, cid2, loo_index)
    if s == "C4":
        if loo_index is None:
            raise ValueError("C4 needs --loo-index (which triplet to hold out).")
        return split_C4(cid1, cid2, cell, loo_index, seed=seed)
    raise ValueError(f"Unknown setup: {setup}")


def verify_no_leak(setup: str, train_idx, val_idx, test_idx, *, cid1, cid2, cell) -> dict:
    """Verify the document's no-leak constraint.

    The held-out drug set under the OR rule is exactly the drugs that appear
    in test triplets but NEVER in train triplets — those are the genuinely
    cold-start drugs. The constraint to verify is therefore: every test
    triplet contains at least one such cold-start drug (i.e. no test triplet
    is composed entirely of training drugs). A train-partner drug legitimately
    appearing in a test triplet is NOT a leak.
    """
    cid1 = np.asarray(cid1); cid2 = np.asarray(cid2); cell = np.asarray(cell)
    out = {"setup": setup, "n_train": len(train_idx), "n_val": len(val_idx),
           "n_test": len(test_idx)}
    train_drugs = set(cid1[train_idx].tolist()) | set(cid2[train_idx].tolist())
    train_cells = set(cell[train_idx].tolist())
    s = setup.upper()
    if s in ("B3", "B3A", "C3", "B4", "C4"):
        # Leak = a test triplet whose BOTH drugs are seen in training.
        d1_in = np.isin(cid1[test_idx], list(train_drugs)) if train_drugs else np.zeros(len(test_idx), bool)
        d2_in = np.isin(cid2[test_idx], list(train_drugs)) if train_drugs else np.zeros(len(test_idx), bool)
        out["test_triplets_fully_seen_drugs"] = int((d1_in & d2_in).sum())
        if s == "B3A":
            # B3A's claim is strictly stronger than B3's: not "at least one drug is cold" but
            # "NEITHER drug was ever trained on". One seen drug is enough to restore a partner
            # marginal, which is the whole quantity the setting exists to remove.
            out["test_rows_with_a_seen_drug"] = int((d1_in | d2_in).sum())
    if s in ("B2", "C2", "B4", "C4"):
        c_in = np.isin(cell[test_idx], list(train_cells)) if train_cells else np.zeros(len(test_idx), bool)
        out["test_triplets_seen_cell"] = int(c_in.sum())
    if s in ("LCO", "LCOW"):
        # Two constraints, and they pull in OPPOSITE directions -- both must hold or the setting is
        # not what it claims:
        #   (a) no test PAIR may appear in train, in any cell   -> that is the held-out unit
        #   (b) both drugs of a test row SHOULD appear in train -> otherwise the row is really B3
        def _pk(a, b):
            return set(zip(np.minimum(a, b).tolist(), np.maximum(a, b).tolist()))
        tr_pairs = _pk(cid1[train_idx], cid2[train_idx])
        te_pk = list(zip(np.minimum(cid1[test_idx], cid2[test_idx]).tolist(),
                         np.maximum(cid1[test_idx], cid2[test_idx]).tolist()))
        out["test_pairs_seen_in_train"] = int(sum(p in tr_pairs for p in te_pk))
        d1_in = np.isin(cid1[test_idx], list(train_drugs)) if train_drugs else np.zeros(len(test_idx), bool)
        d2_in = np.isin(cid2[test_idx], list(train_drugs)) if train_drugs else np.zeros(len(test_idx), bool)
        out["test_rows_with_cold_drug"] = int((~(d1_in & d2_in)).sum())
        if s == "LCOW":
            c_in = np.isin(cell[test_idx], list(train_cells)) if train_cells else np.zeros(len(test_idx), bool)
            out["test_rows_with_cold_cell"] = int((~c_in).sum())
    return out


def verify_no_leak_loso(train_idx, val_idx, test_idx, study) -> dict:
    """LOSO constraint: the held-out study must not appear in train or val at all."""
    study = np.asarray(study)
    test_studies = set(study[test_idx].tolist())
    return {"setup": "LOSO", "n_train": len(train_idx), "n_val": len(val_idx),
            "n_test": len(test_idx), "test_studies": sorted(test_studies),
            "train_rows_from_test_study": int(np.isin(study[train_idx], list(test_studies)).sum()),
            "val_rows_from_test_study": int(np.isin(study[val_idx], list(test_studies)).sum())}


# ═════════════════════════════════════════════════════════════════════════════════════════════
# dpsyn additions — everything below is NEW; the splitters above are the ported originals.
# ═════════════════════════════════════════════════════════════════════════════════════════════

#: Human-readable description of every setting, so `--setup` is self-documenting and the taxonomy
#: lives in code rather than in a markdown file that drifts.
SETUPS = {
    "B1":   ("ID / transductive", "stratified triplet K-fold; drugs AND cells shared train<->test",
             "fold"),
    "B2":   ("Cell-line OOD", "K-fold over cell lines; the held-out cell never appears in train",
             "fold"),
    "B3":   ("Drug OOD", "K-fold over drugs, OR rule: a triplet is test if >=1 drug is held out",
             "fold"),
    "B3A":  ("Drug OOD, AND rule", "K-fold over drugs, AND rule: a triplet is test only if BOTH "
             "drugs are absent from training. B3's train and val exactly; test restricted to the "
             "both-cold subset, so the model is B3's model", "fold"),
    "B4":   ("Joint drug+cell OOD", "test = (>=1 held-out drug) AND a held-out cell", "fold"),
    # NOT in the spec. Named LCO, not "B5" -- the spec's B5 is the GRAPH CONSTRUCTION section,
    # not a split, and reusing the number would collide with it.
    "B2P":  ("Cell-line OOD, pair also held out",
             "K-fold over cells AND pairs jointly; the test cell is unseen and the test pair was "
             "never scored in any training cell, so pair_memorizer collapses to the prior", "fold"),
    "LCO":  ("Combination OOD", "K-fold over drug PAIRS; both drugs stay in train with other "
             "partners, the pair itself is unseen (leave-combination-out)", "fold"),
    "LCOW": ("Pair OOD, warm drugs and cell", "K-fold over drug PAIRS; the pair is unseen while "
             "both individual drugs and the cell line appear in training", "fold"),
    "C2":   ("Cell LOO", "one held-out cell at a time", "loo"),
    "C3":   ("Drug LOO", "one held-out drug at a time (OR rule)", "loo"),
    "C4":   ("Triplet LOO", "one triplet; its two drugs and its cell are excluded from train", "loo"),
    "LOSO": ("Leave-one-study-out", "hold out an entire source study (ALMANAC / CLOUD / ONEIL / ...)",
             "loo"),
    "EXT":  ("Independent Jaaks validation", "train/validation use DrugCombDB only; rows marked "
             "fold=-2 from the Jaaks 2022 validation screen are test-only", "fold"),
}


def describe(setup: str) -> str:
    name, how, kind = SETUPS[setup.upper()]
    return f"{setup.upper()} — {name}: {how}"


def feature_identical_leak(train_idx, test_idx, *, cid1, cid2, drug_x) -> dict:
    """How much of the "cold-start" test set is not actually cold for a feature-based model.

    A 2D fingerprint cannot separate stereoisomers: omeprazole/esomeprazole, quinine/quinidine and
    betamethasone/dexamethasone have BIT-IDENTICAL AtomPair vectors. If one member of such a pair
    is in train and the other is held out at B3, the "unseen" drug arrives with a representation
    the model has already fitted. That is not label leakage -- the label is still unseen -- but it
    does make B3/B4 easier than the split implies, and it is invisible to `verify_no_leak`, which
    only looks at identities.

    Returns the number of test rows containing a held-out drug whose feature vector is identical to
    some training drug's.
    """
    import numpy as _np
    X = drug_x.numpy() if hasattr(drug_x, "numpy") else _np.asarray(drug_x)
    cid1 = _np.asarray(cid1); cid2 = _np.asarray(cid2)
    train_drugs = set(cid1[train_idx].tolist()) | set(cid2[train_idx].tolist())
    test_drugs = set(cid1[test_idx].tolist()) | set(cid2[test_idx].tolist())
    cold = test_drugs - train_drugs
    # Round before hashing. Under `feat_transform=whiten` the rows come out of LAPACK's SVD, which
    # gives no bitwise-identical-row guarantee, so two BIT-IDENTICAL input fingerprints can differ
    # in the last ulp and an exact-bytes hash misses them. Measured before this fix: 135.1 shadowed
    # B3 test rows under `real` vs 6.8 under `real_std` on the SAME splits -- a ~20x under-report of
    # a diagnostic that must be transform-invariant, since whitening is a fixed linear map.
    sig = lambda v: _np.round(v, 6).tobytes()
    train_sigs = {sig(X[i]) for i in train_drugs if X[i].any()}
    shadowed = {i for i in cold if X[i].any() and sig(X[i]) in train_sigs}
    if not shadowed:
        return {"cold_drugs": len(cold), "shadowed_drugs": 0, "shadowed_test_rows": 0}
    sh = _np.fromiter(shadowed, dtype=cid1.dtype, count=len(shadowed))
    rows = int((_np.isin(cid1[test_idx], sh) | _np.isin(cid2[test_idx], sh)).sum())
    return {"cold_drugs": len(cold), "shadowed_drugs": len(shadowed), "shadowed_test_rows": rows}


def split_summary(setup, tr, va, te, *, cid1, cid2, cell, y, study=None, drug_x=None,
                  n_total=None) -> dict:
    """One dict describing a split: sizes, prevalences, cold-entity counts, and leak checks.

    Printed on every run. A degenerate split (empty test, single-class test, a train set of 1,275
    rows at B4) is the single most common reason a number is meaningless, and it is only visible
    if the split is described rather than assumed.
    """
    import numpy as _np
    cid1 = _np.asarray(cid1); cid2 = _np.asarray(cid2); cell = _np.asarray(cell); y = _np.asarray(y)
    out = {"setup": setup, "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te))}
    for nm, ix in (("train", tr), ("val", va), ("test", te)):
        out[f"prev_{nm}"] = float(y[ix].mean()) if len(ix) else float("nan")
    tr_d = set(cid1[tr].tolist()) | set(cid2[tr].tolist())
    tr_c = set(cell[tr].tolist())
    te_d = set(cid1[te].tolist()) | set(cid2[te].tolist())
    te_c = set(cell[te].tolist())
    out["train_drugs"] = len(tr_d); out["test_drugs"] = len(te_d)
    out["cold_drugs"] = len(te_d - tr_d)
    out["train_cells"] = len(tr_c); out["test_cells"] = len(te_c)
    out["cold_cells"] = len(te_c - tr_c)
    if str(setup).upper() == "LOSO" and study is not None:
        out.update(verify_no_leak_loso(tr, va, te, study))
    else:
        out.update(verify_no_leak(setup, tr, va, te, cid1=cid1, cid2=cid2, cell=cell))
    if drug_x is not None and len(te):
        out.update(feature_identical_leak(tr, te, cid1=cid1, cid2=cid2, drug_x=drug_x))
    # degeneracy flags -- these make the run's metrics uninterpretable, so name them explicitly
    flags = []
    if out["n_test"] == 0:
        flags.append("EMPTY_TEST")
    elif len(_np.unique(y[te])) < 2:
        flags.append("SINGLE_CLASS_TEST")
    if out["n_train"] < 500:
        flags.append(f"TINY_TRAIN({out['n_train']})")
    # Train fraction must be measured against the WHOLE dataset, not against the rows this split
    # happened to assign. B4 discards every row outside a coherent drug x cell quadrant -- 72.4% of
    # rows at num_folds=4 -- so the assigned-rows denominator reported train_frac=0.689 for a split
    # that actually trains on 19.0% of the data, and the LOW_TRAIN_FRAC guard never fired.
    #
    # The (num_folds-2)/num_folds rule holds ONLY for B1, which folds rows. Entity-level folds give
    # very different fractions because the constraint compounds: B3 needs BOTH drugs to survive, so
    # it is ~((k-2)/k)^2 (24.6% at k=4, not 50%); B2 goes the other way (72.7%) because cell folds
    # are wildly unbalanced. Measured on extmap:
    #     nf     B1      B2      B3      B4
    #      3  0.333   0.149   0.120   0.086
    #      4  0.500   0.727   0.246   0.190
    #      5  0.600   0.806   0.343   0.288
    # A B3 or B4 number is therefore NOT comparable to a B1 number at equal num_folds -- it is a
    # smaller-train model. `train_frac` is written into every sweep row so the confound is visible.
    assigned = out["n_train"] + out["n_val"] + out["n_test"]
    tot = int(n_total) if n_total else assigned
    out["n_total"] = tot
    out["n_dropped"] = max(0, tot - assigned)
    out["train_frac"] = out["n_train"] / tot if tot else 0.0
    out["assigned_frac"] = assigned / tot if tot else 0.0
    if tot and out["train_frac"] < 0.40:
        flags.append(f"LOW_TRAIN_FRAC({out['train_frac']:.0%})")
    if tot and out["n_dropped"] > 0.10 * tot:
        flags.append(f"DROPPED_ROWS({out['n_dropped'] / tot:.0%})")
    if out["n_val"] == 0:
        flags.append("EMPTY_VAL")
    # LCO's whole claim is "the pair is unseen but the drugs are not" -- flag either half failing.
    if out.get("test_pairs_seen_in_train"):
        flags.append(f"PAIR_LEAK({out['test_pairs_seen_in_train']})")
    if str(setup).upper() in ("LCO", "LCOW") and out.get("test_rows_with_cold_drug"):
        flags.append(f"COLD_DRUG_IN_LCO({out['test_rows_with_cold_drug']})")
    if str(setup).upper() == "LCOW" and out.get("test_rows_with_cold_cell"):
        flags.append(f"COLD_CELL_IN_LCOW({out['test_rows_with_cold_cell']})")
    if out.get("test_rows_with_a_seen_drug"):
        flags.append(f"SEEN_DRUG_IN_B3A({out['test_rows_with_a_seen_drug']})")
    # NOT `elif` -- chaining this off the LCO branch made SINGLE_CLASS_VAL unreachable for any LCO
    # fold that also had a cold drug, i.e. exactly the folds most likely to be degenerate.
    if len(va) and len(_np.unique(y[va])) < 2:
        flags.append("SINGLE_CLASS_VAL")

    # ── THE ENTITY-LEAK FLAGS ────────────────────────────────────────────────────────────────
    # `verify_no_leak` above already computes these two counts and puts them in `out`, but nothing
    # ever read them: the flag list was built without consulting either, and neither was a sweep
    # column. A B2 split in which 100% of test rows used a training cell produced `degenerate = []`
    # and a CSV row that read as clean. (`leak_check` is a different thing -- it is the random_row
    # floor control.) These are the document's own constraints, so a violation is a FAILED RUN, not
    # a footnote.
    if out.get("test_triplets_seen_cell"):
        flags.append(f"CELL_LEAK({out['test_triplets_seen_cell']})")
    if out.get("test_triplets_fully_seen_drugs"):
        flags.append(f"DRUG_LEAK({out['test_triplets_fully_seen_drugs']})")
    if out.get("train_rows_from_test_study") or out.get("val_rows_from_test_study"):
        flags.append(f"STUDY_LEAK({out.get('train_rows_from_test_study', 0)}"
                     f"/{out.get('val_rows_from_test_study', 0)})")
    out["degenerate"] = flags
    return out


def format_summary(s: dict) -> str:
    bits = [f"train={s['n_train']:,}(p={s['prev_train']:.3f})",
            f"val={s['n_val']:,}", f"test={s['n_test']:,}(p={s['prev_test']:.3f})",
            f"cold_drugs={s['cold_drugs']}/{s['test_drugs']}",
            f"cold_cells={s['cold_cells']}/{s['test_cells']}"]
    if s.get("shadowed_test_rows"):
        bits.append(f"fp-shadowed_test_rows={s['shadowed_test_rows']}")
    if s.get("degenerate"):
        bits.append("!! " + " ".join(s["degenerate"]))
    return "[split] " + "  ".join(bits)
