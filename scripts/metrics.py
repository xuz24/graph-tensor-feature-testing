"""Metrics, with the caveats attached to the metric rather than left in a markdown file.

Three families, reported together because each is misleading alone:

  auroc / ap          threshold-free, but DOMINATED BY MAIN EFFECTS. An additive score built from
                      three lookup tables (drug1's rate, drug2's rate, the cell's rate) reaches
                      AUROC 0.8695 on this data with no model and no interaction term. So an AUROC
                      of 0.91 is not "0.41 above chance", it is ~0.04 above a lookup table.

  mrr / hits@10       per-cell ranking. SATURATES at two-band prevalence (~31%): every arm scores
                      0.95-1.00 and several score exactly 1.0. Informative ONLY in the ~5%
                      single-threshold regime (`folds="zip_single"`). `rank_metrics` refuses to
                      report them above `saturation_prevalence` rather than emitting a number that
                      looks like a result.

  wq_*                within-query ranking: fix (anchor drug, cell) and rank the observed partners.
                      The anchor's and the cell's main effects are CONSTANT within a query, so they
                      cannot move the ranking -- they are removed by conditioning, not arithmetic.
                      Exactly one main effect survives: the partner's own marginal rate. So the bar
                      is `partner_marginal`, not 0.5. Beating it is the only evidence of genuine
                      pair-specific interaction this project has been able to construct.

Why not residualisation: subtracting sigmoid(main_logit) from a BINARY label does not orthogonalise
it. If drug A is positive in all its rows then residual = 1 - p_A > 0 everywhere, and one-hot
simply relearns "A -> positive residual" (it scored 0.8610 on the residual, which should have been
chance). Conditioning works; subtraction does not.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)

SATURATION_PREVALENCE = 0.15


def select_antagonism_threshold(y_true, score) -> float:
    """Select a synergy-score cutoff that maximises antagonism F1.

    Selection belongs on validation predictions.  A row is predicted antagonistic when its
    synergy probability is at or below the returned cutoff.  The deterministic last-maximum rule
    favours the most precise antagonism threshold when several cutoffs have the same F1 because
    ``precision_recall_curve`` returns thresholds in ascending antagonist-score order.
    """
    y = np.asarray(y_true)
    s = np.asarray(score)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5
    ant_y, ant_s = 1 - y, 1.0 - s
    precision, recall, thresholds = precision_recall_curve(ant_y, ant_s)
    if not len(thresholds):
        return 0.5
    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denom,
                   out=np.zeros_like(denom), where=denom > 0)
    maxima = np.flatnonzero(np.isclose(f1, np.nanmax(f1)))
    return float(1.0 - thresholds[int(maxima[-1])])


def basic(y_true, score, *, decision_threshold=0.5) -> dict:
    """Global binary metrics from both the synergy and antagonism viewpoints.

    ``antagonism_auroc`` is intentionally emitted even though it is mathematically identical to
    ``auroc``: reversing both the label and score preserves every positive-negative ordering.  AP
    and the thresholded metrics are not symmetric and therefore carry the new information the
    reviewer requested.  ``decision_threshold`` is expressed on the synergy score; antagonism is
    predicted when ``score <= decision_threshold``.
    """
    y = np.asarray(y_true)
    s = np.asarray(score)
    if len(np.unique(y)) < 2:
        return {k: float("nan") for k in (
            "auroc", "ap", "antagonism_auroc", "antagonism_ap",
            "antagonism_precision", "antagonism_recall", "antagonism_f1")
        } | {"antagonism_threshold": float(decision_threshold)}
    ant_y, ant_s = 1 - y, 1.0 - s
    ant_pred = (s <= decision_threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ant_y, ant_pred, average="binary", zero_division=0)
    return {
        "auroc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "antagonism_auroc": float(roc_auc_score(ant_y, ant_s)),
        "antagonism_ap": float(average_precision_score(ant_y, ant_s)),
        "antagonism_precision": float(precision),
        "antagonism_recall": float(recall),
        "antagonism_f1": float(f1),
        "antagonism_threshold": float(decision_threshold),
    }


def rank_metrics(y_true, score, group, *, prevalence=None, min_size=2) -> dict:
    """Per-group (group = cell line) MRR and Hits@10.

    Returns NaN when prevalence is above `SATURATION_PREVALENCE`: at 31% positives these metrics
    are pinned near 1.0 for every arm and reporting them invites a comparison that cannot
    discriminate. Explicit NaN beats a saturated number that reads like a result.
    """
    y = np.asarray(y_true); g = np.asarray(group); s = np.asarray(score)
    p = float(y.mean()) if prevalence is None else prevalence
    if p > SATURATION_PREVALENCE:
        return {"mrr": float("nan"), "hits10": float("nan"),
                "rank_note": f"suppressed: prevalence {p:.3f} > {SATURATION_PREVALENCE} (saturates)"}
    rr, h10 = [], []
    for q in np.unique(g):
        m = g == q
        if m.sum() < min_size or y[m].sum() == 0:
            continue
        order = y[m][np.argsort(-s[m], kind="mergesort")]
        rr.append(1.0 / (int(np.argmax(order == 1)) + 1))
        h10.append(1.0 if order[:10].sum() > 0 else 0.0)
    return {"mrr": float(np.mean(rr)) if rr else float("nan"),
            "hits10": float(np.mean(h10)) if h10 else float("nan"),
            "n_rank_groups": len(rr)}


def build_queries(d1, d2, cell, test_idx, min_candidates=3):
    """(anchor drug, cell) -> its candidate partner rows, over TEST rows only.

    Each test row joins two queries, once per drug acting as the anchor.
    """
    qmap = defaultdict(list)
    for k in test_idx:
        qmap[(int(d1[k]), int(cell[k]))].append(int(k))
        qmap[(int(d2[k]), int(cell[k]))].append(int(k))
    return [v for v in qmap.values() if len(v) >= min_candidates]


def within_query(queries, y, score) -> dict:
    """Average AUROC / MRR / Hits@1 over queries, each ranked independently. TIE-CORRECTED.

    Ties are not a corner case here, they are the dominant case at OOD. `partner_marginal` assigns
    *exactly* the prior to every drug with no training rows, and at B3 that is 375 of 576 test
    drugs -- so 614 of 1,293 queries are FULLY tied. Ranking those by a stable argsort orders them
    by fold-CSV row position, which is arbitrary, and it inflated the reported bar by **+0.065
    AUROC at B3 and +0.071 at B4**. On an all-tied query the old code returned 0.25 or 0.75
    depending on row order where the answer is 0.5.

    Corrections applied:
      auroc   midranks for tied scores (the standard Mann-Whitney tie correction; matches
              sklearn's roc_auc_score exactly).
      mrr     EXACT expected reciprocal rank under uniform random tie-breaking, E[1/rank].
              NOT 1/E[rank] (the midrank shortcut): by Jensen those differ, and on a fully-tied
              7-candidate query the shortcut gives 0.25 against a true 0.53 -- it penalises tied
              arms against arms whose continuous scores never tie.
      hits1   expected value: (# positives in the top tie group) / (size of the top tie group).
    """
    from scipy.stats import rankdata
    y = np.asarray(y); s = np.asarray(score)
    au, mrr, h1 = [], [], []
    for rows in queries:
        yy = y[rows]
        if len(rows) < 2 or yy.sum() == 0 or yy.sum() == len(rows):
            continue
        sc = s[rows]
        P, N = int(yy.sum()), int(len(yy) - yy.sum())
        # midranks -> tie-corrected AUC
        r = rankdata(sc, method="average")
        au.append((r[yy == 1].sum() - P * (P + 1) / 2) / (P * N))
        # EXPECTED RECIPROCAL RANK under random tie-breaking, computed exactly.
        #
        # The obvious shortcut -- give each item its midrank and take 1/midrank -- computes
        # 1/E[rank], not E[1/rank]. By Jensen those differ, and the gap is large exactly where ties
        # dominate: a fully-tied query of 7 candidates gets 1/4 = 0.25 from the shortcut but ~0.45
        # from an honest random tie-break. That systematically penalises heavily-tied arms
        # (constant features, partner_marginal at cold-start) against arms whose continuous scores
        # never tie, which is precisely the comparison this grid exists to make.
        #
        # Exact form: let g = items scoring strictly higher than the best-scoring positive's tie
        # group, t = size of that group, p = positives inside it. The first positive lands at
        # position k within the group with P(K=k) = C(t-k, p-1) / C(t, p), so
        #     E[1/rank] = sum_k P(K=k) / (g + k)
        best = sc[yy == 1].max()
        g = int((sc > best).sum())
        grp = sc == best
        t, pp = int(grp.sum()), int((grp & (yy == 1)).sum())
        from math import comb
        denom = comb(t, pp)
        mrr.append(float(sum(comb(t - k, pp - 1) / denom / (g + k)
                             for k in range(1, t - pp + 2))))
        # expected Hits@1 = P(the item drawn from the top tie group is positive)
        top = sc == sc.max()
        h1.append(float(yy[top].sum() / top.sum()))
    return {"wq_auroc": float(np.mean(au)) if au else float("nan"),
            "wq_mrr": float(np.mean(mrr)) if mrr else float("nan"),
            "wq_hits1": float(np.mean(h1)) if h1 else float("nan"),
            "n_queries": len(au)}


def evaluate(y, score, *, test_idx, cell, d1, d2, queries=None, min_candidates=3,
             prevalence=None, decision_threshold=0.5) -> dict:
    """Every metric for one set of test scores. `score` is indexed like `y` (full length)."""
    y = np.asarray(y); score = np.asarray(score)
    out = basic(y[test_idx], score[test_idx], decision_threshold=decision_threshold)
    out.update(rank_metrics(y[test_idx], score[test_idx], np.asarray(cell)[test_idx],
                            prevalence=prevalence))
    if queries is None:
        queries = build_queries(d1, d2, cell, test_idx, min_candidates)
    out.update(within_query(queries, y, score))
    return out
