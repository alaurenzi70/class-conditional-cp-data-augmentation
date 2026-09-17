"""
scores.py
=========
Nonconformity scores for the simulation study.

Three score functions are implemented, all built on the randomised APS score:

    plain APS          S(x, y, u)      = cumulative mass above y, minus u * p_y
    prediction-avg     S(fbar(x), y, u)   average the softmax over the orbit,
                                          then score once  (TTA-style)
    score-avg          mean_b S(f(g_b x), y, u)   score each transformed point,
                                                  then average  (orbit average)

The two averaged scores differ because APS is not linear in the softmax vector:
it depends on the ordering of the probabilities and on cumulative sums, so

    S(mean_b f(g_b x), y, u)  !=  mean_b S(f(g_b x), y, u)

in general. Which of the two is preferable is an empirical question; both are
fixed measurable functions of (x, y, u) conditionally on the transformations
g_1, ..., g_B, so both inherit the split-conformal coverage guarantee.

Relation to the reference implementation
----------------------------------------
The deterministic part of the APS calculation follows Ding et al.
(https://github.com/tiffanyding/class-conditional-conformal, utils/): sort the
predicted probabilities in decreasing order, take cumulative sums, map back to
the original label order.

The randomisation differs from their vectorised implementation in two ways.

First, U is passed in rather than generated internally from
`np.random.seed(seed)`. The original helper reseeds the global NumPy RNG on
every call, so calling it with the same seed for the calibration and the test
softmax matrices makes the two U matrices coincide on their first
min(n_cal, n_test) rows. The pairs ((X_i, Y_i), u_i) are then no longer
exchangeable with the test pair, and the guarantee no longer applies.

Second, one scalar U_i per observation is shared across candidate labels,
consistently with the original APS definition, whereas the reference
implementation draws an independent U_iy for every (observation, label) pair.
With a shared U_i the prediction set stays nested with respect to the ranking
of the predicted probabilities; with independent draws a label with smaller
cumulative mass can be excluded while a later one is included.

Neither change affects coverage. The calibration score uses only the entry of
the true class, and S_A(X_i, Y_i, U_{i,Y_i}) has the same law under both
schemes, so the quantile is unchanged in distribution and the baselines remain
comparable. What changes is the geometry of the set, hence its size.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "draw_uniform",
    "aps_scores",
    "orbit_softmax",
    "prediction_averaged_scores",
    "score_averaged_scores",
    "within_orbit_score_variance",
    "true_class_rank",
]


# ---------------------------------------------------------------------------
# Randomisation
# ---------------------------------------------------------------------------

def draw_uniform(n: int, n_classes: int, rng: np.random.Generator) -> np.ndarray:
    """Draw one APS randomisation variable per observation.

    The same scalar U_i is used for every candidate label and for every
    transformed version of observation i. The returned matrix repeats U_i
    across labels for compatibility with the vectorised score implementation.

    Calibration and test draws must come from independent generator streams.
    The intended usage is one `np.random.SeedSequence` per replication, spawned
    into separate children for the calibration set, the test sets and the
    transformations.
    """
    u = rng.random((n, 1))
    return np.repeat(u, n_classes, axis=1)


# ---------------------------------------------------------------------------
# Plain APS
# ---------------------------------------------------------------------------

def aps_scores(softmax: np.ndarray, U: np.ndarray | None = None) -> np.ndarray:
    """Randomised APS score for every candidate label.

    Parameters
    ----------
    softmax : np.ndarray of shape (n, n_classes)
        Predicted class probabilities, rows summing to one.
    U : np.ndarray of shape (n, n_classes) or None
        Tie-breaking variables. If None the deterministic (non-randomised)
        score is returned, which is conservative.

    Returns
    -------
    np.ndarray of shape (n, n_classes): score of each label for each row.
    Lower scores mean the label conforms better.
    """
    softmax = np.asarray(softmax, dtype=float)

    # Cumulative probability mass up to and including each label, when labels
    # are ranked by decreasing predicted probability, then mapped back to the
    # original label order.
    order = np.argsort(-softmax, axis=1)
    sorted_probs = np.take_along_axis(softmax, order, axis=1)
    cumulative = np.cumsum(sorted_probs, axis=1)
    inverse = np.argsort(order, axis=1)
    scores = np.take_along_axis(cumulative, inverse, axis=1)

    if U is None:
        return scores - softmax
    U = np.asarray(U, dtype=float)
    if U.shape != softmax.shape:
        raise ValueError(f"U has shape {U.shape}, expected {softmax.shape}")
    return scores - U * softmax


# ---------------------------------------------------------------------------
# Orbit machinery
# ---------------------------------------------------------------------------

def orbit_softmax(model, X: np.ndarray, group, eps_batch: np.ndarray) -> np.ndarray:
    """Predicted probabilities for every transformed copy of X.

    Parameters
    ----------
    model : fitted classifier exposing `predict_proba`
    X : np.ndarray of shape (n, n_features)
    group : SignFlipGroup
    eps_batch : np.ndarray of shape (B, n_nuisance), shared across all rows

    Returns
    -------
    np.ndarray of shape (B, n, n_classes)
    """
    return np.stack(
        [model.predict_proba(group.apply(X, eps)) for eps in np.atleast_2d(eps_batch)],
        axis=0,
    )


def prediction_averaged_scores(
    orbit_probs: np.ndarray, U: np.ndarray | None = None
) -> np.ndarray:
    """Average the softmax over the orbit, then compute APS once.

    This is the aggregation point used by test-time augmentation: predictions
    are pooled first, so the score sees a single, denoised probability vector.

    Averaging is uniform over the sampled transformations. This is TTA-Avg, not
    the learned-policy variant of Shanmugam et al., which fits augmentation
    weights on a separate split. Uniform averaging is used deliberately, to
    isolate the aggregation point from the choice of policy.

    Parameters
    ----------
    orbit_probs : np.ndarray of shape (B, n, n_classes), from `orbit_softmax`
    U : np.ndarray of shape (n, n_classes) or None

    Returns
    -------
    np.ndarray of shape (n, n_classes)
    """
    mean_probs = np.asarray(orbit_probs, dtype=float).mean(axis=0)
    # Renormalise: averaging keeps rows summing to one up to floating point,
    # but APS relies on the cumulative mass reaching exactly one.
    mean_probs = mean_probs / mean_probs.sum(axis=1, keepdims=True)
    return aps_scores(mean_probs, U)


def score_averaged_scores(
    orbit_probs: np.ndarray, U: np.ndarray | None = None
) -> np.ndarray:
    """Compute APS on each transformed copy, then average the scores.

    This is the orbit-averaged score of the theory: a fixed measurable function
    of (x, y, u) conditionally on the transformations, hence a valid
    nonconformity score with no invariance assumption required.

    The same U is used for every b. Each observation therefore keeps a single
    tie-breaking variable, and the resulting score remains a fixed measurable
    function of (x, y, u).

    Parameters
    ----------
    orbit_probs : np.ndarray of shape (B, n, n_classes), from `orbit_softmax`
    U : np.ndarray of shape (n, n_classes) or None

    Returns
    -------
    np.ndarray of shape (n, n_classes)
    """
    orbit_probs = np.asarray(orbit_probs, dtype=float)
    per_transform = np.stack([aps_scores(p, U) for p in orbit_probs], axis=0)
    return per_transform.mean(axis=0)


# ---------------------------------------------------------------------------
# Mechanism diagnostics
# ---------------------------------------------------------------------------

def within_orbit_score_variance(
    orbit_probs: np.ndarray, labels: np.ndarray, U: np.ndarray | None = None
) -> np.ndarray:
    """Per-observation variance of the true-label score across the orbit.

    This estimates Var_g[S(gX, Y)], the quantity governing the variance
    reduction: orbit averaging can only help to the extent that the score
    varies within an orbit. Where this is close to zero the theory predicts no
    gain, so a null result here is a confirmation rather than a failure.

    Parameters
    ----------
    orbit_probs : np.ndarray of shape (B, n, n_classes)
    labels : np.ndarray of shape (n,), true class indices
    U : np.ndarray of shape (n, n_classes) or None

    Returns
    -------
    np.ndarray of shape (n,): within-orbit variance for each observation.
    """
    orbit_probs = np.asarray(orbit_probs, dtype=float)
    labels = np.asarray(labels).astype(int)
    rows = np.arange(labels.shape[0])
    per_transform = np.stack(
        [aps_scores(p, U)[rows, labels] for p in orbit_probs], axis=0
    )
    return per_transform.var(axis=0, ddof=0)


def true_class_rank(softmax: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Rank of the true class in the predicted probabilities, 1 = top.

    Reported for the plain and the averaged predictions. APS accumulates the
    probability mass of every label ranked above the true one, so a lower rank
    directly shrinks the score of the true label and hence the prediction set.
    Tracking the rank separates "the set got smaller because the ranking
    improved" from "the set got smaller because the threshold moved".

    Parameters
    ----------
    softmax : np.ndarray of shape (n, n_classes)
    labels : np.ndarray of shape (n,)

    Returns
    -------
    np.ndarray of shape (n,), integer ranks in 1..n_classes.
    """
    softmax = np.asarray(softmax, dtype=float)
    labels = np.asarray(labels).astype(int)
    rows = np.arange(labels.shape[0])
    true_prob = softmax[rows, labels][:, None]
    # Number of labels with strictly greater probability, plus one.
    return (softmax > true_prob).sum(axis=1) + 1
