"""
calibration.py
==============
Calibration procedures compared in the simulation study.

Every baseline delegates to the reference implementation of Ding et al. (2023),
vendored in `ding_conformal_utils.py`. Nothing in the calibration logic is
reimplemented here: this module only fixes the interface, so that all methods
take scores and return prediction sets in the same shape.

Methods
-------
marginal   one quantile over all calibration scores               (Ding)
classwise  one quantile per class, from that class's scores       (Ding)
clustered  classes grouped by score distribution, quantile per group (Ding)
naive_multicopy  classwise, but each transformed copy of a calibration
                 point is entered as a separate calibration unit

The last one is not a proposed method. It is the procedure that treats
|G| * n_y transformed copies as if they were n_y exchangeable calibration
units. Copies of the same observation are deterministic functions of a common
random variable, so the split-conformal guarantee does not cover it, and the
non-exchangeable machinery of Barber et al. (2023) does not rescue it either:
the total variation terms attain their maximum and the bound is vacuous. It is
included precisely so that the size of the resulting coverage error can be
measured rather than assumed.

Note that `naive_multicopy` is literally `classwise` applied to the augmented
scores. The only difference is that n_y is replaced by |G| * n_y in the
quantile, which is what makes the comparison between the two arms a clean
isolation of the exchangeability question.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np

import ding_conformal_utils as ding

__all__ = ["marginal_sets", "classwise_sets", "clustered_sets",
           "naive_multicopy_sets", "METHOD_GUARANTEE"]


# What each method actually targets. Used when reporting results: comparing the
# set size of a method targeting class-conditional coverage against one
# targeting only marginal coverage is not a like-for-like comparison, and the
# tables should group by this rather than list every method in one column.
METHOD_GUARANTEE = {
    "marginal": "marginal",
    "clustered": "cluster-conditional",   # degrades to marginal for rare classes
    "classwise": "class-conditional",
    "naive_multicopy": "none",
}


def _silent(fn, *args, **kwargs):
    """Call a reference function, suppressing its diagnostic output.

    Two things are suppressed, both expected in the low-data regime and both
    already recorded in the returned quantities:

      - Ding's helpers print a message whenever a class has too few calibration
        points to admit a finite quantile. That is reported through the
        returned q_hat being infinite.
      - NumPy emits an empty-slice warning, and the invalid-divide warning that
        follows it, when clustered conformal ends up with an empty cluster.
        That happens when the rare classes hold at most 1/alpha - 1 points and
        cannot be assigned, and it is reported through the procedure collapsing
        onto the marginal method, which `metrics.clustered_diagnostic` counts.

    At the scale of hundreds of replications both are noise. The filters are
    deliberately narrow: any other warning still surfaces.
    """
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Mm]ean of empty slice.*",
                                category=RuntimeWarning)
        warnings.filterwarnings("ignore",
                                message=".*invalid value encountered in.*",
                                category=RuntimeWarning)
        return fn(*args, **kwargs)


def _to_boolean_matrix(set_preds, n_classes: int) -> np.ndarray:
    """Convert Ding's prediction sets to a dense boolean matrix.

    The reference implementation returns, for each instance, an array of the
    included label indices. A dense (n_instances, n_classes) boolean matrix is
    easier to aggregate per class, which is the unit of the output table.
    """
    out = np.zeros((len(set_preds), n_classes), dtype=bool)
    for i, labels in enumerate(set_preds):
        out[i, np.asarray(labels, dtype=int)] = True
    return out


def marginal_sets(cal_scores, cal_labels, test_scores, alpha):
    """Split conformal with a single marginal quantile.

    Parameters
    ----------
    cal_scores : (n_cal, n_classes) array of scores for every candidate label
    cal_labels : (n_cal,) true labels
    test_scores : (n_test, n_classes)
    alpha : float

    Returns
    -------
    sets : (n_test, n_classes) boolean array, True where the label is included
    q_hat : float
    """
    q_hat = _silent(ding.compute_qhat, cal_scores, cal_labels, alpha=alpha)
    sets = _silent(ding.create_prediction_sets, test_scores, q_hat)
    return _to_boolean_matrix(sets, test_scores.shape[1]), float(np.asarray(q_hat).ravel()[0])


def classwise_sets(cal_scores, cal_labels, test_scores, alpha, n_classes):
    """Split conformal with one quantile per class.

    A class with n_y calibration points admits a finite quantile only when
    ceil((n_y + 1)(1 - alpha)) <= n_y. Below that threshold the reference
    implementation returns an infinite q_hat, so the label is included in every
    prediction set. This is the conservative behaviour that makes classwise
    calibration expensive in set size when classes are rare, and it is the
    regime the study is about.

    Returns
    -------
    sets : (n_test, n_classes) boolean array
    q_hats : (n_classes,) array, possibly containing np.inf
    """
    q_hats = _silent(
        ding.compute_class_specific_qhats,
        cal_scores, cal_labels,
        num_classes=n_classes, alpha=alpha, default_qhat=np.inf,
    )
    sets = _silent(ding.create_classwise_prediction_sets, test_scores, q_hats)
    return _to_boolean_matrix(sets, n_classes), np.asarray(q_hats, dtype=float)


def clustered_sets(cal_scores, cal_labels, test_scores, test_labels, alpha, seed=0):
    """Clustered conformal prediction.

    Classes with similar score distributions share a quantile, which buys
    stability at the cost of a weaker, cluster-conditional guarantee. When rare
    classes have at most 1/alpha - 1 samples they cannot be assigned to any
    cluster, clustering is skipped, and the procedure reduces to the marginal
    method. That degradation is expected in the low-data regime and is reported
    rather than hidden.

    Returns
    -------
    sets : (n_test, n_classes) boolean array
    """
    _, sets, _, _ = _silent(
        ding.clustered_conformal,
        cal_scores, cal_labels, alpha,
        val_scores_all=test_scores, val_labels=test_labels, seed=seed,
    )
    return _to_boolean_matrix(sets, test_scores.shape[1])


def naive_multicopy_sets(aug_scores, aug_labels, test_scores, alpha, n_classes):
    """Classwise calibration treating each transformed copy as its own unit.

    `aug_scores` stacks the scores of all transformed copies of every
    calibration point, and `aug_labels` repeats the corresponding labels. The
    per-class quantile is then computed over |G| * n_y values.

    This has no finite-sample guarantee. Under exact invariance each transformed
    score has the correct marginal distribution, although the collection of
    transformed scores is not exchangeable. The arm is run to measure the size
    of the resulting coverage error rather than to assume it.
    """
    return classwise_sets(aug_scores, aug_labels, test_scores, alpha, n_classes)
