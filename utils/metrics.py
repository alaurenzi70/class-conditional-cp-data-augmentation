"""
metrics.py
==========
Aggregation of the long-format simulation output.

`simulate.run` writes one row per (replication, train_aug, method, class) and
performs no aggregation. Everything the paper reports is derived here, so a new
question never requires re-running the simulation.

Definitions
-----------
Let Cov_y be the empirical coverage of class y on the test set of a given
replication, and let 1 - alpha be the nominal level.

    MCCD  = mean_y |Cov_y - (1 - alpha)|          macro coverage deviation
    WCCD  = max_y  |Cov_y - (1 - alpha)|          worst-class deviation
    WCU   = max_y  [(1 - alpha) - Cov_y]_+        worst-class UNDER-coverage

MCCD is the CovGap of the submitted version and is retained for continuity. It
penalises over- and under-coverage symmetrically, so it does not distinguish a
conservative method from one that undercovers by the same amount. WCU is the
quantity that matters when a rare class is a rare defect, since only
under-coverage is a failure there, and it is the primary metric here.

Per-class quantities come from the balanced test set, marginal ones from the
population-distributed test set. Keeping the two separate avoids reporting a
macro average as if it were marginal coverage, which is what an equal-weight
average over classes actually is.

All three are computed within a replication and then averaged across
replications, with the Monte Carlo standard error of that average reported
alongside. Averaging classes before replications, or vice versa, gives
different numbers; the order used here is the one the definitions imply.

The distribution-free reference band
------------------------------------
For a continuous score distribution, the coverage of classwise conformal
conditional on the calibration set follows

    Beta(k_y, n_y + 1 - k_y),      k_y = ceil((n_y + 1)(1 - alpha)),

a law depending on the data only through n_y and alpha, and invariant to the
choice of nonconformity score. No score transformation can beat it. Plotting
the observed spread of class-conditional coverage against this band shows
directly what orbit averaging can and cannot do: it can move the score
distribution, hence set size, but not the stability of coverage at fixed n_y.

`beta_reference` returns that band, optionally inflated for the binomial noise
of a finite test set. Writing C_y for the calibration-conditional coverage and
Chat_y for its estimate on m_y test points,

    Var(Chat_y) = Var(C_y) + E[C_y (1 - C_y)] / m_y,

and for C_y ~ Beta(a, b) the second term is ab / [m_y (a+b)(a+b+1)]. The
reference to compare against an empirical spread is therefore

    sqrt( ab / [(a+b)^2 (a+b+1)] + ab / [m_y (a+b)(a+b+1)] ).

The comparison must be made class by class: the Beta law describes one class
with n_y calibration points, so contrasting it with the spread of a statistic
aggregated over classes, such as the minimum coverage, is not like-for-like.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

__version__ = "2026-08-rev3"

__all__ = ["load", "per_replication", "summarize", "ablation_table",
           "beta_reference", "beta_reference_table", "mechanism_table",
           "pareto_front", "rare_class_spread", "paired_delta",
           "clustered_diagnostic", "DISPLAY_NAMES", "finite_threshold_min",
           "degenerate_fraction", "infinite_threshold_curve",
           "dispersion_ratio", "paired_reduction", "classifier_robustness"]


# Names used in tables and figures. The distinction between TTA-Avg and the
# learned-policy variant matters: uniform averaging over sampled
# transformations is not the method of Shanmugam et al., which fits
# augmentation weights on a separate split.
DISPLAY_NAMES = {
    "marginal_plain":     "Marginal APS",
    "classwise_plain":    "Classwise APS",
    "clustered":          "Clustered APS",
    "marginal_ttaavg":    "Marginal TTA-Avg APS",
    "classwise_ttaavg":   "Classwise TTA-Avg APS",
    "marginal_orbitavg":  "Marginal orbit-averaged APS",
    "classwise_orbitavg": "Classwise orbit-averaged APS",
    "naive_multicopy":    "Naive multi-copy APS",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(outdir: str = "results_new") -> pd.DataFrame:
    """Load and concatenate every long table written into `outdir`."""
    # Checkpoints are partial tables written mid-cell. Loading one as if it
    # were a finished cell would silently mix partial and complete runs.
    paths = sorted(p for p in
                   glob.glob(os.path.join(outdir, "*__long.parquet")) +
                   glob.glob(os.path.join(outdir, "*__long.csv.gz"))
                   if "__ckpt" not in p)
    if not paths:
        raise FileNotFoundError(f"no long tables found in {outdir!r}")
    frames = [pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
              for p in paths]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-replication reduction
# ---------------------------------------------------------------------------

# Columns identifying an experimental cell. A replication is one draw within a
# cell, so these are what we group by before collapsing classes.
CELL = ["scenario", "model", "alpha", "n_cal", "n_transforms",
        "exact_invariance", "train_aug", "method", "guarantee"]


def per_replication(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the class dimension, one row per replication and cell.

    Coverage metrics are computed across classes within a replication; set size
    and marginal coverage are already replication-level and are simply carried
    through.
    """
    target = 1.0 - df["alpha"]
    dev = df["coverage_cls"] - target
    work = df.assign(_dev=dev, _absdev=dev.abs(), _under=(-dev).clip(lower=0))

    agg = (work.groupby(CELL + ["rep"], observed=True)
                .agg(MCCD=("_absdev", "mean"),
                     WCCD=("_absdev", "max"),
                     WCU=("_under", "max"),
                     min_coverage=("coverage_cls", "min"),
                     frac_undercovered=("_dev", lambda s: float((s < 0).mean())),
                     MacroSize=("avg_size_cls", "mean"),
                     AvgSize=("avg_size_marginal", "first"),
                     CovMarginal=("coverage_marginal", "first"),
                     macro_f1=("macro_f1", "first"),
                     n_infinite_qhat=("n_infinite_qhat", "first"),
                     n_cal_rarest=("n_cal_cls", "min"))
                .reset_index())

    # Mechanism diagnostics are constant within a replication when present.
    for col in ["var_within_orbit", "rank_plain", "rank_ttaavg",
                "n_cal_realized_rarest", "n_cal_expected_rarest"]:
        if col in df.columns:
            extra = (df.groupby(CELL + ["rep"], observed=True)[col]
                       .first().reset_index())
            agg = agg.merge(extra, on=CELL + ["rep"], how="left")
    return agg


# ---------------------------------------------------------------------------
# Across-replication summaries
# ---------------------------------------------------------------------------

def _mcse(s: pd.Series) -> float:
    """Monte Carlo standard error of the mean across replications."""
    n = s.notna().sum()
    return float(s.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan


def summarize(rep_df: pd.DataFrame,
              metrics=("MCCD", "WCCD", "WCU", "AvgSize", "MacroSize",
                       "CovMarginal", "n_infinite_qhat"),
              by=None) -> pd.DataFrame:
    """Average each metric across replications, with its Monte Carlo SE.

    The SE quantifies how precisely the mean is estimated, and is what decides
    whether R is large enough. It is not the spread across replications: that
    is `.std()`, and for coverage it is the quantity the Beta band bounds.
    """
    by = by or [c for c in CELL if c in rep_df.columns]
    out = {}
    grouped = rep_df.groupby(by, observed=True)
    for m in metrics:
        if m not in rep_df.columns:
            continue
        out[m] = grouped[m].mean()
        out[m + "_mcse"] = grouped[m].apply(_mcse)
        out[m + "_sd"] = grouped[m].std(ddof=1)
    out["n_rep"] = grouped.size()
    return pd.DataFrame(out).reset_index()


def ablation_table(rep_df: pd.DataFrame, metric: str = "WCU") -> pd.DataFrame:
    """The 2 x 2 ablation: training augmentation against calibration arm.

    Rows are the calibration arms (plain, prediction-averaged,
    score-averaged, naive multi-copy), columns are train_aug in {False, True}.
    The four cells of the referee's 2 x 2 are the corners of this table; the
    extra rows separate the two aggregation points, which the 2 x 2 alone
    conflates.
    """
    keys = [c for c in ["scenario", "model", "alpha", "n_cal"]
            if c in rep_df.columns]
    piv = (rep_df.groupby(keys + ["method", "train_aug"], observed=True)[metric]
                 .mean().unstack("train_aug"))
    piv.columns = [f"train_aug={bool(c)}" for c in piv.columns]
    if piv.shape[1] == 2:
        piv["delta"] = piv.iloc[:, 1] - piv.iloc[:, 0]
    return piv.reset_index()


def mechanism_table(rep_df: pd.DataFrame) -> pd.DataFrame:
    """Diagnostics linking the observed effect to the mechanism in the theory.

    `var_within_orbit` estimates Var_g[S(gX, Y)], the quantity governing the
    variance reduction. Where it is near zero the theory predicts no gain from
    orbit averaging, so observing no gain there confirms the account rather
    than contradicting it. The true-class ranks say whether any reduction in
    set size came from a better ranking of the true label, which is the
    mechanism by which averaging shrinks an APS set.
    """
    cols = [c for c in ["var_within_orbit", "rank_plain", "rank_ttaavg",
                        "macro_f1"] if c in rep_df.columns]
    if not cols:
        return pd.DataFrame()
    keys = [c for c in ["scenario", "model", "n_cal", "n_transforms",
                        "exact_invariance", "train_aug"] if c in rep_df.columns]
    out = rep_df.groupby(keys, observed=True)[cols].mean().reset_index()
    if {"rank_plain", "rank_ttaavg"} <= set(out.columns):
        out["rank_delta"] = out["rank_ttaavg"] - out["rank_plain"]
    return out


# ---------------------------------------------------------------------------
# Distribution-free reference
# ---------------------------------------------------------------------------

def beta_reference(n_y: int, alpha: float, n_test: int | None = None,
                   quantiles=(0.05, 0.5, 0.95)) -> dict:
    """Beta law of the calibration-conditional coverage for a class with n_y points.

    Parameters
    ----------
    n_y : int
        Calibration points in the class.
    alpha : float
    n_test : int or None
        Test points of that class. When given, `sd_observable` adds the
        binomial noise of estimating the coverage on a finite test set, which
        is what an empirical spread should be compared against; `sd` remains
        the noiseless Beta standard deviation.
    quantiles : tuple of float
        Quantiles of the Beta law, for a reference band.

    Returns
    -------
    dict with the Beta parameters, mean, standard deviations and quantiles.
    If ceil((n_y + 1)(1 - alpha)) > n_y no finite quantile exists, the
    procedure returns an infinite threshold and coverage is one by
    construction; this is flagged rather than silently returning a Beta.
    """
    k = int(np.ceil((n_y + 1) * (1.0 - alpha)))
    if k > n_y:
        return {"n_y": n_y, "k": k, "degenerate": True,
                "mean": 1.0, "sd": 0.0, "sd_observable": 0.0,
                **{f"q{int(q * 100):02d}": 1.0 for q in quantiles}}

    a, b = k, n_y + 1 - k
    dist = stats.beta(a, b)
    var_beta = a * b / ((a + b) ** 2 * (a + b + 1))
    # E[C(1-C)] = ab / [(a+b)(a+b+1)] for C ~ Beta(a, b)
    var_obs = var_beta
    if n_test:
        var_obs = var_beta + a * b / (n_test * (a + b) * (a + b + 1))

    return {"n_y": n_y, "k": k, "degenerate": False, "a": a, "b": b,
            "mean": float(dist.mean()),
            "sd": float(np.sqrt(var_beta)),
            "sd_observable": float(np.sqrt(var_obs)),
            **{f"q{int(q * 100):02d}": float(dist.ppf(q)) for q in quantiles}}


def beta_reference_table(n_values, alpha: float,
                         n_test: int | None = None) -> pd.DataFrame:
    """Beta reference band over a grid of per-class calibration sizes."""
    return pd.DataFrame([beta_reference(int(n), alpha, n_test=n_test)
                         for n in n_values])


# ---------------------------------------------------------------------------
# Coverage / efficiency trade-off
# ---------------------------------------------------------------------------

def pareto_front(summary: pd.DataFrame,
                 coverage_metric: str = "WCU",
                 size_metric: str = "AvgSize",
                 within: str | None = "guarantee") -> pd.DataFrame:
    """Flag the methods on the coverage / set-size Pareto front.

    Reported instead of a single weighted index. A scalar combination of a
    percentage and a label count requires a weight that has no natural value,
    and it aggregates methods pursuing different guarantees into one ranking.
    The front says which methods are not dominated without committing to a
    weight; the trade-off is then the reader's to make.

    By default the front is computed WITHIN each guarantee class. Comparing the
    set size of a method targeting class-conditional coverage against one that
    only targets marginal coverage is not like-for-like: the larger sets are
    the price of the stronger guarantee, not a defect. Pass `within=None` for a
    global front, but then the guarantee column must be read alongside it.

    Adds a boolean `on_front` column. A row is dominated when another row in
    the same comparison group has both a smaller coverage metric and a smaller
    set size.
    """
    out = summary.copy()

    def _front(block: pd.DataFrame) -> np.ndarray:
        cov = block[coverage_metric].to_numpy()
        size = block[size_metric].to_numpy()
        dominated = np.zeros(len(block), dtype=bool)
        for i in range(len(block)):
            dominated[i] = np.any((cov <= cov[i]) & (size <= size[i]) &
                                  ((cov < cov[i]) | (size < size[i])))
        return ~dominated

    if within and within in out.columns:
        out["on_front"] = False
        for _, idx in out.groupby(within, observed=True).groups.items():
            out.loc[idx, "on_front"] = _front(out.loc[idx])
    else:
        out["on_front"] = _front(out)
    return out


def rare_class_spread(df: pd.DataFrame, alpha: float,
                      methods=("classwise_plain", "classwise_ttaavg",
                               "classwise_orbitavg"),
                      min_reps: int = 10, n_bins: int = 12) -> pd.DataFrame:
    """Observed spread of the rarest class's coverage against the Beta floor.

    The Beta law describes one class with n_y calibration points, so the
    comparison is made class by class and conditional on the realised count.
    With i.i.d. calibration that count is random and, at large n_cal, spreads
    over many distinct values with too few replications each to estimate a
    standard deviation. Counts are therefore binned on a log scale and the
    reference is evaluated at the median count of each bin.

    Binning mixes slightly different Beta laws within a bin. Their means are all
    close to 1 - alpha, so the inflation of the observed spread is second order,
    but bins should be kept narrow.

    `min_reps` drops bins supported by too few replications: a standard
    deviation estimated on a handful of values has relative error about
    1/sqrt(2(n-1)) and says nothing about the floor.
    """
    cols = ["method", "n_cal_cls", "observed_sd", "observed_mean", "n_rep",
            "n_test", "k", "degenerate", "sd", "sd_observable"]

    sub = df[df["method"].isin(methods)].copy()
    sub = sub[sub["cls"] == sub["cls"].max()]          # the rarest class
    if sub.empty:
        return pd.DataFrame(columns=cols)

    counts = sub["n_cal_cls"].to_numpy()
    lo, hi = counts.min(), counts.max()
    if hi > lo:
        edges = np.unique(np.round(
            np.logspace(np.log10(max(lo, 1)), np.log10(hi + 1), n_bins + 1)))
        sub["_bin"] = pd.cut(sub["n_cal_cls"], bins=edges,
                             include_lowest=True, duplicates="drop")
    else:
        sub["_bin"] = 0

    out = (sub.groupby(["method", "_bin"], observed=True)
              .agg(n_cal_cls=("n_cal_cls", "median"),
                   observed_sd=("coverage_cls", "std"),
                   observed_mean=("coverage_cls", "mean"),
                   n_rep=("coverage_cls", "size"),
                   n_test=("n_test_cls", "first"))
              .reset_index()
              .drop(columns="_bin"))
    out = out[(out["n_rep"] >= min_reps) & out["observed_sd"].notna()]
    out = out.reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(columns=cols)

    ref = out.apply(
        lambda r: beta_reference(int(round(r["n_cal_cls"])), alpha,
                                 n_test=int(r["n_test"])),
        axis=1, result_type="expand")
    return pd.concat([out, ref[["k", "degenerate", "sd", "sd_observable"]]], axis=1)

def paired_delta(rep_df: pd.DataFrame, metric: str,
                 baseline: str, treatment: str,
                 by=("scenario", "model", "alpha", "n_cal", "train_aug")) -> pd.DataFrame:
    """Replication-by-replication difference between two arms.

    Arms within a replication share the dataset, the splits, the fitted model,
    the transformations and the tie-breaking variables, so differencing them
    removes all of that variation. The paired standard error is far smaller
    than what the two marginal standard errors would suggest, which matters for
    the small but systematic gaps - orbit averaging against prediction
    averaging, for instance - that an unpaired comparison cannot resolve.
    """
    by = [c for c in by if c in rep_df.columns]
    a = rep_df[rep_df.method == baseline].set_index(by + ["rep"])[metric]
    b = rep_df[rep_df.method == treatment].set_index(by + ["rep"])[metric]
    d = (b - a).dropna().rename("delta").reset_index()
    out = (d.groupby(by, observed=True)["delta"]
             .agg(mean_delta="mean", sd="std", n="size").reset_index())
    out["mcse"] = out["sd"] / np.sqrt(out["n"])
    out["frac_negative"] = (d.groupby(by, observed=True)["delta"]
                              .apply(lambda s: float((s < 0).mean())).to_numpy())
    out["baseline"], out["treatment"], out["metric"] = baseline, treatment, metric
    return out


def clustered_diagnostic(rep_df: pd.DataFrame,
                         reference: str = "marginal_plain",
                         tol: float = 1e-6) -> pd.DataFrame:
    """How often clustered conformal collapses onto the marginal method.

    When the rare classes hold at most 1/alpha - 1 points they cannot be
    assigned to a cluster, clustering is skipped, and the procedure reduces to
    the marginal one. The fraction of replications in which that happens is the
    honest way to report clustered in the low-data regime, and it explains any
    non-monotone behaviour across calibration sizes: the method is not degrading
    smoothly, it is switching between two different procedures.
    """
    by = [c for c in ["scenario", "model", "alpha", "n_cal", "train_aug"]
          if c in rep_df.columns]
    cl = rep_df[rep_df.method == "clustered"].set_index(by + ["rep"])["AvgSize"]
    mg = rep_df[rep_df.method == reference].set_index(by + ["rep"])["AvgSize"]
    same = (cl - mg).abs() < tol
    return (same.rename("collapsed").reset_index()
                .groupby(by, observed=True)["collapsed"]
                .agg(frac_collapsed="mean", n_rep="size").reset_index())

# ---------------------------------------------------------------------------
# Paper deliverables
# ---------------------------------------------------------------------------

def finite_threshold_min(alpha: float) -> int:
    """Smallest per-class calibration count admitting a finite threshold.

    A finite class-specific threshold exists iff
    k_y = ceil((n_y + 1)(1 - alpha)) <= n_y, which for integer n_y is
    n_y >= (1 - alpha)/alpha. With alpha = 0.10 and 0.05 this is 9 and 19.
    """
    return int(np.ceil((1.0 - alpha) / alpha))


def degenerate_fraction(df: pd.DataFrame, alpha: float,
                        methods=("classwise_plain",)) -> pd.DataFrame:
    """Fraction of replications whose rarest class has no finite threshold.

    Design points near the boundary straddle two qualitatively different
    regimes, so their average describes neither. This is the quantity to report
    alongside those points, and to condition on when summarising them.
    """
    nmin = finite_threshold_min(alpha)
    sub = df[df["method"].isin(methods) & (~df["train_aug"])]
    sub = sub[sub["cls"] == sub["cls"].max()]
    by = [c for c in ["scenario", "model", "alpha", "n_cal"] if c in sub.columns]
    out = (sub.groupby(by + ["rep"], observed=True)["n_cal_cls"].first()
              .reset_index()
              .groupby(by, observed=True)["n_cal_cls"]
              .agg(frac_degenerate=lambda s: float((s < nmin).mean()),
                   median_count="median",
                   q05=lambda s: float(s.quantile(0.05)),
                   q95=lambda s: float(s.quantile(0.95)),
                   n_rep="size")
              .reset_index())
    out["n_min"] = nmin
    return out


def infinite_threshold_curve(df: pd.DataFrame, alpha: float,
                             methods=("classwise_plain",),
                             min_reps: int = 5) -> pd.DataFrame:
    """Rare-class threshold status against the realised rare-class count.

    Reports P(no finite threshold FOR THE RAREST CLASS | its realised count),
    which is what the operational rule is about. This is not the same as the
    probability that some class lacks a finite threshold: with multinomial
    calibration counts a second-rarest class can fall below the requirement
    while the rarest one does not, and conditioning that event on the rarest
    count alone gives intermediate, non-monotone values describing neither
    quantity. `also_any_class` reports it separately so the two are not
    confused.

    The rare-class status is deterministic given n_y and alpha - the threshold
    is infinite iff ceil((n_y + 1)(1 - alpha)) > n_y - so that curve is a step
    at n_min = ceil((1 - alpha)/alpha). Plotting it verifies that the realised
    counts behave as the theory says, and locates the boundary on the axis a
    practitioner can observe.
    """
    nmin = finite_threshold_min(alpha)
    sub = df[df["method"].isin(methods) & (~df["train_aug"])]
    sub = sub[sub["cls"] == sub["cls"].max()]          # the rarest class
    sub = sub.assign(rare_infinite=(sub["n_cal_cls"] < nmin).astype(float))

    out = (sub.groupby("n_cal_cls", observed=True)
              .agg(frac_infinite=("rare_infinite", "mean"),
                   also_any_class=("n_infinite_qhat",
                                   lambda s: float((s > 0).mean())),
                   mean_n_infinite=("n_infinite_qhat", "mean"),
                   n_rep=("n_cal_cls", "size"))
              .reset_index())
    out = out[out["n_rep"] >= min_reps].reset_index(drop=True)
    out["n_min"] = nmin
    return out

def dispersion_ratio(df: pd.DataFrame, alpha: float,
                     methods=("classwise_plain", "classwise_ttaavg",
                              "classwise_orbitavg")) -> pd.DataFrame:
    """Observed rare-class coverage spread against its distribution-free floor.

    The reference is a MIXTURE, not a single Beta law. Within a design level the
    realised calibration count is binomial, so the replications draw their
    coverage from Beta laws with different parameters, and some may have no
    finite threshold at all. Evaluating the reference at one representative
    count understates the spread the design induces, which pushes every ratio
    below one and invites the reading that the methods beat a floor they cannot
    beat.

    Writing C_r for the calibration-conditional coverage of replication r, with
    realised count n_r and k_r = ceil((n_r + 1)(1 - alpha)):

        finite threshold (k_r <= n_r): C_r ~ Beta(k_r, n_r + 1 - k_r), and the
            observable variance adds the binomial noise of the test set,
            E[C(1-C)]/m = ab / [m (a+b)(a+b+1)];
        infinite threshold (k_r > n_r): every label is included, so C_r = 1
            with zero variance.

    By the law of total variance the level reference is

        v_mix = mean_r Var(Chat_r | n_r) + Var_r( E[Chat_r | n_r] ),

    where the second term carries the mixing across regimes. A ratio of one
    means the method sits exactly on the floor; no score transformation can go
    below it.
    """
    sub = df[df["method"].isin(methods) & (~df["train_aug"])]
    sub = sub[sub["cls"] == sub["cls"].max()]
    by = [c for c in ["scenario", "model", "alpha", "n_cal"] if c in sub.columns]

    obs = (sub.groupby(by + ["method"], observed=True)
              .agg(observed_sd=("coverage_cls", "std"),
                   observed_mean=("coverage_cls", "mean"),
                   n_rep=("coverage_cls", "size"))
              .reset_index())

    # One (mu_r, v_r) per replication, then pooled by design level.
    per_rep = (sub.groupby(by + ["rep"], observed=True)
                  .agg(n_y=("n_cal_cls", "first"),
                       m=("n_test_cls", "first"))
                  .reset_index())

    def _moments(n_y: int, m: int):
        ref = beta_reference(int(n_y), alpha, n_test=int(m))
        if ref["degenerate"]:
            return 1.0, 0.0
        return ref["mean"], ref["sd_observable"] ** 2

    per_rep[["mu", "v"]] = per_rep.apply(
        lambda r: _moments(r["n_y"], r["m"]), axis=1, result_type="expand")

    ref = (per_rep.groupby(by, observed=True)
                  .agg(mean_v=("v", "mean"),
                       var_mu=("mu", "var"),
                       median_count=("n_y", "median"),
                       frac_degenerate=("v", lambda s: float((s == 0).mean())),
                       n_test=("m", "first"))
                  .reset_index())
    ref["sd_mixture"] = np.sqrt(ref["mean_v"] + ref["var_mu"].fillna(0.0))

    out = obs.merge(ref, on=by, how="left")
    out["ratio"] = out["observed_sd"] / out["sd_mixture"].replace(0.0, np.nan)
    return out

def paired_reduction(rep_df: pd.DataFrame,
                     baseline: str = "classwise_plain",
                     treatment: str = "classwise_orbitavg",
                     metric: str = "AvgSize",
                     relative: bool = True) -> pd.DataFrame:
    """Per-replication reduction of `treatment` relative to `baseline`.

    Reported instead of the change in true-class rank, which is real but too
    small to read off a plot. The relative form answers the question a
    practitioner asks - by what percentage do the sets shrink - and its paired
    standard error is much smaller than the two marginal ones, because the arms
    share the data, the splits, the fitted model, the transformations and the
    randomisation.
    """
    by = [c for c in ["scenario", "model", "alpha", "n_cal", "train_aug"]
          if c in rep_df.columns]
    a = rep_df[rep_df.method == baseline].set_index(by + ["rep"])[metric]
    b = rep_df[rep_df.method == treatment].set_index(by + ["rep"])[metric]
    d = (100.0 * (a - b) / a if relative else a - b).dropna().rename("reduction")

    out = (d.reset_index().groupby(by, observed=True)["reduction"]
             .agg(mean="mean", sd="std", n="size").reset_index())
    out["mcse"] = out["sd"] / np.sqrt(out["n"])
    out["t"] = out["mean"] / out["mcse"]
    out["frac_positive"] = (d.reset_index().groupby(by, observed=True)["reduction"]
                              .apply(lambda s: float((s > 0).mean())).to_numpy())
    out["baseline"], out["treatment"] = baseline, treatment
    out["units"] = "%" if relative else "labels"
    return out


def classifier_robustness(rep_df: pd.DataFrame, summ_df: pd.DataFrame,
                          levels=(250, 1250),
                          models=("XGBoost", "RandomForest", "MLP"),
                          pi_rare: float = 0.04,
                          baseline: str = "classwise_plain",
                          treatment: str = "classwise_orbitavg") -> pd.DataFrame:
    """Does the orbit-averaging gain survive a change of classifier?

    Absolute set sizes are NOT comparable across classifiers: they depend on
    predictive accuracy, and a more accurate model produces smaller sets for
    reasons unrelated to the score construction. The comparable quantity is the
    relative within-model reduction, computed replication by replication so
    that the two arms share the data, the splits, the fitted model, the
    transformations and the randomisation.

    `within_orbit_var` is reported alongside because it is the quantity the
    theory says governs the gain: averaging can only help to the extent that
    the score varies along the orbit, so a classifier with more within-orbit
    variation should show a larger reduction.
    """
    rows = []
    for model in models:
        for n in levels:
            sel = ((summ_df.model == model) & (summ_df.n_cal == n)
                   & (~summ_df.train_aug))
            a = summ_df[sel & (summ_df.method == baseline)]
            o = summ_df[sel & (summ_df.method == treatment)]
            if a.empty or o.empty:
                continue
            a, o = a.iloc[0], o.iloc[0]

            sub = rep_df[(rep_df.model == model) & (~rep_df.train_aug)]
            d = paired_reduction(sub, baseline, treatment, "AvgSize",
                                 relative=True)
            d = d[d.n_cal == n]
            wov = sub[sub.n_cal == n]["var_within_orbit"].mean() \
                if "var_within_orbit" in sub.columns else np.nan

            rows.append({
                "Classifier": model,
                "E[N_rare]": int(round(n * pi_rare)),
                "WCU APS": a.WCU, "WCU orbit": o.WCU,
                "AvgSize APS": a.AvgSize, "AvgSize orbit": o.AvgSize,
                "Reduction (%)": float(d.iloc[0]["mean"]) if len(d) else np.nan,
                "MCSE": float(d.iloc[0]["mcse"]) if len(d) else np.nan,
                "P(delta<0)": float(d.iloc[0]["frac_positive"]) if len(d) else np.nan,
                "within_orbit_var": float(wov),
                "macro_f1": float(sub[sub.n_cal == n]["macro_f1"].mean()),
                "n_rep": int(d.iloc[0]["n"]) if len(d) else 0,
            })
    return pd.DataFrame(rows)
