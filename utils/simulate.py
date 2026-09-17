"""
simulate.py
===========
Simulation driver for the CSDA revision.

Sampling design
---------------
Everything is drawn i.i.d. from the same population within a replication.
Calibration is NOT stratified: the per-class counts N_cal,y are random,
multinomial with the population class probabilities. This matters. Stratifying
calibration to fixed per-class counts while testing on a different class
mixture leaves the calibration points non-exchangeable with the test point, so
the marginal baseline loses its guarantee and the comparison stops being
like-for-like. Reweighting the evaluation would fix the estimand but not the
construction of the quantile.

Two test sets are drawn from disjoint pools:

  population test  Y ~ Categorical(pi). Used for marginal coverage and
                   population-weighted set size, and to validate the marginal
                   baseline against its own target.
  balanced test    equal counts per class. Used for per-class coverage, WCU,
                   WCCD and per-class set size, where the rare classes need a
                   precise estimate that a population draw cannot give.

Order of drawing matters: the pools are separated first, so that carving out
the balanced test set does not deplete the rare classes in the pool that
training and calibration are drawn from. Doing it the other way round
reintroduces exactly the bias this design removes.

The experimental axis
---------------------
With i.i.d. calibration the number of calibration points of the rarest class
can no longer be fixed exactly. The design parameter is therefore its expected
value,

    E[N_cal,rare] = n_cal * pi_rare,

and the realised count is recorded for every replication. Analysis should
condition on the realised count rather than group by the nominal one: near
k_y = ceil((n_y + 1)(1 - alpha)) > n_y the behaviour of classwise calibration
is discontinuous, not gradual, so averaging over a spread of realised counts
mixes qualitatively different regimes. Conditioning on n_y is also what the
theory does: the Beta law of calibration-conditional coverage and the classwise
corollary are both stated conditionally on the class counts.

Ablation
--------
Training augmentation is neutral with respect to conformal validity, since
split conformal treats the fitted classifier as fixed; calibration augmentation
acts on the scores and, in its naive multi-copy form, forfeits the finite-
sample guarantee. Two classifiers are fitted per replication, plain and
train-augmented, sharing one model seed as common random numbers, and every
calibration arm runs on top of each. Only the fitting is duplicated.

Output
------
One row per (replication, train_aug, method, class). Per-class quantities come
from the balanced test set, marginal quantities from the population test set
and are repeated on each row. No aggregation inside the loop.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import calibration as cal
from groups import SignFlipGroup, make_nuisance_block
from scores import (aps_scores, draw_uniform, orbit_softmax,
                    prediction_averaged_scores, score_averaged_scores,
                    true_class_rank, within_orbit_score_variance)

__all__ = ["SimConfig", "run", "run_replication", "IMBALANCE_PROFILE"]


# Population class probabilities for the imbalanced scenarios. The rarest class
# is the last one, and its probability sets the relation between n_cal and the
# expected number of calibration points in that class.
IMBALANCE_PROFILE = np.array([0.40, 0.20, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04])


@dataclass
class SimConfig:
    """One cell of the experimental grid."""

    # --- scenario -----------------------------------------------------------
    scenario: str = "s3_nuisance"
    n_classes: int = 8
    n_informative: int = 5
    n_redundant: int = 0
    n_nuisance: int = 20            # sign-flip block; 0 disables the group arms
    class_sep: float = 0.8
    balanced: bool = False

    # --- sizes --------------------------------------------------------------
    n_cal: int = 500                # the rare-class count is random, not fixed
    n_train: int = 4000
    n_test_pop: int = 4000
    n_test_balanced_per_class: int = 500

    # --- conformal ----------------------------------------------------------
    alpha: float = 0.10
    n_transforms: int = 16          # B in the Monte Carlo orbit average

    # --- design -------------------------------------------------------------
    model: str = "XGBoost"          # XGBoost | RandomForest | MLP
    n_reps: int = 30
    base_seed: int = 20260808
    run_train_aug: bool = True
    exact_invariance: bool = True

    # Stamped into the config file written alongside every long table, so a set
    # of results can be traced back to the code that produced it. Bump it
    # whenever a change alters the numbers a configuration would produce.
    code_version: str = "2026-08-rev3"# Stamped into the config file written alongside every long table, so a set
    # of results can be traced back to the code that produced it. Bump it
    # whenever a change alters the numbers a configuration would produce.
    code_version: str = "2026-08-rev3"
    # Per-replication assertions. Keep them on for a validation run, then turn
    # them off for the campaign: they build Python sets over index arrays, which
    # is not free at several hundred replications.
    strict: bool = True

    outdir: str = "results_new"

    def class_probs(self) -> np.ndarray:
        if self.balanced:
            return np.full(self.n_classes, 1.0 / self.n_classes)
        w = IMBALANCE_PROFILE[:self.n_classes]
        return w / w.sum()

    def expected_rare_cal(self) -> float:
        """E[N_cal,rare]: the quantity the grid is indexed by."""
        return float(self.n_cal * self.class_probs().min())

    def tag(self) -> str:
        inv = "exact" if self.exact_invariance else "approx"
        return (f"{self.scenario}_{self.model}_ncal{self.n_cal:05d}"
                f"_a{str(self.alpha).replace('.', '')}_B{self.n_transforms}_{inv}")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_model(name: str, seed: int):
    """Instantiate a classifier with library defaults."""
    if name == "RandomForest":
        return RandomForestClassifier(random_state=seed)
    if name == "MLP":
        # Standardisation is part of the fitted pipeline, so the scaler is
        # estimated on the training sample only and applied unchanged to
        # calibration and test. Re-estimating it on the calibration set would
        # leak information across the split the conformal guarantee rests on,
        # and would stop the score from being a fixed measurable function.
        # Group transformations act on the raw features, before scaling.
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(random_state=seed, max_iter=500),
        )
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, verbosity=0,
                             tree_method="hist", eval_metric="mlogloss")
    raise ValueError(f"unknown model: {name}")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _population_size(cfg: SimConfig) -> int:
    """Total draw needed so that the rarest class can fill every subset.

    The binding constraint is the balanced test set, which needs a fixed count
    of the rarest class regardless of how rare it is. A safety factor absorbs
    multinomial fluctuation.
    """
    p_rare = float(cfg.class_probs().min())
    rare_needed = (cfg.n_test_balanced_per_class
                   + int(np.ceil(cfg.n_test_pop * p_rare))
                   + int(np.ceil((cfg.n_train + cfg.n_cal) * p_rare)))
    return int(np.ceil(2.0 * rare_needed / p_rare))


def generate_population(cfg: SimConfig, rng: np.random.Generator):
    """Draw one i.i.d. sample whose nuisance block is exactly sign-flip invariant.

    The informative block comes from make_classification. The nuisance block is
    drawn independently of it and of the label; when symmetric about zero,
    (gX, Y) and (X, Y) have the same law for every sign flip g, so exact
    invariance holds by construction rather than by assumption.
    """
    n = _population_size(cfg)
    weights = None if cfg.balanced else cfg.class_probs().tolist()
    seed = int(rng.integers(0, 2 ** 31 - 1))

    X_base, y = make_classification(
        n_samples=n,
        n_features=cfg.n_informative + cfg.n_redundant,
        n_informative=cfg.n_informative,
        n_redundant=cfg.n_redundant,
        n_repeated=0,
        n_classes=cfg.n_classes,
        n_clusters_per_class=1,
        weights=weights,
        class_sep=cfg.class_sep,
        flip_y=0.0,
        random_state=seed,
    )
    if cfg.n_nuisance > 0:
        X_nuis = make_nuisance_block(X_base.shape[0], cfg.n_nuisance, rng,
                                     symmetric=cfg.exact_invariance)
        return np.hstack([X_base, X_nuis]), y
    return X_base, y


def split_population(X, y, cfg: SimConfig, rng: np.random.Generator):
    """Split into a fit pool and a test pool, then carve the four subsets.

    The two pools are separated FIRST. The balanced test set is built inside
    the test pool only, so removing its rare-class points cannot deplete the
    pool training and calibration are drawn from. Training and calibration are
    plain random draws, hence i.i.d. from the population, and the calibration
    class counts are multinomial rather than fixed.
    """
    idx = rng.permutation(len(y))
    n_fit = cfg.n_train + cfg.n_cal
    if len(idx) < n_fit + cfg.n_test_pop:
        raise RuntimeError("population too small for the requested sizes")

    fit_pool, test_pool = idx[:n_fit], idx[n_fit:]
    train_idx = fit_pool[:cfg.n_train]
    cal_idx = fit_pool[cfg.n_train:]

    # Population-distributed test set: a plain draw from the test pool.
    test_pop_idx = test_pool[:cfg.n_test_pop]

    # Balanced test set: equal counts per class, from the remainder.
    remainder = test_pool[cfg.n_test_pop:]
    picked = []
    for c in range(cfg.n_classes):
        avail = remainder[y[remainder] == c]
        k = cfg.n_test_balanced_per_class
        if len(avail) < k:
            raise RuntimeError(
                f"balanced test set: class {c} has {len(avail)} points available "
                f"but {k} are required; increase the population size"
            )
        picked.append(rng.choice(avail, size=k, replace=False))
    test_bal_idx = np.concatenate(picked)

    return train_idx, cal_idx, test_pop_idx, test_bal_idx


# ---------------------------------------------------------------------------
# One replication
# ---------------------------------------------------------------------------

def run_replication(cfg: SimConfig, rep: int) -> list[dict]:
    """Run every arm for one external replication. Returns long-format rows."""
    # Independent streams: data, orbit transformations, training-augmentation
    # flips, the three U draws, and the model seed. The training flips need a
    # stream of their own: drawing them from the orbit stream would make them
    # depend on B, so two cells differing only in B would also end up with
    # different train-augmented models.
    seeds = np.random.SeedSequence([cfg.base_seed, rep]).spawn(7)
    (rng_data, rng_g, rng_gtr, rng_ucal,
     rng_utpop, rng_utbal, rng_model) = (np.random.default_rng(s) for s in seeds)

    X, y = generate_population(cfg, rng_data)
    tr, ca, tp, tb = split_population(X, y, cfg, rng_data)
    X_tr, y_tr = X[tr], y[tr]
    X_ca, y_ca = X[ca], y[ca]
    X_tp, y_tp = X[tp], y[tp]
    X_tb, y_tb = X[tb], y[tb]

    k = cfg.n_classes
    use_group = cfg.n_nuisance > 0
    group = SignFlipGroup(X.shape[1], cfg.n_nuisance) if use_group else None

    # Transformations: drawn once, independently of the data, and shared by
    # every calibration and test point. This keeps the averaged score a fixed
    # measurable function of (x, y, u) conditionally on them.
    eps = group.sample(cfg.n_transforms, rng_g) if use_group else None

    # One scalar U per observation, from separate streams per subset.
    U_ca = draw_uniform(len(y_ca), k, rng_ucal)
    U_tp = draw_uniform(len(y_tp), k, rng_utpop)
    U_tb = draw_uniform(len(y_tb), k, rng_utbal)

    cal_counts = np.bincount(y_ca, minlength=k)
    n_rare_realised = int(cal_counts.min())

    if cfg.strict:
        assert set(np.unique(y_tr)) == set(range(k)), "a class is absent from training"
        assert np.all(np.bincount(y_tb, minlength=k) == cfg.n_test_balanced_per_class)
        assert not (set(tr) & set(ca)), "train and calibration overlap"
        assert not (set(tp) & set(tb)), "the two test sets overlap"
        assert not (set(np.concatenate([tr, ca])) & set(np.concatenate([tp, tb]))), \
            "fit and test pools overlap"
        if use_group:
            flipped = group.apply(X_ca, eps[0])
            assert np.allclose(X_ca[:, :-cfg.n_nuisance],
                               flipped[:, :-cfg.n_nuisance]), "base block altered"
            assert np.allclose(np.abs(X_ca[:, -cfg.n_nuisance:]),
                               np.abs(flipped[:, -cfg.n_nuisance:])), "not a sign flip"

    # Common random numbers across the two training arms: the difference
    # between them is then attributable to the augmentation, not to a different
    # random initialisation of the classifier.
    model_seed = int(rng_model.integers(0, 2 ** 31 - 1))

    rows: list[dict] = []
    train_variants = [False, True] if (cfg.run_train_aug and use_group) else [False]

    for train_aug in train_variants:
        if train_aug:
            # One random flip per training row: teaches the invariance without
            # inflating anything downstream.
            eps_tr = group.sample(len(y_tr), rng_gtr)
            X_flip = X_tr.copy()
            X_flip[:, -cfg.n_nuisance:] *= eps_tr
            X_fit = np.vstack([X_tr, X_flip])
            y_fit = np.concatenate([y_tr, y_tr])
        else:
            X_fit, y_fit = X_tr, y_tr

        model = build_model(cfg.model, seed=model_seed)
        model.fit(X_fit, y_fit)

        p_ca, p_tp, p_tb = (model.predict_proba(A) for A in (X_ca, X_tp, X_tb))
        if cfg.strict:
            for P in (p_ca, p_tp, p_tb):
                assert np.allclose(P.sum(axis=1), 1.0), "probabilities do not sum to one"
        macro_f1 = f1_score(y_tb, p_tb.argmax(1), average="macro")

        # --- scores: (calibration, population test, balanced test) ----------
        arms: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
            "plain": (aps_scores(p_ca, U_ca), aps_scores(p_tp, U_tp),
                      aps_scores(p_tb, U_tb)),
        }
        diag: dict[str, float] = {}

        if use_group:
            orb_ca = orbit_softmax(model, X_ca, group, eps)
            orb_tp = orbit_softmax(model, X_tp, group, eps)
            orb_tb = orbit_softmax(model, X_tb, group, eps)
            arms["ttaavg"] = (prediction_averaged_scores(orb_ca, U_ca),
                              prediction_averaged_scores(orb_tp, U_tp),
                              prediction_averaged_scores(orb_tb, U_tb))
            arms["orbitavg"] = (score_averaged_scores(orb_ca, U_ca),
                                score_averaged_scores(orb_tp, U_tp),
                                score_averaged_scores(orb_tb, U_tb))
            diag["var_within_orbit"] = float(
                within_orbit_score_variance(orb_tb, y_tb, U_tb).mean())
            diag["rank_plain"] = float(true_class_rank(p_tb, y_tb).mean())
            diag["rank_ttaavg"] = float(true_class_rank(orb_tb.mean(0), y_tb).mean())

        # --- calibration arms ------------------------------------------------
        # (name, guarantee, sets on population test, sets on balanced test,
        #  number of classes left with an infinite threshold)
        built: list[tuple[str, str, np.ndarray, np.ndarray, int]] = []

        for score_name, (s_ca, s_tp, s_tb) in arms.items():
            sets_p, _ = cal.marginal_sets(s_ca, y_ca, s_tp, cfg.alpha)
            sets_b, _ = cal.marginal_sets(s_ca, y_ca, s_tb, cfg.alpha)
            built.append((f"marginal_{score_name}", "marginal", sets_p, sets_b, 0))

            sets_p, q = cal.classwise_sets(s_ca, y_ca, s_tp, cfg.alpha, k)
            sets_b, _ = cal.classwise_sets(s_ca, y_ca, s_tb, cfg.alpha, k)
            built.append((f"classwise_{score_name}", "class-conditional",
                          sets_p, sets_b, int(np.isinf(q).sum())))

        sets_p = cal.clustered_sets(arms["plain"][0], y_ca, arms["plain"][1],
                                    y_tp, cfg.alpha, seed=cfg.base_seed + rep)
        sets_b = cal.clustered_sets(arms["plain"][0], y_ca, arms["plain"][2],
                                    y_tb, cfg.alpha, seed=cfg.base_seed + rep)
        built.append(("clustered", "cluster-conditional", sets_p, sets_b, 0))

        if use_group:
            aug_scores = np.vstack(
                [aps_scores(model.predict_proba(group.apply(X_ca, e)), U_ca)
                 for e in eps])
            aug_labels = np.tile(y_ca, cfg.n_transforms)
            sets_p, q = cal.naive_multicopy_sets(aug_scores, aug_labels,
                                                 arms["plain"][1], cfg.alpha, k)
            sets_b, _ = cal.naive_multicopy_sets(aug_scores, aug_labels,
                                                 arms["plain"][2], cfg.alpha, k)
            built.append(("naive_multicopy", "none", sets_p, sets_b,
                          int(np.isinf(q).sum())))

        # --- record ----------------------------------------------------------
        for method, guarantee, sets_p, sets_b, n_inf in built:
            cov_p = sets_p[np.arange(len(y_tp)), y_tp]
            size_p = sets_p.sum(1)
            cov_b = sets_b[np.arange(len(y_tb)), y_tb]
            size_b = sets_b.sum(1)

            base = {
                "rep": rep,
                "scenario": cfg.scenario,
                "model": cfg.model,
                "alpha": cfg.alpha,
                "n_cal": cfg.n_cal,
                "n_cal_expected_rarest": cfg.expected_rare_cal(),
                "n_cal_realized_rarest": n_rare_realised,
                "n_transforms": cfg.n_transforms,
                "exact_invariance": cfg.exact_invariance,
                "train_aug": train_aug,
                "method": method,
                "guarantee": guarantee,
                "n_infinite_qhat": n_inf,
                # marginal quantities: population-distributed test set
                "coverage_marginal": float(cov_p.mean()),
                "avg_size_marginal": float(size_p.mean()),
                "macro_f1": float(macro_f1),
                **diag,
            }
            # per-class quantities: balanced test set
            for c in range(k):
                m = y_tb == c
                rows.append({
                    **base,
                    "cls": c,
                    "n_cal_cls": int(cal_counts[c]),
                    "n_test_cls": int(m.sum()),
                    "coverage_cls": float(cov_b[m].mean()),
                    "avg_size_cls": float(size_b[m].mean()),
                })
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _write(df: pd.DataFrame, stem: str) -> str:
    """Write a table, preferring parquet and falling back to gzipped CSV.

    Returns the path actually written. The fallback keeps the code runnable
    where pyarrow is unavailable; `metrics.load` reads either format.
    """
    try:
        path = stem + ".parquet"
        df.to_parquet(path)
    except (ImportError, ValueError):
        path = stem + ".csv.gz"
        df.to_csv(path, index=False)
    return path


def _find(stem: str) -> str | None:
    """Return the existing file for `stem`, in either format, or None."""
    for ext in (".parquet", ".csv.gz"):
        if os.path.exists(stem + ext):
            return stem + ext
    return None


def _read(path: str) -> pd.DataFrame:
    """Read a table written by `_write`."""
    return (pd.read_parquet(path) if path.endswith(".parquet")
            else pd.read_csv(path))


def run(cfg: SimConfig, verbose: bool = True,
        checkpoint_every: int = 10) -> pd.DataFrame:
    """Run all replications for one configuration and write the long table.

    A checkpoint is written every `checkpoint_every` replications and reloaded
    on restart, so an interrupted session costs at most that many replications
    rather than the whole cell. Resuming is exact: the seeds depend only on
    (base_seed, rep), so replication 137 gives the same result whenever it runs.

    A cell whose final table already exists is never recomputed, which makes
    the driver safe to re-run over a whole grid. Set `checkpoint_every=0` to
    disable checkpointing.
    """
    os.makedirs(cfg.outdir, exist_ok=True)
    stem = os.path.join(cfg.outdir, f"{cfg.tag()}__long")
    ckpt_stem = stem + "__ckpt"

    finished = _find(stem)
    if finished:
        if verbose:
            print(f"[done] {cfg.tag()} already complete", flush=True)
        return _read(finished)

    rows: list[dict] = []
    start = 0
    ckpt = _find(ckpt_stem)
    if ckpt:
        try:
            done = _read(ckpt)
            rows = done.to_dict("records")
            start = int(done["rep"].max()) + 1
            print(f"[resume] {cfg.tag()} from replication {start}", flush=True)
        except Exception as exc:      # e.g. a checkpoint truncated mid-write
            print(f"[warn] unreadable checkpoint, restarting: {exc}", flush=True)
            rows, start = [], 0

    for rep in range(start, cfg.n_reps):
        rows.extend(run_replication(cfg, rep))

        if checkpoint_every and (rep + 1) % checkpoint_every == 0:
            # Write to a temporary stem and rename. os.replace is atomic, so an
            # interruption during the write leaves the previous checkpoint
            # intact rather than a half-written file.
            tmp = _write(pd.DataFrame(rows), ckpt_stem + ".tmp")
            os.replace(tmp, tmp.replace(".tmp", ""))
            if verbose:
                print(f"[{cfg.tag()}] {rep + 1}/{cfg.n_reps} (checkpoint)",
                      flush=True)
        elif verbose and (rep + 1) % 10 == 0:
            print(f"[{cfg.tag()}] {rep + 1}/{cfg.n_reps}", flush=True)

    df = pd.DataFrame(rows)
    _write(df, stem)
    with open(os.path.join(cfg.outdir, f"{cfg.tag()}__config.json"), "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)
    for leftover in (_find(ckpt_stem), _find(ckpt_stem + ".tmp")):
        if leftover:
            os.remove(leftover)
    if verbose:
        print(f"[done] {cfg.tag()}  {len(df)} rows", flush=True)
    return df
