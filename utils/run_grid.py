#!/usr/bin/env python3
"""
run_grid.py
===========
Command-line runner: one process per cell of the experimental grid.

Each invocation runs a single configuration to completion and writes its own
long table, so cells are independent and can be launched in parallel, retried
individually, or resumed after a crash without touching the others.

Examples
--------
Single cell:

    python run_grid.py --scenario s3_nuisance --model XGBoost \\
        --n-cal 500 --n-reps 200 --outdir results_new

Print the whole grid, one command per line, then run 8 at a time:

    python run_grid.py --print-grid | xargs -P 8 -I {} sh -c '{}'

SLURM array (one task per line of the grid file):

    python run_grid.py --print-grid > grid.txt
    sbatch --array=1-$(wc -l < grid.txt) run_grid.sbatch
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

from simulate import SimConfig, run


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------
# Design decisions encoded here:
#   - the calibration size is the primary axis and is swept only in the nuisance
#     scenario, where the theory predicts a signal. Other scenarios are run at a
#     single level, as a check that the ordering of methods does not flip.
#     Calibration is i.i.d., so the axis is indexed by E[N_cal,rare]; the
#     realised count is recorded per replication and is what analysis should
#     condition on.
#   - the sweep uses one model; the single-level cells use all three.
#   - B is swept only in the pilot, to pick the value beyond which the Monte
#     Carlo error of the orbit average stops mattering.

# Calibration sizes chosen so that E[N_cal,rare] = n_cal * 0.04 hits
# 5, 10, 20, 50, 100, 200 for the rarest class.
NCAL_SWEEP = [125, 250, 500, 1250, 2500, 5000]
NCAL_SINGLE = 500
ALPHAS = [0.10, 0.05]
MODELS = ["XGBoost", "RandomForest", "MLP"]

SCENARIOS = {
    # name              n_informative  n_nuisance  class_sep  balanced
    "s1_benchmark":     dict(n_informative=15, n_nuisance=5,  class_sep=1.0, balanced=True),
    "s2_imbalanced":    dict(n_informative=5,  n_nuisance=5,  class_sep=1.0, balanced=False),
    "s3_nuisance":      dict(n_informative=5,  n_nuisance=20, class_sep=0.8, balanced=False),
    "s4_overlapping":   dict(n_informative=5,  n_nuisance=5,  class_sep=0.2, balanced=False),
}


def build_grid():
    """Yield (scenario, model, n_cal, alpha) for every cell to be run."""
    for alpha in ALPHAS:
        # Primary sweep: one scenario, one model, full n_cal_min axis.
        for n in NCAL_SWEEP:
            yield ("s3_nuisance", "XGBoost", n, alpha)
        # Control cells: every scenario and model, single calibration size.
        for scen, model in itertools.product(SCENARIOS, MODELS):
            if scen == "s3_nuisance" and model == "XGBoost":
                continue          # already covered by the sweep
            yield (scen, model, NCAL_SINGLE, alpha)


def make_config(scenario, model, n_cal, alpha, args) -> SimConfig:
    return SimConfig(
        scenario=scenario,
        model=model,
        n_cal=n_cal,
        alpha=alpha,
        n_reps=args.n_reps,
        n_transforms=args.n_transforms,
        n_test_pop=args.n_test_pop,
        n_test_balanced_per_class=args.n_test_balanced,
        n_train=args.n_train,
        run_train_aug=not args.no_train_aug,
        exact_invariance=not args.approx_invariance,
        base_seed=args.base_seed,
        outdir=args.outdir,
        **SCENARIOS[scenario],
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", choices=sorted(SCENARIOS))
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--n-cal", type=int)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--n-reps", type=int, default=200)
    p.add_argument("--n-transforms", type=int, default=16)
    p.add_argument("--n-test-pop", type=int, default=4000)
    p.add_argument("--n-test-balanced", type=int, default=500)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--base-seed", type=int, default=20260808)
    p.add_argument("--no-train-aug", action="store_true",
                   help="skip the train-augmented model (halves the fitting cost)")
    p.add_argument("--approx-invariance", action="store_true",
                   help="break exact invariance in a controlled way")
    p.add_argument("--outdir", default="results_new")
    p.add_argument("--print-grid", action="store_true",
                   help="print one shell command per grid cell and exit")
    p.add_argument("--skip-existing", action="store_true",
                   help="do nothing if the output table already exists")
    args = p.parse_args(argv)

    if args.print_grid:
        exe = f"{sys.executable} {os.path.abspath(__file__)}"
        for scen, model, n, alpha in build_grid():
            print(f"{exe} --scenario {scen} --model {model} --n-cal {n} "
                  f"--alpha {alpha} --n-reps {args.n_reps} "
                  f"--n-transforms {args.n_transforms} --outdir {args.outdir} "
                  f"--skip-existing")
        return 0

    missing = [f for f in ("scenario", "model", "n_cal")
               if getattr(args, f) is None]
    if missing:
        p.error(f"missing required options for a single run: {missing}")

    cfg = make_config(args.scenario, args.model, args.n_cal, args.alpha, args)

    stem = os.path.join(cfg.outdir, f"{cfg.tag()}__long")
    if args.skip_existing and (os.path.exists(stem + ".parquet") or
                               os.path.exists(stem + ".csv.gz")):
        print(f"[skip] {cfg.tag()} already present")
        return 0

    t0 = time.time()
    df = run(cfg, verbose=True)
    print(f"[done] {cfg.tag()}  {len(df)} rows  {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
