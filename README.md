# class-conditional-cp-data-augmentation

Code, data and notebooks for the article

**Enhancing Class-Conditional Conformal Prediction for Multiclass Scenarios
with Data Augmentation**
Andrea Laurenzi, Matteo Borrotti

[DOI / arXiv link – add here]

This repository accompanies the revised version of the manuscript. The method
proposed in the first submission treated transformed copies of a calibration
point as additional calibration units; that construction has no finite-sample
conformal guarantee, and the revision replaces it with an orbit-averaged
nonconformity score, which retains exact validity. The earlier construction is
retained in the experiments as a diagnostic benchmark, not as a method.

---

## Repository structure

| Path | Contents |
|---|---|
| `dataset/` | WM-811K subset and the train / validation / calibration / test splits |
| `models/` | Fitted classifiers used in the real-data study |
| `utils/` | Conformal scores, calibration procedures, transformation groups, simulation driver, and the data-handling utilities of the wafer-map application |
| `notebooks/simulations/` | Synthetic experiments (Section 3 and Appendix B) |
| `notebooks/real_examples/` | Wafer-map application (Section 4 and Appendix C) |

---

## Modules in `utils/`

Conformal machinery:

- `scores.py` — randomised APS, the orbit-averaged score, the
  prediction-averaged (TTA-Avg) score, and the mechanism diagnostics
- `calibration.py` — marginal, classwise, clustered and naive multi-copy
  calibration, all delegating to the reference implementation
- `groups.py` — the sign-flip group used in the simulations
- `image_groups.py` — the dihedral group $D_4$ acting on square images
- `metrics.py` — aggregation of the simulation output into the reported tables
- `simulate.py`, `run_grid.py` — data-generating process, replication loop and
  command-line driver
- `ding_conformal_utils.py`, `clustering_utils.py` — vendored verbatim from
  [tiffanyding/class-conditional-conformal](https://github.com/tiffanyding/class-conditional-conformal).
  Two lines were changed, both marked `[VENDORED]` in the file: `import torch`
  is commented out, since the score helpers it serves are replaced by
  `scores.py`, and a package-relative import is made flat. No calibration logic
  was modified, so the baselines are the authors' own implementation.

Wafer-map application: `general_utils.py`, `effnet_utils.py`, `cnn_utils.py`,
`cfg_effnet.py`, `cfg_cnn.py`.

---

## `notebooks/simulations/`

Run in order; each notebook states what it needs from the previous one.

**1. `1_pilot_colab.ipynb`** — fixes the design choices: the number of
transformations $B$, the number of replications $R$, and the cost per
replication. Also locates the empty-slice warnings emitted by clustered
conformal prediction in the low-data regime.

**2. `2_campaign.ipynb`** — runs the main experiment at $R=300$ over the
calibration-size grid. Written for Google Colab: results are written to Drive, a
checkpoint is taken every ten replications, and resuming is exact, so an
interrupted session costs at most ten replications.

**3. `3_paper_figures.ipynb`** — produces the figures and tables of the main
text from the long tables, together with the LaTeX export and the figures printed
as sentences for the manuscript.

**4. `4_robustness_classifier.ipynb`** — repeats the comparison with Random
Forest and an MLP at two calibration sizes, to check that the result is not
specific to XGBoost.

**5. `5_clustered_results.ipynb`** — evaluates clustered conformal prediction and
documents why its clustering step rarely runs in a problem with eight classes:
the tuning heuristic of Ding et al. was calibrated for 100 to 1000 classes.

The notebooks read their inputs from Google Drive. Set `BASE` in the setup cell
of each one before running.

---

## `notebooks/real_examples/`

**1. `WMDD_dataset_selection_from_wm811k.ipynb`** — builds the subset of
WM-811K and its splits.

**2–5.** Training notebooks for the four classifiers: a CNN trained from
scratch, EfficientNet-B0, ResNeXt-50 and CoaT-Tiny, the last three fine-tuned
from ImageNet weights.

**6. `wm-train-export.ipynb`** — computes, for every held-out wafer and every
element of $D_4$, the predicted probabilities of each fitted model, and writes
them as `<tag>_probs.npy` of shape `(8, n, 8)` together with the labels and a
metadata file. The eight forward passes are done once here; the conformal
notebook then resamples calibration/test splits of an array already in memory.
Exports are reused when the checkpoint and the held-out set are unchanged, which
is verified through a SHA1 of the held-out index.

**7. `7_WM_real_data_models.ipynb`** — the conformal analysis: 300
calibration/test splits at $\alpha \in \{0.10, 0.01\}$, comparing classwise APS,
classwise TTA-Avg and classwise orbit-averaged APS, with marginal and clustered
conformal prediction as baselines and the naive multi-copy construction as a
diagnostic.

---

## Reproducing the results

The simulations are self-contained: `2_campaign.ipynb` regenerates everything
from scratch. The full grid takes several hours; `3_paper_figures.ipynb` reads
the long tables and needs minutes.

The wafer-map study needs the fitted models. Run `wm-train-export.ipynb` first
to produce the probability arrays, then `7_WM_real_data_models.ipynb`.

Paths are hard-coded to Google Drive or Kaggle mounts and must be edited for a
local run.

