"""
groups.py
=========
Transformation group used for calibration augmentation in the simulation study.

The group is the sign-flip group acting on a block of nuisance features:

    G = {-1, +1}^q,    g_eps(x_base, x_nuis) = (x_base, eps * x_nuis)

It is finite, abelian, and the uniform distribution on G is its Haar measure.

Why this group
--------------
The simulated design is X = [X_base, X_nuis] with

    (X_base, Y) ~ make_classification(...),    X_nuis ~ N(0, I_q)   independent.

Because the nuisance block is Gaussian, symmetric around zero, and independent
of (X_base, Y), exact invariance holds BY CONSTRUCTION:

    (g X, Y) =d (X, Y)    for every g in G.

This lets the simulation verify the theory in a setting where the invariance
assumption is satisfied exactly, rather than merely plausible. Approximate
invariance is studied separately by breaking one of the three conditions above
(see `make_nuisance_block(symmetric=False)`).

Sampling contract
-----------------
Transformations must be drawn ONCE PER REPLICATION, independently of the data,
and then SHARED across every calibration and test point in that replication.
This is what makes the orbit-averaged score a fixed measurable function of
(x, y) conditionally on the transformations, which is the condition under which
the split-conformal coverage guarantee applies unchanged.

Drawing a fresh transformation per observation would also be valid (the
randomisation is then i.i.d. across calibration and test units), but that is a
different estimator and is not what this module implements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SignFlipGroup", "make_nuisance_block"]


@dataclass(frozen=True)
class SignFlipGroup:
    """Sign flips applied to the last `n_nuisance` columns of the feature matrix.

    Parameters
    ----------
    n_features : int
        Total number of columns of X.
    n_nuisance : int
        Number of trailing columns forming the nuisance block. The first
        `n_features - n_nuisance` columns are left untouched by every group
        element, since invariance is only guaranteed on the nuisance block.

    Notes
    -----
    The identity element corresponds to eps = (+1, ..., +1). `sample` never
    forces the identity to be included: with B draws the orbit average is a
    Monte Carlo approximation, and including the identity by construction would
    bias it towards the untransformed point.
    """

    n_features: int
    n_nuisance: int

    def __post_init__(self) -> None:
        if not 0 < self.n_nuisance <= self.n_features:
            raise ValueError(
                f"n_nuisance must be in (0, n_features]; "
                f"got n_nuisance={self.n_nuisance}, n_features={self.n_features}"
            )

    @property
    def size(self) -> int:
        """Cardinality |G| = 2 ** n_nuisance.

        Reported for completeness. It is NOT the effective calibration sample
        size: orbit averaging leaves the number of exchangeable calibration
        units unchanged.
        """
        return 2 ** self.n_nuisance

    def sample(self, n_transforms: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `n_transforms` group elements, shared across all observations.

        Parameters
        ----------
        n_transforms : int
            Number of transformations B used to approximate the orbit average.
        rng : np.random.Generator
            Generator seeded independently of the data. Passing the same
            generator state twice reproduces the same transformations, which is
            what makes a replication reproducible.

        Returns
        -------
        np.ndarray of shape (n_transforms, n_nuisance), entries in {-1, +1}.
        """
        if n_transforms < 1:
            raise ValueError(f"n_transforms must be >= 1; got {n_transforms}")
        return rng.choice([-1.0, 1.0], size=(n_transforms, self.n_nuisance))

    def apply(self, X: np.ndarray, eps: np.ndarray) -> np.ndarray:
        """Apply a single group element to every row of X.

        Parameters
        ----------
        X : np.ndarray of shape (n, n_features)
        eps : np.ndarray of shape (n_nuisance,), entries in {-1, +1}

        Returns
        -------
        np.ndarray of shape (n, n_features). A copy; X is never modified.
        """
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X has {X.shape[1]} columns, expected {self.n_features}"
            )
        eps = np.asarray(eps, dtype=float).ravel()
        if eps.shape[0] != self.n_nuisance:
            raise ValueError(
                f"eps has length {eps.shape[0]}, expected {self.n_nuisance}"
            )

        out = X.copy()
        out[:, -self.n_nuisance:] *= eps
        return out

    def orbit(self, X: np.ndarray, eps_batch: np.ndarray) -> np.ndarray:
        """Apply B group elements to X, returning the whole orbit sample.

        Parameters
        ----------
        X : np.ndarray of shape (n, n_features)
        eps_batch : np.ndarray of shape (B, n_nuisance) as returned by `sample`.

        Returns
        -------
        np.ndarray of shape (B, n, n_features), where entry [b] is
        `apply(X, eps_batch[b])`.
        """
        eps_batch = np.atleast_2d(np.asarray(eps_batch, dtype=float))
        return np.stack([self.apply(X, eps) for eps in eps_batch], axis=0)


def make_nuisance_block(
    n_samples: int,
    n_nuisance: int,
    rng: np.random.Generator,
    symmetric: bool = True,
    scale: float = 1.0,
) -> np.ndarray:
    """Generate the nuisance feature block.

    Parameters
    ----------
    n_samples, n_nuisance : int
        Shape of the block.
    rng : np.random.Generator
    symmetric : bool, default True
        If True, draw from N(0, scale^2 I), which is symmetric about zero and
        therefore invariant under sign flips: exact invariance holds.
        If False, draw from a shifted (non-symmetric) distribution so that
        invariance is violated in a controlled way. Used for the approximate
        invariance study, where the variance-reduction interpretation no longer
        applies although conformal validity is unaffected.
    scale : float, default 1.0
        Standard deviation of the nuisance features.

    Returns
    -------
    np.ndarray of shape (n_samples, n_nuisance)
    """
    Z = rng.normal(loc=0.0, scale=scale, size=(n_samples, n_nuisance))
    if not symmetric:
        # A location shift breaks (gX, Y) =d (X, Y) while keeping the block
        # independent of the label. The size of the shift indexes the departure
        # from exact invariance.
        Z = Z + scale
    return Z
