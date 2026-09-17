"""
image_groups.py
===============
The dihedral group D4 acting on square images, used in the wafer-map case
study.

D4 has eight elements: four rotations by multiples of 90 degrees and the four
reflections obtained by composing them with a horizontal flip. It is finite, so
the uniform distribution on it is its Haar measure, and it maps the pixel grid
onto itself exactly - no interpolation, no padding, no loss of information at
the borders. That matters here: an arbitrary rotation would have to resample the
image, and the resampling itself would change the score in ways unrelated to the
group.

Approximate invariance
----------------------
Unlike the sign-flip group of the simulation study, exact invariance is NOT
guaranteed by construction. Whether P(gX, Y) = P(X, Y) depends on the defect
class:

  Edge-Ring, Center, Donut, Random, Near-full
      defined by radial structure or by the absence of directional structure,
      so a rotation or reflection plausibly preserves the class;

  Scratch, Loc, Edge-Loc
      defined by a linear trace or by a localised region, so the transformed
      wafer remains a valid instance of the class but the class-conditional
      distribution of orientations need not be invariant.

This is the point of the case study rather than a limitation of it. Conformal
validity does not require invariance: conditionally on the sampled
transformations the orbit-averaged score is a fixed measurable function of the
observation, so coverage holds regardless. Invariance is what licenses reading
orbit averaging as variance reduction, and its failure should therefore show up
in efficiency, not in coverage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DihedralGroup"]


@dataclass(frozen=True)
class DihedralGroup:
    """D4 acting on the last two axes of an image batch.

    Images are expected with shape (n, H, W) or (n, H, W, C). Group elements are
    encoded as integers 0..7: element ``k`` applies ``k % 4`` quarter turns and,
    when ``k >= 4``, a horizontal flip first.

    Parameters
    ----------
    include_reflections : bool, default True
        When False the group reduces to the cyclic group C4 of rotations only.
        Useful for checking whether reflections in particular break invariance
        for the directional defect classes.
    """

    include_reflections: bool = True

    @property
    def size(self) -> int:
        """Cardinality: 8 for D4, 4 for C4.

        Reported for completeness. It is not the effective calibration sample
        size: orbit averaging leaves the number of exchangeable calibration
        units unchanged.
        """
        return 8 if self.include_reflections else 4

    def sample(self, n_transforms: int, rng: np.random.Generator) -> np.ndarray:
        """Draw `n_transforms` group elements, shared across all observations.

        Drawn independently of the data and reused for every calibration and
        test image within a replication, which is what keeps the averaged score
        a fixed measurable function conditionally on them.

        With B at least as large as the group, sampling with replacement still
        leaves some elements out; `full_orbit` gives the exhaustive alternative.
        """
        if n_transforms < 1:
            raise ValueError(f"n_transforms must be >= 1; got {n_transforms}")
        return rng.integers(0, self.size, size=n_transforms)

    def full_orbit(self) -> np.ndarray:
        """Every group element, in order. Exact orbit average, no Monte Carlo.

        Preferable to `sample` whenever the group is small enough to enumerate:
        the orbit average is then exact and B drops out of the analysis
        entirely.
        """
        return np.arange(self.size)

    def apply(self, X: np.ndarray, g: int) -> np.ndarray:
        """Apply a single group element to a batch of images.

        Parameters
        ----------
        X : np.ndarray of shape (n, H, W) or (n, H, W, C)
        g : int in 0..size-1

        Returns
        -------
        np.ndarray, same shape as X. A copy; X is never modified.
        """
        g = int(g)
        if not 0 <= g < self.size:
            raise ValueError(f"g must be in 0..{self.size - 1}; got {g}")
        if X.ndim not in (3, 4):
            raise ValueError(f"X must have 3 or 4 axes; got shape {X.shape}")

        out = X
        if g >= 4:                      # reflect first, then rotate
            out = np.flip(out, axis=2)
        k = g % 4
        if k:
            out = np.rot90(out, k=k, axes=(1, 2))
        return np.ascontiguousarray(out)

    def orbit(self, X: np.ndarray, elements: np.ndarray) -> np.ndarray:
        """Apply several group elements, returning the whole orbit sample.

        Returns
        -------
        np.ndarray of shape (B, ...) where entry [b] is `apply(X, elements[b])`.
        """
        return np.stack([self.apply(X, g) for g in np.atleast_1d(elements)],
                        axis=0)
