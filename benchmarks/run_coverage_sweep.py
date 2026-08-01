"""Measure predictive-interval coverage against sample size.

Run from the repository root with::

    python benchmarks/run_coverage_sweep.py             # 40 replicates
    python benchmarks/run_coverage_sweep.py --reps 10   # quicker

This exists so the coverage table in ``README.md`` is reproducible rather than
quoted.  The design deliberately matches ``TestCoverage`` in
``tests/test_statistical.py`` -- same additive truth, same noise level, same
seeds -- so the documented numbers and the asserted bounds describe the same
experiment.  The tests only pin ranges and directions, which is right for a
test but leaves the published figures unreproducible on their own.

The truth is a well-specified smooth additive function with i.i.d. Gaussian
inputs, so this measures the *best* case: coverage here is an upper bound on
what to expect from correlated or misspecified real data.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import numpy as np
from scipy.stats import norm

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from srae import SRAERegressor, SRAERegressorSI  # noqa: E402

warnings.filterwarnings("ignore")

#: Kept identical to ``tests/test_statistical.py``; change both together.
SIGMA = 0.5
LEVEL = 0.90
N_TEST = 400


def _additive(X):
    return 1.2 * np.sin(1.5 * X[:, 0]) + 0.8 * X[:, 1] - 0.5 * X[:, 2]


def _sample(seed, n, sigma=SIGMA, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    return X, _additive(X) + rng.normal(0, sigma, n)


def sweep(cls, n, reps, **kw):
    """Mean empirical coverage of the nominal ``LEVEL`` interval, and sigma2_."""
    z = norm.ppf(0.5 + LEVEL / 2)
    cov, s2 = [], []
    for r in range(reps):
        X, y = _sample(500 + r, n)
        Xt, yt = _sample(9000 + r, N_TEST)
        m = cls(interactions=False, **kw).fit(X, y)
        mean, std = m.predict(Xt, return_std=True)
        cov.append(np.mean(np.abs(yt - mean) <= z * std))
        s2.append(float(getattr(m, "sigma2_", np.nan)))
    return float(np.mean(cov)), float(np.mean(s2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40,
                    help="replicates per sample size (default 40)")
    args = ap.parse_args()

    print(f"nominal {LEVEL:.0%} predictive intervals, "
          f"sigma={SIGMA} (true sigma^2={SIGMA**2:.2f}), reps={args.reps}")
    print(f"{'n_train':>8} {'coverage':>10} {'sigma2_':>9}")
    for n in (60, 100, 200, 400):
        cov, s2 = sweep(SRAERegressor, n, args.reps)
        print(f"{n:8d} {cov * 100:9.1f}% {s2:9.3f}")

    # The shortfall at small n is unpropagated smoothing-parameter
    # uncertainty; integrating that scale is what SI is for.
    plain, _ = sweep(SRAERegressor, 100, args.reps)
    integ, _ = sweep(SRAERegressorSI, 100, args.reps,
                     n_samples=60, n_burn=40, n_chains=2)
    print(f"\nat n=100:  Type-II {plain * 100:.1f}%   "
          f"scale-integrated {integ * 100:.1f}%")


if __name__ == "__main__":
    main()
