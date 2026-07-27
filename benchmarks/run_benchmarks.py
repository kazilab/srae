"""Benchmark SRAE against standard baselines on public sklearn datasets.

Run from the repository root with::

    python benchmarks/run_benchmarks.py            # full run
    python benchmarks/run_benchmarks.py --quick    # skip the slow SI variants

The script adds the repository root to ``sys.path`` so a fresh checkout works
without ``pip install -e .``.  Optional baselines (e.g. XGBoost) still need
``pip install -e ".[benchmark]"`` if you want them enabled.

Everything is 5-fold cross-validated with a fixed seed, so numbers are
reproducible.  Regression additionally reports empirical coverage of the
nominal 90% predictive interval and its mean width -- an interval that is
narrow *and* under-covering is worse than a wide one that covers, and neither
shows up in R^2.

The point is not to win.  It is to have a standing, reproducible measurement
of where an interpretable additive model sits against a linear baseline, an
additive spline baseline, and unrestricted tree ensembles.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import platform
import sys
import time
import warnings
from datetime import date

# Repo root on sys.path so `import srae` works without an editable install.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from scipy.stats import norm
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_wine,
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from srae import (
    SRAEClassifier,
    SRAEClassifierPooled,
    SRAEClassifierSI,
    SRAERegressor,
    SRAERegressorPooled,
    SRAERegressorSI,
)

# Optional peers. EBM is the closest comparator in the literature: a GA2M,
# i.e. an additive model with automatically screened *pairwise* interactions,
# reached by boosting rather than by evidence maximization. TabPFN is not
# interpretable and serves only as a small-n accuracy ceiling; it also needs a
# one-time interactive license acceptance before its weights can be downloaded,
# so it stays optional.
try:
    from interpret.glassbox import (
        ExplainableBoostingClassifier,
        ExplainableBoostingRegressor,
    )
    HAS_EBM = True
except ImportError:                                   # pragma: no cover
    HAS_EBM = False

try:
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    HAS_TABPFN = True
except ImportError:                                   # pragma: no cover
    HAS_TABPFN = False

#: Directory holding TabPFN ``.ckpt`` files, from ``$TABPFN_WEIGHTS_DIR``.
#:
#: Pointing at local weights via ``model_path`` sidesteps TabPFN's download
#: path entirely, which otherwise needs a Prior Labs API token *and* access to
#: a gated HuggingFace repo. Without this the model is simply skipped -- the
#: benchmark must stay runnable by someone who has neither.
TABPFN_WEIGHTS = os.environ.get("TABPFN_WEIGHTS_DIR", "").strip()


def _tabpfn_ckpt(kind):
    """Local checkpoint for ``kind`` in {'regressor', 'classifier'}, or None."""
    if not TABPFN_WEIGHTS:
        return None
    p = pathlib.Path(TABPFN_WEIGHTS).expanduser() / f"tabpfn-v3-{kind}-v3_default.ckpt"
    return str(p) if p.is_file() else None


def _make_tabpfn(kind):
    """Build a TabPFN estimator, preferring local weights when available."""
    ckpt = _tabpfn_ckpt(kind)
    cls = TabPFNRegressor if kind == "regressor" else TabPFNClassifier
    return cls(model_path=ckpt) if ckpt else cls()

warnings.filterwarnings("ignore")
SEED = 0
LEVEL = 0.90
Z90 = norm.ppf(0.5 + LEVEL / 2)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def regression_datasets(subsample=2000, allow_download=True):
    """Bundled datasets first; california needs a download on a fresh machine.

    ``load_diabetes`` ships with scikit-learn, so the benchmark always has at
    least one regression dataset even with no network.
    """
    X, y = load_diabetes(return_X_y=True)
    yield "diabetes", X, y

    try:
        cal = fetch_california_housing(download_if_missing=allow_download)
    except Exception as exc:                      # offline and not cached
        print(f"  [skipped california: {type(exc).__name__}: {exc}]", flush=True)
        return
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(cal.data), size=min(subsample, len(cal.data)),
                     replace=False)
    yield "california(sub)", cal.data[idx], cal.target[idx]


def classification_datasets():
    X, y = load_breast_cancer(return_X_y=True)
    yield "breast_cancer", X, y
    X, y = load_wine(return_X_y=True)
    yield "wine(3cls)", X, y


# --------------------------------------------------------------------------
# Model grids
# --------------------------------------------------------------------------

def regressors(quick):
    grid = {
        "SRAE (Type-II)": lambda: SRAERegressor(),
        "SRAE Pooled": lambda: SRAERegressorPooled(),
        "Ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Spline+Ridge (additive)": lambda: make_pipeline(
            StandardScaler(),
            SplineTransformer(n_knots=8, degree=3),
            Ridge(alpha=1.0),
        ),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=300, random_state=SEED, n_jobs=-1),
        "HistGBR": lambda: HistGradientBoostingRegressor(random_state=SEED),
    }
    if HAS_EBM:
        grid["EBM (GA2M)"] = lambda: ExplainableBoostingRegressor(random_state=SEED)
    if HAS_TABPFN:
        grid["TabPFN"] = lambda: _make_tabpfn("regressor")
    if not quick:
        grid["SRAE SI"] = lambda: SRAERegressorSI()
    return grid


def classifiers(quick):
    grid = {
        "SRAE (Type-II)": lambda: SRAEClassifier(),
        "SRAE Pooled": lambda: SRAEClassifierPooled(),
        "Logistic": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000)),
        "Spline+Logistic (additive)": lambda: make_pipeline(
            StandardScaler(),
            SplineTransformer(n_knots=6, degree=3),
            LogisticRegression(max_iter=5000),
        ),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1),
        "HistGBC": lambda: HistGradientBoostingClassifier(random_state=SEED),
    }
    if HAS_EBM:
        grid["EBM (GA2M)"] = lambda: ExplainableBoostingClassifier(random_state=SEED)
    if HAS_TABPFN:
        grid["TabPFN"] = lambda: _make_tabpfn("classifier")
    if not quick:
        grid["SRAE SI"] = lambda: SRAEClassifierSI()
    return grid


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def eval_regression(make, X, y, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    r2, r2_tr, rmse, cov, width, secs = [], [], [], [], [], []
    for tr, te in cv.split(X):
        m = make()
        t0 = time.time()
        m.fit(X[tr], y[tr])
        secs.append(time.time() - t0)
        # Train score on the same fold: the train-test gap separates a model
        # that generalizes from one that overfits and is rescued by the test
        # split being easy. Neither shows up in the test column alone.
        r2_tr.append(r2_score(y[tr], m.predict(X[tr])))
        pred = m.predict(X[te])
        r2.append(r2_score(y[te], pred))
        rmse.append(float(np.sqrt(np.mean((y[te] - pred) ** 2))))
        if hasattr(m, "predict_interval"):
            lo, hi = m.predict_interval(X[te], level=LEVEL)
            cov.append(float(np.mean((y[te] >= lo) & (y[te] <= hi))))
            width.append(float(np.mean(hi - lo)))
    return dict(
        r2=np.mean(r2), r2_se=np.std(r2, ddof=1) / np.sqrt(len(r2)),
        r2_train=np.mean(r2_tr), gap=np.mean(r2_tr) - np.mean(r2),
        rmse=np.mean(rmse),
        coverage=np.mean(cov) if cov else None,
        width=np.mean(width) if width else None,
        secs=np.mean(secs),
    )


def eval_classification(make, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    acc, acc_tr, ll_tr, auc, ll, brier, secs = [], [], [], [], [], [], []
    n_classes = len(np.unique(y))
    for tr, te in cv.split(X, y):
        m = make()
        t0 = time.time()
        m.fit(X[tr], y[tr])
        secs.append(time.time() - t0)
        acc_tr.append(accuracy_score(y[tr], m.predict(X[tr])))
        ll_tr.append(log_loss(y[tr], m.predict_proba(X[tr]), labels=np.unique(y)))
        P = m.predict_proba(X[te])
        acc.append(accuracy_score(y[te], m.predict(X[te])))
        ll.append(log_loss(y[te], P, labels=np.unique(y)))
        if n_classes == 2:
            auc.append(roc_auc_score(y[te], P[:, 1]))
            brier.append(brier_score_loss(y[te], P[:, 1]))
        else:
            auc.append(roc_auc_score(y[te], P, multi_class="ovr",
                                     average="macro"))
    return dict(
        acc=np.mean(acc), acc_se=np.std(acc, ddof=1) / np.sqrt(len(acc)),
        acc_train=np.mean(acc_tr), gap=np.mean(acc_tr) - np.mean(acc),
        ll_train=np.mean(ll_tr), ll_gap=np.mean(ll) - np.mean(ll_tr),
        auc=np.mean(auc), logloss=np.mean(ll),
        brier=np.mean(brier) if brier else None,
        secs=np.mean(secs),
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

class _Tee:
    """Echo to stdout and collect the same text for the results artifact."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip the scale-integrated variants (much slower)")
    ap.add_argument("--no-download", action="store_true",
                    help="never fetch datasets; skip any that are not cached")
    ap.add_argument("--out", default="benchmarks/RESULTS.md",
                    help="write a committed results artifact here ('' to skip)")
    args = ap.parse_args()
    out = _Tee()

    out("=" * 78)
    out("REGRESSION   5-fold CV, seed 0   (nominal interval level 90%)")
    out("=" * 78)
    for name, X, y in regression_datasets(allow_download=not args.no_download):
        out(f"\n{name}   n={X.shape[0]}  p={X.shape[1]}")
        out(f"  {'model':28}{'R2 test':>16}{'R2 train':>10}{'gap':>8}"
            f"{'cover':>8}{'width':>9}{'s/fit':>8}")
        for label, make in regressors(args.quick).items():
            try:
                r = eval_regression(make, X, y)
            except Exception as exc:
                out(f"  {label:28}  [skipped: {type(exc).__name__}: {str(exc)[:40]}]")
                continue
            cov = f"{r['coverage']*100:6.1f}%" if r["coverage"] is not None else "     --"
            wid = f"{r['width']:9.1f}" if r["width"] is not None else "       --"
            out(f"  {label:28}{r['r2']:8.4f}+/-{r['r2_se']:.3f}"
                f"{r['r2_train']:10.4f}{r['gap']:8.4f}{cov}{wid}{r['secs']:8.2f}")

    out("\n" + "=" * 78)
    out("CLASSIFICATION   5-fold stratified CV, seed 0")
    out("=" * 78)
    for name, X, y in classification_datasets():
        out(f"\n{name}   n={X.shape[0]}  p={X.shape[1]}  classes={len(np.unique(y))}")
        out(f"  {'model':28}{'acc test':>16}{'acc tr':>9}{'gap':>8}"
            f"{'logloss':>9}{'ll tr':>8}{'ll gap':>8}{'s/fit':>8}")
        for label, make in classifiers(args.quick).items():
            try:
                r = eval_classification(make, X, y)
            except Exception as exc:
                out(f"  {label:28}  [skipped: {type(exc).__name__}: {str(exc)[:40]}]")
                continue
            br = f"{r['brier']:8.4f}" if r["brier"] is not None else "      --"
            out(f"  {label:28}{r['acc']:8.4f}+/-{r['acc_se']:.3f}"
                f"{r['acc_train']:9.4f}{r['gap']:8.4f}{r['logloss']:9.4f}"
                f"{r['ll_train']:8.4f}{r['ll_gap']:8.4f}{r['secs']:8.2f}")


    if args.out:
        import srae
        header = [
            "# SRAE benchmark results",
            "",
            "Generated by `python benchmarks/run_benchmarks.py"
            + (" --quick" if args.quick else "") + "`.",
            "Regenerate rather than edit by hand.",
            "",
            f"- srae {srae.__version__}",
            f"- python {sys.version.split()[0]}, {platform.system()} {platform.machine()}",
            f"- date {date.today().isoformat()}",
            f"- EBM (interpret): {'yes' if HAS_EBM else 'not installed'}",
            f"- TabPFN: {'yes' if HAS_TABPFN else 'not installed'}"
            + (" (local weights)" if _tabpfn_ckpt("regressor") else ""),
            "",
            "Baselines run at library defaults with no hyperparameter search,",
            "which flatters SRAE since it tunes its own shrinkage internally.",
            "`california` is a 2000-row subsample for runtime.",
            "",
            "```text",
        ]
        text = "\n".join(header + out.lines + ["```", ""])
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
