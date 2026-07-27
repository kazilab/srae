"""Project metadata is a single source of truth.

``srae/_metadata.py`` feeds the package version, the packaging metadata and
the Sphinx configuration, so these values must not drift apart.
"""

from __future__ import annotations

from pathlib import Path

import srae
from srae import _metadata

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_exported_from_metadata():
    assert srae.__version__ == _metadata.VERSION


def test_license_copyright_matches_license_file():
    """LICENSE is the operative document; the metadata mirrors it verbatim."""
    text = (ROOT / "LICENSE").read_text()
    assert _metadata.LICENSE_COPYRIGHT in text


def test_copyright_holder_is_distinct_from_credited_author():
    """The credited author and the legal holder are separate fields."""
    assert _metadata.COPYRIGHT_HOLDER in _metadata.COPYRIGHT
    assert _metadata.COPYRIGHT_YEAR in _metadata.COPYRIGHT


def test_pyproject_version_is_dynamic():
    """A literal version in pyproject.toml would reintroduce the duplication."""
    text = (ROOT / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in text
    assert f'version = "{_metadata.VERSION}"' not in text
