"""Smoke tests for the packaging itself: that `sknm` installs, imports and versions."""

from importlib import metadata


def test_sknm_is_importable():
    import sknm

    assert sknm is not None


def test_module_version_matches_installed_distribution():
    """`__version__` is written by hand, so it can drift from `pyproject.toml`.

    The version is static rather than derived from version control, so this is what
    catches the two falling out of step.
    """
    import sknm

    assert sknm.__version__ == metadata.version("sknm")
