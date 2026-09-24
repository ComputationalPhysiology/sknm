"""The markup the package's docstrings are written in.

They are rendered as markdown on the reference pages, so what is valid there is what is
valid in a docstring.
"""

import ast
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "sknm"


def _docstrings():
    """Yield every docstring in the package with enough context to report where it is.

    Yields
    ------
    tuple of (Path, str, str)
        The file, the name of the thing documented, and the docstring itself.
    """
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, holders):
                text = ast.get_docstring(node, clean=False)
                if text:
                    yield path, getattr(node, "name", "<module>"), text


def test_the_docstrings_are_written_in_the_markdown_the_pages_render():
    """The reference pages are markdown, and reStructuredText does not survive the trip.

    A trailing `::` opens a literal block in reStructuredText and is plain text in markdown,
    where it shows up as a stray colon pair; a `.. ` directive is dropped along with whatever
    it introduced. Both read as mistakes on the page, so neither belongs in a docstring.
    """
    offenders = [
        f"{path.name}:{name}: {line.strip()}"
        for path, name, text in _docstrings()
        for line in text.splitlines()
        if line.strip().endswith("::") or line.strip().startswith(".. ")
    ]
    assert offenders == []
