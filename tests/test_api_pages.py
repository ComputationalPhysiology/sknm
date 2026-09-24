"""The API reference generator.

The pages under `docs/api/` are written from the installed package at build time, so the
reference always describes the code in the repository. These tests cover what goes on a page
and what is left off it.
"""

import pytest

# `griffe` reads the source rather than importing it, and is a documentation dependency. It
# is in the test extra because these tests are the only thing that checks the generator, but
# a bare install without it should skip rather than fail to collect.
generate_api_pages = pytest.importorskip(
    "generate_api_pages", reason="griffe is a documentation extra"
)


def test_a_module_offers_the_classes_and_functions_it_defines():
    module = generate_api_pages.load()["linalg"]

    names = [member.name for member in generate_api_pages.documented_members(module)]

    assert "CGSolver" in names
    assert "as_solver" in names


def test_what_a_module_imports_is_left_to_the_module_that_defines_it():
    """A name is documented once, on the page for the module it is written in.

    Almost every module imports from its neighbours, and the package re-exports seventeen
    names from its submodules. Documenting what a module merely imports would put most of
    the package on several pages at once and none of them authoritative.
    """
    package = generate_api_pages.load()

    def names(module):
        return [member.name for member in generate_api_pages.documented_members(module)]

    assert "CellNetwork" in names(package["network"])
    assert "CellNetwork" not in names(package["presets"])
    assert names(package) == []


def test_a_module_that_declares_all_offers_only_what_it_declares(tmp_path):
    (tmp_path / "toy.py").write_text(
        '"""A module that narrows its own surface."""\n\n'
        '__all__ = ["kept"]\n\n\n'
        'def kept():\n    """Documented."""\n\n\n'
        'def also_public():\n    """Not in __all__."""\n'
    )

    module = generate_api_pages.load("toy", search_paths=[tmp_path])

    assert [member.name for member in generate_api_pages.documented_members(module)] == ["kept"]


def test_names_beginning_with_an_underscore_are_left_out():
    module = generate_api_pages.load()["linalg"]

    assert "_preconditioner" in module.members
    names = [member.name for member in generate_api_pages.documented_members(module)]
    assert "_preconditioner" not in names


def test_a_function_is_shown_with_the_signature_it_is_called_by():
    page = generate_api_pages.render(generate_api_pages.load()["linalg"])

    assert "as_solver(solver: Solver | str) -> Solver" in page


def test_the_parameters_a_docstring_documents_reach_the_page():
    page = generate_api_pages.render(generate_api_pages.load()["linalg"])

    assert "corresponding solver with its default settings." in page


def test_what_a_function_returns_and_raises_reaches_the_page():
    page = generate_api_pages.render(generate_api_pages.load()["linalg"])

    assert "The solver itself, or the object the string names." in page
    assert "If `solver` is a string naming no solver." in page


def test_a_worked_example_in_a_docstring_is_shown_as_code():
    page = generate_api_pages.render(generate_api_pages.load()["linalg"])

    assert '>>> as_solver("cg")' in page
    assert "```pycon" in page


def test_a_class_is_followed_by_the_methods_it_offers():
    page = generate_api_pages.render(generate_api_pages.load()["linalg"])

    assert page.index("`CGSolver`") < page.index("CGSolver.factorize")


def test_a_module_that_is_mostly_constants_still_documents_them():
    """`sknm.units` is twenty-eight unit aliases and three functions.

    Left to classes and functions its page would be almost empty, and the module a reader
    imports `ms` and `mV` from would not say that it has them.
    """
    page = generate_api_pages.render(generate_api_pages.load()["units"])

    assert "`ms`" in page
    assert "`mV`" in page


def test_the_page_is_titled_with_the_module_it_documents():
    page = generate_api_pages.render(generate_api_pages.load()["membrane"]["protocol"])

    assert page.startswith("---\ntitle: sknm.membrane.protocol\n---\n")


def test_a_page_is_written_for_every_module_in_the_package(tmp_path):
    """A module with no page is the failure this cannot afford to make quietly.

    MyST does not complain about a table of contents entry with no file behind it, so a
    module that gained no page would simply be missing from the site.
    """
    package = generate_api_pages.load()

    written = generate_api_pages.write_pages(tmp_path)

    assert [path.name for path in written] == [
        f"{module.path}.md" for module in generate_api_pages.modules(package)
    ]
    assert all(path.read_text() for path in written)


def test_a_class_that_defines_no_constructor_is_not_shown_with_a_call_that_takes_nothing():
    """`ConvergenceError` is a bare `RuntimeError` subclass and `Solver` a protocol.

    Neither writes an `__init__`, so there is no signature to show, and a fence holding the
    class name and nothing else reads as a constructor that takes no arguments.
    """
    page = generate_api_pages.render(generate_api_pages.load()["linalg"])

    assert "```python\nConvergenceError\n```" not in page
    assert "```python\nSolver\n```" not in page
    assert "```python\nCGSolver(rtol: float = 1e-06" in page


def test_a_column_no_row_fills_is_not_shown():
    """Most unit aliases carry neither an annotation nor a docstring of their own.

    `sknm.units` is twenty-eight of them, so a table with a column per thing an attribute
    might have would be mostly empty down its whole length.
    """
    rows = [("`ms`", "", "`ureg.ms`", ""), ("`mV`", "", "`ureg.mV`", "")]

    table = generate_api_pages._table(rows, ("Name", "Type", "Value", "Description"))

    assert table[0] == "| Name | Value |"
    assert table[2] == "| `ms` | `ureg.ms` |"


def test_a_package_page_points_at_the_modules_beneath_it():
    """`sknm` and `sknm.membrane` define nothing themselves; they re-export.

    Both would otherwise be a docstring and nothing else, which is the one page a reader
    arriving at the reference is most likely to land on first.
    """
    package = generate_api_pages.render(generate_api_pages.load())
    membrane = generate_api_pages.render(generate_api_pages.load()["membrane"])

    assert "[`sknm.network`](sknm.network.md)" in package
    assert "[`sknm.membrane`](sknm.membrane.md)" in package
    assert "[`sknm.membrane.PBM`](sknm.membrane.PBM.md)" in membrane
    assert "Membrane models and the seam they plug into." in package


def test_a_module_that_has_no_modules_under_it_gets_no_such_list():
    page = generate_api_pages.render(generate_api_pages.load()["linalg"])

    assert "## Modules" not in page


def test_the_page_follows_the_order_the_module_is_written_in():
    """Source order is the author's order, and groups related things together.

    Alphabetical would separate `CGSolver` from `BiCGSTABSolver` and put the exception in
    the middle of the solvers.
    """
    page = generate_api_pages.render(generate_api_pages.load()["linalg"])

    assert page.index("## `ConvergenceError`") < page.index("## `DirectSolver`")
    assert page.index("## `DirectSolver`") < page.index("## `as_solver`")


def test_a_column_some_rows_fill_is_kept():
    rows = [("`ms`", "", "`ureg.ms`"), ("`BASE_UNITS`", "`dict[str, str]`", "`{...}`")]

    table = generate_api_pages._table(rows, ("Name", "Type", "Value"))

    assert table[0] == "| Name | Type | Value |"
    assert table[2] == "| `ms` |  | `ureg.ms` |"


def test_a_description_that_runs_on_stays_part_of_its_definition():
    """A definition list ends at the first line that is not indented under it.

    Most of these descriptions are longer than one line, and an unindented continuation
    leaves the rest of the sentence outside the entry as a loose paragraph.
    """
    page = generate_api_pages.render(generate_api_pages.load()["network"])

    assert (
        "\n  junctions are shut, which blocks intracellular current without disconnecting"
    ) in page


def test_a_package_page_is_written_before_the_modules_beneath_it(tmp_path):
    written = generate_api_pages.write_pages(tmp_path)

    names = [path.name for path in written]
    assert names[0] == "sknm.md"
    assert names.index("sknm.membrane.md") < names.index("sknm.membrane.PBM.md")


def test_a_constant_is_listed_once_and_given_no_section_of_its_own():
    """Twenty-eight unit aliases with a heading each would be the whole page."""
    page = generate_api_pages.render(generate_api_pages.load()["units"])

    assert "## `ms`" not in page
    assert page.count("| `ms` |") == 1
