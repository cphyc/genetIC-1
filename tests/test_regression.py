"""Run the unchanged upstream fixtures and tolerances against the Python port."""
import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from genetic.cli import Interpreter

REPOSITORY = Path(__file__).resolve().parents[1]
FIXTURES = REPOSITORY / "genetIC/tests"
SPLICE_FIXTURES = REPOSITORY / "tests/fixtures/splicing"
CASES = sorted({p.name for root in (FIXTURES, SPLICE_FIXTURES) for p in root.glob("test_*")})
MAPPERS = sorted(p.name for p in FIXTURES.glob("mapper_test_*"))


@pytest.fixture(scope="session")
def fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp("legacy")
    files = subprocess.check_output(["git", "ls-files", "genetIC/tests"], cwd=REPOSITORY, text=True).splitlines()
    for file in files:
        source = REPOSITORY/file
        target = root/source.relative_to(FIXTURES)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for source in SPLICE_FIXTURES.glob("test_*"):
        shutil.copytree(source, root/source.name, dirs_exist_ok=True)
    return root


@pytest.fixture(scope="session")
def compare():
    sys.path.insert(0, str(REPOSITORY/"tools"))
    import compare
    # Keep upstream assertions and tolerances, without launching desktop viewers.
    compare.post_compare_diagnostic_plot = lambda *args: None
    compare.post_compare_grid_diagnostic_plot = lambda *args: None
    return compare


@pytest.mark.parametrize("name", CASES)
def test_legacy(name, fixtures, compare):
    p = fixtures/name
    output = io.StringIO()
    failed = False
    with contextlib.redirect_stdout(output):
        try:
            Interpreter().run(p/"paramfile.txt")
        except ValueError as error:
            print(str(error))
            failed = True
    (p/"IC_output.txt").write_text(output.getvalue())
    assert failed == name.endswith("_error"), output.getvalue()
    previous = sys.argv
    try:
        sys.argv = ["compare.py", str(p)+"/"]
        compare.default_comparisons()
    finally:
        sys.argv = previous


@pytest.mark.parametrize("name", MAPPERS)
def test_mapper(name, fixtures):
    p = fixtures/name
    previous = Path.cwd()
    try:
        os.chdir(p)
        a = Interpreter(geometry_only=True)
        a.run("paramfile_a.txt" if Path("paramfile_a.txt").exists() else "paramfile.txt")
        if Path("ID_a.txt").exists():
            b = Interpreter(geometry_only=True)
            b.run("paramfile_b.txt" if Path("paramfile_b.txt").exists() else "paramfile.txt")
            ids = np.atleast_1d(np.loadtxt("ID_a.txt", dtype=int))
            b.transfer_ids(a, ids)
            result = b.flagged_ids()
        else:
            result = a.flagged_ids()
        np.testing.assert_array_equal(result, np.atleast_1d(np.loadtxt("reference.txt", dtype=int)))
    finally:
        os.chdir(previous)
