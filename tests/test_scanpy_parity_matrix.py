"""CREST against the installed scanpy on real data, one test per comparison, with the tolerances of ``_tolerances.py``.

``_parity_cases.py`` runs every comparison once per data set (the PBMC 3k run takes about two minutes) and the tests assert
``deviation <= tolerance``. Both tools start from the same raw counts; see ``_tolerances.py`` for how each tolerance was set
and which platforms and scanpy versions it covers (1.11.5, 1.12.6 and 1.13.0 on x86-64 and arm64).

Data (environment variables):

* ``CREST_TEST_PBMC3K``   directory with ``matrix.mtx``, ``genes.tsv`` and ``barcodes.tsv`` of PBMC 3k (always run when set)
* ``CREST_TEST_PBMC10K``  path of ``pbmc10k.h5`` (PBMC 10k v3; the larger variant, about ten minutes, run when set; not exported
  by ``fetch_test_data.py``: ``export CREST_TEST_PBMC10K=$CREST_TEST_DATA/pbmc10k.h5``)

The module needs scanpy and the reference packages of the ``test`` extra. Without either data variable it skips as a whole.
"""

import functools
import os
import re

import pytest

sc = pytest.importorskip("scanpy")
pytest.importorskip("leidenalg")
pytest.importorskip("igraph")
pytest.importorskip("sklearn")

import _parity_cases as cases  # noqa: E402
import _tolerances as tol  # noqa: E402

PATHS = {name: os.environ.get(var) for name, var in (("pbmc3k", "CREST_TEST_PBMC3K"), ("pbmc10k", "CREST_TEST_PBMC10K"))}
AVAILABLE = [name for name, path in PATHS.items() if path]
if not AVAILABLE:
    pytest.skip("CREST_TEST_PBMC3K / CREST_TEST_PBMC10K not set", allow_module_level=True)


@functools.lru_cache(maxsize=None)
def measured(dataset):
    return cases.measure(dataset, PATHS[dataset])


def _scanpy_version():
    from importlib import metadata
    return metadata.version("scanpy")


CASES = [(ds, comparison) for ds in AVAILABLE for comparison in tol.TOLERANCES]


@pytest.mark.parametrize("dataset,comparison", CASES, ids=[f"{ds}-{c}" for ds, c in CASES])
def test_deviation_from_scanpy_is_within_the_tolerance(dataset, comparison):
    run = measured(dataset)
    value = run["metrics"].get(comparison)
    assert value is not None, f"{comparison} was not measured on {dataset} with scanpy {_scanpy_version()}: {run['errors'] or 'no error recorded'}"
    limit = getattr(tol, tol.TOLERANCES[comparison][dataset])
    assert value <= limit, (f"{comparison} on {dataset}, scanpy {_scanpy_version()}: deviation {value:.4g} exceeds the tolerance {limit:.4g} "
                            f"(constant {tol.TOLERANCES[comparison][dataset]})")


@pytest.mark.parametrize("dataset", AVAILABLE)
def test_every_measured_comparison_has_a_tolerance_or_is_informational(dataset):
    informational = re.compile(r"^(knn\.scanpy\.recall(_shortfall)?|knn\.crest_vs_scanpy\.overlap_shortfall|ingest\.centre_(query|reference)\..*)$")
    missing = [k for k in measured(dataset)["metrics"] if k not in tol.TOLERANCES and not informational.match(k)]
    assert not missing, f"comparisons without a tolerance constant: {missing}"
    assert not measured(dataset)["errors"], f"sections that failed on {dataset}: {list(measured(dataset)['errors'])}"


def test_no_tolerance_is_looser_than_an_existing_test_on_pbmc3k():
    loose = [(k, getattr(tol, tol.TOLERANCES[k]["pbmc3k"]), cap) for k, (cap, _) in tol.EXISTING_TESTS.items()
             if getattr(tol, tol.TOLERANCES[k]["pbmc3k"]) > cap]
    assert not loose, f"tolerance above the value an existing test applies to the same comparison: {loose}"


def test_tolerances_are_finite_and_not_negative():
    for k, per_dataset in tol.TOLERANCES.items():
        for name in per_dataset.values():
            value = getattr(tol, name)
            assert value == value and 0 <= value < float("inf"), (k, name, value)


def test_ingest_centring_follows_the_scanpy_version():
    """CREST's ``center`` option names the two conventions: scanpy >= 1.12.3 centres the query on the reference mean."""
    from packaging.version import Version
    expected = "reference" if Version(_scanpy_version()) >= Version("1.12.3") else "query"
    assert cases.scanpy_ingest_centering() == expected
