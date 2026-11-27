"""Verify that the public repository is self-contained as source software."""
from pathlib import Path
import ast
import nbformat
import pytest

ROOT=Path(__file__).resolve().parents[1]

NOTEBOOKS = (
    "01_01_sleep_fundamentals.ipynb",
    "01_02_eeg_signal_foundations.ipynb",
    "01_03_signal_processing.ipynb",
    "01_04_sleep_edf.ipynb",
    "01_05_literature.ipynb",
    "01_06_research_gap.ipynb",
    "01_07_research_protocol.ipynb",
    "02_data_pipeline.ipynb",
    "03_classical_ml.ipynb",
    "04_deep_learning.ipynb",
    "05_ensemble_learning.ipynb",
    "06_explainability.ipynb",
    "07_scientific_report.ipynb",
)

@pytest.mark.parametrize("name",NOTEBOOKS)
def test_notebook_code_is_syntactically_valid(name):
    notebook=nbformat.read(ROOT/"notebooks"/name,as_version=4)
    nbformat.validate(notebook)
    code=[cell for cell in notebook.cells if cell.cell_type=="code"]
    assert code, "Each notebook must contain executable learning material"
    for cell in code:
        ast.parse(cell.source)


def test_release_contains_readme_and_images():
    readme=(ROOT/"README.md").read_text(encoding="utf8")
    assert "## Real-data pilot results" in readme
    assert "not clinical validation" in readme.lower()
    assert (ROOT/"figures/03_sleep_stage_patterns.png").exists()
    assert (ROOT/".github/workflows/ci.yml").exists()
    assert (ROOT/"CITATION.cff").exists()
    assert (ROOT/"docs/GITHUB_PUBLISH.md").exists()


def test_no_raw_eeg_is_bundled():
    import os
    from pathlib import Path

    excluded_dirs = {
        ".venv",
        "venv",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }

    excluded_data = {
        ("data", "raw"),
        ("data", "processed"),
    }

    for current_dir, dirs, files in os.walk(ROOT):
        relative = Path(current_dir).relative_to(ROOT)

        # Exclude local datasets and experiment outputs
        if (
            relative.parts[:2] in excluded_data
            or relative.parts[:1] == ("results",)
        ):
            dirs[:] = []
            continue

        # Exclude virtual environments and caches
        dirs[:] = [
            d for d in dirs
            if d not in excluded_dirs
        ]

        for filename in files:
            assert not filename.lower().endswith(
                (".edf", ".fif")
            ), f"Unexpected EEG file: {Path(current_dir) / filename}"
