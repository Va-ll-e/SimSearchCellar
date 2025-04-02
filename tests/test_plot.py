"""Test the plot.py script."""

from pathlib import Path

from app import main as plot_main


def test_plot_runs_without_crashing():
    """Run app without crashing."""
    try:
        plot_main()
    except Exception as err:
        raise AssertionError(f"plot_main crashed with error: {err}") from err
    assert True, "plot_main ran successfully"

    # Assert that the plot file was created
    assert Path("./plots/first_similarity_search.png").exists(), "Plot file was not created."
