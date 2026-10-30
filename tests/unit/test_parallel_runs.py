"""Run-completeness checks: a failed worker must stop the study, not shrink it."""

import json

import pytest

from scripts import run_e6_monte_carlo as mc
from scripts.parallel_runs import RunFailure, completeness, run_tasks


def _square(x):
    return x * x


def _fail_on_three(x):
    if x == 3:
        raise ValueError("simulated worker failure")
    return x


@pytest.mark.parametrize("jobs", [1, 2])
def test_all_results_returned_in_task_order(jobs):
    assert run_tasks(_square, [4, 1, 3, 2], jobs=jobs) == [16, 1, 9, 4]


@pytest.mark.parametrize("jobs", [1, 2])
def test_worker_failure_raises_and_names_the_run(jobs):
    with pytest.raises(RunFailure, match=r"1 of 5 runs did not complete.*\[3\]"):
        run_tasks(_fail_on_three, [0, 1, 2, 3, 4], jobs=jobs)


def test_every_failure_is_reported_not_only_the_first():
    with pytest.raises(RunFailure, match=r"2 of 4"):
        run_tasks(_fail_on_three, [3, 3, 0, 1], jobs=1)


def test_completeness_record_and_mismatches():
    record = completeness(3, [42, 1042, 2042], [42, 1042, 2042])
    assert record == {"n_requested": 3, "n_completed": 3, "seeds": [42, 1042, 2042]}
    with pytest.raises(RunFailure):
        completeness(3, [42, 1042, 2042], [42, 1042])          # a run is missing
    with pytest.raises(RunFailure):
        completeness(3, [42, 1042, 2042], [42, 1042, 1042])    # duplicate / wrong seed


def test_monte_carlo_fails_loudly_when_a_run_errors(monkeypatch):
    def flaky(task):
        if task["index"] == 2:
            raise RuntimeError("simulated worker failure")
        return {"seed": task["seed"]}

    monkeypatch.setattr(mc, "run_one", flaky)
    tasks = [{"index": i, "seed": 42 + 1000 * i} for i in range(4)]
    with pytest.raises(RunFailure, match="1 of 4 runs did not complete"):
        mc.execute_runs(tasks, jobs=1)


def test_monte_carlo_detects_a_run_with_the_wrong_seed(monkeypatch):
    monkeypatch.setattr(mc, "run_one", lambda task: {"seed": task["seed"] + 1})
    with pytest.raises(RunFailure, match="do not match"):
        mc.execute_runs([{"index": 0, "seed": 42}], jobs=1)


def test_published_e6_monte_carlo_is_complete():
    """The shipped 200-run result has 200 unique seeds and 200 per-run records."""
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "docs" / "e6_monte_carlo.json"
    data = json.loads(path.read_text())
    assert data["n_runs"] == 200 == len(data["runs"])
    assert len(set(data["seeds"])) == 200
    assert [r["seed"] for r in data["runs"]] == data["seeds"]
