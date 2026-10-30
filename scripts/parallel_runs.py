"""Run independent tasks serially or in a process pool, and fail loudly.

A study that silently loses runs would report probabilities over fewer trials
than it claims. ``run_tasks`` therefore (a) collects every worker exception
instead of stopping at the first, (b) raises ``RunFailure`` naming the failed
task indices and their errors, and (c) checks that exactly as many results came
back as tasks were requested. Results are returned in task order, so they do not
depend on the worker count or on completion order.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Sequence


class RunFailure(RuntimeError):
    """One or more requested runs did not complete."""


def run_tasks(func: Callable[[Any], Any], tasks: Sequence[Any], jobs: int = 1,
              label: str = "run") -> List[Any]:
    n_requested = len(tasks)
    results: List[Any] = [None] * n_requested
    done = [False] * n_requested
    failures: Dict[int, str] = {}

    if jobs > 1 and n_requested > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(func, task) for task in tasks]
            for index, future in enumerate(futures):
                try:
                    results[index] = future.result()
                    done[index] = True
                except BaseException as error:  # includes BrokenProcessPool
                    failures[index] = f"{type(error).__name__}: {error}"
    else:
        for index, task in enumerate(tasks):
            try:
                results[index] = func(task)
                done[index] = True
            except Exception as error:
                failures[index] = f"{type(error).__name__}: {error}"

    n_completed = sum(done)
    if failures or n_completed != n_requested:
        first = ", ".join(f"{label} {i}: {message}"
                          for i, message in list(failures.items())[:3])
        raise RunFailure(
            f"{n_requested - n_completed} of {n_requested} {label}s did not complete "
            f"(failed indices {sorted(failures)}). First errors: {first}")
    return results


def completeness(n_requested: int, seeds_requested: Sequence[int],
                 seeds_completed: Sequence[int]) -> Dict[str, Any]:
    """Return the completeness record stored in every output JSON; raise on a mismatch."""
    if (len(seeds_completed) != n_requested
            or list(seeds_completed) != list(seeds_requested)
            or len(set(seeds_completed)) != len(seeds_completed)):
        raise RunFailure(
            f"completed seeds do not match the request: requested {n_requested}, "
            f"completed {len(seeds_completed)}, unique {len(set(seeds_completed))}")
    return {"n_requested": int(n_requested), "n_completed": len(seeds_completed),
            "seeds": [int(s) for s in seeds_completed]}
