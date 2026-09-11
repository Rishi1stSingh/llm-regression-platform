from __future__ import annotations

import asyncio
import time
from typing import Iterable

from src.evaluators.base import BaseEvaluator
from src.models import CaseResult, GoldenCase, EvaluationRun


class EvaluationEngine:
    """Run a BaseEvaluator over a set of GoldenCase objects.

    Responsibility:
    - Call `evaluator.evaluate_case(case)` for every case
    - Collect CaseResult objects
    - Build and finalize an EvaluationRun

    The engine is evaluator-agnostic and does not perform baseline/regression logic.
    """

    def __init__(self, evaluator: BaseEvaluator) -> None:
        self.evaluator = evaluator

    def run(self, cases: Iterable[GoldenCase], version: str) -> EvaluationRun:
        results: list[CaseResult] = []
        for case in cases:
            result = self.evaluator.evaluate_case(case)
            results.append(result)
        run = EvaluationRun(version=version, evaluator=self.evaluator.name, cases=results)
        # Propagate strategy if the evaluator exposes one
        strategy = getattr(self.evaluator, "strategy", None)
        if strategy is not None:
            run.strategy = strategy
        run.finalize_score()
        return run


class AsyncEvaluationEngine:
    """Evaluate multiple cases concurrently using asyncio with bounded concurrency.

    Runs sync `evaluate_case` in threads via asyncio.to_thread, with a semaphore
    limiting concurrent evaluations. Preserves original case ordering in results.
    """

    def __init__(self, evaluator: BaseEvaluator, concurrency: int = 5) -> None:
        self.evaluator = evaluator
        self.semaphore = asyncio.Semaphore(concurrency)

    async def _evaluate_one(self, case: GoldenCase) -> CaseResult:
        async with self.semaphore:
            return await asyncio.to_thread(self.evaluator.evaluate_case, case)

    async def run_async(self, cases: Iterable[GoldenCase], version: str) -> EvaluationRun:
        case_list = list(cases)
        start = time.monotonic()
        results = await asyncio.gather(*(self._evaluate_one(c) for c in case_list))
        elapsed = time.monotonic() - start
        run = EvaluationRun(version=version, evaluator=self.evaluator.name, cases=list(results))
        strategy = getattr(self.evaluator, "strategy", None)
        if strategy is not None:
            run.strategy = strategy
        run.finalize_score()
        run.elapsed_seconds = elapsed  # type: ignore[attr-defined]
        return run

    def run(self, cases: Iterable[GoldenCase], version: str) -> EvaluationRun:
        """Synchronous wrapper for run_async."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            # Already in an async context — run in a new thread to avoid nested loop issues
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self.run_async(cases, version))
                return future.result()
        return asyncio.run(self.run_async(cases, version))