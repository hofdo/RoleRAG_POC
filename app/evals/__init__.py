from typing import TYPE_CHECKING, Any

from app.evals.fixtures import EvalFixture, build_eval_fixture
from app.evals.memory_continuity import MemoryContinuityResult, evaluate_memory_continuity
from app.evals.memory_recall import MemoryRecallResult, evaluate_memory_recall
from app.evals.retrieval_quality import RetrievalQualityResult, evaluate_retrieval_quality
from app.evals.role_consistency import RoleConsistencyResult, evaluate_role_consistency

if TYPE_CHECKING:
    from app.evals.regression_runner import CategoryResult, RegressionReport

__all__ = [
    "CategoryResult",
    "EvalFixture",
    "MemoryContinuityResult",
    "MemoryRecallResult",
    "RegressionReport",
    "RetrievalQualityResult",
    "RoleConsistencyResult",
    "build_eval_fixture",
    "evaluate_memory_recall",
    "evaluate_memory_continuity",
    "evaluate_retrieval_quality",
    "evaluate_role_consistency",
    "run_regressions",
]


def __getattr__(name: str) -> Any:
    if name in {"CategoryResult", "RegressionReport", "run_regressions"}:
        from app.evals import regression_runner

        return getattr(regression_runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
