from http import HTTPStatus

from scripts.starase_navigator.errors import AppError
from scripts.starase_navigator.evaluate_agent_metamorphic import run_variant


class _Runtime:
    def __init__(self, fail_on: str) -> None:
        self.fail_on = fail_on
        self.calls: list[str] = []

    def agent_resolve(self, text: str, *, ui_language: str, session_id: str):
        self.calls.append(text)
        if text == self.fail_on:
            raise AppError("synthetic_failure", "failed", HTTPStatus.BAD_GATEWAY, "synthetic")
        return {"agent_execution": {"turn_count": 1}}


def test_metamorphic_context_failure_is_recorded_instead_of_aborting_suite() -> None:
    runtime = _Runtime("prefix")
    result = run_variant(
        runtime,
        case_id="case",
        context=["prefix", "never reached"],
        variant={"transform": "original", "text": "target"},
        ui_language="zh",
    )
    assert result["status"] == "error"
    assert result["error_stage"] == "context"
    assert result["context_index"] == 0
    assert result["error_code"] == "synthetic_failure"
    assert runtime.calls == ["prefix"]


def test_metamorphic_target_failure_is_recorded_with_target_stage() -> None:
    runtime = _Runtime("target")
    result = run_variant(
        runtime,
        case_id="case",
        context=["prefix"],
        variant={"transform": "punctuation", "text": "target"},
        ui_language="zh",
    )
    assert result["status"] == "error"
    assert result["error_stage"] == "target"
    assert result["error_code"] == "synthetic_failure"
    assert runtime.calls == ["prefix", "target"]
