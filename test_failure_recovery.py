# test_failure_recovery.py
from tools.registry import registry
from agents.graph import route_after_review
from agents.state import AgentState


def test_tool_exception_handling():
    print("--- 1. Testing Tool Fault Tolerance ---")
    bad_res = registry.execute(
        "run_python_code",
        specialist="coder",
        code="raise ValueError('Simulated tool exception')"
    )
    # The sandbox catches exceptions and returns the error message in the output payload
    output_str = str(bad_res.get("output", "")) + str(bad_res.get("error", ""))
    assert "Simulated tool exception" in output_str or "Execution Error" in output_str
    print("  [Pass] Unhandled exception in tool intercepted without crashing orchestrator.")


def test_timeout_enforcement():
    print("\n--- 2. Testing Sandbox Timeout Enforcement ---")
    # Simulate an infinite loop in Python execution
    hang_code = "import time\nwhile True:\n    pass"
    res = registry.execute("run_python_code", specialist="coder", code=hang_code, timeout_seconds=1)
    output_str = str(res.get("output", "")) + str(res.get("error", ""))
    assert "exceeded" in output_str.lower() or "limit" in output_str.lower() or "timeout" in output_str.lower()
    print("  [Pass] Long-running loop safely terminated by sandbox alarm.")


def test_circuit_breaker_escalation():
    print("\n--- 3. Testing Quality Rejection Escalation Threshold ---")
    # Case A: 1st Rejection -> retry
    state_retry: AgentState = {
        "reviewer_verdict": "rejected",
        "error_count": 1,
    }
    assert route_after_review(state_retry) == "retry_specialist"
    print("  [Pass] First rejection routes to specialist retry.")

    # Case B: 2nd Rejection (error_count >= 2) -> circuit breaker
    state_escalate: AgentState = {
        "reviewer_verdict": "rejected",
        "error_count": 2,
    }
    assert route_after_review(state_escalate) == "human_escalation"
    print("  [Pass] Repeated rejection (error_count >= 2) triggers human escalation.")


if __name__ == "__main__":
    print("Testing Subtask 19: Edge-Case & Failure-Recovery System...")
    test_tool_exception_handling()
    test_timeout_enforcement()
    test_circuit_breaker_escalation()
    print("\nFailure Recovery Suite (Subtask 19) Verified Successfully!")