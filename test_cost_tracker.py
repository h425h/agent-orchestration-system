# test_cost_tracker.py
from eval.cost_tracker import cost_tracker


def test_cost_and_token_tracking():
    print("Testing Subtask 15: Cost & Token Tracking...")
    run_id = "test_run_metrics_101"

    # 1. Supervisor: Haiku (Input: 1500 tokens, Output: 400 tokens)
    cost_tracker.record_usage(
        run_id=run_id,
        agent_role="supervisor",
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        input_tokens=1500,
        output_tokens=400,
        latency_ms=310.0,
    )

    # 2. Coder: Sonnet (Input: 2000 tokens, Output: 800 tokens)
    cost_tracker.record_usage(
        run_id=run_id,
        agent_role="coder",
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        input_tokens=2000,
        output_tokens=800,
        latency_ms=750.0,
    )

    # 3. Reviewer: Haiku (Input: 1000 tokens, Output: 150 tokens)
    cost_tracker.record_usage(
        run_id=run_id,
        agent_role="reviewer",
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        input_tokens=1000,
        output_tokens=150,
        latency_ms=180.0,
    )

    summary = cost_tracker.get_run_summary(run_id)

    # Validate math:
    # Haiku supervisor: (1.5 * 0.0008) + (0.4 * 0.004) = 0.0012 + 0.0016 = 0.0028
    # Sonnet coder: (2.0 * 0.003) + (0.8 * 0.015) = 0.0060 + 0.0120 = 0.0180
    # Haiku reviewer: (1.0 * 0.0008) + (0.15 * 0.004) = 0.0008 + 0.0006 = 0.0014
    # Expected total: 0.0028 + 0.0180 + 0.0014 = $0.0222
    assert summary["total_tokens"] == 5850
    assert abs(summary["total_cost_usd"] - 0.0222) < 0.0001
    assert "supervisor" in summary["by_agent"]
    assert "coder" in summary["by_agent"]

    print("  [Pass] Exact Bedrock pricing verified ($0.0222 across models).")
    print(f"  [Pass] Total tokens: {summary['total_tokens']} tokens.")
    print("  [Pass] Attribution breakdown by agent verified:")
    for role, stats in summary["by_agent"].items():
        print(f"    - {role}: {stats['input_tokens'] + stats['output_tokens']} tokens (${stats['cost_usd']})")

    print("\nCost & Performance Tracking (Subtask 15) Foundation Verified Successfully!")


if __name__ == "__main__":
    test_cost_and_token_tracking()