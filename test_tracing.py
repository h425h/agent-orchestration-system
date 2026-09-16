# test_tracing.py
import time
from eval.tracer import tracer


def test_trace_lifecycle():
    print("Testing Subtask 13: Execution Tracing...")
    trace_id = "test_run_001"

    # 1. Root Supervisor Span
    root_span = tracer.start_span(
        trace_id=trace_id,
        name="supervisor_planning",
        agent_role="supervisor",
        attributes={"task": "Market analysis and benchmarks"}
    )
    time.sleep(0.05)
    tracer.end_span(
        root_span.span_id,
        status="success",
        attributes={"plan.subtasks_count": 3}
    )

    # 2. Child Specialist Span
    spec_span = tracer.start_span(
        trace_id=trace_id,
        name="execute_subtask",
        agent_role="researcher",
        parent_span_id=root_span.span_id,
        attributes={"subtask_id": "task_1"}
    )
    time.sleep(0.03)

    # 3. Nested Tool Call Span
    tool_span = tracer.start_span(
        trace_id=trace_id,
        name="tool_call_web_search",
        agent_role="researcher",
        parent_span_id=spec_span.span_id,
        attributes={"tool.name": "web_search", "query": "vector databases 2026"}
    )
    time.sleep(0.02)
    tracer.end_span(tool_span.span_id, status="success")

    tracer.end_span(spec_span.span_id, status="success", attributes={"output_length": 450})

    # Assertions
    trace_tree = tracer.get_trace_tree(trace_id)
    assert len(trace_tree) == 3
    assert trace_tree[0]["name"] == "supervisor_planning"
    assert trace_tree[1]["parent_span_id"] == root_span.span_id
    assert trace_tree[2]["parent_span_id"] == spec_span.span_id
    assert trace_tree[0]["latency_ms"] >= 40.0

    print("  [Pass] Root and child spans linked correctly.")
    print(f"  [Pass] Spans recorded with realistic latencies: {trace_tree[0]['latency_ms']}ms, {trace_tree[2]['latency_ms']}ms.")
    print("\nExecution Tracing (Subtask 13) Foundation Verified Successfully!")


if __name__ == "__main__":
    test_trace_lifecycle()