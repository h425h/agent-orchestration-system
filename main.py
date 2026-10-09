# main.py
import uuid
import time
from dotenv import load_dotenv

from agents.graph import create_agent_graph
from agents.state import AgentState
from agents.bedrock_llm import usage_collector
from memory.checkpointer import get_sqlite_checkpointer
from memory.semantic_store import semantic_memory
from eval.tracer import tracer
from eval.cost_tracker import cost_tracker

load_dotenv()

HALT_PREFIX = "[HUMAN ESCALATION HALT]"


def classify_run(final_state: dict) -> str:
    """
    'escalated'  - halted for a human (plan gate, review gate, or retry ceiling)
    'completed'  - every planned subtask was reviewed, approved and committed
    'incomplete' - ended without finishing the plan and without an escalation
    """
    if (final_state.get("final_output") or "").startswith(HALT_PREFIX):
        return "escalated"
    planned = {t["id"] for t in (final_state.get("plan") or {}).get("subtasks", [])}
    done = {t["id"] for t in final_state.get("completed_subtasks", [])}
    return "completed" if planned and planned <= done else "incomplete"


def run_orchestration_pipeline(task_prompt: str, thread_id: str = None, db_path: str = "orchestrator_state.db") -> dict:
    """
    Executes the full end-to-end multi-agent pipeline:
    - LangGraph execution with Sqlite state checkpointing
    - Span tracing with real per-node durations
    - Measured token usage and Bedrock cost, attributed to the node that incurred it
    - Long-term memory distillation, only for runs that actually completed
    """
    thread_id = thread_id or f"run_{uuid.uuid4().hex[:8]}"
    checkpointer = get_sqlite_checkpointer(db_path)
    if hasattr(checkpointer, "setup"):
        checkpointer.setup()

    app = create_agent_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 70)
    print(f"Launching Pipeline Run: {thread_id}")
    print(f"Task: {task_prompt}")
    print("=" * 70)

    root_span = tracer.start_span(
        trace_id=thread_id,
        name="end_to_end_pipeline",
        attributes={"task": task_prompt}
    )

    initial_state: AgentState = {
        "task": task_prompt,
        "plan": None,
        "completed_subtasks": [],
        "current_subtask_id": None,
        "current_specialist_output": None,
        "reviewer_verdict": None,
        "reviewer_feedback": None,
        "reviewer_score": None,
        "candidate_subtask": None,
        "final_output": None,
        "error_count": 0,
        "human_approved": False,
        "require_human_approval": False,
        "pending_escalation": None,
    }

    usage_collector.drain()  # discard calls made before this run started
    start_time = time.time()
    last_event_ts = start_time

    for event in app.stream(initial_state, config=config):
        for node_name, _updates in event.items():
            now = time.time()
            calls = usage_collector.drain()
            records = cost_tracker.record_calls(thread_id, node_name, calls)
            print(f"  [Graph Node Completed]: {node_name} ({len(calls)} LLM calls)")

            # Nodes run sequentially, so a node's duration is the time since the previous event.
            node_span = tracer.start_span(
                trace_id=thread_id,
                name=f"node_{node_name}",
                parent_span_id=root_span.span_id,
                agent_role=node_name
            )
            node_span.start_time = last_event_ts
            tracer.end_span(node_span.span_id, status="success", attributes={
                "llm_calls": len(calls),
                "input_tokens": sum(r.input_tokens for r in records),
                "output_tokens": sum(r.output_tokens for r in records),
                "cost_usd": round(sum(r.cost_usd for r in records), 6),
            })
            last_event_ts = now

    # The checkpoint holds the reduced state (completed_subtasks accumulated across nodes);
    # merging stream updates by hand would keep only the last node's list.
    final_state = dict(app.get_state(config).values)
    status = classify_run(final_state)

    elapsed = round(time.time() - start_time, 2)

    if status == "escalated":
        esc = final_state.get("pending_escalation") or {}
        reason = getattr(esc.get("reason"), "value", esc.get("reason")) or "reviewer_escalation"
        print(f"\nHalted for human review: {reason}")
        print(f"  {esc.get('description') or final_state.get('reviewer_feedback')}")
    tracer.end_span(root_span.span_id, status="success", attributes={"elapsed_s": elapsed, "run_status": status})

    insights = {}
    if status == "completed":
        print("\nDistilling run learnings into ChromaDB long-term memory...")
        insights = semantic_memory.distill_and_store(task_prompt, final_state)
        cost_tracker.record_calls(thread_id, "memory_distill", usage_collector.drain())
    else:
        print(f"\nSkipping memory distillation (run status: {status}).")

    summary = cost_tracker.get_run_summary(thread_id)

    print("=" * 70)
    print(f"Pipeline Run Finished in {elapsed}s — status: {status}")
    print(f"Total Run Cost: ${summary['total_cost_usd']:.5f} ({summary['total_tokens']} tokens, {summary['llm_calls']} LLM calls)")
    if summary["unpriced_calls"]:
        print(f"Warning: {summary['unpriced_calls']} calls used a model with no price entry; cost is understated.")
    if insights:
        print(f"Memory Distilled: {insights.get('summary')}")
    print("=" * 70)

    return {
        "thread_id": thread_id,
        "status": status,
        "elapsed_s": elapsed,
        "final_state": final_state,
        "cost_summary": summary,
        "distilled_insights": insights
    }


if __name__ == "__main__":
    task = "Research 2026 AI agent orchestration frameworks and write a structured executive summary."
    run_orchestration_pipeline(task)
