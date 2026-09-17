# main.py
import uuid
import time
import os
from dotenv import load_dotenv

from agents.graph import create_agent_graph
from agents.state import AgentState
from memory.checkpointer import get_sqlite_checkpointer
from memory.semantic_store import semantic_memory
from eval.tracer import tracer
from eval.cost_tracker import cost_tracker

load_dotenv()


def run_orchestration_pipeline(task_prompt: str, thread_id: str = None) -> dict:
    """
    Executes the full end-to-end multi-agent pipeline:
    - LangGraph execution with Sqlite state checkpointing
    - Span tracing via OrchestrationTracer
    - Token attribution and cost tracking
    - Long-term memory distillation upon completion
    """
    thread_id = thread_id or f"run_{uuid.uuid4().hex[:8]}"
    checkpointer = get_sqlite_checkpointer("orchestrator_state.db")
    if hasattr(checkpointer, "setup"):
        checkpointer.setup()

    app = create_agent_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 70)
    print(f"🚀 Launching Pipeline Run: {thread_id}")
    print(f"📋 Task: {task_prompt}")
    print("=" * 70)

    # Start root execution trace
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
        "final_output": None,
        "error_count": 0,
        "human_approved": False,
        "require_human_approval": False,
        "pending_escalation": None,
    }

    final_state = initial_state
    start_time = time.time()

    for event in app.stream(initial_state, config=config):
        for node_name, updates in event.items():
            print(f"  👉 [Graph Node Completed]: {node_name}")
            final_state.update(updates)

            # Record telemetry span for node
            node_span = tracer.start_span(
                trace_id=thread_id,
                name=f"node_{node_name}",
                parent_span_id=root_span.span_id,
                agent_role=node_name
            )
            tracer.end_span(node_span.span_id, status="success")

            # Track mock token usage for Bedrock accounting
            cost_tracker.record_usage(
                run_id=thread_id,
                agent_role=node_name,
                model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
                input_tokens=850,
                output_tokens=220,
                latency_ms=180.0
            )

    elapsed = round(time.time() - start_time, 2)
    tracer.end_span(root_span.span_id, status="success", attributes={"elapsed_s": elapsed})

    # Long-term semantic distillation
    print("\n🧠 Distilling run learnings into ChromaDB long-term memory...")
    insights = semantic_memory.distill_and_store(task_prompt, final_state)

    summary = cost_tracker.get_run_summary(thread_id)

    print("=" * 70)
    print(f"✅ Pipeline Run Finished in {elapsed}s")
    print(f"💰 Total Run Cost: ${summary['total_cost_usd']:.5f} ({summary['total_tokens']} tokens)")
    print(f"💾 Memory Distilled: {insights.get('summary')}")
    print("=" * 70)

    return {
        "thread_id": thread_id,
        "final_state": final_state,
        "cost_summary": summary,
        "distilled_insights": insights
    }


if __name__ == "__main__":
    task = "Research 2026 AI agent orchestration frameworks and write a structured executive summary."
    run_orchestration_pipeline(task)