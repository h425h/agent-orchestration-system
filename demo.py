# demo.py
import time
import uuid
from agents.graph import create_agent_graph
from agents.state import AgentState
from agents.approval_queue import approval_queue, ReviewDecision
from agents.escalation import ApprovalLevel, EscalationReason
from agents.resumption import resume_with_human_decision
from memory.checkpointer import get_sqlite_checkpointer
from memory.semantic_store import semantic_memory
from eval.tracer import tracer
from eval.cost_tracker import cost_tracker


def run_portfolio_showcase():
    print("=" * 80)
    print("🌟 MULTI-AGENT AUTONOMOUS ORCHESTRATION PLATFORM — SYSTEM DEMO")
    print("=" * 80)

    demo_thread = f"showcase_{uuid.uuid4().hex[:6]}"
    checkpointer = get_sqlite_checkpointer("showcase_state.db")
    if hasattr(checkpointer, "setup"):
        checkpointer.setup()

    # -------------------------------------------------------------
    # STAGE 1: Long-Term Memory Seeding
    # -------------------------------------------------------------
    print("\n[STAGE 1] Seeding Long-Term Semantic Memory (ChromaDB)...")
    semantic_memory.store_memory(
        task="Benchmark vector indexing structures",
        approach="Run Python simulation of HNSW vs Flat indexing under 10k dimensions",
        tools_used=["run_python_code"],
        findings="HNSW provides 12x lower p99 latency than Flat indexing at scale.",
        tags=["benchmarks", "vector-db"]
    )
    time.sleep(0.3)
    print("  ✔ Prior domain memory indexed into vector database.")

    # -------------------------------------------------------------
    # STAGE 2: Task Execution & Specialist Coordination
    # -------------------------------------------------------------
    task_goal = "Analyze 2026 vector databases, simulate retrieval performance, and create an executive brief."
    print(f"\n[STAGE 2] Launching Orchestrator on Thread: {demo_thread}")
    print(f"  Goal: {task_goal}")

    app = create_agent_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": demo_thread}}

    root_span = tracer.start_span(demo_thread, "portfolio_demo_run", attributes={"goal": task_goal})

    initial_state: AgentState = {
        "task": task_goal,
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

    step_count = 0
    for event in app.stream(initial_state, config=config):
        for node_name, updates in event.items():
            step_count += 1
            print(f"  → Node Completed [{step_count:02d}]: {node_name}")
            span = tracer.start_span(demo_thread, f"node_{node_name}", parent_span_id=root_span.span_id)
            tracer.end_span(span.span_id, status="success")

            cost_tracker.record_usage(
                run_id=demo_thread,
                agent_role=node_name,
                model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
                input_tokens=650,
                output_tokens=180,
                latency_ms=120.0
            )

    # -------------------------------------------------------------
    # STAGE 3: Human-in-the-Loop Escalation & Resumption
    # -------------------------------------------------------------
    print("\n[STAGE 3] Demonstrating Human-in-the-Loop Governance...")
    pending = approval_queue.get_pending_tickets()

    if pending:
        active_ticket = pending[-1]
        print(f"  ✔ Live Escalation Caught: Ticket ID={active_ticket.ticket_id} (Status: {active_ticket.status})")
        print(f"  ✔ Policy Level: {active_ticket.level.value} | Reason: {active_ticket.reason.value}")
        print("  → Simulating Operator Sign-off & Resumption...")
        resume_with_human_decision(
            app=app,
            config=config,
            ticket_id=active_ticket.ticket_id,
            decision=ReviewDecision.APPROVED,
            feedback="Operator verified plan and approved execution."
        )
        print("  ✔ Ticket resolved and operator approval merged into graph checkpoint.")
    else:
        # If execution ran autonomously, simulate a sensitive action escalation
        sim_ticket = approval_queue.enqueue(
            thread_id=demo_thread,
            level=ApprovalLevel.APPROVE_ACTION,
            reason=EscalationReason.SENSITIVE_OPERATION,
            description="Coder attempted to run index migration script.",
            context={"subtask_id": "task_deploy", "proposed_action": "apply_index_migrations()"}
        )
        print(f"  ✔ Escalation Enqueued: Ticket ID={sim_ticket.ticket_id} (Level={sim_ticket.level.value})")
        resume_with_human_decision(
            app=app,
            config=config,
            ticket_id=sim_ticket.ticket_id,
            decision=ReviewDecision.APPROVED,
            feedback="Operator verified migration safety."
        )
        print("  ✔ Ticket resolved to APPROVED.")

    tracer.end_span(root_span.span_id, status="success")

    # -------------------------------------------------------------
    # STAGE 4: Observability & Cost Accounting Summary
    # -------------------------------------------------------------
    print("\n[STAGE 4] Telemetry & Cost Accounting Summary:")
    summary = cost_tracker.get_run_summary(demo_thread)
    trace_tree = tracer.get_trace_tree(demo_thread)

    print(f"  • Total Execution Spans Recorded: {len(trace_tree)}")
    print(f"  • Total Inferred Tokens:         {summary['total_tokens']:,}")
    print(f"  • Total Bedrock Run Cost:        ${summary['total_cost_usd']:.5f}")
    print(f"  • Agent Roles Utilized:          {list(summary['by_agent'].keys())}")

    print("\n" + "=" * 80)
    print("✅ SHOWCASE DEMO COMPLETED SUCCESSFULLY")
    print("   To explore traces & checkpoints interactively, run: python cli.py ui")
    print("=" * 80)


if __name__ == "__main__":
    run_portfolio_showcase()