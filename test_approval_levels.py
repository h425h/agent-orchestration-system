# test_approval_levels.py
import uuid
from memory.checkpointer import get_sqlite_checkpointer as get_checkpointer
from agents.graph import create_agent_graph
from agents.approval_queue import approval_queue, ReviewDecision
from agents.escalation import ApprovalLevel, EscalationReason
from agents.resumption import resume_with_human_decision
from agents.state import AgentState


def test_granular_approval_levels():
    print("Testing Subtask 11: Granular Approval Levels...\n")
    approval_queue.clear()
    checkpointer = get_checkpointer("test_hitl.db")
    app = create_agent_graph(checkpointer=checkpointer)

    # ------------------------------------------------------------------
    # Tier 1: NOTIFY (Audit log only; graph does not pause)
    # ------------------------------------------------------------------
    print("--- 1. Testing NOTIFY Level ---")
    notify_ticket = approval_queue.enqueue(
        thread_id="thread_notify",
        level=ApprovalLevel.NOTIFY,
        reason=EscalationReason.USER_REQUESTED,
        description="Routine audit: search engine query executed.",
        context={"query": "top vector databases 2026"}
    )
    assert notify_ticket.level == ApprovalLevel.NOTIFY
    print(f"  [Pass] NOTIFY ticket created: {notify_ticket.ticket_id} (execution proceeds without interrupt)")

    # ------------------------------------------------------------------
    # Tier 2: APPROVE_PLAN with modification
    # ------------------------------------------------------------------
    print("\n--- 2. Testing APPROVE_PLAN (Plan Modification) ---")
    thread_plan = str(uuid.uuid4())
    config_plan = {"configurable": {"thread_id": thread_plan}}

    initial_plan_state: AgentState = {
        "task": "Deploy production database schema",
        "plan": {
            "reasoning": "Standard deployment",
            "confidence": 0.50,  # Below threshold
            "subtasks": [{"id": "t1", "description": "Run raw drop table", "specialist": "coder", "dependencies": []}]
        },
        "completed_subtasks": [],
        "current_subtask_id": None,
        "current_specialist_output": None,
        "reviewer_verdict": None,
        "reviewer_feedback": None,
        "reviewer_score": None,
        "final_output": None,
        "error_count": 0,
        "human_approved": False,
        "require_human_approval": True,
        "pending_escalation": None,
    }

    # Graph pauses
    app.invoke(initial_plan_state, config=config_plan)
    tickets = approval_queue.get_pending_tickets()
    plan_ticket = [t for t in tickets if t.level == ApprovalLevel.APPROVE_PLAN][0]

    # Human modifies the dangerous subtask
    safe_plan = {
        "reasoning": "Human sanitized plan",
        "confidence": 1.0,
        "subtasks": [{"id": "t1", "description": "Perform non-destructive dry-run migration", "specialist": "coder", "dependencies": []}]
    }

    updates = resume_with_human_decision(
        app=app,
        config=config_plan,
        ticket_id=plan_ticket.ticket_id,
        decision=ReviewDecision.MODIFIED,
        modified_plan=safe_plan,
        feedback="Sanitized dangerous drop table step."
    )
    assert updates["human_approved"] is True
    assert updates["plan"]["subtasks"][0]["description"] == "Perform non-destructive dry-run migration"
    print("  [Pass] APPROVE_PLAN successfully applied modified plan.")

    # ------------------------------------------------------------------
    # Tier 3: APPROVE_ACTION (Tool / Deliverable modification)
    # ------------------------------------------------------------------
    print("\n--- 3. Testing APPROVE_ACTION (Action Modification) ---")
    action_ticket = approval_queue.enqueue(
        thread_id="thread_action",
        level=ApprovalLevel.APPROVE_ACTION,
        reason=EscalationReason.SENSITIVE_OPERATION,
        description="Coder attempted to run sensitive shell code.",
        context={"subtask_id": "t2", "proposed_action": "os.system('rm -rf /tmp/test')"}
    )
    sanitized_action = "print('Safe dry-run cleanup')"
    updates_action = resume_with_human_decision(
        app=app,
        config=None,
        ticket_id=action_ticket.ticket_id,
        decision=ReviewDecision.MODIFIED,
        modified_output=sanitized_action,
        feedback="Replaced destructive shell call with safe dry-run."
    )
    assert updates_action["human_approved"] is True
    assert updates_action["current_specialist_output"] == sanitized_action
    print("  [Pass] APPROVE_ACTION successfully replaced proposed specialist output.")

    # ------------------------------------------------------------------
    # Tier 4: TAKE_OVER (Human supplies final result directly)
    # ------------------------------------------------------------------
    print("\n--- 4. Testing TAKE_OVER (Manual Substitution) ---")
    takeover_ticket = approval_queue.enqueue(
        thread_id="thread_takeover",
        level=ApprovalLevel.TAKE_OVER,
        reason=EscalationReason.REPEATED_SUBTASK_FAILURE,
        description="Coder repeatedly failed benchmark syntax.",
        context={"subtask_id": "t3"}
    )
    manual_deliverable = "Human Engineer: Benchmarked cosine similarity @ 2.1us (manual result)."
    updates_to = resume_with_human_decision(
        app=app,
        config=None,
        ticket_id=takeover_ticket.ticket_id,
        decision=ReviewDecision.TAKEN_OVER,
        modified_output=manual_deliverable
    )
    assert updates_to["reviewer_verdict"] == "approved"
    assert updates_to["completed_subtasks"][0]["result"] == manual_deliverable
    assert updates_to["completed_subtasks"][0]["specialist"] == "human_override"
    print("  [Pass] TAKE_OVER injected human deliverable into completed subtasks.")

    print("\nAll Granular Approval Levels (Subtask 11) Verified Successfully!")


if __name__ == "__main__":
    test_granular_approval_levels()