# test_approval_queue.py
from agents.approval_queue import approval_queue, ReviewDecision
from agents.escalation import ApprovalLevel, EscalationReason
from agents.graph import create_agent_graph
from agents.state import AgentState


def test_approval_queue_and_pause():
    print("Testing Subtask 10: Approval Queue & State Interruption...")
    approval_queue.clear()

    app = create_agent_graph()

    # Initial state configured to demand plan review
    state_requiring_approval: AgentState = {
        "task": "Migrate data and purge legacy database tables",
        "plan": {
            "reasoning": "Database cleanup procedure",
            "estimated_complexity": "medium",
            "confidence": 0.50,  # Below threshold triggers pause
            "subtasks": [
                {"id": "task_1", "description": "Scan old records", "specialist": "researcher", "dependencies": []}
            ]
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

    print("\n--- 1. Executing Graph with Low Confidence / User Approval ---")
    output_state = app.invoke(state_requiring_approval)

    # Check that execution halted
    assert output_state["human_approved"] is False
    assert "[HUMAN ESCALATION HALT]" in output_state["final_output"]
    print("  [Pass] Graph paused safely at human_escalation node.")

    # Check ticket registration and context packaging
    pending = approval_queue.get_pending_tickets()
    assert len(pending) == 1
    ticket = pending[0]
    assert ticket.level == ApprovalLevel.APPROVE_PLAN
    assert ticket.reason == EscalationReason.USER_REQUESTED
    assert "task" in ticket.context
    assert "plan" in ticket.context
    print(f"  [Pass] Review Ticket Registered: ID={ticket.ticket_id}")
    print(f"  [Pass] Packaged Context Contains Task: '{ticket.context['task']}'")

    # Resolve ticket
    print("\n--- 2. Resolving Pending Ticket ---")
    approval_queue.resolve_ticket(
        ticket_id=ticket.ticket_id,
        decision=ReviewDecision.APPROVED,
        feedback="Verified safe by database administrator."
    )
    assert ticket.status == "resolved"
    assert ticket.decision == ReviewDecision.APPROVED
    assert ticket.human_feedback == "Verified safe by database administrator."
    print("  [Pass] Ticket marked resolved with human approval and feedback.")

    print("\nSubtask 10 (Approval Queue & State Interruption) Verified Successfully!")


if __name__ == "__main__":
    test_approval_queue_and_pause()