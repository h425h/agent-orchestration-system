# agents/resumption.py
from typing import Dict, Any, Optional

from agents.state import AgentState
from agents.approval_queue import approval_queue, ReviewDecision
from agents.escalation import ApprovalLevel


def resume_with_human_decision(
    app: Any,
    config: Optional[Dict[str, Any]],
    ticket_id: str,
    decision: ReviewDecision,
    feedback: Optional[str] = None,
    modified_plan: Optional[Dict[str, Any]] = None,
    modified_output: Optional[str] = None
) -> Dict[str, Any]:
    """
    Applies the human decision to the ticket and updates LangGraph checkpoint state
    to resume graph execution.
    """
    ticket = approval_queue.get_ticket(ticket_id)
    if not ticket:
        raise KeyError(f"Ticket {ticket_id} not found in approval queue.")

    # 1. Update queue ticket record
    approval_queue.resolve_ticket(
        ticket_id=ticket_id,
        decision=decision,
        feedback=feedback,
        modified_output=modified_output
    )

    # 2. Build state modifications according to the granular tier
    state_update: Dict[str, Any] = {
        "human_approved": (decision in [ReviewDecision.APPROVED, ReviewDecision.MODIFIED, ReviewDecision.TAKEN_OVER]),
        "pending_escalation": None,
    }

    if decision == ReviewDecision.REJECTED:
        state_update["human_approved"] = False
        state_update["reviewer_verdict"] = "rejected"
        state_update["reviewer_feedback"] = feedback or "Rejected by human reviewer."

    elif ticket.level == ApprovalLevel.APPROVE_PLAN:
        if decision == ReviewDecision.MODIFIED and modified_plan:
            state_update["plan"] = modified_plan
        state_update["reviewer_feedback"] = feedback or "Plan approved by human."

    elif ticket.level == ApprovalLevel.APPROVE_ACTION:
        if decision == ReviewDecision.MODIFIED and modified_output:
            state_update["current_specialist_output"] = modified_output
        state_update["reviewer_feedback"] = feedback or "Action approved by human."

    elif ticket.level == ApprovalLevel.TAKE_OVER:
        subtask_id = ticket.context.get("subtask_id")
        takeover_deliverable = modified_output or feedback or "Deliverable provided by human operator."
        state_update["current_specialist_output"] = takeover_deliverable
        state_update["reviewer_verdict"] = "approved"
        state_update["error_count"] = 0
        state_update["completed_subtasks"] = [{
            "id": subtask_id,
            "specialist": "human_override",
            "description": "Manual override deliverable",
            "result": takeover_deliverable
        }]

    # 3. Update checkpointer state if config is provided
    if config:
        app.update_state(config, state_update)

    return state_update