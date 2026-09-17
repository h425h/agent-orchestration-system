# agents/graph.py
from typing import Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from agents.state import AgentState
from agents.supervisor import supervisor_planner
from agents.specialists import researcher_node, coder_node, writer_node
from agents.reviewer import reviewer_node
from agents.escalation import EscalationPolicy, ApprovalLevel
from agents.approval_queue import approval_queue


# ---------------------------------------------------------
# 1. Management & Gate Nodes
# ---------------------------------------------------------

def plan_gate_node(state: AgentState) -> dict:
    """
    #1. Evaluates supervisor plan against escalation policy before any subtask begins.
    """
    escalation = EscalationPolicy.check_plan_escalation(state)
    if escalation:
        if escalation.level == ApprovalLevel.NOTIFY:
            # Informational only: record notification without pausing
            return {"pending_escalation": escalation.model_dump()}

        # Package full context and register review ticket
        ticket = approval_queue.enqueue(
            thread_id="active_thread",
            level=escalation.level,
            reason=escalation.reason,
            description=escalation.description,
            context={
                "task": state.get("task"),
                "plan": state.get("plan"),
                "completed_subtasks": state.get("completed_subtasks", []),
                "current_subtask_id": state.get("current_subtask_id"),
            }
        )
        return {
            "pending_escalation": escalation.model_dump(),
            "human_approved": False,
            "reviewer_feedback": f"Review Ticket Created: {ticket.ticket_id}"
        }

    return {"pending_escalation": None}


def dispatcher_node(state: AgentState) -> dict:
    """
    #2. Evaluates completed subtasks and selects the next runnable subtask.
    """
    plan = state.get("plan", {})
    subtasks = plan.get("subtasks", [])
    completed = state.get("completed_subtasks", [])
    completed_ids = {t["id"] for t in completed}

    for task in subtasks:
        if task["id"] not in completed_ids:
            deps_met = all(dep in completed_ids for dep in task.get("dependencies", []))
            if deps_met:
                return {
                    "current_subtask_id": task["id"],
                    "reviewer_verdict": None,
                    "reviewer_feedback": None,
                    "error_count": 0,
                }

    return {"current_subtask_id": None}


def human_review_gate_node(state: AgentState) -> dict:
    """
    #3. Evaluates deliverable quality and retries to check if execution pause is needed.
    """
    escalation = EscalationPolicy.check_execution_escalation(state)
    if escalation:
        if escalation.level == ApprovalLevel.NOTIFY:
            return {"pending_escalation": escalation.model_dump()}

        # Package snapshot: task, plan, completed subtasks, active step, proposed deliverable
        ticket = approval_queue.enqueue(
            thread_id="active_thread",
            level=escalation.level,
            reason=escalation.reason,
            description=escalation.description,
            context={
                "task": state.get("task"),
                "plan": state.get("plan"),
                "completed_subtasks": state.get("completed_subtasks", []),
                "subtask_id": state.get("current_subtask_id"),
                "proposed_action": state.get("current_specialist_output"),
                "reviewer_feedback": state.get("reviewer_feedback"),
                "reviewer_score": state.get("reviewer_score"),
            }
        )
        return {
            "pending_escalation": escalation.model_dump(),
            "human_approved": False,
            "reviewer_feedback": f"Execution Escalated: {ticket.ticket_id}"
        }

    return {"pending_escalation": None}


def human_escalation_node(state: AgentState) -> dict:
    """
    #4. Terminal pause node when execution cannot proceed autonomously.
    """
    feedback = state.get("reviewer_feedback", "Task paused for human approval.")
    escalation_msg = f"[HUMAN ESCALATION HALT]: {feedback}"
    return {
        "final_output": escalation_msg,
        "human_approved": False,
    }


def update_error_counter(state: AgentState) -> dict:
    """#5. Increments retry error count."""
    return {"error_count": state.get("error_count", 0) + 1}


# ---------------------------------------------------------
# 2. Routing Logic
# ---------------------------------------------------------

def route_plan_gate(state: AgentState) -> str:
    """Decides if plan requires human review before dispatching subtasks."""
    escalation = state.get("pending_escalation")
    if escalation and escalation.get("level") in [ApprovalLevel.APPROVE_PLAN, ApprovalLevel.APPROVE_ACTION]:
        if not state.get("human_approved", False):
            return "human_escalation"
    return "dispatcher"


def route_dispatcher(state: AgentState) -> str:
    """Directs flow to the assigned specialist or terminates."""
    current_id = state.get("current_subtask_id")
    if not current_id:
        return "end"

    for task in state.get("plan", {}).get("subtasks", []):
        if task["id"] == current_id:
            return task["specialist"]

    return "end"


def route_after_review(state: AgentState) -> str:
    """
    Directs flow based on reviewer verdict:
    1. 'approved' ALWAYS moves forward to dispatcher.
    2. 'rejected' checks error_count:
       - if error_count >= 2 -> human_escalation
       - if error_count < 2 -> retry_specialist
    3. Any other verdict ('escalate', unparseable) -> human_escalation
    """
    verdict = state.get("reviewer_verdict")

    if verdict == "approved":
        return "dispatcher"

    if verdict == "rejected":
        errors = state.get("error_count", 0)
        if errors >= 2:
            return "human_escalation"
        return "retry_specialist"

    return "human_escalation"


def route_after_human_gate(state: AgentState) -> str:
    """Routes after human review gate evaluates state."""
    escalation = state.get("pending_escalation")
    if escalation and escalation.get("level") in [ApprovalLevel.APPROVE_ACTION, ApprovalLevel.TAKE_OVER]:
        if not state.get("human_approved", False):
            return "human_escalation"
    return "dispatcher"


# ---------------------------------------------------------
# 3. Assembling the State Machine Graph
# ---------------------------------------------------------

def create_agent_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    workflow = StateGraph(AgentState)

    # Register Nodes
    workflow.add_node("supervisor", supervisor_planner)
    workflow.add_node("plan_gate", plan_gate_node)
    workflow.add_node("dispatcher", dispatcher_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("human_review_gate", human_review_gate_node)
    workflow.add_node("increment_error", update_error_counter)
    workflow.add_node("human_escalation", human_escalation_node)

    # 1. Entry point -> Supervisor -> Plan Gate
    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "plan_gate")

    # 2. Plan Gate conditionally allows execution or pauses
    workflow.add_conditional_edges(
        "plan_gate",
        route_plan_gate,
        {
            "dispatcher": "dispatcher",
            "human_escalation": "human_escalation"
        }
    )

    # 3. Dispatcher branches to specialists or END
    workflow.add_conditional_edges(
        "dispatcher",
        route_dispatcher,
        {
            "researcher": "researcher",
            "coder": "coder",
            "writer": "writer",
            "end": END,
        }
    )

    # 4. Specialists -> Reviewer
    workflow.add_edge("researcher", "reviewer")
    workflow.add_edge("coder", "reviewer")
    workflow.add_edge("writer", "reviewer")

    # 5. Reviewer evaluation
    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "dispatcher": "dispatcher",
            "retry_specialist": "increment_error",
            "human_review_gate": "human_review_gate",
            "human_escalation": "human_escalation",
        }
    )

    # 6. Human Review Gate branches
    workflow.add_conditional_edges(
        "human_review_gate",
        route_after_human_gate,
        {
            "dispatcher": "dispatcher",
            "human_escalation": "human_escalation"
        }
    )

    # 7. Retry path increments error count and re-dispatches
    workflow.add_conditional_edges(
        "increment_error",
        route_dispatcher,
        {
            "researcher": "researcher",
            "coder": "coder",
            "writer": "writer",
            "end": "human_escalation",
        }
    )

    # 8. Clean finish on escalation halt
    workflow.add_edge("human_escalation", END)

    return workflow.compile(checkpointer=checkpointer)