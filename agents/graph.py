# agents/graph.py
from typing import Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from agents.state import AgentState
from agents.supervisor import supervisor_planner
from agents.specialists import researcher_node, coder_node, writer_node
from agents.reviewer import reviewer_node


# ---------------------------------------------------------
# 1. Management Nodes (Dispatcher & Escalation)
# ---------------------------------------------------------

def dispatcher_node(state: AgentState) -> dict:
    """
    Evaluates completed subtasks against dependencies to choose the next subtask,
    or signals completion if all subtasks are finished.
    Resets the error count when selecting a fresh subtask.
    """
    plan = state.get("plan", {})
    subtasks = plan.get("subtasks", [])
    completed = state.get("completed_subtasks", [])
    completed_ids = {t["id"] for t in completed}

    # #1. Find the next subtask whose dependencies are all satisfied
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

    # #2. If all tasks are completed, clear active subtask ID to trigger termination
    return {"current_subtask_id": None}


def human_escalation_node(state: AgentState) -> dict:
    """
    Fallback circuit breaker when confidence is low or retry thresholds are reached.
    """
    reason = state.get("reviewer_feedback", "Maximum retry threshold reached or quality failed.")
    escalation_msg = f"[HUMAN ESCALATION TRIGGERED]: {reason}"
    return {
        "final_output": escalation_msg,
        "human_approved": False,
    }


def update_error_counter(state: AgentState) -> dict:
    """Helper node on rejection to increment the retry error count."""
    return {"error_count": state.get("error_count", 0) + 1}


# ---------------------------------------------------------
# 2. Routing Logic (Conditional Edges)
# ---------------------------------------------------------

def route_dispatcher(state: AgentState) -> str:
    """
    Inspects the current subtask selected by the dispatcher
    and returns the name of the assigned specialist node, or ends execution.
    """
    current_id = state.get("current_subtask_id")
    if not current_id:
        return "end"

    # #1. Match the current task ID to identify which specialist node should execute
    for task in state.get("plan", {}).get("subtasks", []):
        if task["id"] == current_id:
            return task["specialist"]

    return "end"


def route_after_review(state: AgentState) -> str:
    """
    Evaluates reviewer verdict and error counts to direct subsequent execution:
    - 'approved': moves forward to dispatcher (regardless of prior retries)[cite: 1].
    - 'rejected' & error_count < 2: retries the specialist with feedback[cite: 1].
    - 'rejected' & error_count >= 2: breaks circuit and routes to human escalation[cite: 1].
    - 'escalate': safety or formatting failure, routes directly to human escalation[cite: 1].
    """
    verdict = state.get("reviewer_verdict")

    # #1. Approved deliverables always advance to the dispatcher for the next subtask[cite: 1]
    if verdict == "approved":
        return "dispatcher"

    # #2. On rejection, evaluate the error counter before allowing another attempt[cite: 1]
    if verdict == "rejected":
        errors = state.get("error_count", 0)
        if errors >= 2:
            return "human_escalation"
        return "retry_specialist"

    # #3. Route to human escalation on unparseable verdicts or explicit escalation[cite: 1]
    return "human_escalation"


# ---------------------------------------------------------
# 3. Assembling the State Machine Graph with Checkpointing
# ---------------------------------------------------------

def create_agent_graph(checkpointer: Optional[BaseCheckpointSaver] = None):
    """Compiles the multi-agent state machine with an optional persistent checkpointer."""
    # #1. Initialize the StateGraph with the shared schema definition[cite: 1]
    workflow = StateGraph(AgentState)

    # #2. Register all management, worker, and validation nodes[cite: 1]
    workflow.add_node("supervisor", supervisor_planner)
    workflow.add_node("dispatcher", dispatcher_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("increment_error", update_error_counter)
    workflow.add_node("human_escalation", human_escalation_node)

    # #3. Set the supervisor planner as the entry point and link directly to dispatcher[cite: 1]
    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "dispatcher")

    # #4. Connect specialist nodes directly to the reviewer quality gate[cite: 1]
    workflow.add_edge("researcher", "reviewer")
    workflow.add_edge("coder", "reviewer")
    workflow.add_edge("writer", "reviewer")

    # #5. Dispatcher conditionally branches to the assigned specialist or terminates at END[cite: 1]
    workflow.add_conditional_edges(
        "dispatcher",
        route_dispatcher,
        {
            "researcher": "researcher",
            "coder": "coder",
            "writer": "writer",
            "end": END,
        },
    )

    # #6. Reviewer conditionally routes based on deliverable evaluation[cite: 1]
    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "dispatcher": "dispatcher",
            "retry_specialist": "increment_error",
            "human_escalation": "human_escalation",
        },
    )

    # #7. Retry path increments the error count and hands work back to the specialist[cite: 1]
    workflow.add_conditional_edges(
        "increment_error",
        route_dispatcher,
        {
            "researcher": "researcher",
            "coder": "coder",
            "writer": "writer",
            "end": "human_escalation",
        },
    )

    # #8. Terminate workflow cleanly upon human escalation[cite: 1]
    workflow.add_edge("human_escalation", END)

    # #9. Compile into an executable graph, injecting the persistent checkpointer if provided
    return workflow.compile(checkpointer=checkpointer)