# agents/graph.py
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.supervisor import supervisor_planner
from agents.specialists import researcher_node, coder_node, writer_node
from agents.reviewer import reviewer_node

# ---------------------------------------------------------
# 1. Management Nodes (Dispatcher & Escalation)
# ---------------------------------------------------------

def dispatcher_node(state: AgentState) -> dict:
    """
    Dependency-aware dispatcher:
    Inspects the plan, checks what subtasks are already completed,
    and selects the next subtask whose dependencies are satisfied.
    """
    plan = state.get("plan", {})
    subtasks = plan.get("subtasks", [])
    completed = state.get("completed_subtasks", [])
    completed_ids = {t["id"] for t in completed}

    # Find the first pending subtask where all prerequisite dependencies are finished
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

    # If no pending subtasks remain, clear the active ID to signal completion
    return {"current_subtask_id": None}


def human_escalation_node(state: AgentState) -> dict:
    """
    Terminal fallback node:
    Captures failure context when quality checks repeatedly fail.
    """
    feedback = state.get("reviewer_feedback", "Maximum retry threshold reached or quality gate failed.")
    escalation_msg = f"[HUMAN ESCALATION TRIGGERED]: {feedback}"
    return {
        "final_output": escalation_msg,
        "human_approved": False
    }


def update_error_counter(state: AgentState) -> dict:
    """Helper node to increment error count upon task rejection."""
    return {"error_count": state.get("error_count", 0) + 1}


# ---------------------------------------------------------
# 2. Routing Functions (Conditional Edges)
# ---------------------------------------------------------

def route_dispatcher(state: AgentState) -> str:
    """
    Inspects the active subtask chosen by the dispatcher
    and returns the node name of the specialist assigned to it.
    """
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
    1. 'approved' ALWAYS moves forward to dispatcher (regardless of error_count).
    2. 'rejected' checks error_count:
       - if error_count >= 2 -> human_escalation
       - if error_count < 2 -> retry_specialist
    3. Any other verdict ('escalate', unparseable) -> human_escalation
    """
    verdict = state.get("reviewer_verdict")

    # If approved, advance to the next task regardless of how many retries it took
    if verdict == "approved":
        return "dispatcher"

    # Only check retry limits when the deliverable was rejected
    if verdict == "rejected":
        errors = state.get("error_count", 0)
        if errors >= 2:
            return "human_escalation"
        return "retry_specialist"

    # Verdict is 'escalate' or unrecognized
    return "human_escalation"


# ---------------------------------------------------------
# 3. Assembling the State Machine Graph
# ---------------------------------------------------------

def create_agent_graph():
    """Constructs and compiles the complete multi-agent LangGraph workflow."""
    # StateGraph takes our AgentState schema so it knows how to validate and merge state updates
    workflow = StateGraph(AgentState)

    # Register all working nodes
    workflow.add_node("supervisor", supervisor_planner)
    workflow.add_node("dispatcher", dispatcher_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("increment_error", update_error_counter)
    workflow.add_node("human_escalation", human_escalation_node)

    # 1. Entry: Every run starts at the supervisor to decompose the request
    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "dispatcher")

    # 2. Dispatcher dynamically directs flow to the required specialist or END
    workflow.add_conditional_edges(
        "dispatcher",
        route_dispatcher,
        {
            "researcher": "researcher",
            "coder": "coder",
            "writer": "writer",
            "end": END
        }
    )

    # 3. Every specialist forwards their output directly into the Reviewer Quality Gate
    workflow.add_edge("researcher", "reviewer")
    workflow.add_edge("coder", "reviewer")
    workflow.add_edge("writer", "reviewer")

    # 4. Reviewer evaluates the output and routes accordingly
    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "dispatcher": "dispatcher",
            "retry_specialist": "increment_error",
            "human_escalation": "human_escalation"
        }
    )

    # 5. When retrying, increment error count and re-route to the active specialist
    workflow.add_conditional_edges(
        "increment_error",
        route_dispatcher,
        {
            "researcher": "researcher",
            "coder": "coder",
            "writer": "writer",
            "end": "human_escalation"
        }
    )

    # 6. Human escalation terminates the execution cleanly
    workflow.add_edge("human_escalation", END)

    # Compile the directed graph into an executable pipeline
    return workflow.compile()