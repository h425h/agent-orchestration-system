# test_replay.py
import uuid
from eval.replay import ReplayDebugger
from agents.state import AgentState


def test_replay_and_branching():
    print("Testing Subtask 16: Checkpoint Replay & Branching Engine...")
    test_db = "test_replay_state.db"
    debugger = ReplayDebugger(db_path=test_db)
    app = debugger.app

    thread_id = f"test_base_{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "task": "Test replay branching across multiple checkpoints.",
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

    # 1. Execute supervisor step on base thread
    print(f"  1. Running base execution on thread [{thread_id}]...")
    for event in app.stream(initial_state, config=config):
        if "supervisor" in event:
            break

    # 2. Inspect historical steps
    history = debugger.get_execution_history(thread_id)
    assert len(history) >= 1, f"Expected at least 1 populated checkpoint, found {len(history)}"
    print(f"  [Pass] Loaded {len(history)} checkpoint step(s) containing state values.")
    
    last_step_idx = len(history) - 1
    selected_step = history[last_step_idx]
    print(f"    - Forking from Step {last_step_idx} (Next nodes: {selected_step['next_node']})")
    assert "task" in selected_step["values"], "Checkpoint values must contain original task"

    # 3. Fork and branch with mutation
    branch_id = f"test_branch_{uuid.uuid4().hex[:6]}"
    print(f"  2. Forking step {last_step_idx} into branch [{branch_id}] with mutated state...")
    result = debugger.replay_and_branch(
        base_thread_id=thread_id,
        fork_step_index=last_step_idx,
        mutated_state_updates={"reviewer_feedback": "Injected mutation during replay."},
        new_thread_id=branch_id
    )

    assert result["branch_thread_id"] == branch_id
    assert result["mutations_applied"]["reviewer_feedback"] == "Injected mutation during replay."
    print("  [Pass] Branch execution initialized and mutated state successfully merged.")

    print("\nReplay Debugging System (Subtask 16) Foundation Verified Successfully!")


if __name__ == "__main__":
    test_replay_and_branching()