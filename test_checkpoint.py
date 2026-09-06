# test_checkpoint.py
import uuid
from memory.checkpointer import get_checkpointer
from agents.graph import create_agent_graph
from agents.state import AgentState

def run_test():
    checkpointer = get_checkpointer("test_state.db")
    app = create_agent_graph(checkpointer=checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "task": "Test checkpointing with SQLite persistence.",
        "plan": None,
        "completed_subtasks": [],
        "current_subtask_id": None,
        "current_specialist_output": None,
        "reviewer_verdict": None,
        "reviewer_feedback": None,
        "final_output": None,
        "error_count": 0,
        "human_approved": False,
    }

    print(f"Testing state persistence for Thread ID: {thread_id}")

    # Run the initial supervisor node
    for event in app.stream(initial_state, config=config):
        for node_name in event.keys():
            print(f"Executed node: [{node_name}]")
            if node_name == "supervisor":
                break
        break

    # Inspect checkpointed state directly from SQLite
    snapshot = app.get_state(config)
    print("\n--- Checkpointed State Verified ---")
    print(f"Thread ID: {snapshot.config['configurable']['thread_id']}")
    print(f"Next Node to Run: {snapshot.next}")
    print(f"Plan Created: {snapshot.values.get('plan') is not None}")

if __name__ == "__main__":
    run_test()