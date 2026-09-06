# test_memory_planning.py
import json
from agents.state import AgentState
from agents.supervisor import supervisor_planner

def test_planner_with_memory():
    print("Testing Retrieval-Augmented Planning (Supervisor + ChromaDB)...")

    # #1. Create a query that semantically overlaps with the memory stored in Subtask 2
    test_state: AgentState = {
        "task": "Compare vector search engines for low-latency indexing in 2026 and benchmark them.",
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

    # #2. Run the supervisor planner node
    updates = supervisor_planner(test_state)

    print("\n--- Generated Plan with Memory Augmentation ---")
    print(f"Reasoning: {updates['plan'].get('reasoning')}\n")
    print("Subtasks Planned:")
    for subtask in updates["plan"].get("subtasks", []):
        print(f"  - [{subtask['id']}] ({subtask['specialist']}): {subtask['description']}")
        if subtask.get("dependencies"):
            print(f"    Dependencies: {subtask['dependencies']}")

if __name__ == "__main__":
    test_planner_with_memory()