# test_specialists.py
from agents.state import AgentState
from agents.specialists import researcher_node
from agents.reviewer import reviewer_node

test_state: AgentState = {
    "task": "Research top vector databases in 2026.",
    "plan": {
        "reasoning": "Research current landscape",
        "estimated_complexity": "low",
        "subtasks": [
            {
                "id": "task_1",
                "description": "Identify top vector databases in 2026 and gather key features.",
                "specialist": "researcher",
                "dependencies": []
            }
        ]
    },
    "completed_subtasks": [],
    "current_subtask_id": "task_1",
    "current_specialist_output": None,
    "reviewer_verdict": None,
    "reviewer_feedback": None,
    "final_output": None,
    "error_count": 0,
    "human_approved": False,
}

print("1. Running Researcher Specialist (Tool Call + LLM Synthesis)...")
researcher_updates = researcher_node(test_state)
test_state.update(researcher_updates)

print("\n--- Output Preview ---")
print(test_state["current_specialist_output"][:350], "...\n")

print("2. Running Reviewer Quality Gate...")
review_updates = reviewer_node(test_state)
test_state.update(review_updates)

print("Reviewer Verdict:", test_state["reviewer_verdict"])
print("Reviewer Feedback:", test_state["reviewer_feedback"])