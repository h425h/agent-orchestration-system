# main.py
from agents.graph import create_agent_graph
from agents.state import AgentState

def main():
    print("Compiling LangGraph Multi-Agent State Machine...")
    app = create_agent_graph()

    initial_state: AgentState = {
        "task": (
            "Research top vector databases in 2026, benchmark vector math calculation speed "
            "using Python simulation code, and write an executive summary."
        ),
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

    print("\nStarting Autonomous Orchestration...\n" + "=" * 60)

    final_report = None

    for event in app.stream(initial_state):
        for node_name, updates in event.items():
            print(f"\n---> Finished Node: [{node_name}]")

            if "plan" in updates and updates["plan"]:
                tasks = updates["plan"].get("subtasks", [])
                print(f"     Supervisor created plan with {len(tasks)} subtasks.")

            if "current_subtask_id" in updates and updates["current_subtask_id"]:
                print(f"     Active Task: {updates['current_subtask_id']}")

            if "reviewer_verdict" in updates and updates["reviewer_verdict"]:
                print(f"     Reviewer Verdict: {updates['reviewer_verdict']}")
                feedback = updates.get("reviewer_feedback", "")
                if feedback:
                    print(f"     Feedback: {feedback[:100]}...")

            if "error_count" in updates:
                print(f"     Retry Error Count: {updates['error_count']}")

            # Capture the writer deliverable as soon as it emerges
            if "final_output" in updates and updates["final_output"]:
                final_report = updates["final_output"]

    print("\n" + "=" * 60 + "\nOrchestration Complete!\n")

    if final_report:
        print("=== FINAL DELIVERABLE ===\n")
        print(final_report)
        with open("final_report.md", "w") as f:
            f.write(final_report)
        print("\nSaved deliverable to final_report.md")
    else:
        print("No final deliverable produced (run may have terminated in human escalation).")

if __name__ == "__main__":
    main()