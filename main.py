# main.py
import uuid
from agents.graph import create_agent_graph
from agents.state import AgentState
from memory.checkpointer import get_checkpointer
from memory.semantic_store import semantic_memory


def main():
    # #1. Initialize persistent working memory checkpointer
    checkpointer = get_checkpointer("orchestrator_state.db")
    app = create_agent_graph(checkpointer=checkpointer)

    # #2. Unique session identifier for state persistence and resumption
    session_thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_thread_id}}

    user_task = (
        "Research top vector databases in 2026, benchmark vector math calculation speed "
        "using Python simulation code, and write an executive summary."
    )

    initial_state: AgentState = {
        "task": user_task,
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

    print(f"Initializing Multi-Agent Pipeline | Thread ID: {session_thread_id}")
    print("=" * 60)

    final_report = None
    final_state_accumulator = {}

    # #3. Stream node updates through checkpointed state graph
    for event in app.stream(initial_state, config=config):
        for node_name, updates in event.items():
            print(f"\n---> Finished Node: [{node_name}]")
            final_state_accumulator.update(updates)

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

            if "final_output" in updates and updates["final_output"]:
                final_report = updates["final_output"]

    print("\n" + "=" * 60)
    print("Orchestration Pipeline Finished!")

    # #4. Handle deliverable and trigger long-term semantic distillation
    if final_report:
        print("\n=== FINAL DELIVERABLE ===\n")
        print(final_report)

        with open("final_report.md", "w") as f:
            f.write(final_report)
        print("\nSaved deliverable to final_report.md")

        print("\n--- Distilling Knowledge into Long-Term Semantic Memory ---")
        snapshot = app.get_state(config)
        insights = semantic_memory.distill_and_store(user_task, snapshot.values, user_id="default_user")
        print("Distilled Summary:", insights.get("summary"))
        print("Approach Saved:", insights.get("successful_approach"))
    else:
        print("No final deliverable produced (run may have terminated in human escalation).")


if __name__ == "__main__":
    main()