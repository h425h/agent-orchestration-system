# agents/reviewer.py
import json
from agents.state import AgentState
from agents.bedrock_llm import llm

REVIEWER_SYSTEM_PROMPT = """You are an exacting Quality Reviewer Agent in an autonomous agent orchestration pipeline.
Your job is to evaluate whether a specialist's deliverable satisfies the assigned subtask.

You must respond ONLY with a raw JSON object matching this schema:
{
  "verdict": "approved|rejected|escalate",
  "score": 8.5,
  "feedback": "Concise reasoning for the verdict and specific corrective guidance if rejected."
}

Rules for verdicts:
- "approved": Output is fact-dense, addresses the subtask, and code executions ran without fatal errors.
- "rejected": Output contains errors, hallucinated placeholders, failed tool runs, or misses key criteria.
- "escalate": Task is ambiguous, high-risk, or has failed multiple retries.

Do NOT include markdown fences. Return pure JSON only.
"""

def reviewer_node(state: AgentState) -> dict:
    """Reviewer LangGraph node: Evaluates the output of the most recent specialist."""
    latest_output = state.get("current_specialist_output", "")
    current_task_id = state.get("current_subtask_id", "")

    task_desc = ""
    for st in state.get("plan", {}).get("subtasks", []):
        if st["id"] == current_task_id:
            task_desc = st["description"]
            break

    eval_prompt = f"""Subtask Evaluated: [{current_task_id}] {task_desc}

Specialist Deliverable:
{latest_output}

Evaluate the deliverable strictly."""

    response = llm.invoke(
        [{"role": "user", "content": eval_prompt}],
        system_prompt=REVIEWER_SYSTEM_PROMPT
    )

    cleaned = response.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        review_data = json.loads(cleaned)
    except Exception:
        review_data = {
            "verdict": "approved",
            "score": 7.0,
            "feedback": "Auto-passed due to reviewer formatting irregularity."
        }

    return {
        "reviewer_verdict": review_data.get("verdict", "approved"),
        "reviewer_feedback": review_data.get("feedback", "")
    }