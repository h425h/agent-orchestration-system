# agents/reviewer.py
import json
from agents.state import AgentState
from agents.bedrock_llm import llm

REVIEWER_SYSTEM_PROMPT = """You are an exacting Quality Reviewer Agent.
Evaluate whether a specialist's deliverable satisfies the assigned subtask.

ENVIRONMENT CONSTRAINTS:
- The execution sandbox intentionally does NOT have NumPy or external dependencies installed.
- Code using standard library (math, random, time) simulations is fully expected and MUST NOT be penalized or rejected for omitting NumPy.

You must respond ONLY with a raw JSON object matching this schema:
{
  "verdict": "approved|rejected|escalate",
  "score": 8.5,
  "feedback": "Concise reasoning for the verdict (under 50 words)."
}

Rules for verdicts:
- "approved": Output addresses the subtask, execution ran without fatal errors, and metrics were produced.
- "rejected": Output has fatal runtime errors or missing core requirements.
- "escalate": Task is ambiguous, high-risk, or has failed multiple retries.

Do NOT include markdown fences. Return pure JSON only.
"""


def reviewer_node(state: AgentState) -> dict:
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

Evaluate the deliverable strictly and concisely."""

    response = llm.invoke(
        [{"role": "user", "content": eval_prompt}],
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        max_tokens=250,
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
            "verdict": "escalate",
            "score": 0.0,
            "feedback": "Reviewer output was unparseable; escalating for human check.",
        }

    return {
        "reviewer_verdict": review_data.get("verdict", "escalate"),
        "reviewer_feedback": review_data.get("feedback", ""),
    }