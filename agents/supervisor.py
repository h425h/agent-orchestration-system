# agents/supervisor.py
import json
from agents.state import AgentState, ExecutionPlan
from agents.bedrock_llm import llm
from memory.semantic_store import semantic_memory


SUPERVISOR_SYSTEM_PROMPT = """You are an expert AI Supervisor and Project Manager in an autonomous multi-agent system.
Your job is to decompose complex user tasks into a structured, dependency-aware execution plan.

You assign subtasks to one of three specialists:
1. 'researcher': Information retrieval, web search, factual lookup.
2. 'coder': Code generation, sandboxed execution, simulation, data transformation.
3. 'writer': Synthesis, formatting, deliverable collation, executive summaries.

You must respond ONLY with a raw JSON object conforming to the following structure:
{
  "reasoning": "Explanation of approach and how retrieved memories informed this plan",
  "estimated_complexity": "low|medium|high",
  "confidence": 0.85,
  "subtasks": [
    {
      "id": "task_1",
      "description": "Actionable instructions for specialist",
      "specialist": "researcher|coder|writer",
      "dependencies": []
    }
  ]
}

Rules:
- "confidence" is your honest probability (0.0-1.0) that following this plan fully accomplishes the task. Use a value below 0.7 when the task is ambiguous, underspecified, or needs information you do not have.
- Every dependency must reference the id of another subtask in this plan. No cycles, no self-references.
- Express dependencies cleanly (e.g. task_2 depends on ['task_1'] if it needs task_1's output).
- Writer specialist tasks that synthesize deliverables should depend on preceding research/coding tasks.
- If relevant past memories are provided, incorporate proven approaches, avoid known pitfalls, and respect stated constraints.
Do NOT include markdown fences (```json) or conversational commentary. Return pure JSON only.
"""


def clean_json_response(raw_text: str) -> str:
    """Defensively removes markdown fences from model responses."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


MAX_PLAN_ATTEMPTS = 3


class PlanningError(RuntimeError):
    """Raised when the supervisor cannot produce a valid plan within MAX_PLAN_ATTEMPTS."""


def extract_json_object(raw_text: str) -> str:
    """Strips fences and any prose around the outermost JSON object."""
    cleaned = clean_json_response(raw_text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end < start:
        return cleaned
    return cleaned[start:end + 1]


def parse_plan(raw_text: str) -> ExecutionPlan:
    """Parses and validates a supervisor response. Raises ValueError on any problem."""
    parsed = json.loads(extract_json_object(raw_text))
    if not isinstance(parsed, dict):
        raise ValueError("Plan must be a JSON object.")
    # Without this check a missing field silently falls back to the model default,
    # which would make the low-confidence escalation unreachable.
    if "confidence" not in parsed:
        raise ValueError("Missing required field 'confidence' (a number between 0.0 and 1.0).")
    return ExecutionPlan(**parsed)


def supervisor_planner(state: AgentState) -> dict:
    """
    Supervisor LangGraph node:
    #1. Queries long-term semantic memory for relevant past experiences.
    #2. Injects retrieved context into the planning prompt.
    #3. Emits a dependency-aware ExecutionPlan and sets the first active subtask ID.
    """
    user_task = state["task"]

    # #1. Retrieve relevant memories from ChromaDB
    recalled_memories = semantic_memory.recall_similar_memories(user_task, limit=2)

    memory_context_str = "No relevant past experiences found."
    if recalled_memories:
        formatted = []
        for idx, mem in enumerate(recalled_memories, 1):
            formatted.append(
                f"[Past Experience #{idx} (Memory ID: {mem['id']})]:\n{mem['content']}"
            )
        memory_context_str = "\n\n".join(formatted)

    # #2. Construct prompt injecting past experiences into planning
    prompt = f"""USER TASK TO PLAN:
{user_task}

RELEVANT PAST MEMORIES & RETRIEVED KNOWLEDGE:
{memory_context_str}

Decompose this task into an optimal execution plan, leveraging insights from past memories where applicable."""

    plan = None
    last_error = None
    for _attempt in range(MAX_PLAN_ATTEMPTS):
        attempt_prompt = prompt
        if last_error:
            attempt_prompt += (
                f"\n\nYour previous response was rejected: {last_error}\n"
                "Return a corrected plan as valid JSON only."
            )
        response_text = llm.invoke(
            [{"role": "user", "content": attempt_prompt}],
            system_prompt=SUPERVISOR_SYSTEM_PROMPT,
            max_tokens=1500,
        )
        try:
            plan = parse_plan(response_text)
            break
        except (ValueError, TypeError) as exc:
            last_error = str(exc)[:500]

    if plan is None:
        raise PlanningError(
            f"Supervisor failed to produce a valid plan after {MAX_PLAN_ATTEMPTS} attempts: {last_error}"
        )

    first_task_id = plan.subtasks[0].id if plan.subtasks else None

    return {
        "plan": plan.model_dump(),
        "current_subtask_id": first_task_id,
    }