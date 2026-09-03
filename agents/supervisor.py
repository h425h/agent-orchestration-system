# agents/supervisor.py
import json
from agents.state import AgentState, ExecutionPlan
from agents.bedrock_llm import llm

SUPERVISOR_SYSTEM_PROMPT = """You are an expert AI Supervisor and Project Manager.
Your job is to decompose complex user tasks into a structured, dependency-aware execution plan.
You assign subtasks to one of three specialists:
1. 'researcher': Information retrieval, web search, factual synthesis.
2. 'coder': Code generation, sandboxed execution, benchmarks, calculations.
3. 'writer': Synthesis, formatting, and executive summaries.

You must respond ONLY with a raw JSON object conforming to this schema:
{
  "reasoning": "High level strategy explanation",
  "estimated_complexity": "low|medium|high",
  "subtasks": [
    {
      "id": "task_1",
      "description": "Clear instruction for specialist",
      "specialist": "researcher|coder|writer",
      "dependencies": []
    }
  ]
}
Do NOT include Markdown fences (```json) or conversational text. Output pure JSON only.
"""

def clean_json_response(raw_text: str) -> str:
    """Strips accidental markdown code fence formatting from LLM output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()

def supervisor_planner(state: AgentState) -> dict:
    """Supervisor LangGraph node: Decomposes task into an execution plan."""
    user_task = state["task"]
    messages = [
        {"role": "user", "content": f"Decompose this task into subtasks:\n\n{user_task}"}
    ]

    response_text = llm.invoke(messages, system_prompt=SUPERVISOR_SYSTEM_PROMPT)
    cleaned_json = clean_json_response(response_text)

    try:
        parsed_dict = json.loads(cleaned_json)
        plan = ExecutionPlan(**parsed_dict)
    except Exception as e:
        raise ValueError(f"Supervisor produced invalid JSON plan: {e}\nRaw output: {response_text}")

    first_task_id = plan.subtasks[0].id if plan.subtasks else None

    # Return partial state updates to be merged into AgentState
    return {
        "plan": plan.model_dump(),
        "current_subtask_id": first_task_id
    }