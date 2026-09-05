# agents/specialists.py
from agents.state import AgentState
from agents.bedrock_llm import llm, llm_sonnet
from tools.registry import registry

CODE_FENCE = chr(96) * 3


def get_current_subtask(state: AgentState) -> dict:
    current_id = state.get("current_subtask_id")
    plan = state.get("plan", {})
    for task in plan.get("subtasks", []):
        if task["id"] == current_id:
            return task
    return {}


def format_completed_context(state: AgentState) -> str:
    completed = state.get("completed_subtasks", [])
    if not completed:
        return "No prior subtasks completed yet."

    formatted = []
    for t in completed:
        formatted.append(f"Subtask [{t['id']}]: {t['description']}\nResult:\n{t['result'][:500]}...")
    return "\n\n".join(formatted)


# ---------------------------------------------------------
# 1. Researcher Specialist Node
# ---------------------------------------------------------
def researcher_node(state: AgentState) -> dict:
    subtask = get_current_subtask(state)
    prior_context = format_completed_context(state)

    query_prompt = f"""Formulate a single, concise web search query to gather the required data.
Subtask: {subtask.get('description', '')}
Context: {prior_context}

Return ONLY the plain query string."""

    search_query = llm.invoke([{"role": "user", "content": query_prompt}], max_tokens=60).strip()
    tool_res = registry.execute("web_search", specialist="researcher", query=search_query, max_results=3)
    raw_search_data = tool_res.get("output", "No results found.")

    synthesis_prompt = f"""Analyze the search results and fulfill the subtask instructions concisely.
Subtask: {subtask.get('description', '')}
Results:
{raw_search_data}

Provide a concise, fact-dense bulleted summary."""

    result = llm.invoke([{"role": "user", "content": synthesis_prompt}], max_tokens=1000)

    completed_task = {
        "id": subtask.get("id"),
        "specialist": "researcher",
        "description": subtask.get("description"),
        "result": result,
    }

    return {
        "current_specialist_output": result,
        "completed_subtasks": [completed_task],
    }


# ---------------------------------------------------------
# 2. Coder Specialist Node
# ---------------------------------------------------------
def coder_node(state: AgentState) -> dict:
    subtask = get_current_subtask(state)
    prior_context = format_completed_context(state)
    feedback = state.get("reviewer_feedback")

    feedback_prompt = ""
    if feedback and state.get("reviewer_verdict") == "rejected":
        feedback_prompt = f"""
PREVIOUS ATTEMPT REJECTED:
Feedback: {feedback}
Fix the exact issue noted above.
"""

    coder_prompt = f"""You are a Python benchmarking specialist.
Write a compact, lightweight Python script to fulfill this subtask.

Subtask: {subtask.get('description', '')}
Prerequisites: {prior_context}
{feedback_prompt}
CRITICAL BENCHMARK CONSTRAINTS:
- Use ONLY standard library: math, random, time.
- NumPy is NOT installed in this environment. Do NOT import numpy.
- Keep calculations lightweight: use vector dimension <= 32 and sample counts <= 100 so execution finishes in < 0.5s.
- Print final metrics directly to stdout.
- Script must be concise (under 45 lines). Return ONLY code inside standard python markdown fences."""

    code_resp = llm_sonnet.invoke([{"role": "user", "content": coder_prompt}], max_tokens=1500)

    python_fence = CODE_FENCE + "python"
    code = code_resp
    if python_fence in code:
        code = code.split(python_fence)[1].split(CODE_FENCE)[0].strip()
    elif CODE_FENCE in code:
        code = code.split(CODE_FENCE)[1].split(CODE_FENCE)[0].strip()

    tool_res = registry.execute("run_python_code", specialist="coder", code=code)
    execution_output = tool_res.get("output", tool_res.get("error", "No output captured."))

    summary = (
        "### Code Executed:\n"
        f"{python_fence}\n{code}\n{CODE_FENCE}\n\n"
        "### Execution Output:\n"
        f"{CODE_FENCE}\n{execution_output}\n{CODE_FENCE}"
    )

    completed_task = {
        "id": subtask.get("id"),
        "specialist": "coder",
        "description": subtask.get("description"),
        "result": summary,
    }

    return {
        "current_specialist_output": summary,
        "completed_subtasks": [completed_task],
    }


# ---------------------------------------------------------
# 3. Writer Specialist Node
# ---------------------------------------------------------
def writer_node(state: AgentState) -> dict:
    subtask = get_current_subtask(state)
    prior_context = format_completed_context(state)

    writer_prompt = f"""Deliver a crisp, executive summary for this subtask based on prior findings.
Overall Goal: {state.get('task')}
Subtask: {subtask.get('description', '')}
Context:
{prior_context}

Structure with clear bullet points, comparison tables, and key takeaways."""

    result = llm.invoke([{"role": "user", "content": writer_prompt}], max_tokens=1500)

    completed_task = {
        "id": subtask.get("id"),
        "specialist": "writer",
        "description": subtask.get("description"),
        "result": result,
    }

    return {
        "current_specialist_output": result,
        "completed_subtasks": [completed_task],
        "final_output": result,
    }