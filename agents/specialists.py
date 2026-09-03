# agents/specialists.py
import json
from agents.state import AgentState
from agents.bedrock_llm import llm
from tools.registry import registry

def get_current_subtask(state: AgentState) -> dict:
    """Helper to locate the current subtask object from the plan."""
    current_id = state.get("current_subtask_id")
    plan = state.get("plan", {})
    for task in plan.get("subtasks", []):
        if task["id"] == current_id:
            return task
    return {}

def format_completed_context(state: AgentState) -> str:
    """Formats previously completed tasks as prerequisite context."""
    completed = state.get("completed_subtasks", [])
    if not completed:
        return "No prior subtasks completed yet."
    
    formatted = []
    for t in completed:
        formatted.append(f"Subtask [{t['id']}]: {t['description']}\nResult:\n{t['result']}")
    return "\n\n".join(formatted)

# ---------------------------------------------------------
# 1. Researcher Specialist Node
# ---------------------------------------------------------
def researcher_node(state: AgentState) -> dict:
    """Specialist that queries the web and synthesizes factual data."""
    subtask = get_current_subtask(state)
    prior_context = format_completed_context(state)
    
    # 1. Ask Haiku to formulate an optimal web search query based on the subtask
    query_prompt = f"""You are a research specialist. Formulate a single, precise web search query to gather the required data.
Subtask Description: {subtask.get('description', '')}
Prior Context: {prior_context}

Return ONLY the plain search query string. No quotes, no markdown."""
    
    search_query = llm.invoke([{"role": "user", "content": query_prompt}], max_tokens=100).strip()
    
    # 2. Execute search via ToolRegistry
    tool_res = registry.execute("web_search", specialist="researcher", query=search_query, max_results=4)
    raw_search_data = tool_res.get("output", "No results found.")
    
    # 3. Synthesize the findings
    synthesis_prompt = f"""You are a research specialist. Analyze the search results and fulfill the subtask instructions.
Subtask: {subtask.get('description', '')}

Search Query Used: {search_query}
Web Search Results:
{raw_search_data}

Provide a structured, fact-dense summary satisfying the task requirement."""

    result = llm.invoke([{"role": "user", "content": synthesis_prompt}], max_tokens=2048)
    
    # Return partial update to LangGraph state
    completed_task = {
        "id": subtask.get("id"),
        "specialist": "researcher",
        "description": subtask.get("description"),
        "result": result
    }
    
    return {
        "current_specialist_output": result,
        "completed_subtasks": [completed_task]
    }

# ---------------------------------------------------------
# 2. Coder Specialist Node
# ---------------------------------------------------------
def coder_node(state: AgentState) -> dict:
    """Specialist that generates and executes Python code to produce metrics."""
    subtask = get_current_subtask(state)
    prior_context = format_completed_context(state)
    
    # 1. Ask Haiku to generate runnable benchmark / calculation code
    coder_prompt = f"""You are a Python engineering and benchmarking specialist.
Write clean, executable Python code to fulfill this subtask.
Subtask: {subtask.get('description', '')}
Prerequisite Context from Prior Steps:
{prior_context}

CRITICAL REQUIREMENTS:
- The script MUST print its final metrics/summary to stdout.
- Use only standard libraries or safe mathematical simulations.
- Do NOT wrap code in explanation text. Return ONLY pure python inside standard markdown ```python ... ``` fences."""

    code_resp = llm.invoke([{"role": "user", "content": coder_prompt}], max_tokens=2048)
    
    # Extract code from fences
    code = code_resp
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
    
    # 2. Execute the code safely in the sandbox
    tool_res = registry.execute("run_python_code", specialist="coder", code=code)
    execution_output = tool_res.get("output", tool_res.get("error", "No output captured."))
    
    # 3. Formulate final analysis combining code + results
    summary = f"""### Code Executed:"""