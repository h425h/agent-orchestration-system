# tools/registry.py
import time
import io
import contextlib
from typing import Callable, Any, Dict, List
from ddgs import DDGS

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        func: Callable,
        allowed_specialists: List[str]
    ):
        """Registers a tool with access control metadata."""
        self._tools[name] = {
            "name": name,
            "description": description,
            "func": func,
            "allowed_specialists": allowed_specialists,
        }

    def get_tools_for_specialist(self, specialist: str) -> List[Dict[str, Any]]:
        """Returns tool definitions available to a specific specialist."""
        return [
            {"name": t["name"], "description": t["description"]}
            for t in self._tools.values()
            if specialist in t["allowed_specialists"]
        ]

    def execute(self, tool_name: str, specialist: str, **kwargs) -> Dict[str, Any]:
        """Executes a registered tool with telemetry and security access checks."""
        if tool_name not in self._tools:
            return {"error": f"Tool '{tool_name}' is not registered."}

        tool_meta = self._tools[tool_name]
        if specialist not in tool_meta["allowed_specialists"]:
            return {
                "error": f"Security violation: '{specialist}' cannot access tool '{tool_name}'."
            }

        start_time = time.time()
        try:
            output = tool_meta["func"](**kwargs)
            latency = round(time.time() - start_time, 3)
            return {
                "tool": tool_name,
                "output": output,
                "latency_seconds": latency,
                "success": True
            }
        except Exception as e:
            latency = round(time.time() - start_time, 3)
            return {
                "tool": tool_name,
                "error": str(e),
                "latency_seconds": latency,
                "success": False
            }

registry = ToolRegistry()

# ---------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------

def web_search(query: str, max_results: int = 4) -> str:
    """Performs live web search using ddgs."""
    results = []
    try:
        ddgs = DDGS()
        raw_results = list(ddgs.text(query, max_results=max_results))
        for r in raw_results:
            title = r.get("title", "No Title")
            body = r.get("body", "")
            href = r.get("href", "")
            results.append(f"Title: {title}\nSnippet: {body}\nURL: {href}")
        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search failed: {e}"

def run_python_code(code: str) -> str:
    """Executes Python code in a safe string-buffered stdout capture."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    safe_globals = {
        "__builtins__": __builtins__,
        "math": __import__("math"),
        "time": __import__("time"),
        "json": __import__("json"),
    }

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(code, safe_globals)
        output = stdout_buffer.getvalue()
        errors = stderr_buffer.getvalue()

        result = output
        if errors:
            result += f"\nSTDERR:\n{errors}"
        return result.strip() if result.strip() else "Code executed successfully with no stdout."
    except Exception as e:
        return f"Execution Error: {e}"

# ---------------------------------------------------------
# Register Default Tools
# ---------------------------------------------------------
registry.register_tool(
    name="web_search",
    description="Search the web for real-time information, market data, and documentation.",
    func=web_search,
    allowed_specialists=["researcher"]
)

registry.register_tool(
    name="run_python_code",
    description="Execute standalone Python benchmark/calculation code and return printed output.",
    func=run_python_code,
    allowed_specialists=["coder"]
)