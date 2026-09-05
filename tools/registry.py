# tools/registry.py
import time
import io
import signal
import contextlib
from typing import Callable, Dict, Any
from ddgs import DDGS


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out.")


def run_python_code(code: str, timeout_seconds: int = 5) -> str:
    """
    Executes Python code in a restricted namespace with a hard signal-based timeout.
    Eliminates IPC multiprocessing serialization entirely.
    """
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    safe_builtins = {
        "print": print,
        "range": range,
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "sorted": sorted,
        "enumerate": enumerate,
        "zip": zip,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "isinstance": isinstance,
        "type": type,
        "__import__": __import__,
        "__name__": "__main__",
    }

    safe_globals = {
        "__builtins__": safe_builtins,
        "math": __import__("math"),
        "time": __import__("time"),
        "json": __import__("json"),
        "random": __import__("random"),
        "tracemalloc": __import__("tracemalloc"),
    }

    # Register Unix alarm signal for timeout
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(code, safe_globals)
        output = stdout_buffer.getvalue()
        errors = stderr_buffer.getvalue()
        result = output
        if errors:
            result += f"\nSTDERR:\n{errors}"
        return result.strip() if result.strip() else "Code executed successfully with no stdout."
    except TimeoutError:
        return f"Execution Error: Code exceeded {timeout_seconds}s limit. Keep benchmarks lightweight."
    except Exception as e:
        return f"Execution Error: {e}"
    finally:
        # Always disable alarm
        signal.alarm(0)


def web_search(query: str, max_results: int = 3) -> str:
    """Executes live web search via ddgs with low result count to minimize latency."""
    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No web results found."

        formatted = []
        for r in results:
            formatted.append(f"Title: {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search Error: {e}"


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, func: Callable, allowed_roles: list[str]):
        self._tools[name] = {"func": func, "allowed_roles": allowed_roles}

    def execute(self, tool_name: str, specialist: str, **kwargs) -> Dict[str, Any]:
        start = time.time()
        if tool_name not in self._tools:
            return {"error": f"Tool '{tool_name}' not found.", "duration": 0.0}

        tool_meta = self._tools[tool_name]
        if specialist not in tool_meta["allowed_roles"]:
            return {
                "error": f"Security violation: '{specialist}' cannot access tool '{tool_name}'.",
                "duration": round(time.time() - start, 4),
            }

        try:
            output = tool_meta["func"](**kwargs)
            return {"output": output, "duration": round(time.time() - start, 4)}
        except Exception as e:
            return {"error": str(e), "duration": round(time.time() - start, 4)}


registry = ToolRegistry()
registry.register("run_python_code", run_python_code, allowed_roles=["coder"])
registry.register("web_search", web_search, allowed_roles=["researcher"])