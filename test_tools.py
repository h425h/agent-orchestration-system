# test_tools.py
from tools.registry import registry

print("1. Testing Web Search (Researcher)...")
res = registry.execute("web_search", specialist="researcher", query="Top vector databases 2026")
print("Search Result (truncated):", str(res["output"])[:300], "...\n")

print("2. Testing Python Execution (Coder)...")
code_snippet = """
import time
start = time.time()
total = sum([x**2 for x in range(100000)])
latency = time.time() - start
print(f'Computed sum in {latency:.4f}s: {total}')
"""
res_code = registry.execute("run_python_code", specialist="coder", code=code_snippet)
print("Code Output:", res_code["output"], "\n")

print("3. Testing Security Boundary (Writer trying to run code)...")
res_blocked = registry.execute("run_python_code", specialist="writer", code="print(1)")
print("Security Check:", res_blocked)