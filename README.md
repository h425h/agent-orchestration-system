# Multi-Agent Orchestration Platform

An enterprise-grade multi-agent orchestration framework built on **LangGraph** and **AWS Bedrock (Claude Haiku 4.5)**, featuring autonomous task decomposition, role-based tool execution, dynamic state accumulation, and automated quality gating.

---

## Key Architecture & Features

* **Hierarchical Agent Flow**: A Supervisor Agent decomposes complex, ambiguous requests into an ordered, dependency-aware directed acyclic graph (DAG) of subtasks assigned to specialized domain agents.
* **Specialist Layer**:
* **Researcher**: Formulates targeted search queries, interfaces with live web search APIs (`ddgs`), and extracts structured domain insights.
* **Coder**: Generates clean Python scripts, executes them in an isolated stdout/stderr capture runtime, and collects computational benchmarks.
* **Writer**: Consolidates prior findings and data logs into executive-level synthesis deliverables.


* **Hardened Tool Registry & RBAC**: Centralized tool execution catalog enforcing deterministic, code-level role-based access control (RBAC) to block cross-agent unauthorized tool execution, complete with runtime latency telemetry.
* **Reviewer Quality Gate**: A dedicated validation node reviewing all specialist outputs against task criteria, capable of issuing `approved`, `rejected` (with corrective feedback for iterative loops), or `escalate` verdicts.
* **Resilient AWS Bedrock Integration**: Custom client implementation targeting Bedrock's cross-region inference profiles (`us.anthropic.claude-haiku-4-5-20251001-v1:0`) using botocore request lifecycle hooks for Bearer token authorization.
* **Immutable State Reducers**: LangGraph state updates utilize append-only list reducers (`operator.add`) to guarantee intermediate research and computation results accumulate without accidental overwrites.

---

## Project Structure

```text
agent-orchestration-system/
├── agents/
│   ├── __init__.py
│   ├── bedrock_llm.py      # AWS Bedrock Converse API singleton with auth hooks
│   ├── state.py            # TypedDict AgentState & Pydantic execution schemas
│   ├── supervisor.py       # Task decomposition & planning engine
│   ├── specialists.py      # Researcher, Coder, and Writer worker nodes
│   └── reviewer.py         # Output validation and routing gate
├── tools/
│   ├── __init__.py
│   └── registry.py         # Tool catalog with access controls & execution telemetry
├── memory/                 # Persistent memory modules (Short-term & Long-term)
├── eval/                   # Langfuse tracing and prompt evaluation suites
├── tests/                  # Unit and integration test suites
├── test_bedrock.py         # Bedrock inference profile verification script
├── test_tools.py           # Tool registry & RBAC boundary verification
├── test_specialists.py     # Specialist-to-reviewer flow validation
├── main.py                 # Core supervisor planner execution entry point
├── .env.example            # Environment variable template
└── .gitignore              # Strict secret isolation rules

```

---

## Getting Started

### Prerequisites

* Python 3.11+ (Python 3.12 recommended)
* `uv` package manager
* AWS Account with Bedrock model access enabled for Claude models in `us-west-2`

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/agent-orchestration-system.git
cd agent-orchestration-system

```


2. **Set up the virtual environment:**
```bash
uv venv venv --python 3.12
source venv/bin/activate

```


3. **Install dependencies:**
```bash
uv pip install boto3 python-dotenv langgraph langchain-core pydantic ddgs

```


4. **Configure Environment Variables:**
```bash
cp .env.example .env

```


Open `.env` and configure your AWS credentials:
```env
AWS_REGION=us-west-2
AWS_BEARER_TOKEN_BEDROCK="your_actual_bedrock_api_token"

```



---

## Verification & Execution

* **Verify Bedrock Connectivity:**
```bash
python test_bedrock.py

```


* **Verify Tool Registry & Access Controls:**
```bash
python test_tools.py

```


* **Verify Specialist Synthesis & Review Gate:**
```bash
python test_specialists.py

```


* **Run Supervisor Task Decomposition:**
```bash
python main.py

```



---

## Roadmap

* [x] Phase 1: Supervisor planning, Tool Registry, Specialist Nodes, Reviewer Gate
* [ ] Phase 2: LangGraph State Machine compilation with cyclic retry routing
* [ ] Phase 3: Short-term (Redis) and Long-term Semantic Memory (SQLite / ChromaDB)
* [ ] Phase 4: Human-in-the-Loop (HITL) approval queues and escalation triggers
* [ ] Phase 5: Observability, OpenTelemetry traces, and Langfuse integration
