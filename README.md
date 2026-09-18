# Autonomous Multi-Agent Orchestration Platform

An enterprise-ready, observable multi-agent orchestration engine built with **LangGraph**, **AWS Bedrock (Claude 3.5 Sonnet & Haiku)**, **ChromaDB**, and **OpenTelemetry**. 

The system features dynamic task decomposition, specialized tool execution, fail-closed quality review, human-in-the-loop (HITL) escalation gates, persistent SQLite state checkpointing, semantic memory distillation, and an interactive trace explorer with checkpoint replay debugging.

---

## System Architecture

┌─────────────────────────┐
                            │    User Prompt / Goal   │
                            └────────────┬────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │   Supervisor Planner    │◄─── Long-Term Memory
                            │  (Claude 3.5 Haiku)     │     (ChromaDB Retrieval)
                            └────────────┬────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │    Plan Review Gate     │───► Escalation Queue
                            └────────────┬────────────┘     (Low Confidence)
                                         │ (Approved)
                                         ▼
                            ┌─────────────────────────┐
                 ┌─────────►│  Dependency Dispatcher  │◄────────┐
                 │          └────────────┬────────────┘         │
                 │                       │                      │
   ┌─────────────┴──────────┐            │        ┌─────────────┴──────────┐
   │     Researcher Node    │            │        │       Writer Node      │
   │  • DuckDuckGo Search   │            │        │  • Synthesis & Report  │
   └─────────────┬──────────┘            │        └─────────────┬──────────┘
                 │                       ▼                      │
                 │          ┌─────────────────────────┐         │
                 │          │        Coder Node       │         │
                 │          │   (Claude 3.5 Sonnet)   │         │
                 │          │  • Sandboxed Execution  │         │
                 │          └────────────┬────────────┘         │
                 │                       │                      │
                 └───────────────────────┼──────────────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │      Reviewer Gate      │
                            │   (Fail-Closed Check)   │
                            └──────┬───────────┬──────┘
                                   │           │
                      (Approved)   │           │ (Rejected & Retries < 2)
                                   │           └──────► Increment Error & Retry
                                   ▼
                     All Subtasks Complete?
                             /        \
                         (No)          (Yes)
                          /              \
                 Back to Dispatcher       ▼
                                   ┌─────────────────────────┐
                                   │  Distill Semantic Store │
                                   │  (Embed into ChromaDB)  │
                                   └─────────────────────────┘

---

## Core Capabilities Across Phases

### 1. Multi-Agent Coordination (Phase 1)
* **Supervisor Planner**: Decomposes natural language objectives into structured subtasks with explicit dependencies and complexity estimates.
* **Specialist Execution**:
  * **Researcher**: Queries DuckDuckGo for live facts, metrics, and documentation.
  * **Coder (Claude 3.5 Sonnet)**: Formulates and executes benchmarks/simulations inside an in-process, timeout-protected sandbox.
  * **Writer**: Synthesizes multi-step research deliverables into structured executive summaries.
* **Fail-Closed Quality Reviewer**: Grades specialist outputs; rejects deficient deliverables with corrective feedback.
* **Circuit Breaker**: Halts execution and escalates to human intervention if a single subtask fails review more than twice.

### 2. Dual-Layer Memory Systems (Phase 2)
* **Short-Term Checkpointing**: Uses `SqliteSaver` to record state snapshots after each node, enabling deterministic execution recovery and pause-resume flows.
* **Long-Term Semantic Distillation**: Extracts and embeds completed execution methodologies into ChromaDB.
* **Retrieval-Augmented Planning**: Automatically injects relevant past strategies into the supervisor prompt before task decomposition.
* **Lifecycle Management**: Implements access-count importance weighting, decay filters, consolidation, and user data purge endpoints.

### 3. Human-in-the-Loop Governance (Phase 3)
* **Policy-Based Triggers**: Escalates automatically on low plan confidence, destructive operations, review quality failures, or repeated errors.
* **Approval Tiers**: Supports `NOTIFY`, `APPROVE_PLAN`, `APPROVE_ACTION`, and `TAKE_OVER`.
* **State Resumption**: Merges human overrides or approvals back into checkpointed graphs.

### 4. Observability & Debugging (Phase 4)
* **Trace Hierarchy**: Emits OpenTelemetry spans capturing agent decisions, tool execution times, and latencies.
* **Token & USD Cost Attribution**: Computes exact Bedrock inference costs across Claude 3.5 Haiku ($0.0008 / $0.004 per 1k) and Sonnet ($0.003 / $0.015 per 1k).
* **Replay Debugger**: Inspects step-by-step state histories and executes divergent branches with mutated inputs.

### 5. Automated Evaluation & CLI (Phase 5)
* **LLM-as-a-Judge**: Evaluates runs against plan quality, tool accuracy, and groundedness.
* **Fault Tolerance**: Verified against sandbox timeouts, unhandled tool exceptions, and consecutive review failures.
* **Unified CLI**: CLI interface for running jobs, managing checkpointer databases, and serving the Streamlit UI.

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.11+
* AWS Account with Amazon Bedrock model access (Claude 3.5 Sonnet & Claude 3.5 Haiku)

### 1. Clone & Configure
```bash
git clone [https://github.com/h425h/agent-orchestration-system.git](https://github.com/h425h/agent-orchestration-system.git)
cd agent-orchestration-system

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies via uv or pip
pip install -r requirements.txt