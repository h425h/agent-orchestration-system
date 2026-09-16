# eval/cost_tracker.py
from typing import Dict, Any, List
from pydantic import BaseModel, Field
import time

# AWS Bedrock Pricing per 1,000 tokens (USD)
# Claude 3.5 Haiku: $0.0008 input / $0.004 output per 1k
# Claude 3.5 Sonnet: $0.003 input / $0.015 output per 1k
BEDROCK_RATES = {
    "anthropic.claude-3-5-haiku-20241022-v1:0": {
        "input_per_1k": 0.0008,
        "output_per_1k": 0.004,
    },
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "input_per_1k": 0.003,
        "output_per_1k": 0.015,
    },
}

# Default fallback model mapping
DEFAULT_RATE = BEDROCK_RATES["anthropic.claude-3-5-haiku-20241022-v1:0"]


class RunUsageRecord(BaseModel):
    """Stores token and latency usage for a single agent node execution."""
    agent_role: str
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float = 0.0


class CostTracker:
    """Tracks token consumption, Bedrock USD spend, and latency per run and agent."""

    def __init__(self):
        self._runs: Dict[str, List[RunUsageRecord]] = {}
        self._escalation_counts: Dict[str, int] = {}

    def record_usage(
        self,
        run_id: str,
        agent_role: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float
    ) -> RunUsageRecord:
        """Computes USD cost and stores usage record for the run."""
        if run_id not in self._runs:
            self._runs[run_id] = []

        rates = BEDROCK_RATES.get(model_id, DEFAULT_RATE)
        input_cost = (input_tokens / 1000.0) * rates["input_per_1k"]
        output_cost = (output_tokens / 1000.0) * rates["output_per_1k"]
        total_cost = round(input_cost + output_cost, 6)

        record = RunUsageRecord(
            agent_role=agent_role,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=total_cost,
        )
        self._runs[run_id].append(record)
        return record

    def record_escalation(self, run_id: str):
        self._escalation_counts[run_id] = self._escalation_counts.get(run_id, 0) + 1

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Aggregates cost and token attribution by agent role."""
        records = self._runs.get(run_id, [])
        total_cost = sum(r.cost_usd for r in records)
        total_input_tokens = sum(r.input_tokens for r in records)
        total_output_tokens = sum(r.output_tokens for r in records)
        total_latency_ms = sum(r.latency_ms for r in records)

        by_agent = {}
        for r in records:
            if r.agent_role not in by_agent:
                by_agent[r.agent_role] = {
                    "cost_usd": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "invocations": 0,
                }
            by_agent[r.agent_role]["cost_usd"] = round(by_agent[r.agent_role]["cost_usd"] + r.cost_usd, 6)
            by_agent[r.agent_role]["input_tokens"] += r.input_tokens
            by_agent[r.agent_role]["output_tokens"] += r.output_tokens
            by_agent[r.agent_role]["invocations"] += 1

        return {
            "run_id": run_id,
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_latency_ms": round(total_latency_ms, 2),
            "escalations": self._escalation_counts.get(run_id, 0),
            "by_agent": by_agent,
        }


# Global cost tracker instance
cost_tracker = CostTracker()