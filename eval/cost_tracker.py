# eval/cost_tracker.py
import logging
import os
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# AWS Bedrock on-demand list prices, USD per 1,000 tokens (i.e. $/MTok divided by 1000).
# Verify against https://aws.amazon.com/bedrock/pricing/ before quoting numbers; prices change.
#   Claude Haiku 4.5:  $1 / $5  per MTok
#   Claude Sonnet 4.6: $3 / $15 per MTok
# Legacy 3.5 entries are kept so old recorded runs still price correctly.
BEDROCK_RATES = {
    "anthropic.claude-haiku-4-5-20251001-v1:0": {"input_per_1k": 0.001, "output_per_1k": 0.005},
    "anthropic.claude-sonnet-4-6": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "anthropic.claude-3-5-haiku-20241022-v1:0": {"input_per_1k": 0.0008, "output_per_1k": 0.004},
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {"input_per_1k": 0.003, "output_per_1k": 0.015},
}

# Geo/regional inference profiles can carry a premium over global endpoints for newer models.
# Set BEDROCK_PRICE_MULTIPLIER (e.g. 1.1) if your invoice shows one; default is list price.
PRICE_MULTIPLIER = float(os.getenv("BEDROCK_PRICE_MULTIPLIER", "1.0"))

_REGION_PREFIX = re.compile(r"^(us|eu|apac|au|jp|global|us-gov)\.")


def normalize_model_id(model_id: str) -> str:
    """'us.anthropic.claude-sonnet-4-6' -> 'anthropic.claude-sonnet-4-6'."""
    return _REGION_PREFIX.sub("", model_id)


class RunUsageRecord(BaseModel):
    """Stores token and latency usage for a single agent node execution."""
    agent_role: str
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float = 0.0
    priced: bool = True  # False when the model has no entry in BEDROCK_RATES (cost is unknown, not zero)


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

        rates = BEDROCK_RATES.get(normalize_model_id(model_id))
        if rates is None:
            # Never silently price an unknown model at another model's rate.
            logger.warning("No Bedrock rate for model '%s'; cost recorded as unpriced.", model_id)
            total_cost, priced = 0.0, False
        else:
            input_cost = (input_tokens / 1000.0) * rates["input_per_1k"]
            output_cost = (output_tokens / 1000.0) * rates["output_per_1k"]
            total_cost, priced = round((input_cost + output_cost) * PRICE_MULTIPLIER, 6), True

        record = RunUsageRecord(
            agent_role=agent_role,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=total_cost,
            priced=priced,
        )
        self._runs[run_id].append(record)
        return record

    def record_calls(self, run_id: str, agent_role: str, calls: list) -> List[RunUsageRecord]:
        """Records measured LLMCall objects (see agents.bedrock_llm.usage_collector.drain())."""
        return [
            self.record_usage(
                run_id=run_id,
                agent_role=agent_role,
                model_id=c.model_id,
                input_tokens=c.input_tokens,
                output_tokens=c.output_tokens,
                latency_ms=c.latency_ms,
            )
            for c in calls
        ]

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
            "llm_calls": len(records),
            "unpriced_calls": sum(1 for r in records if not r.priced),
            "by_agent": by_agent,
        }


# Global cost tracker instance
cost_tracker = CostTracker()