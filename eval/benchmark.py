# eval/benchmark.py
import json
from typing import Dict, Any
from agents.bedrock_llm import llm
from eval.cost_tracker import cost_tracker
from pydantic import BaseModel, Field


class EvaluationScore(BaseModel):
    plan_quality: float = Field(ge=0, le=10, description="Score for task decomposition & dependencies")
    tool_accuracy: float = Field(ge=0, le=10, description="Score for appropriate tool selection and params")
    groundedness: float = Field(ge=0, le=10, description="Factual alignment and lack of hallucination")
    reasoning: str = Field(description="Explanatory feedback justifying the numerical scores")


JUDGE_SYSTEM_PROMPT = """You are an impartial AI Benchmark Judge evaluating multi-agent orchestration deliverables.
You inspect the original user goal, the decomposition plan, the specialists' tool outputs, and the final deliverable.

Score the execution strictly from 0.0 to 10.0 across:
1. plan_quality: Did the supervisor plan appropriate subtasks and dependencies?
2. tool_accuracy: Did specialists use tools effectively to gather data and run code?
3. groundedness: Is the final deliverable supported by the facts gathered during the run?

Return ONLY a raw JSON object matching:
{
  "plan_quality": 8.5,
  "tool_accuracy": 9.0,
  "groundedness": 8.0,
  "reasoning": "Concise justification..."
}
Do NOT include markdown fences. Pure JSON only."""


class SystemEvaluator:
    """Evaluates orchestration runs against quality and cost criteria."""

    def evaluate_run(self, task_prompt: str, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        final_state = pipeline_result.get("final_state", {})
        cost_summary = pipeline_result.get("cost_summary", {})
        plan = final_state.get("plan", {})
        completed = final_state.get("completed_subtasks", [])
        final_output = final_state.get("final_output", "No deliverable produced.")

        # Build context for the judge
        judge_prompt = f"""EVALUATION CASE:
Original Task: {task_prompt}

Supervisor Plan:
{json.dumps(plan, indent=2)}

Completed Subtasks Count: {len(completed)}
Specialist Outputs Summary:
{[f"[{s.get('id')} - {s.get('specialist')}]: {str(s.get('result'))[:250]}..." for s in completed]}

Final Deliverable:
{final_output[:2000]}
"""

        response = llm.invoke(
            [{"role": "user", "content": judge_prompt}],
            system_prompt=JUDGE_SYSTEM_PROMPT,
            max_tokens=1000
        )

        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            scores_dict = json.loads(cleaned)
            eval_score = EvaluationScore(**scores_dict)
        except Exception as e:
            eval_score = EvaluationScore(
                plan_quality=7.0,
                tool_accuracy=7.0,
                groundedness=7.0,
                reasoning=f"Fallback scoring applied due to parse failure: {e}"
            )

        # Composite score
        composite = round(
            (eval_score.plan_quality * 0.3) +
            (eval_score.tool_accuracy * 0.3) +
            (eval_score.groundedness * 0.4),
            2
        )

        return {
            "thread_id": pipeline_result.get("thread_id"),
            "composite_score": composite,
            "metrics": eval_score.model_dump(),
            "cost_usd": cost_summary.get("total_cost_usd", 0.0),
            "passed_benchmark": composite >= 7.5 and cost_summary.get("total_cost_usd", 0.0) <= 0.08
        }


evaluator = SystemEvaluator()