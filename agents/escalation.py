# agents/escalation.py
import re
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from agents.state import AgentState


class ApprovalLevel(str, Enum):
    NOTIFY = "notify"                  # Informational alert; graph proceeds
    APPROVE_PLAN = "approve_plan"      # Pause before executing any subtask
    APPROVE_ACTION = "approve_action"  # Pause before executing a specific specialist/tool step
    TAKE_OVER = "take_over"            # Specialist stands down; human supplies result


class EscalationReason(str, Enum):
    LOW_PLAN_CONFIDENCE = "low_plan_confidence"
    REPEATED_SUBTASK_FAILURE = "repeated_subtask_failure"
    SENSITIVE_OPERATION = "sensitive_operation"
    LOW_QUALITY_SCORE = "low_quality_score"
    USER_REQUESTED = "user_requested"


class EscalationRequest(BaseModel):
    """Payload packaged when an execution pause is requested."""
    subtask_id: Optional[str] = Field(default=None, description="Subtask requesting escalation")
    level: ApprovalLevel = Field(description="Severity/depth of approval required")
    reason: EscalationReason = Field(description="Trigger reason")
    description: str = Field(description="Explanation for why human intervention is required")
    context: Dict[str, Any] = Field(default_factory=dict, description="Snapshot context for reviewer")


class EscalationPolicy:
    """Evaluates task context and agent state against HITL escalation criteria."""

    SENSITIVE_KEYWORDS = {
        "delete", "drop table", "rm -rf", "truncate", "transfer", "pay",
        "email", "post", "publish", "send", "credential", "api_key", "secret"
    }

    CONFIDENCE_THRESHOLD = 0.70
    QUALITY_SCORE_THRESHOLD = 6.0

    @classmethod
    def check_plan_escalation(cls, state: AgentState) -> Optional[EscalationRequest]:
        """Evaluates whether the supervisor's plan requires human sign-off."""
        plan = state.get("plan") or {}
        confidence = plan.get("confidence", 1.0)
        task_text = state.get("task", "").lower()

        # #1. Check explicit user request
        if state.get("require_human_approval"):
            return EscalationRequest(
                level=ApprovalLevel.APPROVE_PLAN,
                reason=EscalationReason.USER_REQUESTED,
                description="User explicitly configured task to require plan approval prior to start.",
                context={"task": state.get("task"), "plan": plan}
            )

        # #2. Check plan confidence score
        if confidence < cls.CONFIDENCE_THRESHOLD:
            return EscalationRequest(
                level=ApprovalLevel.APPROVE_PLAN,
                reason=EscalationReason.LOW_PLAN_CONFIDENCE,
                description=f"Supervisor plan confidence ({confidence:.2f}) is below threshold ({cls.CONFIDENCE_THRESHOLD}).",
                context={"confidence": confidence, "plan": plan}
            )

        # #3. Check for sensitive domain terms (whole words, so "postgres" is not "post")
        task_match = _SENSITIVE_RE.search(task_text)
        if task_match:
            return EscalationRequest(
                level=ApprovalLevel.APPROVE_ACTION,
                reason=EscalationReason.SENSITIVE_OPERATION,
                description=f"Task text contains sensitive operation '{task_match.group(0)}' requiring confirmation.",
                context={"matched": task_match.group(0), "task": state.get("task")}
            )

        for subtask in plan.get("subtasks", []):
            desc = subtask.get("description", "").lower()
            match = _SENSITIVE_RE.search(desc)
            if match:
                return EscalationRequest(
                    subtask_id=subtask.get("id"),
                    level=ApprovalLevel.APPROVE_ACTION,
                    reason=EscalationReason.SENSITIVE_OPERATION,
                    description=f"Subtask '{subtask.get('id')}' contains sensitive operation '{match.group(0)}' requiring confirmation.",
                    context={"subtask": subtask, "matched": match.group(0)}
                )

        return None

    @classmethod
    def check_execution_escalation(cls, state: AgentState) -> Optional[EscalationRequest]:
        """Evaluates specialist execution, retries, and reviewer scores."""
        # #1. Retry ceiling reached
        if state.get("error_count", 0) >= 2 and state.get("reviewer_verdict") == "rejected":
            return EscalationRequest(
                subtask_id=state.get("current_subtask_id"),
                level=ApprovalLevel.TAKE_OVER,
                reason=EscalationReason.REPEATED_SUBTASK_FAILURE,
                description=f"Subtask '{state.get('current_subtask_id')}' failed review 2 consecutive times.",
                context={
                    "subtask_id": state.get("current_subtask_id"),
                    "reviewer_feedback": state.get("reviewer_feedback"),
                    "last_output": state.get("current_specialist_output")
                }
            )

        # #2. Reviewer quality score floor
        reviewer_score = state.get("reviewer_score", 10.0)
        if reviewer_score is not None and reviewer_score < cls.QUALITY_SCORE_THRESHOLD:
            return EscalationRequest(
                subtask_id=state.get("current_subtask_id"),
                level=ApprovalLevel.APPROVE_ACTION,
                reason=EscalationReason.LOW_QUALITY_SCORE,
                description=f"Deliverable quality score ({reviewer_score:.1f}) is below acceptable threshold ({cls.QUALITY_SCORE_THRESHOLD}).",
                context={
                    "reviewer_score": reviewer_score,
                    "reviewer_feedback": state.get("reviewer_feedback")
                }
            )

        return None


# Whole-word match with simple inflections (delete/deleted/deleting, send/sends/sending).
# Deliberately conservative: a false positive costs a human click, a miss costs a destructive action.
_SENSITIVE_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(k) for k in sorted(EscalationPolicy.SENSITIVE_KEYWORDS, key=len, reverse=True))
    + r")(?:s|es|ed|d|ing)?\b"
)
