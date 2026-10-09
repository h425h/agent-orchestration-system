# agents/state.py
import operator
from typing import TypedDict, Annotated, List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, model_validator

class Subtask(BaseModel):
    """Represents an atomic subtask within the execution plan."""
    id: str = Field(description="Unique identifier for the subtask, e.g. 'task_1'")
    description: str = Field(description="Actionable goal of this subtask")
    # Tightly constrained specialist type
    specialist: Literal["researcher", "coder", "writer"] = Field(
        description="Assigned specialist: must be 'researcher', 'coder', or 'writer'"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="IDs of prerequisite subtasks that must complete first"
    )
    status: str = Field(default="pending", description="pending, in_progress, completed, failed")
    result: Optional[str] = Field(default=None, description="Output generated for this subtask")

class ExecutionPlan(BaseModel):
    """Structured plan produced by the Supervisor agent."""
    reasoning: str = Field(description="Decomposition logic and high-level strategy")
    estimated_complexity: str = Field(description="low, medium, or high")
    confidence: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description="Supervisor self-reported confidence in plan validity (0.0 - 1.0)"
    )
    subtasks: List[Subtask] = Field(description="Ordered list of subtasks")

    @model_validator(mode="after")
    def _validate_dependency_graph(self) -> "ExecutionPlan":
        """Reject plans the dispatcher could never finish: empty, duplicate ids,
        unknown or self dependencies, and dependency cycles."""
        if not self.subtasks:
            raise ValueError("plan must contain at least one subtask")

        ids = [t.id for t in self.subtasks]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate subtask ids: {duplicates}")

        known = set(ids)
        for t in self.subtasks:
            for dep in t.dependencies:
                if dep == t.id:
                    raise ValueError(f"subtask '{t.id}' depends on itself")
                if dep not in known:
                    raise ValueError(f"subtask '{t.id}' depends on unknown subtask '{dep}'")

        # Kahn-style peel: if nothing is ready while tasks remain, there is a cycle.
        remaining = {t.id: set(t.dependencies) for t in self.subtasks}
        while remaining:
            ready = [i for i, deps in remaining.items() if not deps]
            if not ready:
                raise ValueError(f"dependency cycle among subtasks: {sorted(remaining)}")
            for i in ready:
                del remaining[i]
            for deps in remaining.values():
                deps.difference_update(ready)
        return self

class AgentState(TypedDict):
    """The shared memory state flowing through all LangGraph nodes."""
    task: str
    plan: Optional[Dict[str, Any]]
    completed_subtasks: Annotated[List[Dict[str, Any]], operator.add]
    current_subtask_id: Optional[str]
    current_specialist_output: Optional[str]
    reviewer_verdict: Optional[str]  # "approved", "rejected", or "escalate"
    reviewer_feedback: Optional[str]
    reviewer_score: Optional[float]           # Phase 3: Reviewer numerical quality score (0-10)
    candidate_subtask: Optional[Dict[str, Any]]  # Specialist output awaiting review; committed only once approved
    final_output: Optional[str]
    error_count: int
    human_approved: bool
    require_human_approval: bool             # Phase 3: Explicit human review override flag
    pending_escalation: Optional[Dict[str, Any]]  # Phase 3: Serialized EscalationRequest payload