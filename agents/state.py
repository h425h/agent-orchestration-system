# agents/state.py
import operator
from typing import TypedDict, Annotated, List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

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
    subtasks: List[Subtask] = Field(description="Ordered list of subtasks")

class AgentState(TypedDict):
    """The shared memory state flowing through all LangGraph nodes."""
    task: str
    plan: Optional[Dict[str, Any]]
    completed_subtasks: Annotated[List[Dict[str, Any]], operator.add]
    current_subtask_id: Optional[str]
    current_specialist_output: Optional[str]
    reviewer_verdict: Optional[str]  # "approved", "rejected", or "escalate"
    reviewer_feedback: Optional[str]
    final_output: Optional[str]
    error_count: int
    human_approved: bool