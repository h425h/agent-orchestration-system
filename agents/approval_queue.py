# agents/approval_queue.py
import uuid
import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from agents.escalation import ApprovalLevel, EscalationReason


class ReviewDecision(str, Enum):
    APPROVED = "approved"
    MODIFIED = "modified"
    REJECTED = "rejected"
    TAKEN_OVER = "taken_over"


class ReviewQueueItem(BaseModel):
    """
    Represents a paused execution state snapshot awaiting human judgment.
    """
    ticket_id: str = Field(default_factory=lambda: f"rev_{uuid.uuid4().hex[:8]}")
    thread_id: str = Field(description="LangGraph persistent thread identifier")
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    status: str = Field(default="pending", description="pending, resolved")
    level: ApprovalLevel
    reason: EscalationReason
    description: str
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Packaged snapshot: task, plan, completed subtasks, active subtask, proposed output"
    )
    decision: Optional[ReviewDecision] = None
    human_feedback: Optional[str] = None
    modified_output: Optional[str] = None


class ApprovalQueue:
    """Thread-safe review queue managing escalated execution states."""

    def __init__(self):
        self._queue: Dict[str, ReviewQueueItem] = {}

    def enqueue(
        self,
        thread_id: str,
        level: ApprovalLevel,
        reason: EscalationReason,
        description: str,
        context: Dict[str, Any]
    ) -> ReviewQueueItem:
        """#1. Package execution context snapshot and register pending review ticket."""
        item = ReviewQueueItem(
            thread_id=thread_id,
            level=level,
            reason=reason,
            description=description,
            context=context
        )
        self._queue[item.ticket_id] = item
        return item

    def get_pending_tickets(self) -> List[ReviewQueueItem]:
        """#2. Retrieve all open escalations waiting for human review."""
        return [item for item in self._queue.values() if item.status == "pending"]

    def get_ticket(self, ticket_id: str) -> Optional[ReviewQueueItem]:
        """#3. Look up a specific ticket by ticket ID."""
        return self._queue.get(ticket_id)

    def resolve_ticket(
        self,
        ticket_id: str,
        decision: ReviewDecision,
        feedback: Optional[str] = None,
        modified_output: Optional[str] = None
    ) -> ReviewQueueItem:
        """#4. Record human resolution (approve, modify, reject, take over) on an open ticket."""
        if ticket_id not in self._queue:
            raise KeyError(f"Ticket {ticket_id} not found in approval queue.")

        ticket = self._queue[ticket_id]
        ticket.status = "resolved"
        ticket.decision = decision
        ticket.human_feedback = feedback
        ticket.modified_output = modified_output
        return ticket

    def clear(self):
        """Reset queue for test isolation."""
        self._queue.clear()


# Shared singleton instance
approval_queue = ApprovalQueue()