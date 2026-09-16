# eval/tracer.py
import time
import uuid
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class SpanRecord(BaseModel):
    """Represents an atomic execution span within a task trace tree."""
    span_id: str = Field(default_factory=lambda: f"span_{uuid.uuid4().hex[:8]}")
    parent_span_id: Optional[str] = None
    name: str
    agent_role: Optional[str] = None
    status: str = "success"  # success, error, escalated
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    latency_ms: Optional[float] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    events: List[Dict[str, Any]] = Field(default_factory=list)


class OrchestrationTracer:
    """Centralized in-memory trace manager capturing multi-agent execution hierarchies."""

    def __init__(self):
        self._spans: Dict[str, List[SpanRecord]] = {}
        self._active_spans: Dict[str, SpanRecord] = {}

    def start_trace(self, trace_id: str) -> None:
        if trace_id not in self._spans:
            self._spans[trace_id] = []

    def start_span(
        self,
        trace_id: str,
        name: str,
        agent_role: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> SpanRecord:
        self.start_trace(trace_id)
        span = SpanRecord(
            name=name,
            agent_role=agent_role,
            parent_span_id=parent_span_id,
            attributes=attributes or {}
        )
        self._active_spans[span.span_id] = span
        self._spans[trace_id].append(span)
        return span

    def end_span(
        self,
        span_id: str,
        status: str = "success",
        attributes: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> Optional[SpanRecord]:
        span = self._active_spans.pop(span_id, None)
        if not span:
            return None

        span.end_time = time.time()
        span.latency_ms = round((span.end_time - span.start_time) * 1000, 2)
        span.status = status
        if attributes:
            span.attributes.update(attributes)
        if error_message:
            span.events.append({"event": "error", "message": error_message, "time": time.time()})
        return span

    def get_trace_tree(self, trace_id: str) -> List[Dict[str, Any]]:
        """Returns ordered execution spans formatted for inspection and visualization."""
        spans = self._spans.get(trace_id, [])
        return [span.model_dump() for span in spans]

    def clear(self):
        self._spans.clear()
        self._active_spans.clear()


# Global singleton tracer
tracer = OrchestrationTracer()