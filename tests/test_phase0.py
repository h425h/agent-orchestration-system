"""Phase 0: the system behaves as documented. All LLM and tool calls are faked."""
import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import agents.reviewer as reviewer_mod
import agents.specialists as specialists_mod
import agents.supervisor as supervisor_mod
from agents.approval_queue import approval_queue
from agents.bedrock_llm import BedrockLLM, usage_collector, LLMCall
from agents.escalation import EscalationPolicy, EscalationReason, ApprovalLevel
from agents.graph import create_agent_graph, route_after_review
from agents.state import ExecutionPlan
from eval.cost_tracker import CostTracker, normalize_model_id
from main import classify_run


# ----------------------------------------------------------------- helpers
def plan_json(subtasks, confidence=0.9):
    return json.dumps({
        "reasoning": "test plan",
        "estimated_complexity": "low",
        "confidence": confidence,
        "subtasks": subtasks,
    })


TWO_STEP_PLAN = [
    {"id": "t1", "description": "Look up the facts", "specialist": "researcher", "dependencies": []},
    {"id": "t2", "description": "Summarise the findings", "specialist": "writer", "dependencies": ["t1"]},
]


HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _record_fake_usage():
    usage_collector.record(LLMCall(HAIKU, 100, 20, 5.0))


class ScriptedLLM:
    """Returns queued responses in order; records every prompt it receives."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, messages, system_prompt=None, max_tokens=2048):
        self.prompts.append(messages[0]["content"])
        _record_fake_usage()
        return self.responses.pop(0)


class SpecialistLLM:
    """Stands in for the Haiku/Sonnet specialist calls."""

    def __init__(self):
        self.prompts = []

    def invoke(self, messages, system_prompt=None, max_tokens=2048):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        _record_fake_usage()
        if "Formulate a single" in prompt:
            return "fake search query"
        if "Deliver a crisp" in prompt:
            return "FINAL REPORT"
        return "research summary"


def review(verdict, score, feedback="ok"):
    return json.dumps({"verdict": verdict, "score": score, "feedback": feedback})


def install_fakes(monkeypatch, plan_response, reviews):
    sup, rev, spec = ScriptedLLM([plan_response]), ScriptedLLM(reviews), SpecialistLLM()
    monkeypatch.setattr(supervisor_mod, "llm", sup)
    monkeypatch.setattr(supervisor_mod.semantic_memory, "recall_similar_memories", lambda *a, **k: [])
    monkeypatch.setattr(reviewer_mod, "llm", rev)
    monkeypatch.setattr(specialists_mod, "llm", spec)
    monkeypatch.setattr(specialists_mod, "llm_sonnet", spec)
    monkeypatch.setattr(
        specialists_mod.registry, "execute",
        lambda *a, **k: {"output": "tool output", "duration": 0.0},
    )
    return sup, rev, spec


@pytest.fixture
def run_graph(monkeypatch):
    """Returns run(plan_response, reviews) -> (final_state, specialist_llm, supervisor_llm)."""

    def run(plan_response, reviews, thread_id="test-thread"):
        sup, _rev, spec = install_fakes(monkeypatch, plan_response, reviews)
        app = create_agent_graph(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": thread_id}}
        initial = {
            "task": "Summarise topic X", "plan": None, "completed_subtasks": [],
            "current_subtask_id": None, "current_specialist_output": None,
            "reviewer_verdict": None, "reviewer_feedback": None, "reviewer_score": None,
            "candidate_subtask": None, "final_output": None, "error_count": 0,
            "human_approved": False, "require_human_approval": False, "pending_escalation": None,
        }
        for _ in app.stream(initial, config=config):
            pass
        return dict(app.get_state(config).values), spec, sup

    return run


# ----------------------------------------------------------------- plan validation
def _plan(subtasks):
    return ExecutionPlan(reasoning="r", estimated_complexity="low", subtasks=subtasks)


def _st(i, deps=()):
    return {"id": i, "description": "d", "specialist": "writer", "dependencies": list(deps)}


def test_valid_plan_accepted():
    assert len(_plan([_st("a"), _st("b", ["a"])]).subtasks) == 2


@pytest.mark.parametrize("subtasks,fragment", [
    ([], "at least one"),
    ([_st("a"), _st("a")], "duplicate"),
    ([_st("a", ["ghost"])], "unknown subtask"),
    ([_st("a", ["a"])], "depends on itself"),
    ([_st("a", ["b"]), _st("b", ["a"])], "cycle"),
])
def test_invalid_plans_rejected(subtasks, fragment):
    with pytest.raises(ValueError, match=fragment):
        _plan(subtasks)


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValueError):
        ExecutionPlan(reasoning="r", estimated_complexity="low", confidence=1.4, subtasks=[_st("a")])


# ----------------------------------------------------------------- supervisor parsing
def test_parse_plan_requires_confidence():
    raw = json.dumps({"reasoning": "r", "estimated_complexity": "low", "subtasks": [_st("a")]})
    with pytest.raises(ValueError, match="confidence"):
        supervisor_mod.parse_plan(raw)


def test_parse_plan_tolerates_fences_and_prose():
    raw = "Here is the plan:\n```json\n" + plan_json([_st("a")]) + "\n```\nHope that helps."
    assert supervisor_mod.parse_plan(raw).subtasks[0].id == "a"


def test_supervisor_retries_then_succeeds(monkeypatch):
    llm = ScriptedLLM(["not json at all", plan_json([_st("a")], confidence=0.8)])
    monkeypatch.setattr(supervisor_mod, "llm", llm)
    monkeypatch.setattr(supervisor_mod.semantic_memory, "recall_similar_memories", lambda *a, **k: [])
    out = supervisor_mod.supervisor_planner({"task": "do a thing"})
    assert out["plan"]["confidence"] == 0.8
    assert "previous response was rejected" in llm.prompts[1]


def test_supervisor_raises_after_max_attempts(monkeypatch):
    llm = ScriptedLLM(["garbage"] * supervisor_mod.MAX_PLAN_ATTEMPTS)
    monkeypatch.setattr(supervisor_mod, "llm", llm)
    monkeypatch.setattr(supervisor_mod.semantic_memory, "recall_similar_memories", lambda *a, **k: [])
    with pytest.raises(supervisor_mod.PlanningError):
        supervisor_mod.supervisor_planner({"task": "do a thing"})


# ----------------------------------------------------------------- reviewer
def test_reviewer_returns_score(monkeypatch):
    monkeypatch.setattr(reviewer_mod, "llm", ScriptedLLM([review("approved", 8.5, "good")]))
    out = reviewer_mod.reviewer_node({"current_specialist_output": "x", "current_subtask_id": "t1", "plan": {}})
    assert out["reviewer_verdict"] == "approved" and out["reviewer_score"] == 8.5


def test_reviewer_fails_closed(monkeypatch):
    for bad in ["not json", review("banana", 9.0), "[1, 2]"]:
        monkeypatch.setattr(reviewer_mod, "llm", ScriptedLLM([bad]))
        out = reviewer_mod.reviewer_node({"current_specialist_output": "x", "current_subtask_id": "t1", "plan": {}})
        assert out["reviewer_verdict"] == "escalate", bad


def test_reviewer_score_clamped(monkeypatch):
    monkeypatch.setattr(reviewer_mod, "llm", ScriptedLLM([review("approved", 99)]))
    out = reviewer_mod.reviewer_node({"current_specialist_output": "x", "current_subtask_id": "t1", "plan": {}})
    assert out["reviewer_score"] == 10.0


# ----------------------------------------------------------------- routing
def test_routing_table():
    assert route_after_review({"reviewer_verdict": "approved", "reviewer_score": 9.0}) == "commit"
    assert route_after_review({"reviewer_verdict": "approved"}) == "commit"
    assert route_after_review({"reviewer_verdict": "approved", "reviewer_score": 5.0}) == "human_review_gate"
    assert route_after_review({"reviewer_verdict": "rejected", "error_count": 1}) == "retry_specialist"
    assert route_after_review({"reviewer_verdict": "rejected", "error_count": 2}) == "human_review_gate"
    assert route_after_review({"reviewer_verdict": "escalate"}) == "human_escalation"
    assert route_after_review({"reviewer_verdict": None}) == "human_escalation"


# ----------------------------------------------------------------- end-to-end graph behaviour
def test_happy_path_commits_each_subtask_once(run_graph):
    state, _, _ = run_graph(plan_json(TWO_STEP_PLAN), [review("approved", 9), review("approved", 9)])
    assert [t["id"] for t in state["completed_subtasks"]] == ["t1", "t2"]
    assert state["final_output"] == "FINAL REPORT"
    assert classify_run(state) == "completed"


def test_rejected_attempt_is_not_committed_and_retry_sees_feedback(run_graph):
    state, spec, _ = run_graph(
        plan_json(TWO_STEP_PLAN),
        [review("rejected", 3, "missing sources"), review("approved", 9), review("approved", 9)],
    )
    ids = [t["id"] for t in state["completed_subtasks"]]
    assert ids == ["t1", "t2"], "rejected attempt must not create a duplicate entry"
    assert any("missing sources" in p for p in spec.prompts), "retry must include reviewer feedback"
    assert classify_run(state) == "completed"


def test_low_quality_score_reaches_human_gate(run_graph):
    state, _, _ = run_graph(plan_json(TWO_STEP_PLAN), [review("approved", 5.0, "thin")])
    assert classify_run(state) == "escalated"
    assert state["completed_subtasks"] == [], "low-scoring output must not be committed"
    ticket = approval_queue.get_pending_tickets()[0]
    assert ticket.reason == EscalationReason.LOW_QUALITY_SCORE
    assert ticket.level == ApprovalLevel.APPROVE_ACTION
    assert ticket.thread_id == "test-thread"


def test_retry_ceiling_creates_take_over_ticket(run_graph):
    rejects = [review("rejected", 2, "bad")] * 3
    state, _, _ = run_graph(plan_json(TWO_STEP_PLAN), rejects, thread_id="run-42")
    assert classify_run(state) == "escalated"
    assert state["completed_subtasks"] == []
    ticket = approval_queue.get_pending_tickets()[0]
    assert ticket.level == ApprovalLevel.TAKE_OVER
    assert ticket.reason == EscalationReason.REPEATED_SUBTASK_FAILURE
    assert ticket.thread_id == "run-42"


def test_low_confidence_plan_halts_before_any_specialist(run_graph):
    state, spec, _ = run_graph(plan_json(TWO_STEP_PLAN, confidence=0.4), [])
    assert classify_run(state) == "escalated"
    assert spec.prompts == [], "no specialist may run before the plan is approved"
    ticket = approval_queue.get_pending_tickets()[0]
    assert ticket.reason == EscalationReason.LOW_PLAN_CONFIDENCE
    assert ticket.level == ApprovalLevel.APPROVE_PLAN


# ----------------------------------------------------------------- escalation keywords
def _plan_state(task, descriptions):
    return {
        "task": task,
        "plan": {"confidence": 0.95, "subtasks": [
            {"id": f"t{i}", "description": d} for i, d in enumerate(descriptions)]},
    }


@pytest.mark.parametrize("task,descs", [
    ("Benchmark postgres query payload sizes", ["Measure the sender throughput"]),
    ("Compare vector databases", ["Summarise pricing"]),
])
def test_keywords_do_not_match_inside_words(task, descs):
    assert EscalationPolicy.check_plan_escalation(_plan_state(task, descs)) is None


@pytest.mark.parametrize("task,descs", [
    ("Send an email to the team", ["Draft the text"]),
    ("Clean up staging", ["Delete old records"]),
    ("Clean up staging", ["Deleted records are logged", "x"]),
])
def test_sensitive_keywords_trigger(task, descs):
    esc = EscalationPolicy.check_plan_escalation(_plan_state(task, descs))
    assert esc is not None and esc.reason == EscalationReason.SENSITIVE_OPERATION


def test_task_level_keyword_triggers_even_with_empty_subtasks():
    esc = EscalationPolicy.check_plan_escalation(_plan_state("Delete the production table", []))
    assert esc is not None and esc.level == ApprovalLevel.APPROVE_ACTION


# ----------------------------------------------------------------- cost & usage
def test_cost_uses_current_model_rates():
    t = CostTracker()
    haiku = t.record_usage("r", "supervisor", "us.anthropic.claude-haiku-4-5-20251001-v1:0", 1000, 1000, 10.0)
    sonnet = t.record_usage("r", "coder", "us.anthropic.claude-sonnet-4-6", 1000, 1000, 10.0)
    assert haiku.cost_usd == pytest.approx(0.006)   # $1/$5 per MTok
    assert sonnet.cost_usd == pytest.approx(0.018)  # $3/$15 per MTok
    assert normalize_model_id("us.anthropic.claude-sonnet-4-6") == "anthropic.claude-sonnet-4-6"


def test_unknown_model_is_flagged_not_silently_priced():
    t = CostTracker()
    rec = t.record_usage("r", "x", "vendor.unknown-model", 1000, 1000, 5.0)
    assert rec.priced is False and rec.cost_usd == 0.0
    assert t.get_run_summary("r")["unpriced_calls"] == 1


def test_invoke_records_measured_usage():
    llm = BedrockLLM("us.anthropic.claude-haiku-4-5-20251001-v1:0")

    class FakeClient:
        def converse(self, **kwargs):
            return {
                "output": {"message": {"content": [{"text": "hello"}]}},
                "usage": {"inputTokens": 123, "outputTokens": 45},
                "stopReason": "end_turn",
            }

    llm.client = FakeClient()
    assert llm.invoke([{"role": "user", "content": "hi"}]) == "hello"
    calls = usage_collector.drain()
    assert len(calls) == 1
    assert (calls[0].input_tokens, calls[0].output_tokens) == (123, 45)
    assert calls[0].latency_ms >= 0
    assert usage_collector.drain() == []


def test_record_calls_attributes_to_agent():
    t = CostTracker()
    calls = [LLMCall("us.anthropic.claude-sonnet-4-6", 2000, 500, 800.0)]
    t.record_calls("r", "coder", calls)
    summary = t.get_run_summary("r")
    assert summary["by_agent"]["coder"]["input_tokens"] == 2000
    assert summary["total_cost_usd"] == pytest.approx(0.006 + 0.0075)


# ----------------------------------------------------------------- full pipeline (main.py)
def test_pipeline_end_to_end(monkeypatch, tmp_path):
    import main
    from eval.cost_tracker import cost_tracker
    from eval.tracer import tracer

    install_fakes(monkeypatch, plan_json(TWO_STEP_PLAN), [review("approved", 9), review("approved", 9)])
    distilled = []
    monkeypatch.setattr(
        main.semantic_memory, "distill_and_store",
        lambda task, state: distilled.append(state) or {"summary": "stub"},
    )

    result = main.run_orchestration_pipeline("Summarise topic X", thread_id="e2e-1", db_path=str(tmp_path / "state.db"))

    assert result["status"] == "completed"
    # Regression: merging stream updates by hand kept only the last node's completed_subtasks.
    assert [t["id"] for t in result["final_state"]["completed_subtasks"]] == ["t1", "t2"]
    assert len(distilled) == 1 and len(distilled[0]["completed_subtasks"]) == 2

    summary = cost_tracker.get_run_summary("e2e-1")
    # supervisor 1 + researcher 2 (query+summary) + reviewer 1 + writer 1 + reviewer 1
    assert summary["llm_calls"] == 6
    assert summary["total_input_tokens"] == 600 and summary["total_output_tokens"] == 120
    assert {"supervisor", "researcher", "writer", "reviewer"} <= set(summary["by_agent"])

    spans = tracer.get_trace_tree("e2e-1")
    assert spans and all(sp["latency_ms"] is not None for sp in spans)


def test_pipeline_skips_distillation_for_escalated_runs(monkeypatch, tmp_path):
    import main

    install_fakes(monkeypatch, plan_json(TWO_STEP_PLAN), [review("approved", 4.0)])
    called = []
    monkeypatch.setattr(main.semantic_memory, "distill_and_store", lambda *a, **k: called.append(1) or {})

    result = main.run_orchestration_pipeline("Summarise topic X", thread_id="e2e-2", db_path=str(tmp_path / "state.db"))

    assert result["status"] == "escalated"
    assert called == [], "failed or escalated runs must not be stored as successful strategies"


# ----------------------------------------------------------------- keyword gate scope
def _typed_state(task, subtasks):
    return {"task": task, "plan": {"confidence": 0.95, "subtasks": subtasks}}


def _st2(i, specialist, desc):
    return {"id": i, "specialist": specialist, "description": desc}


def test_researcher_and_writer_wording_is_not_screened():
    state = _typed_state("Research agent frameworks", [
        _st2("t1", "researcher", "Cover publish-subscribe patterns and published analyst reports"),
        _st2("t2", "writer", "Write blog posts style summary"),
    ])
    assert EscalationPolicy.check_plan_escalation(state) is None


def test_coder_subtask_is_still_screened_in_mixed_plan():
    state = _typed_state("Research then clean up", [
        _st2("t1", "researcher", "Look up published docs"),
        _st2("t2", "coder", "Drop table staging_users"),
    ])
    esc = EscalationPolicy.check_plan_escalation(state)
    assert esc is not None and esc.subtask_id == "t2"


def test_unknown_specialist_is_screened():
    state = _typed_state("x", [{"id": "t1", "description": "delete old records"}])
    assert EscalationPolicy.check_plan_escalation(state) is not None


def test_halt_reason_is_printed(monkeypatch, tmp_path, capsys):
    import main
    install_fakes(monkeypatch, plan_json(TWO_STEP_PLAN), [review("approved", 4.0)])
    monkeypatch.setattr(main.semantic_memory, "distill_and_store", lambda *a, **k: {})
    main.run_orchestration_pipeline("Summarise topic X", thread_id="halt-1", db_path=str(tmp_path / "s.db"))
    out = capsys.readouterr().out
    assert "Halted for human review: low_quality_score" in out


# ----------------------------------------------------------------- retry feedback reaches the search query
def test_researcher_retry_query_includes_reviewer_feedback(monkeypatch):
    spec = SpecialistLLM()
    monkeypatch.setattr(specialists_mod, "llm", spec)
    monkeypatch.setattr(specialists_mod.registry, "execute", lambda *a, **k: {"output": "x", "duration": 0.0})
    state = {
        "plan": {"subtasks": [{"id": "t1", "description": "Research frameworks", "specialist": "researcher"}]},
        "current_subtask_id": "t1", "completed_subtasks": [],
        "reviewer_verdict": "rejected", "reviewer_feedback": "missing AutoGen and CrewAI",
    }
    specialists_mod.researcher_node(state)
    query_prompt = spec.prompts[0]
    assert "missing AutoGen and CrewAI" in query_prompt and "NEW query" in query_prompt


def test_first_attempt_query_has_no_retry_text(monkeypatch):
    spec = SpecialistLLM()
    monkeypatch.setattr(specialists_mod, "llm", spec)
    monkeypatch.setattr(specialists_mod.registry, "execute", lambda *a, **k: {"output": "x", "duration": 0.0})
    state = {
        "plan": {"subtasks": [{"id": "t1", "description": "Research frameworks", "specialist": "researcher"}]},
        "current_subtask_id": "t1", "completed_subtasks": [],
    }
    specialists_mod.researcher_node(state)
    assert "NEW query" not in spec.prompts[0] and "REJECTED" not in spec.prompts[0]


# ----------------------------------------------------------------- checkpoint holds plain strings, not Enums
def test_pending_escalation_is_checkpointed_as_plain_strings(run_graph):
    state, _, _ = run_graph(plan_json(TWO_STEP_PLAN, confidence=0.4), [])
    esc = state["pending_escalation"]
    assert type(esc["level"]) is str and type(esc["reason"]) is str
    assert esc["level"] == "approve_plan" and esc["reason"] == "low_plan_confidence"
