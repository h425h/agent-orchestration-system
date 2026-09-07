# test_escalation_triggers.py
from agents.escalation import EscalationPolicy, ApprovalLevel, EscalationReason
from agents.state import AgentState


def test_escalations():
    print("Testing Phase 3 Escalation Triggers...")

    # Case 1: Low plan confidence
    state_low_conf: AgentState = {
        "task": "Perform migration across undefined legacy clusters",
        "plan": {"confidence": 0.55, "subtasks": []},
        "completed_subtasks": [],
        "current_subtask_id": None,
        "current_specialist_output": None,
        "reviewer_verdict": None,
        "reviewer_feedback": None,
        "reviewer_score": None,
        "final_output": None,
        "error_count": 0,
        "human_approved": False,
        "require_human_approval": False,
        "pending_escalation": None,
    }
    esc_conf = EscalationPolicy.check_plan_escalation(state_low_conf)
    assert esc_conf is not None
    assert esc_conf.level == ApprovalLevel.APPROVE_PLAN
    assert esc_conf.reason == EscalationReason.LOW_PLAN_CONFIDENCE
    print("  [Pass] Low Plan Confidence -> Triggers APPROVE_PLAN")

    # Case 2: Sensitive operation detected
    state_sensitive: AgentState = {
        "task": "Delete test user database records from production",
        "plan": {
            "confidence": 0.95,
            "subtasks": [{"id": "task_1", "description": "Drop table test_records in database"}]
        },
        "completed_subtasks": [],
        "current_subtask_id": "task_1",
        "current_specialist_output": None,
        "reviewer_verdict": None,
        "reviewer_feedback": None,
        "reviewer_score": None,
        "final_output": None,
        "error_count": 0,
        "human_approved": False,
        "require_human_approval": False,
        "pending_escalation": None,
    }
    esc_sens = EscalationPolicy.check_plan_escalation(state_sensitive)
    assert esc_sens is not None
    assert esc_sens.level == ApprovalLevel.APPROVE_ACTION
    assert esc_sens.reason == EscalationReason.SENSITIVE_OPERATION
    print("  [Pass] Sensitive Operation -> Triggers APPROVE_ACTION")

    # Case 3: Repeated subtask failure
    state_failure: AgentState = {
        "task": "Generate high-frequency math script",
        "plan": None,
        "completed_subtasks": [],
        "current_subtask_id": "task_2",
        "current_specialist_output": "Broken code",
        "reviewer_verdict": "rejected",
        "reviewer_feedback": "Syntax error persistent",
        "reviewer_score": 4.0,
        "final_output": None,
        "error_count": 2,
        "human_approved": False,
        "require_human_approval": False,
        "pending_escalation": None,
    }
    esc_fail = EscalationPolicy.check_execution_escalation(state_failure)
    assert esc_fail is not None
    assert esc_fail.level == ApprovalLevel.TAKE_OVER
    assert esc_fail.reason == EscalationReason.REPEATED_SUBTASK_FAILURE
    print("  [Pass] Double Rejection -> Triggers TAKE_OVER")

    print("\nAll Escalation Trigger Rules Verified Successfully!")


if __name__ == "__main__":
    test_escalations()