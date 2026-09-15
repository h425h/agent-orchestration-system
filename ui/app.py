# ui/app.py
import os
import sys

# Ensure project root is on the Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import streamlit as st

from agents.approval_queue import approval_queue, ReviewDecision
from agents.escalation import ApprovalLevel, EscalationReason
from agents.resumption import resume_with_human_decision
from agents.graph import create_agent_graph
from memory.checkpointer import get_sqlite_checkpointer

st.set_page_config(
    page_title="Multi-Agent HITL Orchestrator",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Multi-Agent HITL Control Center")
st.caption("Inspect paused executions, review agent decisions, and issue granular approvals.")

# Initialize graph instance with checkpointer
checkpointer = get_sqlite_checkpointer("orchestrator_state.db")
app = create_agent_graph(checkpointer=checkpointer)

# ---------------------------------------------------------
# Sidebar: Ticket Queue & Refresh
# ---------------------------------------------------------
st.sidebar.header("📥 Approval Queue")

# Seed mock tickets if queue is empty for UI testing
if st.sidebar.button("➕ Seed Test Escalations"):
    approval_queue.enqueue(
        thread_id="test_session_101",
        level=ApprovalLevel.APPROVE_PLAN,
        reason=EscalationReason.LOW_PLAN_CONFIDENCE,
        description="Supervisor confidence is 0.58. High uncertainty on data migration order.",
        context={
            "task": "Execute zero-downtime schema migration across shards",
            "plan": {
                "reasoning": "Sequence relies on unverified shard partitions",
                "confidence": 0.58,
                "subtasks": [
                    {"id": "t1", "description": "Lock shard writes", "specialist": "coder", "dependencies": []},
                    {"id": "t2", "description": "Sync replica records", "specialist": "coder", "dependencies": ["t1"]}
                ]
            }
        }
    )
    approval_queue.enqueue(
        thread_id="test_session_102",
        level=ApprovalLevel.APPROVE_ACTION,
        reason=EscalationReason.SENSITIVE_OPERATION,
        description="Coder attempted to drop table partitions.",
        context={
            "task": "Clean test records from staging clusters",
            "subtask_id": "task_2",
            "proposed_action": "DROP TABLE test_users_staging CASCADE;",
            "reviewer_feedback": "Destructive query detected in specialist output.",
            "reviewer_score": 4.0
        }
    )
    st.sidebar.success("Seeded test tickets.")

pending_tickets = approval_queue.get_pending_tickets()

if not pending_tickets:
    st.sidebar.info("No tickets currently pending review.")
    st.info("✅ No active agent escalations. Autonomous pipeline running normally.")
    st.stop()

ticket_options = {f"[{t.ticket_id}] {t.level.value.upper()} - {t.reason.value}": t.ticket_id for t in pending_tickets}
selected_label = st.sidebar.selectbox("Select Pending Ticket", list(ticket_options.keys()))
selected_ticket_id = ticket_options[selected_label]
ticket = approval_queue.get_ticket(selected_ticket_id)

# ---------------------------------------------------------
# Main Ticket Inspection Panel
# ---------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Execution Context")
    st.markdown(f"**Ticket ID:** `{ticket.ticket_id}`")
    st.markdown(f"**Thread ID:** `{ticket.thread_id}`")
    st.markdown(f"**Approval Level:** `{ticket.level.value}`")
    st.markdown(f"**Escalation Reason:** `{ticket.reason.value}`")
    st.warning(f"**Trigger Explanation:** {ticket.description}")

    if "task" in ticket.context:
        st.markdown(f"**Root User Task:**\n> {ticket.context['task']}")

    if "plan" in ticket.context:
        st.markdown("**Current Supervisor Plan:**")
        st.json(ticket.context["plan"])

with col2:
    st.subheader("🤖 Proposed Specialist Action & Review")

    if "subtask_id" in ticket.context:
        st.markdown(f"**Active Subtask:** `{ticket.context['subtask_id']}`")

    if "proposed_action" in ticket.context:
        st.markdown("**Proposed Deliverable / Code:**")
        st.code(ticket.context["proposed_action"], language="sql")

    if "reviewer_feedback" in ticket.context:
        st.markdown(f"**Reviewer Critique:**\n> {ticket.context['reviewer_feedback']}")

    if "reviewer_score" in ticket.context:
        score = ticket.context["reviewer_score"]
        st.metric(label="Deliverable Quality Score", value=f"{score}/10", delta="-Low Quality" if score < 6.0 else "Acceptable")

# ---------------------------------------------------------
# Human Resolution Center
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🛠️ Human Resolution Center")

feedback = st.text_area(
    "Operator Critique / Feedback (sent to agent if modifying or rejecting):",
    placeholder="Explain required changes or sign-off notes..."
)

override_box = None
if ticket.level in [ApprovalLevel.APPROVE_ACTION, ApprovalLevel.TAKE_OVER]:
    override_box = st.text_area(
        "Deliverable Override / Modification Buffer:",
        value=ticket.context.get("proposed_action", ""),
        height=140
    )

action_col1, action_col2, action_col3, action_col4 = st.columns(4)

with action_col1:
    if st.button("✅ Approve Action / Plan", use_container_width=True):
        resume_with_human_decision(
            app=app,
            config={"configurable": {"thread_id": ticket.thread_id}},
            ticket_id=ticket.ticket_id,
            decision=ReviewDecision.APPROVED,
            feedback=feedback or "Approved without changes by operator."
        )
        st.success(f"Ticket {ticket.ticket_id} APPROVED. Execution resumed.")
        st.rerun()

with action_col2:
    if st.button("✏️ Apply Modifications", use_container_width=True):
        resume_with_human_decision(
            app=app,
            config={"configurable": {"thread_id": ticket.thread_id}},
            ticket_id=ticket.ticket_id,
            decision=ReviewDecision.MODIFIED,
            feedback=feedback,
            modified_output=override_box
        )
        st.success(f"Ticket {ticket.ticket_id} MODIFIED and resumed.")
        st.rerun()

with action_col3:
    if st.button("❌ Reject & Force Retry", use_container_width=True):
        resume_with_human_decision(
            app=app,
            config={"configurable": {"thread_id": ticket.thread_id}},
            ticket_id=ticket.ticket_id,
            decision=ReviewDecision.REJECTED,
            feedback=feedback or "Rejected by human operator."
        )
        st.error(f"Ticket {ticket.ticket_id} REJECTED. Re-routing to specialist.")
        st.rerun()

with action_col4:
    if st.button("👤 Manual Take Over", use_container_width=True):
        resume_with_human_decision(
            app=app,
            config={"configurable": {"thread_id": ticket.thread_id}},
            ticket_id=ticket.ticket_id,
            decision=ReviewDecision.TAKEN_OVER,
            feedback=feedback,
            modified_output=override_box or feedback
        )
        st.warning(f"Ticket {ticket.ticket_id} TAKEN OVER. Subtask resolved manually.")
        st.rerun()