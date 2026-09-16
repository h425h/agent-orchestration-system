# ui/trace_explorer.py
import streamlit as st
import time
import uuid
from eval.tracer import tracer
from eval.cost_tracker import cost_tracker
from eval.replay import replay_debugger


def render_trace_explorer():
    st.title("🔍 Observability, Cost & Replay Debugger")
    st.caption("Inspect trace spans, model token costs, and branch historical checkpoints.")

    tab_traces, tab_cost, tab_replay = st.tabs([
        "🌲 Trace Explorer",
        "💰 Cost & Token Attribution",
        "⏪ Replay & Branch Debugger"
    ])

    # -------------------------------------------------------------
    # TAB 1: Trace Explorer (Subtask 14)
    # -------------------------------------------------------------
    with tab_traces:
        # Trace generation helper
        with st.sidebar:
            st.subheader("Trace Controls")
            if st.button("➕ Generate Sample Trace & Usage"):
                sample_id = f"trace_{int(time.time())}"
                # 1. Supervisor
                s1 = tracer.start_span(sample_id, "supervisor_planning", agent_role="supervisor", attributes={"task": "Benchmark vector search 2026"})
                time.sleep(0.04)
                tracer.end_span(s1.span_id, status="success", attributes={"subtasks_created": 3})
                cost_tracker.record_usage(sample_id, "supervisor", "anthropic.claude-3-5-haiku-20241022-v1:0", 1200, 350, 400.0)

                # 2. Researcher & Tool
                s2 = tracer.start_span(sample_id, "execute_researcher", agent_role="researcher", parent_span_id=s1.span_id)
                time.sleep(0.02)
                t1 = tracer.start_span(sample_id, "tool_web_search", agent_role="researcher", parent_span_id=s2.span_id, attributes={"query": "vector databases 2026"})
                time.sleep(0.03)
                tracer.end_span(t1.span_id, status="success", attributes={"results": 3})
                tracer.end_span(s2.span_id, status="success")
                cost_tracker.record_usage(sample_id, "researcher", "anthropic.claude-3-5-haiku-20241022-v1:0", 1800, 450, 520.0)

                # 3. Reviewer
                s3 = tracer.start_span(sample_id, "reviewer_evaluation", agent_role="reviewer", parent_span_id=s2.span_id)
                time.sleep(0.02)
                tracer.end_span(s3.span_id, status="success", attributes={"verdict": "approved", "score": 9.2})
                cost_tracker.record_usage(sample_id, "reviewer", "anthropic.claude-3-5-haiku-20241022-v1:0", 900, 120, 190.0)

                st.success(f"Generated trace: {sample_id}")
                st.rerun()

        trace_ids = list(tracer._spans.keys())
        if not trace_ids:
            st.info("No execution traces found. Use the sidebar button or run a task to record spans.")
        else:
            selected_trace_id = st.selectbox("Select Execution Trace ID", trace_ids, index=len(trace_ids) - 1)
            spans = tracer.get_trace_tree(selected_trace_id)

            col1, col2, col3 = st.columns(3)
            total_latency = sum(s.get("latency_ms", 0.0) or 0.0 for s in spans)
            col1.metric("Total Spans", len(spans))
            col2.metric("Accumulated Span Latency", f"{total_latency:.1f} ms")
            col3.metric("Trace Health", "Clean" if not any(s.get("status") == "error" for s in spans) else "Errors Detected")

            st.divider()
            tree_col, detail_col = st.columns([1.2, 1.8])

            with tree_col:
                st.subheader("Span Hierarchy")
                span_options = {}
                for s in spans:
                    indent = "    " if s.get("parent_span_id") else ""
                    prefix = "🛠️ " if "tool" in s["name"] else ("⚖️ " if "reviewer" in s["name"] else "🤖 ")
                    span_options[s["span_id"]] = f"{indent}{prefix}{s['name']} ({s.get('latency_ms', 0)}ms)"

                selected_span_id = st.radio("Select Span", list(span_options.keys()), format_func=lambda x: span_options[x])

            with detail_col:
                st.subheader("Span Attributes & Payload")
                selected_span = next((s for s in spans if s["span_id"] == selected_span_id), None)
                if selected_span:
                    st.json(selected_span)

    # -------------------------------------------------------------
    # TAB 2: Cost & Token Attribution (Subtask 15)
    # -------------------------------------------------------------
    with tab_cost:
        st.subheader("Bedrock Token & Cost Ledger")
        tracked_runs = list(cost_tracker._runs.keys())
        if not tracked_runs:
            st.info("No token cost records tracked yet.")
        else:
            selected_run = st.selectbox("Select Task Run ID", tracked_runs, key="cost_run_sel")
            summary = cost_tracker.get_run_summary(selected_run)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Run Cost", f"${summary['total_cost_usd']:.5f}")
            c2.metric("Total Tokens", f"{summary['total_tokens']:,}")
            c3.metric("Input Tokens", f"{summary['total_input_tokens']:,}")
            c4.metric("Output Tokens", f"{summary['total_output_tokens']:,}")

            st.write("#### Agent Role Breakdown")
            agent_data = []
            for role, stats in summary["by_agent"].items():
                agent_data.append({
                    "Agent Role": role,
                    "Invocations": stats["invocations"],
                    "Input Tokens": stats["input_tokens"],
                    "Output Tokens": stats["output_tokens"],
                    "Total Tokens": stats["input_tokens"] + stats["output_tokens"],
                    "Cost (USD)": f"${stats['cost_usd']:.5f}",
                })
            st.dataframe(agent_data, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: Replay & Branch Debugger (Subtask 16)
    # -------------------------------------------------------------
    with tab_replay:
        st.subheader("Historical Checkpoint Stepping & Branching")
        thread_input = st.text_input("Enter Thread ID to Inspect", value="orchestrator_session")

        if st.button("Load Checkpoint History"):
            history = replay_debugger.get_execution_history(thread_input)
            st.session_state["replay_history"] = history
            st.session_state["active_replay_thread"] = thread_input

        history = st.session_state.get("replay_history", [])
        if history:
            st.write(f"Found **{len(history)}** checkpoint states for `{st.session_state.get('active_replay_thread')}`:")
            step_labels = [f"Step {h['step_index']}: Next Node -> {h['next_node']}" for h in history]
            selected_step_idx = st.selectbox("Select Step to Inspect", range(len(history)), format_func=lambda i: step_labels[i])

            step_data = history[selected_step_idx]
            st.json({
                "step_index": step_data["step_index"],
                "checkpoint_id": step_data["checkpoint_id"],
                "next_node": step_data["next_node"],
                "values": step_data["values"],
            })

            st.divider()
            st.subheader("🔀 Branch Execution from This Step")
            mutation_key = st.text_input("State Key to Override", value="reviewer_feedback")
            mutation_val = st.text_area("Override Value", value="Fix performance metrics table format.")

            if st.button("🚀 Replay Branch with Mutation"):
                new_branch_id = f"branch_{uuid.uuid4().hex[:6]}"
                result = replay_debugger.replay_and_branch(
                    base_thread_id=st.session_state.get("active_replay_thread"),
                    fork_step_index=selected_step_idx,
                    mutated_state_updates={mutation_key: mutation_val},
                    new_thread_id=new_branch_id
                )
                st.success(f"Execution branched to `{new_branch_id}` successfully!")
                st.json(result)