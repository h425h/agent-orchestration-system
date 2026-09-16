# ui/trace_explorer.py
import streamlit as st
import time
from eval.tracer import tracer, SpanRecord


def render_trace_explorer():
    st.title("🔍 Multi-Agent Execution Trace Explorer")
    st.caption("Inspect hierarchical execution trees, tool call latencies, and agent reasoning traces.")

    # Sidebar: Seed mock traces if empty
    with st.sidebar:
        st.subheader("Trace Controls")
        if st.button("➕ Generate Sample Execution Trace"):
            sample_id = f"trace_{int(time.time())}"
            # 1. Supervisor
            s1 = tracer.start_span(sample_id, "supervisor_planning", agent_role="supervisor", attributes={"task": "Benchmark vector databases 2026"})
            time.sleep(0.04)
            tracer.end_span(s1.span_id, status="success", attributes={"subtasks_created": 3})

            # 2. Researcher
            s2 = tracer.start_span(sample_id, "execute_researcher", agent_role="researcher", parent_span_id=s1.span_id)
            time.sleep(0.02)
            t1 = tracer.start_span(sample_id, "tool_web_search", agent_role="researcher", parent_span_id=s2.span_id, attributes={"query": "vector databases 2026 benchmarks"})
            time.sleep(0.03)
            tracer.end_span(t1.span_id, status="success", attributes={"results_returned": 5})
            tracer.end_span(s2.span_id, status="success", attributes={"output_length": 512})

            # 3. Reviewer
            s3 = tracer.start_span(sample_id, "reviewer_evaluation", agent_role="reviewer", parent_span_id=s2.span_id)
            time.sleep(0.02)
            tracer.end_span(s3.span_id, status="success", attributes={"score": 9.0, "verdict": "approved"})

            st.success(f"Generated trace: {sample_id}")
            st.rerun()

    # Load trace IDs
    trace_ids = list(tracer._spans.keys())
    if not trace_ids:
        st.info("No execution traces found. Click 'Generate Sample Execution Trace' in the sidebar or run an orchestration task.")
        return

    selected_trace_id = st.selectbox("Select Execution Trace ID", trace_ids, index=len(trace_ids) - 1)
    spans = tracer.get_trace_tree(selected_trace_id)

    if not spans:
        st.warning("Selected trace contains no spans.")
        return

    # Top-level Trace Metrics
    total_spans = len(spans)
    total_latency = sum(s.get("latency_ms", 0.0) or 0.0 for s in spans)
    has_errors = any(s.get("status") == "error" for s in spans)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spans", total_spans)
    col2.metric("Accumulated Span Latency", f"{total_latency:.1f} ms")
    col3.metric("Trace Health", "Clean" if not has_errors else "Errors Detected")

    st.divider()

    # Layout: Tree on left, Node detail on right
    tree_col, detail_col = st.columns([1.2, 1.8])

    with tree_col:
        st.subheader("Span Hierarchy")
        span_options = {}
        for s in spans:
            indent = "    " if s.get("parent_span_id") else ""
            prefix = "🛠️ " if "tool" in s["name"] else ("⚖️ " if "reviewer" in s["name"] else "🤖 ")
            label = f"{indent}{prefix}{s['name']} ({s.get('latency_ms', 0)}ms)"
            span_options[s["span_id"]] = label

        selected_span_id = st.radio(
            "Select Span to Inspect",
            options=list(span_options.keys()),
            format_func=lambda x: span_options[x]
        )

    # Detailed node inspection
    with detail_col:
        st.subheader("Span Attributes & Payload")
        selected_span = next((s for s in spans if s["span_id"] == selected_span_id), None)
        if selected_span:
            st.json({
                "span_id": selected_span["span_id"],
                "parent_span_id": selected_span["parent_span_id"],
                "name": selected_span["name"],
                "agent_role": selected_span["agent_role"],
                "status": selected_span["status"],
                "latency_ms": selected_span["latency_ms"],
                "attributes": selected_span["attributes"],
                "events": selected_span["events"],
            })