# cli.py
import argparse
import sys
import subprocess
from main import run_orchestration_pipeline
from eval.replay import ReplayDebugger


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Multi-Agent Orchestration Platform CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available CLI commands")

    # 1. run command
    run_parser = subparsers.add_parser("run", help="Execute an autonomous multi-agent task")
    run_parser.add_argument("task", type=str, help="The prompt/goal for the orchestrator to execute")
    run_parser.add_argument("--thread-id", type=str, default=None, help="Optional custom thread ID")

    # 2. ui command
    subparsers.add_parser("ui", help="Launch the Streamlit Review & Trace Explorer Dashboard")

    # 3. history command
    hist_parser = subparsers.add_parser("history", help="Inspect checkpoint step history for a thread")
    hist_parser.add_argument("thread_id", type=str, help="Thread ID to inspect")
    hist_parser.add_argument("--db", type=str, default="orchestrator_state.db", help="Path to SQLite state database")

    args = parser.parse_args()

    if args.command == "run":
        run_orchestration_pipeline(args.task, thread_id=args.thread_id)

    elif args.command == "ui":
        print("🚀 Starting Streamlit Observability & HITL Dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ui/app.py"])

    elif args.command == "history":
        debugger = ReplayDebugger(db_path=args.db)
        steps = debugger.get_execution_history(args.thread_id)
        print(f"\nExecution history for thread: {args.thread_id} ({len(steps)} checkpoints) from [{args.db}]")
        for s in steps:
            print(f"  • Step {s['step_index']}: Next Node -> {s['next_node']} | Checkpoint: {s['checkpoint_id']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()