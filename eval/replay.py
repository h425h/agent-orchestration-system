# eval/replay.py
import copy
from typing import Dict, Any, List, Optional
from langgraph.graph.state import CompiledStateGraph
from memory.checkpointer import get_sqlite_checkpointer
from agents.graph import create_agent_graph
from agents.state import AgentState


class ReplayDebugger:
    """Manages stepping through checkpoint histories and branching execution from past states."""

    def __init__(self, db_path: str = "orchestrator_state.db"):
        self.db_path = db_path
        self.checkpointer = get_sqlite_checkpointer(db_path)
        if hasattr(self.checkpointer, "setup"):
            self.checkpointer.setup()
        self.app: CompiledStateGraph = create_agent_graph(checkpointer=self.checkpointer)

    def get_execution_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Retrieves chronological state checkpoints for a given thread."""
        config = {"configurable": {"thread_id": thread_id}}
        history = []
        try:
            states = list(self.app.get_state_history(config))
            for idx, state_snapshot in enumerate(reversed(states)):
                # Only include steps that have recorded state values
                if state_snapshot.values:
                    history.append({
                        "step_index": len(history),
                        "checkpoint_id": state_snapshot.config.get("configurable", {}).get("checkpoint_id", f"chk_{idx}"),
                        "next_node": state_snapshot.next,
                        "values": copy.deepcopy(state_snapshot.values),
                        "created_at": getattr(state_snapshot, "created_at", None),
                    })
        except Exception as e:
            print(f"Error reading history: {e}")
        return history

    def replay_and_branch(
        self,
        base_thread_id: str,
        fork_step_index: int,
        mutated_state_updates: Dict[str, Any],
        new_thread_id: str
    ) -> Dict[str, Any]:
        """
        Forks from a specific historical step, applies mutations,
        and runs execution on a new branched thread_id.
        """
        history = self.get_execution_history(base_thread_id)
        if not history or fork_step_index >= len(history):
            raise IndexError(f"Fork step index {fork_step_index} out of range for thread {base_thread_id}")

        base_snapshot = history[fork_step_index]
        base_values: AgentState = copy.deepcopy(base_snapshot["values"])
        base_values.update(mutated_state_updates)

        # Initialize branch thread with mutated snapshot
        branch_config = {"configurable": {"thread_id": new_thread_id}}
        
        # Seed the new branch thread with base values
        self.app.update_state(branch_config, base_values)

        final_state = copy.deepcopy(base_values)
        try:
            # Continue execution from the updated state
            for event in self.app.stream(None, config=branch_config):
                for node_name, updates in event.items():
                    final_state.update(updates)
                    break
                break
        except Exception as e:
            final_state["replay_error"] = str(e)

        return {
            "branch_thread_id": new_thread_id,
            "forked_from_step": fork_step_index,
            "mutations_applied": mutated_state_updates,
            "final_state": final_state,
        }


# Global replay engine
replay_debugger = ReplayDebugger()