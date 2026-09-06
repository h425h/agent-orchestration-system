import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

def get_sqlite_checkpointer(db_path: str = "orchestrator_state.db") -> SqliteSaver:
    """Returns a SQLite-backed checkpointer for persistent task state."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)