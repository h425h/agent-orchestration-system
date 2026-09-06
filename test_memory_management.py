# test_memory_management.py
from memory.semantic_store import semantic_memory

def run_memory_management_test():
    user = "test_developer"
    print(f"--- 1. Testing Memory Dashboard for [{user}] ---")

    # #1. Distill two sample runs to give consolidation material
    mock_run_1 = {
        "plan": {"subtasks": [{"id": "t1", "specialist": "coder"}]},
        "completed_subtasks": [{"id": "t1", "result": "HNSW index reduces search to O(log n)."}],
        "final_output": "Benchmark proves HNSW logarithmic scaling."
    }
    semantic_memory.distill_and_store("Benchmark HNSW indexing algorithm in 2026", mock_run_1, user_id=user)

    mock_run_2 = {
        "plan": {"subtasks": [{"id": "t2", "specialist": "researcher"}]},
        "completed_subtasks": [{"id": "t2", "result": "Pinecone serverless pricing offers high cost efficiency."}],
        "final_output": "Pinecone serverless delivers 40% cost reduction."
    }
    semantic_memory.distill_and_store("Evaluate Pinecone 2026 pricing models", mock_run_2, user_id=user)

    # #2. Check dashboard
    dashboard = semantic_memory.get_memory_dashboard(user_id=user)
    print(f"Total memories in dashboard for {user}: {len(dashboard)}")
    for item in dashboard:
        print(f"  - [{item['id']}] Access Count: {item['metadata'].get('access_count')} | Importance: {item['metadata'].get('importance_score')}")

    # #3. Query memory and verify access count increments
    print(f"\n--- 2. Testing Importance Scoring via Semantic Recall ---")
    query = "How does HNSW index scale?"
    recalled = semantic_memory.recall_similar_memories(query, user_id=user, limit=1)
    if recalled:
        mem = recalled[0]
        print(f"Queried: '{query}' -> Recalled: [{mem['id']}]")
        print(f"Updated Access Count: {mem['metadata']['access_count']}")
        print(f"Updated Importance Score: {mem['metadata']['importance_score']}")

    # #4. Test memory consolidation
    print(f"\n--- 3. Testing Memory Consolidation ---")
    consolidated_id = semantic_memory.consolidate_memories(user_id=user)
    print(f"Consolidation complete! Generated Master Memory ID: {consolidated_id}")

    # #5. Test user deletion endpoint
    print(f"\n--- 4. Testing User Data Deletion Endpoint ---")
    deleted_count = semantic_memory.delete_user_memories(user_id=user)
    print(f"Deleted {deleted_count} memories for user [{user}].")

    remaining = semantic_memory.get_memory_dashboard(user_id=user)
    print(f"Remaining memories for user [{user}]: {len(remaining)}")

if __name__ == "__main__":
    run_memory_management_test()