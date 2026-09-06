# test_semantic_memory.py
from memory.semantic_store import semantic_memory

def test_store_and_recall():
    print("Testing Long-Term Semantic Memory Distillation...")

    # #1. Simulate a completed run state
    mock_final_state = {
        "plan": {
            "subtasks": [
                {"id": "task_1", "specialist": "researcher"},
                {"id": "task_2", "specialist": "coder"}
            ]
        },
        "completed_subtasks": [
            {"id": "task_1", "result": "Qdrant and Pinecone lead in sub-10ms latency."},
            {"id": "task_2", "result": "Vector dimensions have linear impact O(n) on dot product speed."}
        ],
        "final_output": "Executive report: Qdrant delivers 5-12ms p99 with fast indexing."
    }

    task_description = "Research 2026 vector database performance and run vector math benchmarks."

    # #2. Distill and store
    insights = semantic_memory.distill_and_store(task_description, mock_final_state)
    print("\n--- Extracted Insights ---")
    for key, val in insights.items():
        print(f"{key}: {val}")

    # #3. Recall using a semantic query
    print("\nTesting Semantic Retrieval...")
    query = "What vector database is best for fast indexing and low latency?"
    recalled = semantic_memory.recall_similar_memories(query, limit=1)

    print(f"\nQuery: '{query}'")
    print(f"Recalled {len(recalled)} relevant past memory:")
    for mem in recalled:
        print(f"\n[ID: {mem['id']}]")
        print(mem["content"])

if __name__ == "__main__":
    test_store_and_recall()