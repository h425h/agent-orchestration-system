# test_benchmark.py
from eval.benchmark import evaluator


def test_evaluator():
    print("Testing Subtask 18: Automated Benchmark & LLM Judge...")

    mock_pipeline_result = {
        "thread_id": "bench_test_001",
        "final_state": {
            "plan": {
                "subtasks": [
                    {"id": "t1", "specialist": "researcher", "description": "Find vector databases"},
                    {"id": "t2", "specialist": "coder", "description": "Benchmark latency", "dependencies": ["t1"]},
                    {"id": "t3", "specialist": "writer", "description": "Synthesize summary", "dependencies": ["t2"]},
                ]
            },
            "completed_subtasks": [
                {"id": "t1", "specialist": "researcher", "result": "Qdrant and Pinecone lead in p99 latency."},
                {"id": "t2", "specialist": "coder", "result": "Benchmark finished: 12ms p99 at 10k vectors."},
                {"id": "t3", "specialist": "writer", "result": "Executive Summary: Recommended vector database is Qdrant."},
            ],
            "final_output": "Executive Summary: Recommended vector database is Qdrant due to 12ms p99 latency."
        },
        "cost_summary": {
            "total_cost_usd": 0.0245,
            "total_tokens": 12400
        }
    }

    result = evaluator.evaluate_run(
        task_prompt="Benchmark vector database options for 2026.",
        pipeline_result=mock_pipeline_result
    )

    print(f"  [Pass] Composite Benchmark Score: {result['composite_score']}/10.0")
    print(f"  [Pass] Metrics Breakdown: {result['metrics']}")
    print(f"  [Pass] Benchmark Passed: {result['passed_benchmark']}")
    assert result["composite_score"] >= 6.0
    assert "reasoning" in result["metrics"]

    print("\nBenchmark Suite (Subtask 18) Verified Successfully!")


if __name__ == "__main__":
    test_evaluator()