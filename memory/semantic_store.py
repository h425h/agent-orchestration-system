# memory/semantic_store.py
import json
import os
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.utils import embedding_functions

from agents.bedrock_llm import llm


DISTILLATION_PROMPT = """You are a Memory Distillation Agent in an autonomous AI orchestration system.
Analyze the following completed task execution and extract key learnings to store in long-term semantic memory.

Your extraction must produce a valid JSON object matching this schema:
{
  "summary": "1-2 sentence high-level overview of the task and final deliverable.",
  "successful_approach": "Specific workflow strategy, ordering, and specialist choices that worked well.",
  "tools_used": ["tool_a", "tool_b"],
  "domain_facts": ["Specific fact, formula, benchmark finding, or discovery from this run"],
  "user_preferences": ["Formatting, style, or explicit user constraints observed"]
}

Do NOT include markdown formatting or fences (```json). Output pure JSON only.
"""

class SemanticMemoryStore:
    """Manages persistent long-term semantic embeddings and memory recall via ChromaDB."""

    def __init__(self, persist_directory: str = "orchestrator_chroma"):
        # #1. Initialize local persistent Chroma client
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        # #2. Use local default embeddings (runs fully offline, zero token cost)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # #3. Get or create the primary memories collection
        self.collection = self.client.get_or_create_collection(
            name="task_memories",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def distill_and_store(self, task: str, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Distills completed execution context into structured insights
        and saves them into the vector database.
        """
        plan = final_state.get("plan", {})
        completed_subtasks = final_state.get("completed_subtasks", [])

        # #1. Format run summary for the distillation LLM
        run_context = (
            f"User Task: {task}\n"
            f"Execution Plan: {json.dumps(plan, indent=2)}\n"
            f"Completed Steps Count: {len(completed_subtasks)}\n"
            f"Final Deliverable Preview: {str(final_state.get('final_output', ''))[:1000]}"
        )

        messages = [
            {"role": "user", "content": f"Extract memory from this completed run:\n\n{run_context}"}
        ]

        raw_distillation = llm.invoke(messages, system_prompt=DISTILLATION_PROMPT, max_tokens=1000)

        # #2. Clean JSON formatting defensively
        cleaned = raw_distillation.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            distilled = json.loads(cleaned)
        except Exception:
            distilled = {
                "summary": f"Completed task: {task[:60]}",
                "successful_approach": "Multi-agent sequential decomposition",
                "tools_used": ["web_search", "run_python_code"],
                "domain_facts": [],
                "user_preferences": []
            }

        # #3. Build rich search document and associated metadata
        document_text = (
            f"Task: {task}\n"
            f"Approach: {distilled.get('successful_approach', '')}\n"
            f"Facts: {'; '.join(distilled.get('domain_facts', []))}"
        )

        import uuid
        memory_id = f"mem_{uuid.uuid4().hex[:8]}"

        # #4. Insert document into ChromaDB
        self.collection.add(
            ids=[memory_id],
            documents=[document_text],
            metadatas=[{
                "task": task[:200],
                "summary": distilled.get("summary", ""),
                "tools_used": ",".join(distilled.get("tools_used", [])),
                "access_count": 1
            }]
        )

        return distilled

    def recall_similar_memories(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Queries the vector database for prior task learnings relevant to the new query."""
        # #1. Query ChromaDB by semantic similarity
        results = self.collection.query(
            query_texts=[query],
            n_results=limit
        )

        memories = []
        if results and results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                mem_id = results["ids"][0][i]
                memories.append({
                    "id": mem_id,
                    "content": doc,
                    "metadata": meta
                })

        return memories


# Singleton memory instance
semantic_memory = SemanticMemoryStore()