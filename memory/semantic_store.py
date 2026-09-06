# memory/semantic_store.py
import json
import time
import uuid
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

CONSOLIDATION_PROMPT = """You are an Executive Memory Consolidation Agent.
Review the following collection of related memories and consolidate them into a single, high-level strategic summary that eliminates redundancies and captures the strongest operational rules and factual discoveries.

Output pure JSON matching this schema:
{
  "consolidated_approach": "Unified high-level execution strategy synthesizing all entries.",
  "core_domain_facts": ["Deduplicated, essential facts discovered across runs"],
  "user_preferences": ["Consolidated user preference rules"]
}

Do NOT include markdown formatting or fences (```json). Output pure JSON only.
"""


class SemanticMemoryStore:
    """
    Manages persistent long-term semantic embeddings, retrieval-augmented recall,
    importance tracking, consolidation, and TTL cleanup via ChromaDB.
    """

    def __init__(self, persist_directory: str = "orchestrator_chroma"):
        # #1. Initialize local persistent Chroma client
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)

        # #2. Local default embeddings (runs offline with zero token cost)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # #3. Get or create the persistent collection
        self.collection = self.client.get_or_create_collection(
            name="task_memories",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def distill_and_store(self, task: str, final_state: Dict[str, Any], user_id: str = "default_user") -> Dict[str, Any]:
        """Distills completed execution context into structured insights and persists them."""
        plan = final_state.get("plan", {})
        completed_subtasks = final_state.get("completed_subtasks", [])

        # #1. Build context for the distillation LLM
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
                "user_preferences": [],
            }

        # #3. Prepare document and metadata schema with importance counters and timestamps
        document_text = (
            f"Task: {task}\n"
            f"Approach: {distilled.get('successful_approach', '')}\n"
            f"Facts: {'; '.join(distilled.get('domain_facts', []))}"
        )

        memory_id = f"mem_{uuid.uuid4().hex[:8]}"
        now_ts = float(time.time())

        # #4. Save into ChromaDB
        self.collection.add(
            ids=[memory_id],
            documents=[document_text],
            metadatas=[{
                "memory_id": memory_id,
                "user_id": user_id,
                "task": task[:200],
                "summary": distilled.get("summary", ""),
                "tools_used": ",".join(distilled.get("tools_used", [])),
                "access_count": 1,
                "importance_score": 1.0,
                "created_at": now_ts,
                "last_accessed_at": now_ts,
            }],
        )

        return distilled

    def recall_similar_memories(self, query: str, user_id: str = "default_user", limit: int = 3) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB by semantic similarity and increments the access count
        and importance score for retrieved memories.
        """
        # #1. Query ChromaDB filtered by user_id
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where={"user_id": user_id} if user_id else None,
        )

        memories = []
        if results and results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                mem_id = results["ids"][0][i]

                # #2. Increment access count and importance score on access
                current_accesses = meta.get("access_count", 1) + 1
                updated_importance = round(1.0 + (current_accesses * 0.5), 2)
                now_ts = float(time.time())

                meta["access_count"] = current_accesses
                meta["importance_score"] = updated_importance
                meta["last_accessed_at"] = now_ts

                # #3. Update metadata directly in ChromaDB
                self.collection.update(
                    ids=[mem_id],
                    metadatas=[meta],
                )

                memories.append({
                    "id": mem_id,
                    "content": doc,
                    "metadata": meta,
                })

        return memories

    def consolidate_memories(self, user_id: str = "default_user") -> Optional[str]:
        """
        Consolidates stored memories for a user into a single unified summary
        when multiple memory records exist.
        """
        # #1. Retrieve all stored records for user
        all_records = self.collection.get(where={"user_id": user_id})
        if not all_records or len(all_records["ids"]) < 2:
            return None

        # #2. Collate contents to pass to the consolidation agent
        combined_docs = "\n---\n".join(all_records["documents"])
        messages = [{"role": "user", "content": f"Consolidate these memories:\n\n{combined_docs}"}]

        response = llm.invoke(messages, system_prompt=CONSOLIDATION_PROMPT, max_tokens=1000)

        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            consolidated_data = json.loads(cleaned)
        except Exception:
            consolidated_data = {
                "consolidated_approach": "Aggregated execution strategies across prior runs.",
                "core_domain_facts": [],
                "user_preferences": [],
            }

        # #3. Store the consolidated parent memory
        new_id = f"consolidated_{uuid.uuid4().hex[:6]}"
        new_doc = (
            f"Consolidated Strategy: {consolidated_data.get('consolidated_approach')}\n"
            f"Core Facts: {'; '.join(consolidated_data.get('core_domain_facts', []))}\n"
            f"Preferences: {'; '.join(consolidated_data.get('user_preferences', []))}"
        )

        now_ts = float(time.time())
        self.collection.add(
            ids=[new_id],
            documents=[new_doc],
            metadatas=[{
                "memory_id": new_id,
                "user_id": user_id,
                "task": f"Consolidated summary of {len(all_records['ids'])} past tasks",
                "summary": "Master consolidated knowledge artifact",
                "tools_used": "multi-agent-system",
                "access_count": 1,
                "importance_score": 5.0,
                "created_at": now_ts,
                "last_accessed_at": now_ts,
            }],
        )

        return new_id

    def expire_stale_memories(self, max_age_seconds: float = 604800, min_importance: float = 2.0) -> int:
        """
        Removes memories older than max_age_seconds (default 7 days)
        unless their importance score meets or exceeds min_importance.
        """
        # #1. Fetch all items
        all_memories = self.collection.get()
        if not all_memories or not all_memories["ids"]:
            return 0

        now_ts = float(time.time())
        ids_to_purge = []

        # #2. Identify records exceeding age and under importance threshold
        for i, meta in enumerate(all_memories["metadatas"]):
            created_at = meta.get("created_at", now_ts)
            importance = meta.get("importance_score", 1.0)
            age = now_ts - created_at

            if age > max_age_seconds and importance < min_importance:
                ids_to_purge.append(all_memories["ids"][i])

        # #3. Delete identified items
        if ids_to_purge:
            self.collection.delete(ids=ids_to_purge)

        return len(ids_to_purge)

    def get_memory_dashboard(self, user_id: str = "default_user") -> List[Dict[str, Any]]:
        """Returns all memories stored for a given user for dashboard inspection."""
        records = self.collection.get(where={"user_id": user_id})
        dashboard = []

        if records and records["ids"]:
            for i in range(len(records["ids"])):
                dashboard.append({
                    "id": records["ids"][i],
                    "document": records["documents"][i],
                    "metadata": records["metadatas"][i],
                })

        return dashboard

    def delete_user_memories(self, user_id: str) -> int:
        """Privacy and user-request deletion endpoint."""
        records = self.collection.get(where={"user_id": user_id})
        count = len(records["ids"]) if records and records["ids"] else 0

        if count > 0:
            self.collection.delete(where={"user_id": user_id})

        return count


# Singleton memory instance
semantic_memory = SemanticMemoryStore()