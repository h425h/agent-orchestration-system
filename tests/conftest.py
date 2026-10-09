"""
Test bootstrap. Runs before any project module is imported, so the module-level
singletons (Bedrock clients, ChromaDB store) are created in an offline-safe, isolated way.
"""
import os
import tempfile

os.environ.setdefault("AWS_REGION", "us-west-2")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")

# Singletons create orchestrator_chroma/ and *.db files relative to the cwd.
os.chdir(tempfile.mkdtemp(prefix="orchestrator_tests_"))

import hashlib

import chromadb
from chromadb.utils import embedding_functions


class _FakeEmbedding(chromadb.EmbeddingFunction):
    """Deterministic 16-d embedding so tests never download the default ONNX model."""

    def __init__(self):
        pass

    def __call__(self, input):
        vectors = []
        for text in input:
            digest = hashlib.sha256(text.encode()).digest()
            vectors.append([b / 255.0 for b in digest[:16]])
        return vectors

    @staticmethod
    def name() -> str:
        return "fake-test-embedding"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return _FakeEmbedding()


embedding_functions.DefaultEmbeddingFunction = _FakeEmbedding

import pytest


@pytest.fixture(autouse=True)
def _clean_shared_state():
    from agents.approval_queue import approval_queue
    from agents.bedrock_llm import usage_collector
    approval_queue.clear()
    usage_collector.drain()
    yield
    approval_queue.clear()
