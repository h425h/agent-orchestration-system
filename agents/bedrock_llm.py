# agents/bedrock_llm.py
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import List

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMCall:
    """Measured usage for one Bedrock Converse call."""
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: float      # wall clock for the call, including client-side retries
    stop_reason: str = ""  # "max_tokens" means the output was truncated


class UsageCollector:
    """
    Thread-safe buffer of real LLM usage. Every BedrockLLM.invoke() appends here;
    the pipeline drains it after each graph node to attribute tokens, cost and latency.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._calls: List[LLMCall] = []

    def record(self, call: LLMCall) -> None:
        with self._lock:
            self._calls.append(call)

    def drain(self) -> List[LLMCall]:
        with self._lock:
            calls, self._calls = self._calls, []
        return calls


usage_collector = UsageCollector()


class BedrockLLM:
    def __init__(self, model_id: str):
        self.region = os.getenv("AWS_REGION", "us-west-2")
        self.bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
        self.model_id = model_id

        session = boto3.Session()
        self.client = session.client(
            service_name="bedrock-runtime",
            region_name=self.region,
            config=Config(retries={"max_attempts": 5, "mode": "standard"})
        )

        if self.bearer_token:
            def add_bearer_token(request, **kwargs):
                request.headers["Authorization"] = f"Bearer {self.bearer_token}"
            self.client.meta.events.register("before-send.bedrock-runtime.*", add_bearer_token)

    def invoke(self, messages: list, system_prompt: str = None, max_tokens: int = 2048) -> str:
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg["role"],
                "content": [{"text": msg["content"]}]
            })

        kwargs = {
            "modelId": self.model_id,
            "messages": formatted_messages,
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.2}
        }
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        start = time.perf_counter()
        response = self.client.converse(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        usage = response.get("usage", {}) or {}
        stop_reason = response.get("stopReason", "")
        if stop_reason == "max_tokens":
            logger.warning("Response from %s truncated at max_tokens=%s", self.model_id, max_tokens)
        usage_collector.record(LLMCall(
            model_id=self.model_id,
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
            latency_ms=round(latency_ms, 1),
            stop_reason=stop_reason,
        ))

        return response["output"]["message"]["content"][0]["text"]

# Specialized Model Instances
llm_haiku = BedrockLLM(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")
llm_sonnet = BedrockLLM(model_id="us.anthropic.claude-sonnet-4-6")

# Default fallback
llm = llm_haiku
