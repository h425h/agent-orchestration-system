# agents/bedrock_llm.py
import os
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

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

        response = self.client.converse(**kwargs)
        return response["output"]["message"]["content"][0]["text"]

# Specialized Model Instances
llm_haiku = BedrockLLM(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0")
llm_sonnet = BedrockLLM(model_id="us.anthropic.claude-sonnet-4-6")

# Default fallback
llm = llm_haiku