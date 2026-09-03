# agents/bedrock_llm.py
import os
import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

class BedrockLLM:
    """Wrapper around AWS Bedrock Runtime Converse API for Claude models."""

    def __init__(self, model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"):
        self.region = os.getenv("AWS_REGION", "us-west-2")
        self.bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "").strip().strip('"').strip("'")
        self.model_id = model_id

        if self.bearer_token:
            os.environ["AWS_BEDROCK_API_KEY"] = self.bearer_token

        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region,
            config=Config(retries={"max_attempts": 3, "mode": "standard"})
        )

        def add_bearer_auth(request, **kwargs):
            if self.bearer_token:
                auth_val = (
                    self.bearer_token
                    if self.bearer_token.lower().startswith("bearer ")
                    else f"Bearer {self.bearer_token}"
                )
                request.headers["Authorization"] = auth_val

        self.client.meta.events.register("request-created.bedrock-runtime.*", add_bearer_auth)

    def invoke(
        self,
        messages: list[dict],
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048
    ) -> str:
        """
        Sends formatted messages to the Converse API.
        messages format: [{"role": "user"|"assistant", "content": "text string"}]
        """
        converse_messages = [
            {
                "role": msg["role"],
                "content": [{"text": msg["content"]}]
            }
            for msg in messages
        ]

        kwargs = {
            "modelId": self.model_id,
            "messages": converse_messages,
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature
            }
        }
        if system_prompt:
            kwargs["system"] = [{"text": system_prompt}]

        response = self.client.converse(**kwargs)
        return response["output"]["message"]["content"][0]["text"]

# Singleton instance ready for import across agent nodes
llm = BedrockLLM()