# test_bedrock.py
# --------------------------------------------------------------------------------------
# Purpose: Verify end-to-end communication with AWS Bedrock Converse API using Claude.
# Validates:
#   1. Loading environment variables (.env)
#   2. Initializing boto3 bedrock-runtime client in the target region (us-west-2)
#   3. Attaching the AWS Bedrock Bearer Token to outgoing requests via botocore hooks
#   4. Executing an inference call against the Haiku 4.5 cross-region inference profile
# --------------------------------------------------------------------------------------

import os
import boto3
from botocore.config import Config
from dotenv import load_dotenv

# Step 1: Load environment variables from local .env file
load_dotenv()

# Step 2: Read configuration values
# AWS Bedrock cross-region inference profile for us-west-2
region = os.getenv("AWS_REGION", "us-west-2")

# Read the raw bearer token/API key and strip accidental whitespace or quotes
raw_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "").strip().strip('"').strip("'")

# Set the standard Bedrock environment variable fallback for boto3
if raw_token:
    os.environ["AWS_BEDROCK_API_KEY"] = raw_token

# Step 3: Instantiate the Bedrock Runtime client
# The Config object configures automatic retries using AWS standard backoff strategy
client = boto3.client(
    service_name="bedrock-runtime",
    region_name=region,
    config=Config(retries={"max_attempts": 3, "mode": "standard"}),
)

# Step 4: Register an event hook to attach the Bearer token to request headers
# In boto3/botocore, request.headers is a case-insensitive dictionary-like object.
# Direct key indexing (request.headers["Authorization"]) is required because
# botocore HTTPHeaders does not implement an .update() method.
def add_bearer_auth(request, **kwargs):
    if raw_token:
        # Prepend 'Bearer ' if it is not already included in the token string
        auth_val = (
            raw_token
            if raw_token.lower().startswith("bearer ")
            else f"Bearer {raw_token}"
        )
        request.headers["Authorization"] = auth_val

# Attach the handler to the bedrock-runtime request lifecycle
client.meta.events.register(
    "request-created.bedrock-runtime.*",
    add_bearer_auth,
)

# Step 5: Specify the Inference Profile ID
# On-demand calls to Claude Haiku 4.5 require the US cross-region inference profile ('us.' prefix)
model_id = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Step 6: Construct the Converse API message payload
# AWS Bedrock Converse API enforces a strict typed structure:
# [{"role": "user"|"assistant", "content": [{"text": "..."}]}]
messages = [
    {
        "role": "user",
        "content": [{"text": "Respond with 'Bedrock is online' and nothing else."}],
    }
]

# Step 7: Send the request and parse the output
try:
    # client.converse() sends the request payload to the model
    response = client.converse(
        modelId=model_id,
        messages=messages,
        # inferenceConfig controls token budget and deterministic sampling (temperature=0.0)
        inferenceConfig={"maxTokens": 50, "temperature": 0.0},
    )

    # Extract the assistant's reply text from the structured response payload:
    # response -> 'output' -> 'message' -> 'content' -> list of blocks -> first block -> 'text'
    output_text = response["output"]["message"]["content"][0]["text"]
    print(f"\nSuccess! Bedrock Response: {output_text}\n")

except Exception as error:
    # Print runtime exceptions (AccessDenied, ValidationException, network drops, etc.)
    print(f"\nBedrock Invocation Error: {error}\n")