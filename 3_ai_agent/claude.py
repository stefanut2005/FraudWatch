import boto3  # install with command: pip install boto3
from dotenv import load_dotenv  # install with command: pip install python-dotenv
import os


# Load environment variables from the .env file (make sure you set your Bedrock api-key in the AWS_BEARER_TOKEN_BEDROCK variable)
load_dotenv()

# Create a Bedrock Runtime client with bearer token auth
client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-west-2", # Oregon region (this is the correct AWS region for this hackathon)
    aws_session_token=os.environ['AWS_BEARER_TOKEN_BEDROCK']  # Load Bedrock api-key from environment
)

# Set the model ID, e.g., Claude 4 Sonnet.
model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"

# Start a conversation with the user message.
user_message = "Salut, ce faci"
conversation = [
    {
        "role": "user",
        "content": [{"text": user_message}],
    }
]

# Send the message to the model, using a basic inference configuration.
response = client.converse(
    modelId=model_id,
    messages=conversation,
    inferenceConfig={"maxTokens": 512, "temperature": 0.5},  # Example configuration for LLM inference
)

# Extract and print the response text.
response_text = response["output"]["message"]["content"][0]["text"]
print(response_text)