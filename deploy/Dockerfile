FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# For live Bedrock calls, pass AWS credentials + region as environment
# variables (or an attached IAM role when running on AgentCore Runtime/ECS):
#   USE_LIVE_BEDROCK=true
#   AWS_REGION=us-west-2
#   BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0

CMD ["python3", "main.py"]
