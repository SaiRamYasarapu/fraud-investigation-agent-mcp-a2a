FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# For live Bedrock calls, pass AWS credentials + region as environment
# variables (or an attached IAM role when running on AgentCore Runtime/ECS):
#   USE_LIVE_BEDROCK=true
#   AWS_REGION=us-east-1
#   BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0

CMD ["python3", "main.py"]
