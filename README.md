# FRIA_MCP_A2A_RAG
Multi-agent fraud investigation system using A2A protocol, MCP, and RAG — built on Amazon Bedrock architecture with LangGraph orchestration.
## How to run

Make the sandbox (keeps this project's parts separate from the rest of my laptop)
py -V:3.12 -m venv .venv

Switch it on (so everything I install goes inside that sandbox)
.venv\Scripts\activate

Install the pieces (all the tools the project needs)
pip install -r requirements.txt

Set up the .env file (tells the project which AWS account, region, and model to use)
AWS_PROFILE=fraud-agent
AWS_REGION=us-west-2
USE_LIVE_BEDROCK=false
BEDROCK_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0

Run it (starts the project and works through the fraud scenarios)
python main.py