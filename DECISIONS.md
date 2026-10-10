# Decisions

## Models
Using Haiku while building — it's cheap and fast, good enough for development.Switching to Sonnet later, only when I want to properly test. Saves money and time.
## AWS
Using Oregon (us-west-2) — that's where the Bedrock models I want are available.
## Running it
Keeping USE_LIVE_BEDROCK=false for now (practice mode) so I don't spend money while building.I will flip it to true when I'm ready to test for real.
## Code quality
Using ruff to keep the code neat and consistent, and pytest for tests.Turned off the nested-with rule because the longer version is easier to read.