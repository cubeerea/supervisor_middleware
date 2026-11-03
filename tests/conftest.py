import os

# Belt and braces: tests never reach real services.
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_API_KEY"] = ""
