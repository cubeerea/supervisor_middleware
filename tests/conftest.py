import os

# Belt and braces: tests never reach real services.
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["LANGFUSE_TRACING"] = "false"
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
