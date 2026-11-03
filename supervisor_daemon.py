"""Entrypoint named in PROJECT_PLAN.md: `python supervisor_daemon.py [--backend ollama|rules] ...`"""
from supervisor_middleware.supervisor.daemon import main

if __name__ == "__main__":
    raise SystemExit(main())
