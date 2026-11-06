"""An isolated, offline Langfuse client for tests: spans go to an in-memory exporter, API calls (scores) to a
mock HTTP transport. A unique public key per client avoids Langfuse's per-key singleton."""
from __future__ import annotations

import json
import uuid

import httpx
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from supervisor_middleware.tracing import Tracer


class FakeLangfuse:
    def __init__(self):
        self.exporter = InMemorySpanExporter()
        self.requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, json={"successes": [], "errors": []})

        self.client = Langfuse(public_key=f"pk-lf-test-{uuid.uuid4().hex}", secret_key="sk-lf-test",
                               host="http://langfuse.invalid", tracer_provider=TracerProvider(),
                               span_exporter=self.exporter, httpx_client=httpx.Client(transport=httpx.MockTransport(
                                   handler)), flush_at=1, flush_interval=0.1)
        self.tracer = Tracer(self.client)

    def spans(self):
        self.client.flush()
        return list(self.exporter.get_finished_spans())

    def scores(self) -> list[dict]:
        """Score bodies POSTed to /api/public/scores (a JSON list per request in Langfuse 5)."""
        self.client.flush()
        out: list[dict] = []
        for r in self.requests:
            if not str(r.url).endswith("/api/public/scores"):
                continue
            try:
                body = json.loads(r.content or b"[]")
            except ValueError:
                continue
            out += body if isinstance(body, list) else [body]
        return out

    def close(self):
        self.client.shutdown()
