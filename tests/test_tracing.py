"""Distributed tracing: the supervisor's run is stitched under the executor's generation run."""
import json
from dataclasses import replace
from unittest import mock

import fakeredis
import requests
from langsmith import Client, run_helpers

from supervisor_middleware.config import Settings
from supervisor_middleware.executor.run import run_episode
from supervisor_middleware.executor.scripted import ScriptedSource
from supervisor_middleware.redis_bus import AsyncRedisBus
from supervisor_middleware.sandbox_runner import prepare_workdir
from supervisor_middleware.supervisor.daemon import SupervisorDaemon
from supervisor_middleware.supervisor.judge import RuleJudge
from supervisor_middleware.telemetry import EventLog


class _FakeSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.requests = []

    def request(self, method, url, *a, **kw):
        self.requests.append((method, url, kw))
        resp = mock.Mock(status_code=200, text="{}", headers={})
        resp.json.return_value = {}
        resp.raise_for_status.return_value = None
        return resp


async def test_supervisor_run_is_child_of_executor_run(tmp_path):
    import asyncio

    session = _FakeSession()
    client = Client(api_url="http://langsmith.invalid", api_key="fake", session=session, auto_batch_tracing=False,
                    info={})
    settings = replace(Settings(), tracing=True, events_path=tmp_path / "ev.jsonl", max_steps=8)
    events = EventLog(settings.events_path)
    server = fakeredis.FakeServer()

    def bus():
        return AsyncRedisBus(fakeredis.FakeAsyncRedis(server=server, decode_responses=True))

    sup_bus, ex_bus = bus(), bus()
    daemon = SupervisorDaemon(sup_bus, RuleJudge(), events, settings, tracing=True)
    with run_helpers.tracing_context(enabled=True, client=client):
        task = asyncio.create_task(daemon.run())
        await asyncio.wait_for(daemon.ready.wait(), 5)
        wd = prepare_workdir(tmp_path / "wd", rows=60)
        r = await run_episode(settings, source=ScriptedSource("explicit", chars_per_s=500, first_token_s=0.01),
                              bus=ex_bus, events=events, workdir=wd)
        await daemon.drain(5)
        task.cancel()
    client.flush()
    assert r.halts and r.outcome == "finished"

    runs: dict[str, dict] = {}
    for _method, url, kw in session.requests:
        data = kw.get("data")
        if not isinstance(data, (bytes, str)) or "/runs" not in url:
            continue
        body = json.loads(data)
        rid = body.get("id") or url.rstrip("/").split("/")[-1]
        runs.setdefault(rid, {}).update({k: v for k, v in body.items() if v is not None})
    gens = {k: v for k, v in runs.items() if v.get("name") == "executor_generate"}
    evals = [v for v in runs.values() if v.get("name") == "supervisor_evaluate"]
    assert gens and evals
    for e in evals:
        assert e["parent_run_id"] in gens, "supervisor_evaluate must be a child of an executor_generate run"
    halted = [g for g in gens.values() if g.get("extra", {}).get("metadata", {}).get("intercepted")]
    assert len(halted) == 1
    md = halted[0]["extra"]["metadata"]
    assert md["cancel_latency_ms"] >= 0 and md["intended_violation"] is True
