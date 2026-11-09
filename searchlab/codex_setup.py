from __future__ import annotations

import datetime
import json
import os
import tomllib
import uuid
from pathlib import Path

from .common import ROOT, digest, save
from .datasets import validate_splits
from .gateway import BudgetStop, Gateway, Ledger


def initialize(output, model=None, benchmarks=("gsm8k",)):
    output = Path(output)
    if output.exists():
        raise ValueError("Configuration already exists; edit it or choose another --output path")
    if not model:
        home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
        path = home / "config.toml"
        local = tomllib.loads(path.read_text()) if path.exists() else {}
        model = local.get("model")
    if not model:
        raise ValueError("Specify --model or select a model in your local Codex configuration")
    config = json.loads((ROOT / "configs/live.example.json").read_text())
    config["seeds"] = [11]
    config["datasets"] = []
    for name in benchmarks:
        path = ROOT / "data" / (name + ".json")
        if not path.exists():
            raise ValueError(f"Prepare {name} first using the prepare command")
        source = json.loads(path.read_text())
        original = source.pop("sha256")
        if digest(source) != original or source.get("fixture"):
            raise ValueError("Pilot requires a valid real dataset, not fixtures")
        source["source_sha256"] = original
        for split, size in (("train", 1), ("validation", 2), ("test", 3)):
            source[split] = source[split][:size]
        source["sha256"] = digest(source)
        validate_splits(source)
        target = ROOT / "data/codex-pilot" / f"{name}-{source['sha256'][:12]}.json"
        save(target, source)
        config["datasets"].append(str(target.relative_to(ROOT)))
    for role in ("search", "execution"):
        config["models"][role] = {"provider": "codex", "binary": "codex", "model": model,
                                   "reasoning_effort": "low", "timeout_seconds": 120, "max_prompt_bytes": 200000,
                                   "input_usd_per_million": None, "output_usd_per_million": None}
    config["budgets"] = {
        "search": {"max_calls": 80, "max_tokens": 500000, "max_usd": None, "max_seconds": 1200},
        "evaluation": {"max_calls": 120, "max_tokens": 500000, "max_usd": None, "max_seconds": 1800}}
    config["search"]["iterations"] = 1
    config["execution"]["task_timeout_seconds"] = 600
    save(output, config)
    return config


def connection_check(config, output=None):
    if config["fixture"] or any(m.get("provider") != "codex" for m in config["models"].values()):
        raise ValueError("check-codex requires a real Codex configuration")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(output) if output else ROOT / "runs" / ("codex-connection-" + stamp + "-" + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    ledger = Ledger({"max_calls": 3, "max_tokens": 100000, "max_seconds": 240, "max_usd": None}, directory / "calls.jsonl")
    gateway = Gateway(config["models"], ledger, 11)
    results = []
    status = "failed"
    try:
        response = gateway.call([{"role": "user", "content": "Return exactly SEARCH_COMPARE_OK."}], role="search")
        if response["content"].strip() != "SEARCH_COMPARE_OK":
            raise ValueError("Search-role text check failed")
        results.append({"check": "search_role", "passed": True})
        response = gateway.call([{"role": "user", "content": "What is 7 + 5? Put only the decimal integer in the answer field."}], fields=["answer"])
        if str(json.loads(response["content"])["answer"]).strip() != "12":
            raise ValueError("Execution-role structured answer check failed")
        results.append({"check": "execution_role_json", "passed": True})
        tools = [{"type": "function", "function": {"name": "add_numbers", "description": "Add two integers", "parameters": {
            "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"], "additionalProperties": False}}}]
        response = gateway.call([{"role": "user", "content": "Call add_numbers with a=7 and b=5. Do not answer directly."}], tools=tools)
        calls = response.get("tool_calls", [])
        if len(calls) != 1 or calls[0]["function"]["name"] != "add_numbers" or json.loads(calls[0]["function"]["arguments"]) != {"a": 7, "b": 5}:
            raise ValueError("Benchmark-tool relay check failed")
        results.append({"check": "benchmark_tool_request", "passed": True})
        status = "complete"
    except (Exception, BudgetStop) as exc:
        results.append({"check": "connection", "passed": False, "error": str(exc)})
    save(directory / "connection.json", {"kind": "real_codex_connection_check", "status": status,
         "research_comparison": False, "models": config["models"], "checks": results, "budget": ledger.snapshot()})
    (directory / "report.md").write_text(
        "# Real Codex connection check\n\n" +
        "This checks real model responses and metering. It is not an AFlow-versus-ADAS benchmark.\n\n" +
        f"Status: {status}\n\n" + "\n".join(f"- {r['check']}: {'passed' if r['passed'] else r.get('error', 'failed')}" for r in results) +
        f"\n\nCodex launches: {ledger.codex_runs}\nToken usage complete: {ledger.token_usage_complete}\nInput tokens reported: {ledger.input_tokens}\nCached input tokens reported: {ledger.cached_input_tokens}\nOutput tokens reported: {ledger.output_tokens}\nSeconds: {ledger.snapshot()['seconds']:.2f}\nUSD: unknown (ChatGPT authentication)\n")
    return directory, status == "complete"
