from __future__ import annotations

import copy
import datetime
import importlib.metadata
import json
import math
import os
import random
import hashlib
import subprocess
import uuid
from pathlib import Path

from .common import BASELINE, ROOT, digest, save, verify_sources
from .datasets import fixtures, validate_splits
from .evaluation import Evaluator
from .execution import Worker
from .gateway import BudgetStop, Gateway, Ledger
from .methods import SearchContext, run_adas, run_aflow


def load_config(path):
    config = json.loads(Path(path).read_text())
    if type(config.get("fixture")) is not bool:
        raise ValueError("Explicit fixture boolean required")
    if not config.get("seeds") or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Provide distinct nonnegative integer seeds")
    if any(type(s) is not int or s < 0 or s >= 2**32 for s in config["seeds"]):
        raise ValueError("Seeds must be nonnegative 32-bit integers")
    if set(config["models"]) != {"search", "execution"}:
        raise ValueError("Configure search and execution models")
    providers = {m.get("provider", "openai") for m in config["models"].values()}
    if not providers <= {"openai", "codex"} or len(providers) != 1:
        raise ValueError("Use the same backend type (openai or codex) for both model roles")
    codex = providers == {"codex"}
    for role, model in config["models"].items():
        if not model.get("model") or model["model"].startswith("SET_"):
            raise ValueError(f"Set the {role} model name")
        if codex:
            if model.get("input_usd_per_million") is not None or model.get("output_usd_per_million") is not None:
                raise ValueError("Codex subscription cost is unknown; leave USD rates null")
            if model.get("timeout_seconds", 120) <= 0 or model.get("max_prompt_bytes", 200000) <= 0:
                raise ValueError("Codex timeout and prompt byte limit must be positive")
        else:
            for rate in ("input_usd_per_million", "output_usd_per_million"):
                value = model[rate]
                if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Set {role}.{rate} to your provider's rate (zero only for free/local models)")
            for limit in ("max_input_tokens", "max_output_tokens"):
                if type(model[limit]) is not int or model[limit] <= 0:
                    raise ValueError(f"Invalid {role}.{limit}")
    for phase in ("search", "evaluation"):
        limits = config["budgets"][phase]
        for key in ("max_calls", "max_tokens", "max_seconds", "max_usd"):
            value = limits[key]
            if key == "max_usd" and codex:
                if value is not None:
                    raise ValueError("Codex has no measurable USD budget; use null and limit calls, tokens and time")
                continue
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0 or (key != "max_usd" and value == 0):
                raise ValueError(f"Invalid {phase}.{key}")
    for key in ("iterations", "aflow_sample", "adas_debug_max"):
        if type(config["search"][key]) is not int or config["search"][key] < 1:
            raise ValueError(f"Invalid search.{key}")
    for key in ("max_model_calls_per_task", "max_tool_steps"):
        if type(config["evaluation"][key]) is not int or config["evaluation"][key] < 1:
            raise ValueError(f"Invalid evaluation.{key}")
    if config["evaluation"].get("agentdojo_attack") not in (None, "direct", "ignore_previous", "system_message", "injecagent"):
        raise ValueError("Use a fixed built-in attack, not an adaptive attacker, for paired comparisons")
    return config


def bundles_for(config):
    if config["fixture"]:
        return list(fixtures())
    bundles = []
    for path in config["datasets"]:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        bundle = json.loads(resolved.read_text())
        expected = bundle.pop("sha256")
        if digest(bundle) != expected:
            raise ValueError("Dataset fingerprint mismatch")
        bundle["sha256"] = expected
        if bundle.get("fixture"):
            raise ValueError("Fixture datasets cannot be used for live research")
        validate_splits(bundle)
        bundles.append(bundle)
    if not bundles or len({b["name"] for b in bundles}) != len(bundles):
        raise ValueError("Provide at least one dataset; benchmark names must be distinct")
    return bundles


def expand_agentdojo(bundle, evaluation):
    bundle = copy.deepcopy(bundle)
    if not bundle["name"].startswith("agentdojo_") or not evaluation.get("agentdojo_attack"):
        return bundle
    for split in ("train", "validation", "test"):
        expanded = []
        for task in bundle[split]:
            expanded.append({**task, "id": task["id"] + "-clean"})
            for goal in evaluation["agentdojo_injection_tasks"]:
                expanded.append({**task, "id": task["id"] + "-" + goal, "injection_task": goal})
        bundle[split] = expanded
    validate_splits(bundle)
    bundle["source_sha256"] = bundle.pop("sha256")
    bundle["sha256"] = digest(bundle)
    return bundle


def preflight(config, live):
    if not config["fixture"]:
        if not live:
            raise ValueError("Live experiments require the --live flag")
        for model in config["models"].values():
            if model.get("provider") == "codex":
                from .codex_backend import codex_environment
                login = subprocess.run([model.get("binary", "codex"), "login", "status"],
                                       env=codex_environment(), capture_output=True, text=True, timeout=15)
                if login.returncode or "logged in using chatgpt" not in (login.stdout + login.stderr).lower():
                    raise ValueError("Sign in to Codex with ChatGPT first: codex login")
            elif not os.getenv(model.get("api_key_env", "OPENAI_API_KEY")):
                raise ValueError("Set the configured model API key environment variable")
        result = subprocess.run(["docker", "image", "inspect", config["execution"]["docker_image"], "--format", "{{.Id}}"],
                                check=True, capture_output=True, text=True, timeout=15)
        return result.stdout.strip()
    return None


def run(config, output=None, live=False):
    upstream = verify_sources()
    bundles = bundles_for(config)
    image_id = preflight(config, live)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(output) if output else ROOT / "runs" / (stamp + "-" + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"config": config, "config_sha256": digest(config), "fixture": config["fixture"],
                "datasets": {b["name"]: b["sha256"] for b in bundles},
                "upstream": upstream,
                "implementation": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted((ROOT / "searchlab").glob("*.py"))},
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "pydantic", "openai", "scipy")},
                "docker_image_id": image_id,
                "started_utc": stamp, "status": "running"}
    if any(m.get("provider") == "codex" for m in config["models"].values()):
        from .codex_backend import codex_environment
        binary = config["models"]["search"].get("binary", "codex")
        version = subprocess.run([binary, "--version"], env=codex_environment(), capture_output=True, text=True, timeout=15)
        manifest["codex"] = {"version": version.stdout.strip(), "authentication": "chatgpt",
                             "fresh_session_per_call": True, "native_tools_enabled": False,
                             "token_budget_enforcement": "between_codex_runs", "usd_cost_available": False,
                             "seed_applied": False, "temperature_applied": False}
    save(directory / "manifest.json", manifest)
    all_results = []
    execution = {**config["execution"], **({"docker_image": image_id} if image_id else {})}
    worker = Worker(execution, fixture=config["fixture"])
    for original in bundles:
        bundle = expand_agentdojo(original, config["evaluation"])
        save(directory / bundle["name"] / "splits.json", bundle)
        eval_config = {**config["evaluation"], "answer_instruction": {
            "gsm8k": "For mathematics, return only the final numeric answer.",
            "drop": "For reading comprehension, return only the concise answer span(s).",
            "mbpp": "Return only complete Python code implementing the requested function."
        }.get(bundle["name"], "Complete the user's request using the available tools.")}
        for seed in config["seeds"]:
            # Rotate method order to reduce systematic provider/time effects.
            methods = ["aflow", "adas"]
            random.Random(seed).shuffle(methods)
            frozen = []
            for method in ["baseline"] + methods:
                unit = directory / bundle["name"] / f"seed_{seed}" / method
                unit.mkdir(parents=True)
                print(f"{bundle['name']} / seed {seed} / {method}: validation/search", flush=True)
                ledger = Ledger(config["budgets"]["evaluation" if method == "baseline" else "search"], unit / "search_calls.jsonl")
                gateway = Gateway(config["models"], ledger, seed, config["fixture"])
                evaluator = Evaluator(bundle["name"], bundle["validation"], gateway, worker, eval_config, unit, "validation")
                context = SearchContext(evaluator, gateway, unit, bundle["train"], config["search"])
                status, error = "complete", None
                try:
                    if method == "baseline":
                        context.evaluate(BASELINE, "baseline")
                    else:
                        (run_aflow if method == "aflow" else run_adas)(context, seed)
                except BudgetStop as exc:
                    status, error = "budget_stopped", str(exc)
                except Exception as exc:
                    status, error = "failed", type(exc).__name__ + ": " + str(exc)
                search_result = {"status": status, "error": error, "budget": ledger.snapshot(),
                                 "completed_candidates": len(context.history)}
                summary = {"benchmark": bundle["name"], "seed": seed, "method": method,
                           "fixture": config["fixture"], "search": search_result, "test": {"status": "not_run"}}
                if context.best is not None and status != "failed":
                    artifact = {"candidate": context.best["candidate"], "sha256": digest(context.best["candidate"]),
                                "selection": {"split": "validation", "score": context.best["result"]["score"],
                                              "dataset_sha256": original["sha256"]}}
                    save(unit / "frozen.json", artifact)
                    summary["frozen_sha256"] = artifact["sha256"]
                    summary["validation"] = context.best["result"]
                    frozen.append((unit, summary))
                save(unit / "summary.json", summary)
                all_results.append(summary)
                save(directory / "results.json", all_results)
            # Search for *all* methods finishes and artifacts are written before
            # any held-out test feedback exists for this benchmark/seed.
            for unit, summary in frozen:
                artifact = json.loads((unit / "frozen.json").read_text())
                if digest(artifact["candidate"]) != artifact["sha256"]:
                    raise ValueError("Frozen candidate was modified")
                print(f"{bundle['name']} / seed {seed} / {summary['method']}: held-out test", flush=True)
                ledger = Ledger(config["budgets"]["evaluation"], unit / "test_calls.jsonl")
                gateway = Gateway(config["models"], ledger, seed, config["fixture"])
                evaluator = Evaluator(bundle["name"], bundle["test"], gateway, worker, eval_config, unit, "test")
                try:
                    result = evaluator.evaluate(artifact["candidate"], "frozen")
                    summary["test"] = {"status": "complete", "result": result}
                except BudgetStop as exc:
                    summary["test"] = {"status": "incomplete", "error": str(exc)}
                except Exception as exc:
                    summary["test"] = {"status": "failed", "error": type(exc).__name__ + ": " + str(exc)}
                summary["test"]["budget"] = ledger.snapshot()
                save(unit / "summary.json", summary)
                save(directory / "results.json", all_results)
    manifest["status"] = "complete" if all(r["test"]["status"] == "complete" for r in all_results) else "partial"
    save(directory / "manifest.json", manifest)
    from .report import report
    report(directory)
    return directory
