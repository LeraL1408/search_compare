from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import ROOT


def main():
    parser = argparse.ArgumentParser(description="Compare upstream AFlow and ADAS under shared budgets")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "plan"):
        p = commands.add_parser(name)
        p.add_argument("--config", type=Path, default=ROOT / "configs/smoke.json")
        if name == "run":
            p.add_argument("--output", type=Path)
            p.add_argument("--live", action="store_true")
    p = commands.add_parser("prepare")
    p.add_argument("benchmark", choices=["gsm8k", "drop", "mbpp", "agentdojo_workspace", "agentdojo_banking", "agentdojo_travel", "agentdojo_slack"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train", type=Path)
    p.add_argument("--test", type=Path)
    p.add_argument("--train-size", type=int, default=4)
    p.add_argument("--validation-size", type=int, default=12)
    p.add_argument("--test-size", type=int, default=20)
    p.add_argument("--split-seed", type=int, default=1729)
    p = commands.add_parser("report")
    p.add_argument("directory", type=Path)
    commands.add_parser("verify-sources")
    p = commands.add_parser("check-executor")
    p.add_argument("--image", default="python:3.12-slim")
    p.add_argument("--pull", action="store_true")
    p = commands.add_parser("init-codex", help="Create a small real-model pilot using your Codex ChatGPT login")
    p.add_argument("--output", type=Path, default=ROOT / "configs/codex.json")
    p.add_argument("--model")
    p.add_argument("--benchmarks", nargs="+", default=["gsm8k"], choices=["gsm8k", "drop", "mbpp", "agentdojo_workspace", "agentdojo_banking", "agentdojo_travel", "agentdojo_slack"])
    p = commands.add_parser("check-codex", help="Make three real model calls to test the Codex relay")
    p.add_argument("--config", type=Path, default=ROOT / "configs/codex.json")
    p.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "init-codex":
            from .codex_setup import initialize
            config = initialize(args.output, args.model, args.benchmarks)
            print(json.dumps({"configuration": str(args.output), "model": config["models"]["execution"]["model"],
                              "benchmarks": args.benchmarks, "calls_made": 0}, indent=2))
        elif args.command == "check-codex":
            from .codex_setup import connection_check
            from .runner import load_config
            directory, passed = connection_check(load_config(args.config), args.output)
            print(directory)
            return 0 if passed else 2
        elif args.command == "prepare":
            from .datasets import prepare
            if min(args.train_size, args.validation_size, args.test_size) <= 0:
                raise ValueError("All split sizes must be positive")
            bundle = prepare(args.benchmark, args.output, args.train_size, args.validation_size, args.test_size,
                             args.split_seed, args.train, args.test)
            print(json.dumps({"file": str(args.output), "sizes": {s: len(bundle[s]) for s in ("train", "validation", "test")}, "sha256": bundle["sha256"]}, indent=2))
        elif args.command == "report":
            from .report import report
            print(report(args.directory))
        elif args.command == "check-executor":
            import subprocess
            from .common import BASELINE
            from .execution import Worker
            if args.pull:
                subprocess.run(["docker", "pull", args.image], check=True)
            worker = Worker({"docker_image": args.image, "task_timeout_seconds": 30})
            def answer(request):
                return json.dumps({"answer": "isolated-worker-ok"})
            value = worker.run({"candidate": BASELINE, "input": "test", "benchmark": "executor-check"}, answer)
            if value != "isolated-worker-ok":
                raise ValueError("Container model relay check failed")
            worker.run({"op": "grade_code", "code": "import math\ndef square(x):\n    return math.pow(x, 2)",
                        "tests": ["assert square(3) == 9"]})
            print("Docker execution and code grading passed; no model API calls.")
        elif args.command == "verify-sources":
            from .common import verify_sources
            verify_sources()
            print("Vendored upstream source fingerprints match.")
        else:
            from .runner import bundles_for, load_config, run
            config = load_config(args.config)
            if args.command == "plan":
                bundles = bundles_for(config)
                n = len(bundles) * len(config["seeds"])
                print(json.dumps({"fixture": config["fixture"], "seeds": config["seeds"],
                    "benchmarks": [{"name": b["name"], "sizes": {s: len(b[s]) for s in ("train", "validation", "test")}} for b in bundles],
                    "methods": ["baseline", "aflow", "adas"], "budgets": config["budgets"],
                    "benchmark_budgets": config.get("benchmark_budgets", {}),
                    "backend": config["models"]["execution"].get("provider", "openai"),
                    "maximum_configured_total_usd": None if config["budgets"]["search"]["max_usd"] is None else n * (2 * config["budgets"]["search"]["max_usd"] + 4 * config["budgets"]["evaluation"]["max_usd"]),
                    "maximum_gateway_calls": min(n * (2 * config["budgets"]["search"]["max_calls"] + 4 * config["budgets"]["evaluation"]["max_calls"]), config["budgets"].get("global", {}).get("max_calls", float("inf")))}, indent=2))
            else:
                directory = run(config, args.output, args.live)
                print(directory)
                if json.loads((directory / "manifest.json").read_text())["status"] != "complete":
                    return 2
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
