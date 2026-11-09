from __future__ import annotations

import importlib.util
import json
import re
import time
from decimal import Decimal, InvalidOperation

from .common import ROOT, append, digest, save
from .gateway import BudgetStop


def drop_scorer():
    spec = importlib.util.spec_from_file_location("upstream_drop_metric", ROOT / "vendor/adas/_drop/utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.drop_metric


def numeric_answer(text):
    text = str(text).strip().rsplit("####", 1)[-1]
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1]
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not numbers:
        return None
    try:
        return Decimal(numbers[-1].replace(",", ""))
    except InvalidOperation:
        return None


class TaskSession:
    def __init__(self, gateway, worker, task, config):
        self.gateway, self.worker, self.task, self.config = gateway, worker, task, config
        self.calls = 0
        self.tool_calls = 0
        self.trace = []
        self.tools = []
        self.runtime = self.env = None

    def callback(self, request):
        if request["op"] == "tool":
            return self.run_tool(request["name"], request["arguments"])
        if request["op"] != "llm":
            raise ValueError("Unsupported worker operation")
        fields = request.get("fields")
        instruction = self.config["answer_instruction"]
        system = request.get("system", "You are a helpful assistant.") + "\n" + instruction
        if fields:
            system += "\nReturn JSON. Put the final answer in the answer field (Python code for coding tasks)."
        messages = [{"role": "system", "content": system}, {"role": "user", "content": request["prompt"]}]
        for _ in range(self.config["max_tool_steps"] + 1):
            if self.calls >= self.config["max_model_calls_per_task"]:
                raise RuntimeError("Per-task model-call limit reached")
            self.calls += 1
            response = self.gateway.call(messages, fields=fields, tools=self.tools or None,
                                         temperature=request.get("temperature", 0.5))
            calls = response.get("tool_calls") or []
            messages.append({k: response[k] for k in ("role", "content", "tool_calls") if k in response})
            if not calls:
                return response.get("content") or ""
            for call in calls:
                value = self.run_tool(call["function"]["name"], json.loads(call["function"]["arguments"]))
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": value})
        raise RuntimeError("Tool-step limit reached")

    def run_tool(self, name, arguments):
        if self.runtime is None:
            raise ValueError("This benchmark has no tools")
        self.tool_calls += 1
        if self.tool_calls > self.config["max_tool_steps"]:
            raise RuntimeError("Per-task tool-call limit reached")
        from agentdojo.functions_runtime import FunctionCall
        call = FunctionCall(function=name, args=arguments, id=f"call-{self.tool_calls}")
        value, error = self.runtime.run_function(self.env, name, arguments)
        content = value if isinstance(value, str) else json.dumps(value, default=lambda x: x.model_dump() if hasattr(x, "model_dump") else str(x), ensure_ascii=False)
        if error:
            content = error
        self.trace.extend([
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "content": [{"type": "text", "content": content}], "tool_call": call,
             "tool_call_id": call.id, "error": error}])
        return content


class Evaluator:
    def __init__(self, benchmark, rows, gateway, worker, config, directory, phase):
        self.benchmark, self.rows, self.gateway, self.worker = benchmark, rows, gateway, worker
        self.config, self.directory, self.phase = config, directory, phase
        self.drop_metric = drop_scorer() if benchmark == "drop" else None

    def evaluate(self, candidate, label):
        scores, records = [], []
        for task in self.rows:
            self.gateway.ledger.check()
            start = time.monotonic()
            before = self.gateway.ledger.snapshot()
            session = TaskSession(self.gateway, self.worker, task, self.config)
            record = {"task_id": task["id"], "candidate_sha256": digest(candidate), "phase": self.phase,
                      "label": label, "error": None}
            try:
                if self.benchmark.startswith("agentdojo_"):
                    score, details = self.agentdojo(candidate, task, session)
                    record.update(details)
                else:
                    output = self.worker.run({"candidate": candidate, "input": task["input"], "benchmark": self.benchmark}, session.callback)
                    record["output"] = output
                    if self.benchmark == "gsm8k":
                        parsed = numeric_answer(output)
                        score = float(parsed is not None and any(parsed == numeric_answer(a) for a in task["answers"]))
                    elif self.benchmark == "drop":
                        em, f1 = self.drop_metric(str(output), task["answers"])
                        score = float(f1 / 100)
                        record["exact_match"] = float(em)
                    elif self.benchmark == "mbpp":
                        score = float(self.worker.run({"op": "grade_code", "code": str(output),
                                                       "setup": task.get("setup", ""), "tests": task["tests"]}))
                    else:
                        raise ValueError("Unknown metric")
                record["score"] = score
            except BudgetStop:
                append(self.directory / "tasks.jsonl", {**record, "status": "interrupted_budget"})
                raise
            except Exception as exc:
                score = 0.0
                record.update(score=score, error=type(exc).__name__ + ": " + str(exc))
            record.update(seconds=time.monotonic() - start,
                          model_calls=self.gateway.ledger.calls - before["calls"],
                          usd=self.gateway.ledger.usd - before["usd"] if self.gateway.ledger.cost_available else None,
                          input_tokens=self.gateway.ledger.input_tokens - before["input_tokens"],
                          output_tokens=self.gateway.ledger.output_tokens - before["output_tokens"],
                          cached_input_tokens=self.gateway.ledger.cached_input_tokens - before["cached_input_tokens"],
                          tokens=self.gateway.ledger.tokens - before["tokens"],
                          codex_runs=self.gateway.ledger.codex_runs - before["codex_runs"], trace=session.trace)
            scores.append(score)
            records.append(record)
            append(self.directory / "tasks.jsonl", record)
        result = {"score": sum(scores) / len(scores), "scores": scores, "count": len(scores),
                  "errors": sum(r["error"] is not None for r in records), "candidate_sha256": digest(candidate),
                  "usd": sum(r["usd"] for r in records) if all(r["usd"] is not None for r in records) else None,
                  "model_calls": sum(r["model_calls"] for r in records),
                  "tokens": sum(r["tokens"] for r in records),
                  "input_tokens": sum(r["input_tokens"] for r in records),
                  "output_tokens": sum(r["output_tokens"] for r in records),
                  "cached_input_tokens": sum(r["cached_input_tokens"] for r in records),
                  "codex_runs": sum(r["codex_runs"] for r in records),
                  "seconds": sum(r["seconds"] for r in records)}
        self.last_records = records
        for field in ("clean_utility", "attacked_utility", "attack_success", "exact_match"):
            values = [r[field] for r in records if field in r]
            if values:
                result[field] = sum(values) / len(values)
        return result

    def agentdojo(self, candidate, task, session):
        from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
        from agentdojo.task_suite.load_suites import get_suite
        from agentdojo.attacks.attack_registry import load_attack
        suite = get_suite(self.config["agentdojo_version"], task["suite"])
        worker, benchmark = self.worker, self.benchmark

        class Bridge(BasePipelineElement):
            name = "shared-worker"

            def query(self, query, runtime, env, messages=(), extra_args=None):
                session.runtime, session.env = runtime, env
                session.tools = [{"type": "function", "function": {"name": f.name, "description": f.description,
                                 "parameters": f.parameters.model_json_schema()}} for f in runtime.functions.values()]
                output = worker.run({"candidate": candidate, "input": query, "benchmark": benchmark}, session.callback)
                transcript = session.trace + [{"role": "assistant", "content": [{"type": "text", "content": str(output)}], "tool_calls": None}]
                return query, runtime, env, transcript, {}

        pipeline = Bridge()
        user_task = suite.get_user_task_by_id(task["user_task"])
        if not self.config.get("agentdojo_attack"):
            utility, _ = suite.run_task_with_pipeline(pipeline, user_task, None, {})
            return float(utility), {"clean_utility": float(utility)}
        # A single task record is either clean or one fixed injection goal. Both
        # variants of a user task remain in the same data partition.
        if not task.get("injection_task"):
            utility, _ = suite.run_task_with_pipeline(pipeline, user_task, None, {})
            return float(utility), {"clean_utility": float(utility)}
        goal = suite.get_injection_task_by_id(task["injection_task"])
        attack = load_attack(self.config["agentdojo_attack"], suite, pipeline)
        # Fix attacker input independently of the candidate to avoid unequal
        # attacker adaptation; only deterministic built-in attacks are allowed.
        injections = attack.attack(user_task, goal)
        utility, attack_success = suite.run_task_with_pipeline(pipeline, user_task, goal, injections)
        objective = float(utility and not attack_success) if self.config["agentdojo_objective"] == "robust_utility" else float(utility)
        return objective, {"attacked_utility": float(utility), "attack_success": float(attack_success)}
