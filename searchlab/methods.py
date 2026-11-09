"""Run original upstream search code with shared model and evaluation hooks.

Trusted definitions are loaded from the vendored source, not rewritten search
algorithms. Only I/O, prompt domain, model access and candidate execution change.
"""
from __future__ import annotations

import ast
import asyncio
import copy
import datetime
import importlib.util
import json
import logging
import os
import random
import re
import time
import types
import typing
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .common import AFLOW_BASELINE, ROOT, digest, save


def trusted_definitions(path, names, namespace):
    tree = ast.parse(Path(path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f"Missing trusted upstream definitions in {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SearchContext:
    def __init__(self, evaluator, gateway, directory, train, config):
        self.evaluator, self.gateway, self.directory = evaluator, gateway, directory
        self.train, self.config = train, config
        self.best = None
        self.history = []

    @property
    def domain(self):
        public = [{"input": r["input"], "answers": r.get("answers", []),
                   "tests": r.get("tests", [])} for r in self.train]
        return (f"Benchmark: {self.evaluator.benchmark}. {self.evaluator.config['answer_instruction']}\n"
                "Public training examples (not validation or final test examples):\n" + json.dumps(public, ensure_ascii=False))

    @property
    def interface(self):
        return """Shared execution contract (overrides incompatible utility examples):
No imports, filesystem, network, direct API clients, reflection or private attributes.
Use only the provided LLM modules, basic Python builtins, Info, tool(name, arguments),
and the provided Custom, AnswerGenerate, ScEnsemble operators. No Programmer/Test operator.
Every model request uses the configured execution model; a model argument cannot change it.
LLMAgentBase returns Info values in output_fields order. Return the answer Info or a string.
For AFlow return a complete class Workflow with __init__(name,llm_config,dataset)
and async __call__(problem), returning (answer_string, 0). The number is ignored;
the evaluator measures real cost externally. create_llm_instance(llm_config) and
operator and prompt_custom are already provided. Prompts must be plain string assignments.
Available operators:
Custom(llm): await custom(input=string, instruction=string) -> {'response': string}
AnswerGenerate(llm): await generate(problem=string) -> {'response': string}
ScEnsemble(llm): await ensemble(solutions=list_of_strings, problem=string) -> {'response': string}
The LLM gateway automatically offers benchmark tools and executes their tool calls.
All modules within one task share that task's environment. Tools can change state;
do not repeat side effects when trying alternative answers. The evaluator and hidden
answers are unavailable. Final test tasks are unavailable during search.
"""

    def evaluate(self, candidate, label, origin="unspecified"):
        result = self.evaluator.evaluate(candidate, label)
        identifier = digest(candidate)
        save(self.directory / "candidates" / f"{identifier}.json", candidate)
        entry = {"candidate": candidate, "result": result, "label": label, "origin": origin,
                 "budget": self.gateway.ledger.snapshot()}
        self.history.append(entry)
        if self.best is None or result["score"] > self.best["result"]["score"]:
            self.best = copy.deepcopy(entry)
        save(self.directory / "history.json", self.history)
        # Persist the incumbent after every *complete* validation evaluation.
        save(self.directory / "incumbent.json", self.best)
        return result


class XMLFormat:
    @classmethod
    def from_model(cls, model):
        return cls()

    def prepare_prompt(self, prompt):
        return prompt + "\nReturn XML fields: <modification>description</modification><graph>complete class Workflow</graph><prompt>prompt constants or empty</prompt>"


class ProposalLLM:
    def __init__(self, context):
        self.context = context

    async def __call__(self, prompt):
        return self.context.gateway.call([{"role": "user", "content": prompt}], role="search", temperature=0.8)["content"]

    async def call_with_format(self, prompt, formatter):
        text = await self(formatter.prepare_prompt(prompt))
        data = {key: match.group(1).strip() if (match := re.search(rf"<{key}>(.*?)</{key}>", text, re.S)) else ""
                for key in ("modification", "graph", "prompt")}
        if not data["graph"] or not data["modification"]:
            raise ValueError("Missing AFlow XML fields")
        return data


def run_aflow(context, seed):
    random.seed(seed)
    np.random.seed(seed)
    base = ROOT / "vendor/aflow/scripts"
    prompts = load_module("aflow_original_prompts", base / "prompts/optimize_prompt.py")
    logger = logging.getLogger("upstream.aflow")
    namespace = {**vars(typing), "Path": Path, "np": np, "pd": pd, "json": json, "os": os,
                 "random": random, "datetime": datetime, "time": time, "asyncio": asyncio,
                 "re": re, "logger": logger, "BaseModel": BaseModel, "Field": Field,
                 "DatasetType": str, "QuestionType": str, "OptimizerType": str,
                 "read_json_file": lambda path, **kw: json.loads(Path(path).read_text()),
                 "write_json_file": lambda path, data, **kw: save(path, data),
                 "XmlFormatter": XMLFormat, "FormatError": ValueError,
                 "create_llm_instance": lambda config: ProposalLLM(context)}
    namespace.update({k: getattr(prompts, k) for k in ("WORKFLOW_TEMPLATE", "WORKFLOW_INPUT", "WORKFLOW_OPTIMIZE_PROMPT", "WORKFLOW_CUSTOM_USE")})

    class EvaluationBridge:
        def __init__(self, eval_path):
            self.path = eval_path

        async def graph_evaluate(self, dataset, graph, params, path, is_test=False):
            if is_test:
                raise ValueError("Upstream search cannot access final test tasks")
            result = context.evaluate(graph, Path(path).name, origin="initial" if Path(path).name == "round_1" else "generated")
            failures = [{"task_id": record["task_id"], "model_output": record.get("output"),
                         "score": record["score"], "error": record["error"],
                         "expected_answers": task.get("answers", [])}
                        for record, task in zip(context.evaluator.last_records, context.evaluator.rows)
                        if record["score"] < 1 or record["error"]]
            save(Path(path) / "log.json", failures)
            return result["score"], result["usd"] / result["count"] if result["usd"] is not None else None, result["usd"]

    namespace["Evaluator"] = EvaluationBridge
    for filename, names in (("data_utils", ["DataUtils"]), ("graph_utils", ["GraphUtils"]),
                            ("experience_utils", ["ExperienceUtils"]), ("evaluation_utils", ["EvaluationUtils"]),
                            ("convergence_utils", ["ConvergenceUtils"])):
        if filename == "experience_utils":
            from collections import defaultdict
            namespace["defaultdict"] = defaultdict
        trusted_definitions(base / "optimizer_utils" / f"{filename}.py", names, namespace)
    trusted_definitions(base / "optimizer.py", ["GraphOptimize", "Optimizer"], namespace)
    optimizer = namespace["Optimizer"](dataset="SHARED", question_type="code" if context.evaluator.benchmark == "mbpp" else "qa",
        opt_llm_config={}, exec_llm_config={}, operators=["Custom", "AnswerGenerate", "ScEnsemble"],
        sample=context.config["aflow_sample"], optimized_path=str(context.directory / "upstream"),
        max_rounds=context.config["iterations"], validation_rounds=1)
    workflows = Path(optimizer.root_path) / "workflows"
    save(workflows / "round_1/artifact.json", AFLOW_BASELINE)
    (workflows / "round_1/graph.py").write_text(AFLOW_BASELINE["code"])
    (workflows / "round_1/prompt.py").write_text("")
    descriptions = {name: {"description": name + " through shared model gateway", "interface": context.interface}
                    for name in optimizer.operators}
    save(workflows / "template/operator.json", descriptions)
    graph_utils = optimizer.graph_utils

    def load_graph(self, round_number, workflows_path):
        return json.loads((Path(workflows_path) / f"round_{round_number}" / "artifact.json").read_text())

    def write_graph_files(self, directory, response, round_number, dataset):
        candidate = {"kind": "aflow", "code": response["graph"], "prompt": response["prompt"],
                     "name": response["modification"]}
        save(Path(directory) / "artifact.json", candidate)
        (Path(directory) / "graph.py").write_text(candidate["code"])
        (Path(directory) / "prompt.py").write_text(candidate["prompt"])

    original_prompt = graph_utils.create_graph_optimize_prompt

    def create_prompt(self, *args, **kwargs):
        return original_prompt(*args, **kwargs) + "\n" + context.domain + "\n" + context.interface

    graph_utils.load_graph = types.MethodType(load_graph, graph_utils)
    graph_utils.write_graph_files = types.MethodType(write_graph_files, graph_utils)
    graph_utils.create_graph_optimize_prompt = types.MethodType(create_prompt, graph_utils)
    # Use the original search step. The outer loop supplies common stopping
    # policy instead of upstream sleeps and broad retry handling.
    for _ in range(context.config["iterations"]):
        context.gateway.ledger.check()
        try:
            asyncio.run(optimizer._optimize_graph())
        except Exception as exc:
            from .common import append
            append(context.directory / "search_errors.jsonl", {"round": optimizer.round, "error": str(exc)})
        optimizer.round += 1


def run_adas(context, seed):
    random.seed(seed)
    np.random.seed(seed)
    base = ROOT / "vendor/adas/_drop"
    prompts = load_module("adas_original_prompt", base / "drop_prompt.py")
    initial_codes = {solution["code"] for solution in prompts.get_init_archive()}
    metrics = load_module("adas_original_utils", base / "utils.py")
    # Keep archive instructions, seed architectures and both reflection prompts;
    # replace only the original DROP domain/example introduction.
    prompts.base = "# Overview\n" + context.domain + "\n\n# The utility code:" + prompts.base.split("# The utility code:", 1)[1]
    original_get_prompt = prompts.get_prompt

    def get_prompt(archive):
        system, prompt = original_get_prompt(archive)
        return system, prompt + "\n" + context.interface

    def propose(messages, model, temperature=0.8):
        message = context.gateway.call(messages, role="search", temperature=temperature,
                                       fields=["thought", "name", "code"])
        result = json.loads(message["content"])
        if not isinstance(result.get("code"), str):
            raise ValueError("ADAS proposal is missing code")
        return result

    def evaluate(args, code):
        candidate = {"kind": "adas", "code": code, "prompt": "", "name": "ADAS " + digest(code)[:10]}
        return context.evaluate(candidate, candidate["name"], origin="initial" if code in initial_codes else "generated")["scores"]

    ns = {"os": os, "json": json, "np": np, "get_init_archive": prompts.get_init_archive,
          "get_prompt": get_prompt, "get_reflexion_prompt": prompts.get_reflexion_prompt,
          "get_json_response_from_gpt_reflect": propose, "evaluate_forward_fn": evaluate,
          "bootstrap_confidence_interval": lambda scores: metrics.bootstrap_confidence_interval([100 * s for s in scores]),
          "SEARCHING_MODE": True}
    trusted_definitions(base / "search.py", ["search"], ns)
    args = types.SimpleNamespace(save_dir=str(context.directory / "upstream"), expr_name="adas",
                                 n_generation=context.config["iterations"], debug_max=context.config["adas_debug_max"], model="shared-search-model")
    ns["search"](args)
