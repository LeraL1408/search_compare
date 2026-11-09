"""Credential-free generated-agent worker. Live runs execute this in Docker.

The local mode exists only for the bundled deterministic fixture provider.
The AST checks are compatibility checks, not a Python security boundary.
"""
import ast
import asyncio
import builtins
import contextlib
import json
import sys
import types
from collections import namedtuple

WIRE_OUT = sys.stdout
WIRE_IN = sys.stdin


def rpc(payload):
    WIRE_OUT.write(json.dumps(payload) + "\n")
    WIRE_OUT.flush()
    response = json.loads(WIRE_IN.readline())
    if "error" in response:
        raise RuntimeError(response["error"])
    return response["value"]


Info = namedtuple("Info", "name author content iteration_idx")


class LLMAgentBase:
    counter = 0

    def __init__(self, output_fields, agent_name, role="helpful assistant", model=None, temperature=0.5):
        self.output_fields, self.agent_name, self.role = output_fields, agent_name, role
        self.temperature = temperature
        LLMAgentBase.counter += 1
        self.id = str(LLMAgentBase.counter)

    def __repr__(self):
        return self.agent_name + " " + self.id

    def generate_prompt(self, input_infos, instruction):
        inputs = "\n\n".join(f"# {i.name} by {i.author}:\n{i.content}" for i in input_infos if isinstance(i, Info))
        system = f"You are a {self.role}. Return a JSON object with these fields in this order: {json.dumps(self.output_fields)}."
        return system, inputs + "\n\n" + instruction

    def query(self, input_infos, instruction, iteration_idx=-1):
        system, prompt = self.generate_prompt(input_infos, instruction)
        response = rpc({"op": "llm", "prompt": prompt, "system": system, "temperature": self.temperature,
                        "fields": self.output_fields})
        data = json.loads(response)
        return [Info(key, str(self), data.get(key, ""), iteration_idx) for key in self.output_fields]

    __call__ = query


class LLM:
    async def __call__(self, prompt, **kwargs):
        return rpc({"op": "llm", "prompt": str(prompt), "system": "You are a helpful assistant.",
                    "temperature": kwargs.get("temperature", 0.5)})

    def get_usage_summary(self):
        # Actual accounting lives in the parent; generated code cannot alter it.
        return {"total_cost": 0}


class Custom:
    def __init__(self, llm, name="Custom"):
        self.llm = llm

    async def __call__(self, input="", instruction="", **kwargs):
        return {"response": await self.llm(instruction + "\n" + input)}


class AnswerGenerate(Custom):
    async def __call__(self, problem, **kwargs):
        return {"response": await self.llm("Solve the task and return only the final answer.\n" + problem)}


class ScEnsemble(Custom):
    async def __call__(self, solutions, problem):
        return {"response": await self.llm("Select or combine the candidate answers. Return only the final answer.\n" + problem + "\n" + json.dumps(solutions))}


def tool(name, arguments):
    return rpc({"op": "tool", "name": name, "arguments": arguments})


ALLOWED_IMPORTS = {"math", "re", "string", "collections", "itertools", "functools", "heapq", "bisect", "statistics", "decimal", "fractions", "array", "random", "operator", "cmath"}


def validate(code, allow_imports=False):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if not allow_imports or any(m not in ALLOWED_IMPORTS for m in modules) or getattr(node, "level", 0):
                raise ValueError("Import is unavailable in the shared execution interface")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private/introspection attributes are unavailable")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Private/introspection names are unavailable")
    return tree


def namespace():
    names = "abs all any bin bool bytes chr complex dict divmod enumerate filter float format frozenset hex int isinstance issubclass iter len list map max min next oct ord pow range repr reversed round set slice sorted str sum tuple zip Exception ValueError TypeError AssertionError IndexError KeyError ZeroDivisionError StopIteration object".split()
    safe = {name: getattr(builtins, name) for name in names}
    safe["__build_class__"] = builtins.__build_class__
    return {"__builtins__": safe, "__name__": "candidate", "Info": Info, "LLMAgentBase": LLMAgentBase,
            "create_llm_instance": lambda config: LLM(), "DatasetType": str,
            "operator": types.SimpleNamespace(Custom=Custom, AnswerGenerate=AnswerGenerate, ScEnsemble=ScEnsemble),
            "List": list, "Dict": dict, "tool": tool}


def execute(request):
    ns = namespace()
    if request.get("op") == "grade_code":
        code = request["code"]
        if "```" in code:
            import re
            blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", code)
            code = "\n".join(blocks)
        # Code grading runs in the same container restrictions as agent execution.
        def allowed_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name not in ALLOWED_IMPORTS or level or any(n.startswith("_") for n in (fromlist or ())):
                raise ValueError("Unsupported import")
            return builtins.__import__(name, globals, locals, fromlist, level)
        ns["__builtins__"]["__import__"] = allowed_import
        exec(compile(validate(request.get("setup", "") + "\n" + code, allow_imports=True), "submission", "exec"), ns)
        for test in request["tests"]:
            exec(compile(validate(test, allow_imports=True), "hidden_test", "exec"), ns)
        return True
    candidate = request["candidate"]
    prompt_ns = namespace()
    exec(compile(validate(candidate.get("prompt", "")), "prompts", "exec"), prompt_ns)
    ns["prompt_custom"] = types.SimpleNamespace(**{k: v for k, v in prompt_ns.items() if not k.startswith("_")})
    exec(compile(validate(candidate["code"]), "candidate", "exec"), ns)
    if candidate["kind"] == "aflow":
        result = asyncio.run(ns["Workflow"]("shared", {}, request["benchmark"])(request["input"]))
        return result[0] if isinstance(result, tuple) else result
    instance = types.SimpleNamespace()
    result = ns["forward"](instance, Info("task", "User", request["input"], -1))
    return result.content if isinstance(result, Info) else str(result)


if __name__ == "__main__":
    try:
        request = json.loads(WIRE_IN.readline())
        with contextlib.redirect_stdout(sys.stderr):
            result = execute(request)
        WIRE_OUT.write(json.dumps({"op": "result", "value": result}) + "\n")
    except BaseException as exc:
        WIRE_OUT.write(json.dumps({"op": "error", "error": type(exc).__name__ + ": " + str(exc)}) + "\n")
    WIRE_OUT.flush()
