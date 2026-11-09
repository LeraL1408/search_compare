from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temp.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_sources():
    manifest = json.loads((ROOT / "vendor/manifest.json").read_text())
    for repo, details in manifest.items():
        for path, expected in details["files"].items():
            if hashlib.sha256((ROOT / "vendor" / repo / path).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Upstream source changed: {repo}/{path}")
    return manifest


BASELINE = {
    "kind": "adas",
    "code": '''def forward(self, taskInfo):
    agent = LLMAgentBase(['answer'], 'Baseline', temperature=0)
    return agent([taskInfo], 'Solve the task. Return only the final answer.')[0]
''',
    "prompt": "",
    "name": "Single model call (tool loop when tools are available)",
}

AFLOW_BASELINE = {
    "kind": "aflow",
    "code": '''class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
    async def __call__(self, problem):
        result = await self.custom(input=problem, instruction='Solve the task. Return only the final answer.')
        return result['response'], 0
''',
    "prompt": "",
    "name": "AFlow initial workflow",
}
