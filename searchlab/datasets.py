from __future__ import annotations

import ast
import gzip
import json
import random
import urllib.request
import warnings
from pathlib import Path

from .common import ROOT, digest, read_rows, save

URLS = {
    "gsm8k_train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "gsm8k_test": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    "mbpp": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
}


def validate_splits(bundle):
    seen_ids, seen_inputs, seen_groups = set(), set(), set()
    for split in ("train", "validation", "test"):
        rows = bundle[split]
        if not rows:
            raise ValueError(f"Empty {split} split")
        ids, inputs, groups = set(), set(), set()
        for row in rows:
            if not isinstance(row.get("input"), str) or not row["input"].strip():
                raise ValueError("Each task needs a nonempty input")
            if row["id"] in ids:
                raise ValueError("Duplicate task ID")
            ids.add(row["id"])
            inputs.add(digest(" ".join(row["input"].lower().split())))
            groups.add(row.get("group_id", row["id"]))
        if ids & seen_ids or inputs & seen_inputs or groups & seen_groups:
            raise ValueError("Task, input or group leakage between splits")
        seen_ids |= ids
        seen_inputs |= inputs
        seen_groups |= groups


def partition(rows, n_train, n_val, n_test, seed, official_test=None):
    rng = random.Random(seed)
    groups = {}
    for row in rows:
        groups.setdefault(row.get("group_id", row["id"]), []).append(row)
    keys = sorted(groups)
    rng.shuffle(keys)
    # Reserve entire groups to one partition. Discard overflow questions from a
    # reserved group instead of moving them into another partition.
    result = {"train": [], "validation": [], "test": []}
    index = 0
    for split, size in (("train", n_train), ("validation", n_val), ("test", 0 if official_test is not None else n_test)):
        while len(result[split]) < size:
            if index >= len(keys):
                raise ValueError("Not enough independent task groups for requested splits")
            result[split].extend(groups[keys[index]][:size - len(result[split])])
            index += 1
    if official_test is not None:
        test_rows = list(official_test)
        rng.shuffle(test_rows)
        result["test"] = test_rows[:n_test]
        if len(result["test"]) < n_test:
            raise ValueError("Not enough test examples")
    return result


def fetch(key):
    path = ROOT / "data" / "raw" / (key + ".jsonl")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URLS[key], timeout=60) as response:
            contents = response.read()
        path.write_bytes(contents)
    return path


def prepare(name, output, n_train=4, n_val=12, n_test=20, seed=1729, train_path=None, test_path=None):
    sources, official_test = [], None
    if name == "gsm8k":
        paths = [Path(train_path) if train_path else fetch("gsm8k_train"), Path(test_path) if test_path else fetch("gsm8k_test")]
        def convert(path, prefix):
            return [{"id": f"{prefix}-{i}", "input": r["question"], "answers": [r["answer"].rsplit("####", 1)[-1].strip()]} for i, r in enumerate(read_rows(path))]
        rows, official_test = convert(paths[0], "train"), convert(paths[1], "test")
        sources = paths
    elif name == "drop":
        path = Path(train_path) if train_path else ROOT / "vendor/adas/dataset/drop_v0_dev.jsonl.gz"
        with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        rows = []
        for i, row in enumerate(records):
            prompt = row.get("inputs") or "Read the passage and answer the question.\n" + row["context"]
            passage = prompt.split("Question:", 1)[0]
            targets = row.get("targets") or row["ref_text"].split("|")
            rows.append({"id": f"drop-{i}", "group_id": digest(passage), "input": prompt,
                         "answers": targets if isinstance(targets, list) else [targets]})
        sources = [path]
    elif name == "mbpp":
        path = Path(train_path) if train_path else fetch("mbpp")
        converted = []
        for row in read_rows(path):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(row["code"])
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
            called = {n.func.id for test in row["test_list"] for n in ast.walk(ast.parse(test))
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            targets = [f for f in functions if f.name in called] or functions[:1]
            signature = "\n".join(ast.unparse(f).split("\n", 1)[0] for f in targets)
            converted.append({"id": str(row["task_id"]), "input": row["text"] + "\nReturn only Python code implementing:\n" + signature,
                              "tests": row["test_list"], "setup": row.get("test_setup_code", "")})
        # Original MBPP task-ID partitions: test 11..510, validation 511..600,
        # training 601..974. Use training for public examples and validation
        # for search feedback; selection never uses original test tasks.
        train_rows = [r for r in converted if int(r["id"]) >= 601]
        val_rows = [r for r in converted if 511 <= int(r["id"]) <= 600]
        test_rows = [r for r in converted if 11 <= int(r["id"]) <= 510]
        rng = random.Random(seed)
        for subset in (train_rows, val_rows, test_rows):
            rng.shuffle(subset)
        splits = {"train": train_rows[:n_train], "validation": val_rows[:n_val], "test": test_rows[:n_test]}
        if any(len(splits[s]) < size for s, size in (("train", n_train), ("validation", n_val), ("test", n_test))):
            raise ValueError("Requested more tasks than the official partition contains")
        sources = [path]
    elif name.startswith("agentdojo_"):
        from agentdojo.task_suite.load_suites import get_suite
        suite_name = name.removeprefix("agentdojo_")
        suite = get_suite("v1.2.2", suite_name)
        rows = [{"id": task.ID, "group_id": task.ID, "input": task.PROMPT,
                 "suite": suite_name, "user_task": task.ID} for task in suite.user_tasks.values()]
        sources = [ROOT / "vendor/agentdojo/src/agentdojo/task_suite/task_suite.py"]
    else:
        raise ValueError("Unsupported benchmark")
    if name != "mbpp":
        splits = partition(rows, n_train, n_val, n_test, seed, official_test)
    bundle = {"name": name, "fixture": False, "split_seed": seed, **splits,
              "provenance": [{"path": str(p), "sha256": __import__("hashlib").sha256(p.read_bytes()).hexdigest()} for p in sources]}
    bundle["sha256"] = digest(bundle)
    validate_splits(bundle)
    save(output, bundle)
    return bundle


def fixtures():
    for kind in ("gsm8k", "drop", "mbpp"):
        bundle = {"name": kind, "fixture": True, "split_seed": 1729}
        for split, offset, count in (("train", 0, 2), ("validation", 10, 2), ("test", 30, 3)):
            rows = []
            for i in range(offset, offset + count):
                outcome = ("correct", "wrong", "error")[(i - offset) % 3] if split == "test" else ("correct", "wrong")[(i - offset) % 2]
                prompt = f"Fixture question: {i} plus 2"
                row = {"id": f"{kind}-{i}", "input": prompt, "answers": [str(i + 2)]}
                if kind == "drop":
                    row.update(input=f"Fixture passage: The flag for town {i} is blue. What color is it?", answers=["blue"])
                if kind == "mbpp":
                    row.update(input=f"Fixture code: Implement add(a, b). Example number {i}.",
                               tests=[f"assert add({i}, 2) == {i + 2}", "assert add(-1, 1) == 0"], setup="")
                row["input"] += f"\n[fixture outcome: {outcome}]"
                row["fixture_outcome"] = outcome
                rows.append(row)
            bundle[split] = rows
        bundle["sha256"] = digest(bundle)
        validate_splits(bundle)
        yield bundle
