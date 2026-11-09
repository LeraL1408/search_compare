import copy
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from searchlab.codex_backend import CodexFailure, build_command, codex_environment, invoke, normalize_response, parse_events
from searchlab.common import AFLOW_BASELINE, BASELINE, ROOT, save
from searchlab.evaluation import Evaluator
from searchlab.execution import Worker
from searchlab.gateway import BudgetStop, Gateway, Ledger
from searchlab.report import report
from searchlab.runner import load_config


def events(answer="12", calls=None, usage=True):
    value = [{"type": "thread.started", "thread_id": "fresh-test"},
             {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps({"content": answer, "tool_calls": calls or []})}}]
    if usage:
        value.append({"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 10, "cached_input_tokens": 20}})
    return value


class CodexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {"provider": "codex", "binary": "codex", "model": "test-model", "timeout_seconds": 10,
                       "input_usd_per_million": None, "output_usd_per_million": None}

    def tearDown(self):
        self.temp.cleanup()

    def fake_binary(self, event_list=None, script=None):
        path = self.root / "fake-codex"
        body = script or "import json,sys\nsys.stdin.read()\nfor event in " + repr(event_list) + ":\n    print(json.dumps(event), flush=True)\n"
        path.write_text("#!" + sys.executable + "\n" + body)
        path.chmod(0o700)
        return str(path)

    def gateway(self, event_list=None, tokens=10000):
        model = {**self.config, "binary": self.fake_binary(events() if event_list is None else event_list)}
        ledger = Ledger({"max_calls": 5, "max_tokens": tokens, "max_usd": None, "max_seconds": 20}, self.root / "calls.jsonl")
        return Gateway({"search": model, "execution": model}, ledger, 11)

    def test_real_subprocess_parsing_and_unknown_dollar_cost(self):
        gateway = self.gateway()
        message = gateway.call([{"role": "user", "content": "7+5"}])
        self.assertEqual(message["content"], "12")
        budget = gateway.ledger.snapshot()
        self.assertEqual(budget["tokens"], 110)
        self.assertEqual(budget["cached_input_tokens"], 20)
        self.assertEqual(budget["codex_runs"], 1)
        self.assertIsNone(budget["usd"])
        self.assertTrue(budget["token_usage_complete"])
        self.assertTrue(list((self.root / "codex_events").glob("*.jsonl")))

    def test_no_api_keys_or_endpoint_overrides_in_child_environment(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret", "CODEX_API_KEY": "secret", "OPENAI_BASE_URL": "secret", "CODEX_HOME": "/saved-login"}):
            env = codex_environment()
        self.assertEqual(env["CODEX_HOME"], "/saved-login")
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("CODEX_API_KEY", env)
        self.assertNotIn("OPENAI_BASE_URL", env)

    def test_fresh_sessions_with_native_tools_disabled(self):
        command = build_command(self.config, "/isolated", "/schema.json")
        for flag in ("--ephemeral", "--ignore-user-config", "--json", "--output-schema"):
            self.assertIn(flag, command)
        self.assertIn('forced_login_method="chatgpt"', command)
        self.assertIn("features.shell_tool=false", command)
        self.assertIn("features.apps=false", command)
        self.assertIn("features.plugins=false", command)
        self.assertIn("project_doc_max_bytes=0", command)
        self.assertNotIn("resume", command)

    def test_benchmark_tool_normalization_and_foreign_tool_rejection(self):
        envelope = json.dumps({"content": "", "tool_calls": [{"name": "add", "arguments": '{"a":7,"b":5}'}]})
        tool = {"type": "function", "function": {"name": "add"}}
        response = normalize_response(envelope, [tool])
        self.assertEqual(response["tool_calls"][0]["function"]["name"], "add")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            normalize_response(envelope, [])

    def test_native_tool_events_invalidate_run(self):
        stream = events()
        stream.insert(1, {"type": "item.started", "item": {"type": "command_execution", "command": "forbidden"}})
        with self.assertRaisesRegex(CodexFailure, "native tool"):
            parse_events("\n".join(json.dumps(e) for e in stream), self.root / "events.jsonl")

    def test_startup_diagnostic_is_not_mistaken_for_a_tool_call(self):
        stream = events()
        stream.insert(1, {"type": "item.completed", "item": {"type": "error", "message": "Optional capability disabled"}})
        answer, usage = parse_events("\n".join(json.dumps(e) for e in stream), self.root / "events.jsonl")
        self.assertEqual(json.loads(answer)["content"], "12")
        self.assertEqual(usage["prompt_tokens"], 100)

    def test_missing_usage_marks_measurement_incomplete(self):
        gateway = self.gateway(events(usage=False))
        with self.assertRaises(BudgetStop):
            gateway.call([{"role": "user", "content": "task"}])
        self.assertFalse(gateway.ledger.token_usage_complete)
        self.assertEqual(gateway.ledger.failures, 1)
        self.assertIsNone(gateway.ledger.snapshot()["usd"])

    def test_token_overrun_stops_before_candidate_can_complete(self):
        gateway = self.gateway(tokens=100)
        with self.assertRaisesRegex(BudgetStop, "token limit"):
            gateway.call([{"role": "user", "content": "task"}])
        self.assertEqual(gateway.ledger.tokens, 110)
        self.assertEqual(gateway.ledger.calls, 1)

    def test_rejected_prompt_does_not_count_as_a_cli_launch(self):
        gateway = self.gateway()
        gateway.models["execution"]["max_prompt_bytes"] = 1
        with self.assertRaises(BudgetStop):
            gateway.call([{"role": "user", "content": "task"}])
        self.assertEqual(gateway.ledger.calls, 1)
        self.assertEqual(gateway.ledger.codex_runs, 0)

    def test_timeout_kills_cli_process(self):
        config = {**self.config, "binary": self.fake_binary(script="import sys,time\nsys.stdin.read()\ntime.sleep(5)\n")}
        with self.assertRaises(subprocess.TimeoutExpired):
            invoke(config, [], None, None, .1, self.root / "events.jsonl")

    def test_both_upstream_searches_work_with_codex_transport_and_null_cost(self):
        from searchlab.methods import SearchContext, run_aflow, run_adas
        config = json.loads((ROOT / "configs/smoke.json").read_text())
        source = "import json,re,sys\npayload=sys.stdin.read()\n"
        source += "if '<graph>' in payload:\n    content=" + repr("<modification>test mutation</modification><graph>" + AFLOW_BASELINE["code"] + "</graph><prompt></prompt>") + "\n"
        source += "else:\n    match=re.search(r'these fields: (\\[.*?\\])\\.',payload)\n    fields=json.loads(match[1]) if match else []\n"
        source += "    data={f:('True' if f=='correct' else '12') for f in fields}\n"
        source += "    if 'code' in fields: data.update(thought='test proposal',name='test agent',code=" + repr(BASELINE["code"]) + ")\n"
        source += "    content=json.dumps(data) if fields else '12'\n"
        source += "print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps({'content':content,'tool_calls':[]})}}))\n"
        source += "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':100,'output_tokens':10,'cached_input_tokens':20}}))\n"
        model = {**self.config, "binary": self.fake_binary(script=source)}
        for method in (run_aflow, run_adas):
            directory = self.root / method.__name__
            directory.mkdir()
            ledger = Ledger({"max_calls": 200, "max_tokens": 50000, "max_seconds": 60, "max_usd": None}, directory / "calls.jsonl")
            gateway = Gateway({"search": model, "execution": model}, ledger, 11)
            task = {"id": "validation", "input": "7+5", "answers": ["12"]}
            evaluator = Evaluator("gsm8k", [task], gateway, Worker(config["execution"], fixture=True),
                                  {**config["evaluation"], "answer_instruction": "Return a number"}, directory, "validation")
            context = SearchContext(evaluator, gateway, directory, [{**task, "id": "train"}],
                                    {**config["search"], "iterations": 1})
            method(context, 11)
            self.assertEqual(len(context.history), 2 if method == run_aflow else 8)
            self.assertTrue(all(item["result"]["usd"] is None for item in context.history))
            self.assertEqual(context.best["result"]["score"], 1)

    def test_codex_config_accepts_null_usd_budget_without_api_key(self):
        config = json.loads((ROOT / "configs/live.example.json").read_text())
        config["models"] = {"search": self.config, "execution": self.config}
        for limits in config["budgets"].values():
            limits["max_usd"] = None
        path = self.root / "config.json"
        save(path, config)
        self.assertEqual(load_config(path)["models"]["execution"]["provider"], "codex")
        config["budgets"]["search"]["max_usd"] = 0
        save(path, config)
        with self.assertRaisesRegex(ValueError, "null"):
            load_config(path)

    def test_unknown_cost_survives_evaluation_and_report(self):
        gateway = self.gateway(events(answer='{"answer":"12"}'))
        config = json.loads((ROOT / "configs/smoke.json").read_text())
        worker = Worker(config["execution"], fixture=True)
        evaluator = Evaluator("gsm8k", [{"id": "real-format-test", "input": "7+5", "answers": ["12"]}],
                              gateway, worker, {**config["evaluation"], "answer_instruction": "Return a number"}, self.root, "test")
        result = evaluator.evaluate(BASELINE, "test")
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["tokens"], 110)
        self.assertIsNone(result["usd"])
        budget = gateway.ledger.snapshot()
        save(self.root / "manifest.json", {"fixture": False})
        save(self.root / "results.json", [{"benchmark": "gsm8k", "method": "aflow", "seed": 11,
             "search": {"status": "complete", "error": None, "budget": budget},
             "test": {"status": "complete", "result": result, "budget": budget}}])
        report(self.root)
        self.assertIn("unknown", (self.root / "report.md").read_text())
        self.assertIn("search_cached_input_tokens", (self.root / "results.csv").read_text())


if __name__ == "__main__":
    unittest.main()
