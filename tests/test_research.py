import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from searchlab.common import AFLOW_BASELINE, BASELINE, ROOT, digest, save
from searchlab.datasets import fixtures, partition, prepare, validate_splits
from searchlab.evaluation import Evaluator, TaskSession, numeric_answer
from searchlab.execution import Worker
from searchlab.gateway import BudgetStop, Gateway, Ledger
from searchlab.methods import SearchContext, run_aflow, run_adas
from searchlab.runner import bundles_for, expand_agentdojo, load_config, run


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.config = load_config(ROOT / "configs/smoke.json")
        self.config["search"]["iterations"] = 1
        self.config["seeds"] = [11]
        self.worker = Worker(self.config["execution"], fixture=True)

    def tearDown(self):
        self.temp.cleanup()

    def gateway(self, **overrides):
        limits = {**self.config["budgets"]["search"], **overrides}
        ledger = Ledger(limits, self.directory / "calls.jsonl")
        return Gateway(self.config["models"], ledger, 11, fixture=True)

    def evaluator(self, gateway=None, rows=None):
        bundle = next(fixtures())
        evaluation = {**self.config["evaluation"], "answer_instruction": "Return a number."}
        return Evaluator("gsm8k", rows or bundle["validation"], gateway or self.gateway(), self.worker,
                         evaluation, self.directory, "validation")

    def test_split_rejects_same_input_with_different_ids(self):
        bundle = next(fixtures())
        bundle["test"][0]["input"] = "  " + bundle["train"][0]["input"].upper()
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_splits(bundle)

    def test_split_reserves_passage_groups_without_exceeding_size(self):
        rows = [{"id": str(i), "input": str(i), "group_id": str(i // 10)} for i in range(50)]
        split = partition(rows, 2, 3, 4, 7)
        self.assertEqual([len(split[s]) for s in ("train", "validation", "test")], [2, 3, 4])
        validate_splits(split)

    def test_call_budget_stops_before_request(self):
        gateway = self.gateway(max_calls=1)
        gateway.call([{"role": "user", "content": "Fixture question: 1 plus 2"}])
        with self.assertRaises(BudgetStop):
            gateway.call([{"role": "user", "content": "should not execute"}])
        self.assertEqual(gateway.ledger.calls, 1)

    def test_dollar_reservation_stops_before_request(self):
        gateway = self.gateway(max_usd=0.000001)
        gateway.models = copy.deepcopy(gateway.models)
        gateway.models["execution"]["input_usd_per_million"] = 10
        with self.assertRaises(BudgetStop):
            gateway.call([{"role": "user", "content": "too expensive"}])
        self.assertEqual(gateway.ledger.calls, 0)

    def test_missing_usage_charges_reservation_and_stops(self):
        gateway = self.gateway()
        gateway.fixture = False
        client = Mock()
        response = Mock()
        response.usage = None
        response.choices = [types.SimpleNamespace(message=types.SimpleNamespace(model_dump=lambda **kw: {"role": "assistant", "content": "answer"}))]
        client.chat.completions.create.return_value = response
        gateway.clients["execution"] = client
        with self.assertRaises(BudgetStop):
            gateway.call([{"role": "user", "content": "hello"}])
        self.assertTrue(gateway.ledger.uncertain)
        self.assertGreater(gateway.ledger.tokens, 0)
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_worker_runs_both_interfaces(self):
        for candidate in (BASELINE, AFLOW_BASELINE):
            result = self.evaluator().evaluate(candidate, "check")
            self.assertEqual(result["score"], 0.5)
            self.assertEqual(result["errors"], 0)

    def test_controlled_wrong_answers_and_failures_are_not_full_credit(self):
        for bundle in fixtures():
            with self.subTest(benchmark=bundle["name"]):
                config = {**self.config["evaluation"], "answer_instruction": "Solve the task."}
                evaluator = Evaluator(bundle["name"], bundle["test"], self.gateway(), self.worker,
                                      config, self.directory, "test")
                result = evaluator.evaluate(BASELINE, "controlled outcomes")
                self.assertEqual(result["scores"], [1, 0, 0])
                self.assertAlmostEqual(result["score"], 1 / 3)
                self.assertEqual(result["errors"], 2 if bundle["name"] == "mbpp" else 1)

    def test_drop_metric_awards_partial_credit(self):
        from searchlab.evaluation import drop_scorer
        _, f1 = drop_scorer()("blue", ["blue flag"])
        self.assertGreater(f1, 0)
        self.assertLess(f1, 100)

    def test_fixture_report_has_no_performance_leaderboard(self):
        from searchlab.report import report
        from searchlab.common import save
        from searchlab.datasets import fixtures
        budget = self.gateway().ledger.snapshot()
        row = {"benchmark": "gsm8k", "method": "aflow", "seed": 11,
               "search": {"status": "complete", "error": None, "budget": budget},
               "test": {"status": "complete", "budget": budget,
                        "result": {"score": 1, "count": 3, "errors": 0, "usd": 0}}}
        save(self.directory / "manifest.json", {"fixture": True})
        save(self.directory / "results.json", [row])
        save(self.directory / "gsm8k/splits.json", next(fixtures()))
        report(self.directory)
        markdown = (self.directory / "report.md").read_text()
        html = (self.directory / "report.html").read_text()
        csv = (self.directory / "results.csv").read_text()
        self.assertNotIn("1.000", markdown)
        self.assertNotIn("Test score", markdown)
        self.assertNotIn("Paired improvement", markdown)
        self.assertNotIn("<svg", html)
        self.assertIn("check_score", csv)
        self.assertIn("software_check", csv)
        self.assertNotIn("test_score", csv)
        save(self.directory / "manifest.json", {"fixture": False})
        report(self.directory)
        self.assertIn("Test score", (self.directory / "report.md").read_text())
        self.assertIn("1.000", (self.directory / "report.md").read_text())
        self.assertIn("test_score", (self.directory / "results.csv").read_text())

    def test_worker_enforces_timeout(self):
        config = {**self.config["execution"], "task_timeout_seconds": 0.15}
        worker = Worker(config, fixture=True)
        candidate = {"kind": "adas", "code": "def forward(self, taskInfo):\n    while True:\n        pass", "prompt": ""}
        with self.assertRaises(TimeoutError):
            worker.run({"candidate": candidate, "input": "task", "benchmark": "test"})

    def test_code_grader_catches_wrong_code_and_accepts_stdlib(self):
        request = {"op": "grade_code", "code": "import math\ndef square(x):\n    return math.pow(x, 2)", "tests": ["assert square(3) == 9"]}
        self.assertTrue(self.worker.run(request))
        request["tests"] = ["assert square(3) == 10"]
        with self.assertRaises(RuntimeError):
            self.worker.run(request)

    def test_worker_does_not_expose_answers_or_credentials(self):
        candidate = {"kind": "adas", "code": "def forward(self, taskInfo):\n    return open('/etc/passwd').read()", "prompt": ""}
        with self.assertRaises(RuntimeError):
            self.worker.run({"candidate": candidate, "input": "task", "benchmark": "test"})

    def test_interrupted_validation_does_not_replace_incumbent(self):
        gateway = self.gateway(max_calls=2)
        context = SearchContext(self.evaluator(gateway), gateway, self.directory, next(fixtures())["train"], self.config["search"])
        context.evaluate(BASELINE, "completed")
        old = copy.deepcopy(context.best)
        with self.assertRaises(BudgetStop):
            context.evaluate(AFLOW_BASELINE, "interrupted")
        self.assertEqual(old, context.best)
        self.assertEqual(len(context.history), 1)

    def test_selection_uses_validation_only(self):
        gateway = self.gateway()
        evaluator = self.evaluator(gateway)
        context = SearchContext(evaluator, gateway, self.directory, next(fixtures())["train"], self.config["search"])
        context.evaluate(BASELINE, "correct")
        wrong = {"kind": "adas", "code": "def forward(self, taskInfo):\n    return 'wrong'", "prompt": ""}
        context.evaluate(wrong, "wrong")
        self.assertEqual(context.best["candidate"], BASELINE)
        self.assertNotIn("Fixture question: 30", context.domain)

    def test_original_aflow_search_step_produces_history_and_experience(self):
        gateway = self.gateway()
        context = SearchContext(self.evaluator(gateway), gateway, self.directory, next(fixtures())["train"], self.config["search"])
        run_aflow(context, 11)
        self.assertEqual(len(context.history), 2)
        self.assertTrue((self.directory / "upstream/SHARED/workflows/round_2/experience.json").exists())
        self.assertTrue((self.directory / "upstream/SHARED/workflows/round_1/log.json").exists())

    def test_original_adas_search_evaluates_archive_and_new_generation(self):
        gateway = self.gateway()
        context = SearchContext(self.evaluator(gateway), gateway, self.directory, next(fixtures())["train"], self.config["search"])
        run_adas(context, 11)
        self.assertEqual(len(context.history), 8)
        archive = json.loads((self.directory / "upstream/adas_run_archive.json").read_text())
        self.assertEqual(archive[-1]["generation"], 1)

    def test_numeric_scoring(self):
        self.assertEqual(numeric_answer("Work: 3 + 4. Final: \\boxed{1,234.50}"), numeric_answer("1234.5"))
        self.assertIsNone(numeric_answer("No answer"))

    def test_dataset_fingerprint_checked(self):
        path = self.directory / "bundle.json"
        bundle = next(fixtures())
        bundle["fixture"] = False
        save(path, bundle)
        config = {**self.config, "fixture": False, "datasets": [str(path)]}
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            bundles_for(config)

    def test_incomplete_test_has_no_reported_score(self):
        self.config["budgets"]["evaluation"]["max_calls"] = 2
        self.config["search"]["iterations"] = 1
        # Limit to one benchmark to test the entire freeze/test/report path.
        with patch("searchlab.runner.bundles_for", return_value=[next(fixtures())]):
            directory = run(self.config, self.directory / "run")
        results = json.loads((directory / "results.json").read_text())
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result["test"]["status"], "incomplete")
            self.assertNotIn("result", result["test"])
            self.assertTrue((directory / "gsm8k/seed_11" / result["method"] / "frozen.json").exists())


class AgentDojoTests(unittest.TestCase):
    def test_ground_truth_through_adapter_and_environment_reset(self):
        from agentdojo.task_suite.load_suites import get_suite
        suite = get_suite("v1.2.2", "workspace")
        task = suite.get_user_task_by_id("user_task_0")

        class GroundTruthWorker:
            def run(self, request, callback):
                session = callback.__self__
                for call in task.ground_truth(session.env):
                    session.run_tool(call.function, call.args)
                return task.GROUND_TRUTH_OUTPUT or ""

        config = load_config(ROOT / "configs/smoke.json")
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(config["budgets"]["evaluation"], Path(tmp) / "calls.jsonl")
            gateway = Gateway(config["models"], ledger, 11, True)
            row = {"id": task.ID, "input": task.PROMPT, "suite": "workspace", "user_task": task.ID}
            evaluator = Evaluator("agentdojo_workspace", [row], gateway, GroundTruthWorker(),
                                  {**config["evaluation"], "answer_instruction": ""}, Path(tmp), "validation")
            for _ in range(2):
                result = evaluator.evaluate(BASELINE, "ground-truth adapter check")
                self.assertEqual(result["score"], 1)
                self.assertEqual(result["errors"], 0)

    def test_attack_success_direction_and_fixed_attack(self):
        from agentdojo.task_suite.load_suites import get_suite
        suite = get_suite("v1.2.2", "workspace")
        user = suite.get_user_task_by_id("user_task_0")
        goal = suite.get_injection_task_by_id("injection_task_0")

        class AttackerGroundTruthWorker:
            def run(self, request, callback):
                session = callback.__self__
                for call in goal.ground_truth(session.env):
                    session.run_tool(call.function, call.args)
                return "Done"

        config = load_config(ROOT / "configs/smoke.json")
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(config["budgets"]["evaluation"], Path(tmp) / "calls.jsonl")
            gateway = Gateway(config["models"], ledger, 11, True)
            row = {"id": "attack", "input": user.PROMPT, "suite": "workspace", "user_task": user.ID, "injection_task": goal.ID}
            evaluation = {**config["evaluation"], "agentdojo_attack": "ignore_previous", "answer_instruction": ""}
            evaluator = Evaluator("agentdojo_workspace", [row], gateway, AttackerGroundTruthWorker(), evaluation, Path(tmp), "validation")
            result = evaluator.evaluate(BASELINE, "attack check")
            self.assertEqual(result["errors"], 0)
            self.assertEqual(result["attack_success"], 1)
            self.assertEqual(result["score"], 0)


if __name__ == "__main__":
    unittest.main()
