import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from searchlab.common import ROOT
from searchlab.datasets import fixtures
from searchlab.gateway import BudgetStop, Gateway, Ledger
from searchlab.runner import load_config, run


class SharedBudgetTests(unittest.TestCase):
    def test_sibling_phases_share_usage_and_call_ceiling(self):
        config = load_config(ROOT / "configs/smoke.json")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            total = Ledger({**config["budgets"]["search"], "max_calls": 2}, root / "total.json")
            benchmark = Ledger(config["budgets"]["search"], root / "benchmark.json", parent=total)
            for phase in ("search", "test"):
                leaf = Ledger(config["budgets"]["search"], root / (phase + ".jsonl"), parent=benchmark)
                Gateway(config["models"], leaf, 11, fixture=True).call([{"role": "user", "content": "Fixture question: 1 plus 2"}])
                self.assertGreater(leaf.tokens, 0)
            self.assertEqual(total.calls, 2)
            self.assertEqual(benchmark.calls, 2)
            self.assertEqual(total.tokens, benchmark.tokens)
            leaf = Ledger(config["budgets"]["search"], root / "third.jsonl", parent=benchmark)
            with self.assertRaises(BudgetStop):
                Gateway(config["models"], leaf, 11, fixture=True).call([{"role": "user", "content": "must not launch"}])
            self.assertEqual(total.calls, 2)
            self.assertEqual(json.loads((root / "total.json").read_text())["calls"], 2)

    def test_global_run_metrics_sum_phases_without_baseline_validation(self):
        config = load_config(ROOT / "configs/smoke.json")
        config["baseline_validation"] = False
        config["seeds"] = [11]
        config["search"]["iterations"] = 1
        config["budgets"]["global"] = {**config["budgets"]["search"], "max_calls": 300}
        with tempfile.TemporaryDirectory() as tmp:
            with patch("searchlab.runner.bundles_for", return_value=[next(fixtures())]):
                path = run(config, Path(tmp) / "run")
            rows = json.loads((path / "results.json").read_text())
            total = json.loads((path / "total_budget.json").read_text())
            expected_calls = sum(r["search"]["budget"]["calls"] + r["test"]["budget"]["calls"] for r in rows)
            expected_tokens = sum(r["search"]["budget"]["tokens"] + r["test"]["budget"]["tokens"] for r in rows)
            self.assertEqual(total["calls"], expected_calls)
            self.assertEqual(total["tokens"], expected_tokens)
            baseline = next(r for r in rows if r["method"] == "baseline")
            self.assertEqual(baseline["search"]["budget"]["calls"], 0)
            self.assertEqual(baseline["test"]["status"], "complete")
            frozen = json.loads((path / "gsm8k/seed_11/baseline/frozen.json").read_text())
            self.assertEqual(frozen["selection"]["split"], "predefined")
            self.assertIsNone(frozen["selection"]["score"])


if __name__ == "__main__":
    unittest.main()
