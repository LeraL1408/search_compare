"""Optional real container grading of trusted upstream reference programs."""
import json
import os
import unittest

from searchlab.common import ROOT
from searchlab.execution import Worker


@unittest.skipUnless(os.getenv("SEARCHLAB_CONTAINER_CHECK") == "1", "Set SEARCHLAB_CONTAINER_CHECK=1 to run Docker reference checks")
class ContainerReferenceTests(unittest.TestCase):
    def test_selected_mbpp_reference_programs_pass_actual_grader(self):
        dataset = json.loads((ROOT / "data/mbpp.json").read_text())
        originals = {str(r["task_id"]): r for r in (json.loads(line) for line in (ROOT / "data/raw/mbpp.jsonl").read_text().splitlines() if line.strip())}
        worker = Worker({"docker_image": "python:3.12-slim", "task_timeout_seconds": 10})
        for split in ("train", "validation", "test"):
            for row in dataset[split]:
                with self.subTest(split=split, task=row["id"]):
                    self.assertTrue(worker.run({"op": "grade_code", "code": originals[row["id"]]["code"],
                                                "setup": row.get("setup", ""), "tests": row["tests"]}))


if __name__ == "__main__":
    unittest.main()
