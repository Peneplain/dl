"""A bounded physical pilot must pass execution checks before expansion."""

import unittest

from scripts.risk_quality_pipeline import check_pair_pilot, check_parent_batch


class QualityGateTests(unittest.TestCase):
    def test_task_failures_count_but_execution_errors_stop(self):
        summary = {"status": "complete", "planned": 5, "completed": 5, "pending": 0,
                   "attempts": [{"status": "stopped"} for _ in range(5)]}
        check_parent_batch(summary)
        for changed in ({"pending": 1}, {"completed": 4}, {"status": "blocked"},
                        {"attempts": [{"status": "failed"}]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                check_parent_batch({**summary, **changed})

    def test_three_valid_pairs_allow_review_not_unvalidated_expansion(self):
        report = {"planned_pairs": 12, "pending_pairs": 9,
                  "jobs": {str(i): {"status": "paired", "recovery_verified": False} for i in range(3)}}
        check_pair_pilot(report)
        for changed in ({"jobs": {}}, {"planned_pairs": 0},
                        {"jobs": {str(i): {"status": "failed"} for i in range(3)}}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                check_pair_pilot({**report, **changed})


if __name__ == "__main__":
    unittest.main()
