"""Protect the user-frozen proposal; this test makes no model claims."""

import hashlib
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PROPOSAL_SHA256 = (
    "fe8ea7bea24429e7543067dfcb4a7cd519fe60cf6aef93764df6cebc000e28bb"
)


class FrozenProposalTests(unittest.TestCase):
    def test_proposal_matches_original_approved_source(self):
        actual = hashlib.sha256((ROOT / "docs/proposal.tex").read_bytes()).hexdigest()
        self.assertEqual(
            actual, ORIGINAL_PROPOSAL_SHA256,
            "docs/proposal.tex is user-frozen. Record implementation updates "
            "in README.md or other documents; do not modify the proposal.",
        )


if __name__ == "__main__":
    unittest.main()
