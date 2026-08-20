from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.catalyst_finder.feedback_report import load_records, public_row, summarize


class FeedbackReportTests(unittest.TestCase):
    def test_summary_and_contact_redaction(self) -> None:
        rows = [
            {
                "feedback_id": "a",
                "submitted_at_unix": 1.0,
                "rating": "helpful",
                "category": "results",
                "message": "很好",
                "contact": "user@example.com",
                "context": {"direction": "reaction_to_enzyme", "result_mode": "全部候选", "route_id": "r2e"},
            },
            {
                "feedback_id": "b",
                "submitted_at_unix": 2.0,
                "rating": "needs_improvement",
                "category": "interaction",
                "message": "希望更清楚",
                "contact": None,
                "context": {"direction": "pathway_compatibility", "result_mode": "联合评估", "route_id": "pathway-compatibility-v1"},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feedback.jsonl"
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\nnot-json\n", encoding="utf-8")
            loaded, invalid = load_records(path)
        summary = summarize(loaded, invalid)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["invalid_lines"], 1)
        self.assertEqual(summary["with_contact"], 1)
        self.assertEqual(summary["direction"]["pathway_compatibility"], 1)
        self.assertEqual(public_row(rows[0])["contact"], "[redacted]")
        self.assertEqual(public_row(rows[0], include_contact=True)["contact"], "user@example.com")


if __name__ == "__main__":
    unittest.main()
