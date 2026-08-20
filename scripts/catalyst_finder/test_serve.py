from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path

from scripts.catalyst_finder.serve import (
    CatalystFinderRuntime,
    _candidate_match,
    _fallback_queries,
    canonical_rhea_id,
)


class CatalystFinderUnitTests(unittest.TestCase):
    def test_canonical_rhea_id(self) -> None:
        self.assertEqual(canonical_rhea_id("RHEA:33983"), "RHEA:33983")
        self.assertEqual(canonical_rhea_id("33983"), "RHEA:33983")

    def test_candidate_match_prefers_requested_orientation(self) -> None:
        equation = "(+)-copalyl diphosphate = miltiradiene + diphosphate"
        score, orientation = _candidate_match(
            equation,
            ["(+)-copalyl diphosphate"],
            ["miltiradiene"],
        )
        self.assertGreater(score, 6.0)
        self.assertEqual(orientation, "forward")

    def test_candidate_match_can_detect_reverse_orientation(self) -> None:
        equation = "miltiradiene + diphosphate = (+)-copalyl diphosphate"
        _, orientation = _candidate_match(
            equation,
            ["(+)-copalyl diphosphate"],
            ["miltiradiene"],
        )
        self.assertEqual(orientation, "reverse")

    def test_fallback_queries_include_both_sides(self) -> None:
        queries = _fallback_queries(["(+)-copalyl diphosphate"], ["miltiradiene"])
        self.assertTrue(any("AND" in query for query in queries))
        self.assertTrue(any("miltiradiene" in query for query in queries))

    def test_feedback_is_persisted_as_jsonl(self) -> None:
        runtime = CatalystFinderRuntime()
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime.feedback_path = Path(tmpdir) / "feedback.jsonl"
            result = runtime.submit_feedback({
                "rating": "helpful",
                "category": "results",
                "message": "候选结果很有帮助",
                "contact": "",
                "context": {"direction": "reaction_to_enzyme", "route_id": "r2e-current-top10-v1"},
            })
            self.assertTrue(result["ok"])
            rows = runtime.feedback_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            payload = json.loads(rows[0])
            self.assertEqual(payload["rating"], "helpful")
            self.assertEqual(payload["category"], "results")
            self.assertEqual(payload["context"]["direction"], "reaction_to_enzyme")
            self.assertEqual(stat.S_IMODE(runtime.feedback_path.stat().st_mode), 0o600)

    def test_missing_key_does_not_block_exact_rhea_mode(self) -> None:
        runtime = CatalystFinderRuntime()
        payload = runtime.resolve("RHEA:33983")
        self.assertEqual(payload["recommended_id"], "RHEA:33983")
        self.assertIn("miltiradiene", payload["candidates"][0]["equation"].lower())

    def test_result_scope_ui_is_prompt_only_not_stateful_selector(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn("scope-prompt-hints", html)
        self.assertIn("只看当前知识库已经记录的反应–酶关联", html)
        self.assertIn("排除当前知识库已经记录的反应–酶关联", html)
        self.assertNotIn("data-result-scope", html)
        self.assertNotIn("resultScopeOverride", js)
        self.assertNotIn("known_association_policy:", js)
        start = js.index("function wirePolicyPromptButtons")
        end = js.index("function resetConversation", start)
        prompt_handler = js[start:end]
        self.assertIn("input.value", prompt_handler)
        self.assertNotIn("sendPrompt(", prompt_handler)
        self.assertNotIn("setRouteMode(", prompt_handler)

    def test_mixed_result_labels_are_not_rendered_in_single_scope_modes(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn("if (mode.mixed)", js)
        self.assertIn('row.known_association ? "已知" : "潜在"', js)

    def test_continuation_carries_scope_as_context_without_locking_direction(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn("previous_association_policy", js)
        self.assertIn("previous_result_mode", js)
        self.assertIn("associationPolicy: continuationMode?.policy", js)
        self.assertIn("const effectiveHint = directionHint", js)
        self.assertNotIn("continuedHint", js)
        self.assertIn("directionHintOneShot", js)

    def test_right_rail_is_default_collapsed_and_expandable(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="railToggle"', html)
        self.assertIn('aria-expanded="false"', html)
        self.assertIn('id="runRail"', html)
        self.assertIn('class="workspace rail-collapsed"', html)
        self.assertIn('.workspace.rail-collapsed{grid-template-columns:minmax(0,1fr)}', css)
        self.assertIn('.workspace.rail-collapsed .run-rail{display:none}', css)
        self.assertIn('function setRailCollapsed(collapsed)', js)
        self.assertIn('setRailCollapsed(true)', js)
        self.assertIn('railToggle?.addEventListener("click"', js)

    def test_single_reaction_starter_is_explicitly_distinct_from_route_design(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        self.assertIn('data-direction-template="reaction_to_enzyme" data-prompt="我想把【底物】转化为【产物】', html)
        self.assertIn("为单步反应寻找候选酶", html)
        self.assertIn("底物 → 产物是一条目标反应", html)
        self.assertIn('data-direction-template="route_design" data-prompt="请推荐从【起始前体】到【目标产物】', html)

    def test_route_design_ui_is_natural_language_first_without_priority_selector(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn("从起始前体设计到目标产物的路线", html)
        self.assertIn('data-direction-template="route_design"', html)
        self.assertIn("起始前体 → 目标产物允许有中间步骤", html)
        self.assertIn('"route_design", "pathway_compatibility"', js)
        self.assertNotIn("data-route-priority", html)
        self.assertNotIn("routePriorityOverride", js)
        self.assertIn('resolution.direction === "route_design"', js)
        self.assertIn('endpoint: "/api/route/design"', js)
        self.assertIn("eQuilibrator MDF", js)
        self.assertIn("iML1515", js)
        self.assertIn("MDF 未覆盖", js)
        self.assertNotIn("data-route-priority", html)
        self.assertNotIn("routePriorityOverride", js)
        start = js.index("function renderRouteDesignResult")
        end = js.index("function renderResult", start)
        result_renderer = js[start:end]
        self.assertIn("route-design-template-action", result_renderer)
        self.assertIn("input.value", result_renderer)
        self.assertNotIn("sendPrompt(", result_renderer)

    def test_pathway_ui_is_natural_language_first_without_new_selector(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn("评估整条反应路径的酶兼容性", html)
        self.assertIn('data-direction-template="pathway_compatibility"', html)
        self.assertNotIn('data-direction="pathway_compatibility"', html)
        self.assertNotIn('data-pathway-mode', html)
        self.assertIn('resolution.direction === "pathway_compatibility"', js)
        self.assertIn('endpoint: "/api/pathway/analyze"', js)


if __name__ == "__main__":
    unittest.main()
