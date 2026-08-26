from __future__ import annotations

import json
import stat
import tempfile
import time
import unittest
from pathlib import Path

from scripts.catalyst_finder.serve import (
    CatalystFinderRuntime,
    _candidate_match,
    _explicit_uniprot_accession,
    _fallback_queries,
    canonical_rhea_id,
)
from scripts.catalyst_finder.protein_resolution import ProteinResolver


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

    def test_explicit_uniprot_accession_is_extracted_from_natural_language(self) -> None:
        self.assertEqual(_explicit_uniprot_accession("查看 UniProt P00338 的 3 个优先反应"), "P00338")
        self.assertEqual(_explicit_uniprot_accession("查看这个酶的优先反应"), "")

    def test_seen_bonus_does_not_create_unrelated_local_match(self) -> None:
        resolver = ProteinResolver.__new__(ProteinResolver)
        row = {
            "id": "E8W6C7",
            "uniprot_id": "E8W6C7",
            "genbank_id": None,
            "name": "germacradienol synthase",
            "species": "Streptomyces pratensis",
            "seen": True,
        }
        score = resolver._local_score(
            row, protein_terms=[], organism_terms=[], gene_terms=[], accession_terms=["P00338"]
        )
        self.assertEqual(score, 0.0)

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

    def test_run_event_is_persisted_with_prompt_and_private_mode(self) -> None:
        runtime = CatalystFinderRuntime()
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime.run_events_path = Path(tmpdir) / "run_events.jsonl"
            result = runtime.record_run_event(
                event_type="candidate_ranking",
                session_id="sess_test",
                run_id="run_test",
                step_id="step_test",
                input_data={"final_user_prompt": "请找催化 A 到 B 的酶"},
                output_data={"candidates": [{"candidate_id": "P123"}]},
                metadata={"card_id": "reaction_to_enzyme", "prompt_source": "shortcut_card"},
            )
            self.assertTrue(result["ok"])
            payload = json.loads(runtime.run_events_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], "run_test")
            self.assertEqual(payload["input"]["final_user_prompt"], "请找催化 A 到 B 的酶")
            self.assertEqual(payload["output"]["candidates"][0]["candidate_id"], "P123")
            self.assertEqual(stat.S_IMODE(runtime.run_events_path.stat().st_mode), 0o600)

    def test_pending_steps_are_grouped_for_one_run(self) -> None:
        runtime = CatalystFinderRuntime()
        runtime.hold_run_step("run_test", {"step_type": "intent_and_entity_resolution"})
        runtime.hold_run_step("run_test", {"step_type": "candidate_ranking"})
        steps = runtime.take_run_steps("run_test")
        self.assertEqual([step["step_type"] for step in steps], ["intent_and_entity_resolution", "candidate_ranking"])
        self.assertEqual(runtime.take_run_steps("run_test"), [])

    def test_stale_pending_run_steps_are_pruned(self) -> None:
        runtime = CatalystFinderRuntime()
        runtime.hold_run_step("stale", {"step_type": "intent_and_entity_resolution"})
        runtime._pending_run_started["stale"] = time.time() - 3700
        runtime.hold_run_step("fresh", {"step_type": "intent_and_entity_resolution"})
        self.assertNotIn("stale", runtime._pending_run_steps)
        self.assertNotIn("stale", runtime._pending_run_started)
        self.assertIn("fresh", runtime._pending_run_steps)

    def test_pending_run_step_cache_is_bounded(self) -> None:
        runtime = CatalystFinderRuntime()
        for index in range(300):
            runtime.hold_run_step(f"run_{index}", {"step_type": "intent", "index": index})
        self.assertLessEqual(len(runtime._pending_run_steps), 256)
        self.assertLessEqual(len(runtime._pending_run_started), 256)
        for index in range(20):
            runtime.hold_run_step("same_run", {"step_type": "intent", "index": index})
        self.assertEqual(len(runtime._pending_run_steps["same_run"]), 8)
        self.assertEqual(runtime._pending_run_steps["same_run"][-1]["index"], 19)

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
        self.assertIn('data-policy-prompt-en="Restore the default scope: keep database-recorded associations as evidence and also show unrecorded candidates."', html)
        self.assertIn('data-policy-prompt-zh="恢复默认结果范围：保留数据库已知关联作为证据，同时展示尚未记录的新关联候选。"', html)
        self.assertIn('data-policy-prompt-en="Show only database-recorded associations."', html)
        self.assertIn('data-policy-prompt-zh="只看数据库已经记录的反应–酶关联。"', html)
        self.assertIn('data-policy-prompt-en="Exclude database-recorded associations and show only unrecorded candidate associations."', html)
        self.assertIn('data-policy-prompt-zh="排除数据库已经记录的反应–酶关联，只看尚未记录的新关联候选。"', html)
        self.assertNotIn("data-result-scope", html)
        self.assertNotIn("resultScopeOverride", js)
        self.assertNotIn("known_association_policy:", js)
        start = js.index("function wirePolicyPromptButtons")
        end = js.index("function resetConversation", start)
        prompt_handler = js[start:end]
        self.assertIn("input.value", prompt_handler)
        self.assertNotIn("sendPrompt(", prompt_handler)
        self.assertNotIn("setRouteMode(", prompt_handler)

    def test_results_separate_database_evidence_from_unrecorded_candidate_ranking(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        self.assertIn('const known = result.known_associations', js)
        self.assertIn('const discoveryRows = mode.knownOnly ? [] : (result.candidates || [])', js)
        self.assertIn('tr("Known enzymes", "已知酶")', js)
        self.assertIn('tr("Database evidence only", "仅数据库证据")', js)
        self.assertIn('tr("Discovery candidates", "新关联候选酶")', js)
        self.assertIn('Recorded database evidence; not a model prediction', js)
        self.assertNotIn('row.known_association ? "已知" : "潜在"', js)
        self.assertIn('.evidence-section', css)
        self.assertIn('.discovery-section', css)

    def test_bilingual_ui_defaults_to_english_isolates_sessions_and_keeps_chinese_jargon_free(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        i18n = (frontend / "i18n.js").read_text(encoding="utf-8")
        serve = (Path(__file__).resolve().parent / "serve.py").read_text(encoding="utf-8")
        self.assertIn('<html lang="en">', html)
        self.assertIn('id="languageToggle"', html)
        self.assertIn('<script src="/i18n.js" defer></script>', html)
        self.assertIn('data-prompt-en=', html)
        self.assertIn('data-prompt-zh=', html)
        self.assertIn('localStorage.getItem(STORAGE_KEY) || "en"', i18n)
        self.assertIn('location.reload()', i18n)
        self.assertIn('catalyst_finder_session_id_${uiLanguage}', js)
        self.assertIn('tr("Follow-up request:", "用户后续要求：")', js)
        self.assertIn('ui_language: uiLanguage', js)
        self.assertNotRegex(html, r'(?i)(?:data-zh|data-prompt-zh|data-placeholder-zh|data-aria-zh)="[^"]*discovery')
        self.assertNotIn('"Discovery 模型已覆盖"', js)
        self.assertNotIn('已知证据 + discovery', js)
        self.assertNotIn('个 discovery 候选', js)
        self.assertNotIn('数据库事实与 discovery 模型覆盖独立展示', serve)
        self.assertNotIn('模型分数只用于 discovery 候选', serve)
        self.assertIn("call unrecorded model-ranked associations '新关联候选'", serve)

    def test_progressive_capability_guide_exposes_supported_research_scenarios(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        css = (frontend / "styles.css").read_text(encoding="utf-8")
        self.assertIn('id="capabilityGuide"', html)
        self.assertEqual(html.count('class="capability-action"'), 23)
        self.assertIn('class="starter-grid primary-task-grid"', html)
        for label in [
            "给一个反应，寻找催化酶",
            "给一个酶，寻找可能反应",
            "从起始前体设计到目标产物的路线",
            "评估多步路径的酶组合",
            "只找尚未记录的新关联",
            "寻找更远缘的蛋白家族",
            "仅真核候选",
            "仅原核候选",
            "分析模型目录外的 UniProt 蛋白",
            "从已有活性继续扩展",
            "优先热力学可行性",
            "优先 E. coli 宿主通量",
            "优先项目模型覆盖",
            "只用数据库已知反应",
            "探索预测反应步骤",
            "一锅多酶反应",
            "分步执行反应",
            "细胞内代谢路径",
            "指定 pH、温度和辅因子",
        ]:
            with self.subTest(label=label):
                self.assertIn(label, html)
        self.assertIn('.capability-guide', css)
        self.assertIn('.capability-actions', css)
        self.assertNotIn('data-route-priority', html)
        self.assertNotIn('data-pathway-mode', html)

    def test_single_reaction_starter_is_explicitly_distinct_from_route_design(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        self.assertIn('data-direction-template="reaction_to_enzyme" data-prompt-en="I want to convert [substrate] to [product].', html)
        self.assertIn('data-prompt-zh="我想把【底物】转化为【产物】', html)
        self.assertIn('data-en="Find enzymes for a reaction"', html)
        self.assertIn('data-direction-template="route_design" data-prompt-en="Recommend biosynthetic routes from [starting precursor] to [target product]', html)
        self.assertIn('data-prompt-zh="请推荐从【起始前体】到【目标产物】', html)

    def test_route_design_ui_is_natural_language_first_without_priority_selector(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        html = (frontend / "index.html").read_text(encoding="utf-8")
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn("从起始前体设计到目标产物的路线", html)
        self.assertIn('data-direction-template="route_design"', html)
        self.assertIn("优先热力学可行性", html)
        self.assertIn("优先 E. coli 宿主通量", html)
        self.assertIn("只用数据库已知反应", html)
        self.assertIn("探索预测反应步骤", html)
        self.assertIn('"route_design", "pathway_compatibility"', js)
        self.assertNotIn("data-route-priority", html)
        self.assertNotIn("routePriorityOverride", js)
        self.assertIn('resolution.direction === "route_design"', js)
        self.assertIn('endpoint: "/api/route/design"', js)
        self.assertIn("eQuilibrator MDF", js)
        self.assertIn("iML1515", js)
        self.assertIn("MDF 未覆盖", js)
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
        self.assertIn("评估多步路径的酶组合", html)
        self.assertIn("一锅多酶反应", html)
        self.assertIn("分步执行反应", html)
        self.assertIn("细胞内代谢路径", html)
        self.assertIn("指定 pH、温度和辅因子", html)
        self.assertIn('data-direction-template="pathway_compatibility"', html)
        self.assertNotIn('data-direction="pathway_compatibility"', html)
        self.assertNotIn('data-pathway-mode', html)
        self.assertIn('resolution.direction === "pathway_compatibility"', js)
        self.assertIn('endpoint: "/api/pathway/analyze"', js)

    def test_edited_starter_template_releases_soft_direction_hint(self) -> None:
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "catalyst_finder"
        js = (frontend / "app.js").read_text(encoding="utf-8")
        self.assertIn('const starterWasEdited = Boolean(run.card_id && run.prompt_template && text !== run.prompt_template);', js)
        self.assertIn('const effectiveHint = directionHintOneShot && starterWasEdited ? "auto" : directionHint;', js)
        self.assertIn('edited_after_card_click: Boolean(run.card_id && text !== run.prompt_template)', js)

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


if __name__ == "__main__":
    unittest.main()
