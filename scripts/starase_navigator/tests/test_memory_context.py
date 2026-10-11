from __future__ import annotations

import unittest

from scripts.starase_navigator.agent_harness.contracts import PrepareCandidateRetrievalArgs
from scripts.starase_navigator.agent_harness.tool_registry import ScientificToolRegistry
from scripts.starase_navigator.retrieval.memory_context import canonical_memory_context
from scripts.starase_navigator.routing.language import _bounded_context_value


class BridgeDualMemoryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.query = {
            "model_expert": "bridge_final",
            "train_memory_relation_count": 26,
            "relation_context_weight": 0.62,
            "episodic_memory_applied": False,
            "episodic_support_count": 0,
        }

    def test_frozen_long_term_memory_remains_without_episodic_support(self) -> None:
        result = canonical_memory_context(self.query, {"seed_source": "none"})
        self.assertEqual(result["mode"], "long_term_only")
        self.assertEqual(result["long_term_memory"]["source"], "clean2023_frozen_training_graph")
        self.assertEqual(result["long_term_memory"]["query_relation_count"], 26)
        self.assertNotIn("episodic_memory", result)

    def test_inductive_e2r_uses_graph_authority_without_fabricated_support_count(self) -> None:
        readout = {
            "model_expert": "bridge_final",
            "route_id": "bridge-e2r-inductive-bipartite-graph",
            "evidence_authority": {"long_term_relation": 0.42},
            "observed_enzyme_neighbors": 16,
            "inductive_reaction_neighbors": 8,
            "episodic_memory_applied": False,
        }
        context = canonical_memory_context(readout, {"seed_source": "none"})
        long_term = context["long_term_memory"]
        self.assertEqual(context["mode"], "long_term_only")
        self.assertEqual(long_term["readout"], "inductive_two_sided_relation_graph")
        self.assertEqual(long_term["gate_weight"], 0.42)
        self.assertNotIn("query_relation_count", long_term)

    def test_confirmed_new_relation_can_activate_query_local_memory(self) -> None:
        self.query.update({
            "episodic_memory_applied": True,
            "episodic_support_count": 2,
            "episodic_memory_weight": 0.43,
            "training_seed_count": 1,
        })
        result = canonical_memory_context(
            self.query, {"seed_source": "user_confirmed"},
        )
        self.assertEqual(result["mode"], "long_term_plus_episodic")
        self.assertEqual(result["episodic_memory"]["support_count"], 2)
        self.assertEqual(result["episodic_memory"]["training_graph_duplicates_ignored"], 1)

    def test_catalog_associations_cannot_be_attested_as_episodic_support(self) -> None:
        self.query.update({
            "episodic_memory_applied": True,
            "episodic_support_count": 1,
            "episodic_memory_weight": 0.5,
        })
        for source in ("catalog_known_associations", "none", "model_prediction"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                canonical_memory_context(self.query, {"seed_source": source})

    def test_training_duplicate_does_not_create_an_episodic_expert(self) -> None:
        self.query.update({
            "episodic_memory_applied": False,
            "requested_seed_count": 1,
            "training_seed_count": 1,
        })
        result = canonical_memory_context(
            self.query, {"seed_source": "user_confirmed"},
        )
        self.assertEqual(result["mode"], "long_term_only")
        self.assertNotIn("episodic_memory", result)

    def test_other_historical_rankers_do_not_claim_bridge_memory(self) -> None:
        self.query["model_expert"] = "legacy_correspondence"
        self.assertEqual(canonical_memory_context(self.query, {}), {})


class ModelVisibleMemorySemanticsTests(unittest.TestCase):
    def test_legacy_seed_fields_are_not_model_visible(self) -> None:
        schema = PrepareCandidateRetrievalArgs.model_json_schema()["properties"]
        self.assertNotIn("seed_policy", schema)
        self.assertEqual(
            set(schema["episodic_memory_policy"]["enum"]),
            {"confirmed_only", "disabled"},
        )

    def test_ordinary_model_tool_call_cannot_auto_select_catalog_memory(self) -> None:
        args = PrepareCandidateRetrievalArgs(
            direction="reaction_to_enzyme",
            full_text="Rank reactions without verified positive evidence",
        )
        self.assertEqual(
            ScientificToolRegistry._episodic_support_route(
                args, has_confirmed_positives=False,
            ),
            "none",
        )
        legacy = PrepareCandidateRetrievalArgs.model_validate({
            "direction": "reaction_to_enzyme",
            "full_text": "An old client requested catalog guidance",
            "seed_policy": "default",
        })
        self.assertEqual(
            ScientificToolRegistry._episodic_support_route(
                legacy, has_confirmed_positives=False,
            ),
            "none",
        )

    def test_model_context_omits_legacy_modes_and_preserves_true_memory(self) -> None:
        source = {
            "ranking": {
                "route_id": "bridge-final-r2e-v1+fewshot",
                "shot_mode": "episodic_few_shot",
                "memory_context": {"mode": "long_term_plus_episodic"},
            },
            "routing": {
                "seed_mode": "catalog_known",
                "seed_source": "catalog_known_associations",
                "known_enzyme_ids": ["A0AA82WP13"],
            },
        }
        bounded = _bounded_context_value(source, max_depth=8)
        self.assertEqual(
            bounded["ranking"]["route_id"],
            "bridge-final-r2e-v1+episodic-memory",
        )
        self.assertEqual(
            bounded["ranking"]["memory_context"]["mode"],
            "long_term_plus_episodic",
        )
        self.assertNotIn("shot_mode", bounded["ranking"])
        self.assertNotIn("seed_mode", bounded["routing"])
        self.assertNotIn("seed_source", bounded["routing"])


if __name__ == "__main__":
    unittest.main()
