"""Tests for the HF model-card generator (models.yaml -> README.md)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from model_card import hf_repo_for, render_card

ENTRY = {
    "task": "diacritization",
    "precision": "fp32",
    "filename": "ara-diac-2.0-fp32.zip",
    "size": 2780299051,
    "metrics": [
        {"name": "der_total_greedy", "value": 2.2864,
         "source": "rababa/docs/RESULTS.md#r7-verdict"},
        {"name": "der_morph_greedy", "value": 1.3343,
         "source": "rababa/docs/RESULTS.md#r7-verdict"},
    ],
    "license": "BSD-3-Clause",
}
VARIANTS = ["ara-diac-2.0-int8"]


class TestHfRepo(unittest.TestCase):
    def test_repo_is_org_plus_index_id(self):
        self.assertEqual(hf_repo_for("ara-diac-2.0"), "interscript/ara-diac-2.0")

    def test_variant_ids_keep_their_own_repo(self):
        self.assertEqual(hf_repo_for("ara-diac-small-2.1-int8"),
                         "interscript/ara-diac-small-2.1-int8")


class TestRenderCard(unittest.TestCase):
    def test_frontmatter_fields(self):
        card = render_card("ara-diac-2.0", ENTRY, VARIANTS, body="## Benchmark\n\n2.29% DER.")
        self.assertIn("license: bsd-3-clause", card)
        self.assertIn("language: ar", card)
        self.assertIn("inference: false", card)
        self.assertIn("secryst", card)

    def test_metrics_table_with_provenance(self):
        card = render_card("ara-diac-2.0", ENTRY, VARIANTS, body="")
        self.assertIn("| der_total_greedy | 2.2864 |", card)
        self.assertIn("rababa/docs/RESULTS.md#r7-verdict", card)

    def test_task_and_size(self):
        card = render_card("ara-diac-2.0", ENTRY, VARIANTS, body="")
        self.assertIn("diacritization", card)
        self.assertIn("2.59 GiB", card)

    def test_variants_listed(self):
        card = render_card("ara-diac-2.0", ENTRY, VARIANTS, body="")
        self.assertIn("ara-diac-2.0-int8", card)

    def test_body_included(self):
        card = render_card("ara-diac-2.0", ENTRY, VARIANTS, body="## Benchmark\n\n2.29% DER.")
        self.assertIn("## Benchmark", card)
        self.assertIn("2.29% DER.", card)

    def test_language_mapping_per_task_language(self):
        card = render_card("tha-g2p-base-1.0", {**ENTRY, "task": "g2p"}, [], body="")
        self.assertIn("language: th", card)


if __name__ == "__main__":
    unittest.main()
