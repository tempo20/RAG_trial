import tempfile
import unittest
from pathlib import Path

import tgrag_setup
from chatter import resolve_query_target


class _FakeResult:
    def __init__(self, rows=None):
        self._rows = rows or []

    def data(self):
        return self._rows


class _FakeSession:
    def __init__(self, rows=None):
        self._rows = rows or []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, *_args, **_kwargs):
        return _FakeResult(self._rows)


class _FakeDriver:
    def __init__(self, rows=None):
        self._rows = rows or []

    def session(self):
        return _FakeSession(self._rows)


class TestFinancialEntityLinker(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        base = Path(self.tmp_dir.name)
        self.ticker_map = base / "ticker_company_map.csv"
        self.fin_map = base / "financial_entity_map.csv"

        self.ticker_map.write_text(
            "\n".join(
                [
                    "ticker,company_name,aliases",
                    "AAPL,Apple Inc.,Apple;Apple Inc",
                    "MSFT,Microsoft Corporation,Microsoft;Microsoft Corp",
                ]
            ),
            encoding="utf-8",
        )

        self.fin_map.write_text(
            "\n".join(
                [
                    "canonical_id,display_name,entity_type,ticker,aliases",
                    "federal-reserve,Federal Reserve,ORG,,Fed;Federal Reserve System;FOMC",
                    "european-central-bank,European Central Bank,ORG,,ECB",
                ]
            ),
            encoding="utf-8",
        )

        self.alias_to_fin = tgrag_setup.load_financial_entity_map(self.fin_map, self.ticker_map)
        self.alias_to_ticker, self.ticker_to_canonical = tgrag_setup.load_ticker_company_map(self.ticker_map)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_exact_company_link_uses_ticker_canonical(self):
        linked = tgrag_setup.link_financial_entity("Apple Inc.", "ORG", self.alias_to_fin)
        self.assertIsNotNone(linked)
        self.assertEqual(linked["canonical_name"], "AAPL")
        self.assertEqual(linked["ticker"], "AAPL")
        self.assertEqual(linked["link_method"], "exact")

    def test_exact_non_ticker_link_uses_slug_canonical(self):
        linked = tgrag_setup.link_financial_entity("Fed", "ORG", self.alias_to_fin)
        self.assertIsNotNone(linked)
        self.assertEqual(linked["canonical_name"], "federal-reserve")
        self.assertIsNone(linked["ticker"])
        self.assertEqual(linked["entity_type"], "ORG")

    def test_fuzzy_link_above_threshold(self):
        if tgrag_setup.fuzz is None:
            self.skipTest("rapidfuzz not installed in this environment")
        linked = tgrag_setup.link_financial_entity(
            "Federal Reserv",
            "ORG",
            self.alias_to_fin,
            fuzzy_threshold=90,
        )
        self.assertIsNotNone(linked)
        self.assertEqual(linked["canonical_name"], "federal-reserve")
        self.assertEqual(linked["link_method"], "fuzzy")

    def test_link_below_threshold_rejected(self):
        if tgrag_setup.fuzz is None:
            self.skipTest("rapidfuzz not installed in this environment")
        linked = tgrag_setup.link_financial_entity(
            "Fedral Reserv",
            "ORG",
            self.alias_to_fin,
            fuzzy_threshold=101,
        )
        self.assertIsNone(linked)

    def test_extract_entities_includes_link_metadata_and_fallbacks(self):
        chunks = [
            {
                "chunk_uid": "c1",
                "text": "Apple rallied while the Fed held rates. John Doe commented from France.",
                "period_key": "2026-W15",
                "article_id": "a1",
            }
        ]

        class FakePipe:
            def __call__(self, _texts):
                return [[
                    {"entity_group": "ORG", "word": "Apple"},
                    {"entity_group": "MISC", "word": "Fed"},
                    {"entity_group": "PER", "word": "John Doe"},
                    {"entity_group": "LOC", "word": "France"},
                    {"entity_group": "ORG", "word": "Unknown Corp"},
                ]]

        mentions = tgrag_setup.extract_entities_from_chunks(chunks, FakePipe(), self.alias_to_fin)
        by_canonical = {m["canonical_name"]: m for m in mentions}

        self.assertIn("AAPL", by_canonical)
        self.assertEqual(by_canonical["AAPL"]["link_method"], "exact")
        self.assertIn("federal-reserve", by_canonical)
        self.assertEqual(by_canonical["federal-reserve"]["link_method"], "exact")
        self.assertIn("john doe", by_canonical)
        self.assertEqual(by_canonical["john doe"]["link_method"], "ner_heuristic")
        self.assertIn("france", by_canonical)
        self.assertNotIn("unknown corp", by_canonical)

    def test_query_resolution_supports_financial_aliases(self):
        driver = _FakeDriver(rows=[])
        target = resolve_query_target(
            query="latest update on Fed policy",
            alias_to_ticker=self.alias_to_ticker,
            ticker_to_canonical=self.ticker_to_canonical,
            alias_to_fin_entity=self.alias_to_fin,
            driver=driver,
        )
        self.assertEqual(target.canonical_name, "federal-reserve")
        self.assertEqual(target.entity_type, "ORG")

    def test_query_resolution_keeps_ticker_path(self):
        driver = _FakeDriver(rows=[])
        target = resolve_query_target(
            query="AAPL news this week",
            alias_to_ticker=self.alias_to_ticker,
            ticker_to_canonical=self.ticker_to_canonical,
            alias_to_fin_entity=self.alias_to_fin,
            driver=driver,
        )
        self.assertEqual(target.canonical_name, "AAPL")
        self.assertEqual(target.ticker, "AAPL")


if __name__ == "__main__":
    unittest.main()
