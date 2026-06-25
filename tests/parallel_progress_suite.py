import contextlib
import io
import sys
import time
import types
import unittest
from unittest import mock

try:
    from rag_trial.chat import chatter
    CHATTER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime environment guard
    chatter = types.ModuleType("chatter")
    sys.modules["chatter"] = chatter
    CHATTER_IMPORT_ERROR = exc


class TestParallelMapProgress(unittest.TestCase):
    def setUp(self) -> None:
        if CHATTER_IMPORT_ERROR is not None:
            self.skipTest(f"chatter import failed: {CHATTER_IMPORT_ERROR}")

    def test_parallel_map_ordered_preserves_input_order(self) -> None:
        items = [3, 2, 1, 0]

        def work(value: int) -> str:
            time.sleep(value * 0.001)
            return f"item-{value}"

        result = chatter._parallel_map_ordered(items, 4, work)

        self.assertEqual(result, ["item-3", "item-2", "item-1", "item-0"])

    def test_parallel_map_ordered_without_progress_is_quiet(self) -> None:
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = chatter._parallel_map_ordered([1, 2, 3], 2, lambda item: item * 10)

        self.assertEqual(result, [10, 20, 30])
        self.assertEqual(output.getvalue(), "")

    def test_parallel_map_ordered_prints_fallback_when_tqdm_unavailable(self) -> None:
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "tqdm":
                raise ImportError("blocked for fallback test")
            return real_import(name, *args, **kwargs)

        output = io.StringIO()
        with mock.patch("builtins.__import__", side_effect=fake_import):
            with contextlib.redirect_stdout(output):
                result = chatter._parallel_map_ordered(
                    [1, 2, 3],
                    2,
                    lambda item: item + 1,
                    progress_label="test progress",
                    progress_unit="case",
                )

        self.assertEqual(result, [2, 3, 4])
        text = output.getvalue()
        self.assertIn("[test progress] starting 3 case(s) with 2 worker(s)", text)
        self.assertIn("[test progress] completed 3/3 case(s)", text)


if __name__ == "__main__":
    unittest.main()
