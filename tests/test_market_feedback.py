import unittest

from rag_trial.analysis.market_feedback import compare_predicted_vs_actual


class TestMarketFeedback(unittest.TestCase):
    def test_predicted_up_and_actual_up_is_hit(self) -> None:
        outcome = compare_predicted_vs_actual(
            "up",
            {"return_1d": 0.01, "return_3d": 0.015, "return_5d": 0.02},
        )
        self.assertEqual(outcome, "hit")

    def test_predicted_up_and_actual_down_is_miss(self) -> None:
        outcome = compare_predicted_vs_actual(
            "up",
            {"return_1d": -0.01, "return_3d": -0.015, "return_5d": -0.02},
        )
        self.assertEqual(outcome, "miss")


if __name__ == "__main__":
    unittest.main()
