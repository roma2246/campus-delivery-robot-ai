"""One-step MDP decision tests (owner: Ali I.).

Run from the project root:
    python3 -m unittest tests.test_mdp -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from main import (  # noqa: E402
    MDPRewards,
    SensorModel,
    bayes_obstacle_probability,
    break_even_probability,
    mdp_action_values,
)

SENSOR = SensorModel(prior_blocked=0.15, alert_if_blocked=0.85, alert_if_clear=0.25)


class MDPTests(unittest.TestCase):
    def test_values_after_alert(self):
        values = mdp_action_values(0.375, MDPRewards())
        self.assertAlmostEqual(values["Move"], -1.644, places=3)
        self.assertAlmostEqual(values["Wait"], 1.178, places=3)
        self.assertEqual(max(values, key=values.get), "Wait")

    def test_values_without_alert(self):
        p_blocked = bayes_obstacle_probability(SENSOR, False)[1]
        values = mdp_action_values(p_blocked, MDPRewards())
        self.assertAlmostEqual(values["Move"], 11.805, places=3)
        self.assertAlmostEqual(values["Wait"], 1.766, places=3)
        self.assertEqual(max(values, key=values.get), "Move")

    def test_values_match_closed_form(self):
        # Q(Move) = 13.15 - 39.45 p and Q(Wait) = 1.825 - 1.725 p (see REPORT.md)
        for p in (0.0, 0.2, 0.5, 1.0):
            values = mdp_action_values(p, MDPRewards())
            self.assertAlmostEqual(values["Move"], 13.15 - 39.45 * p)
            self.assertAlmostEqual(values["Wait"], 1.825 - 1.725 * p)

    def test_break_even_probability(self):
        threshold = break_even_probability()
        self.assertAlmostEqual(threshold, 11.325 / 37.725)
        self.assertAlmostEqual(threshold, 0.300, places=3)
        values = mdp_action_values(threshold, MDPRewards())
        self.assertAlmostEqual(values["Move"], values["Wait"])

    def test_decision_flips_at_threshold(self):
        threshold = break_even_probability()
        below = mdp_action_values(threshold - 0.01, MDPRewards())
        above = mdp_action_values(threshold + 0.01, MDPRewards())
        self.assertEqual(max(below, key=below.get), "Move")
        self.assertEqual(max(above, key=above.get), "Wait")

    def test_transition_probabilities_are_valid(self):
        for action, row in main.TRANSITION_BLOCKED.items():
            for state, probability in row.items():
                self.assertGreaterEqual(probability, 0.0, f"{action}/{state}")
                self.assertLessEqual(probability, 1.0, f"{action}/{state}")


if __name__ == "__main__":
    unittest.main()
