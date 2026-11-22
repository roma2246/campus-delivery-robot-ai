"""Bayesian sensor update tests (owner: Merey S.).

Run from the project root:
    python3 -m unittest tests.test_bayes -v
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import (  # noqa: E402
    SensorModel,
    bayes_obstacle_probability,
)

SENSOR = SensorModel(prior_blocked=0.15, alert_if_blocked=0.85, alert_if_clear=0.25)


class BayesTests(unittest.TestCase):
    def test_alert_posterior_matches_hand_calculation(self):
        evidence, blocked, clear = bayes_obstacle_probability(SENSOR, True)
        self.assertAlmostEqual(evidence, 0.85 * 0.15 + 0.25 * 0.85)
        self.assertAlmostEqual(evidence, 0.34)
        self.assertAlmostEqual(blocked, 0.375)
        self.assertAlmostEqual(clear, 0.625)

    def test_no_alert_posterior_matches_hand_calculation(self):
        evidence, blocked, _ = bayes_obstacle_probability(SENSOR, False)
        self.assertAlmostEqual(evidence, 0.66)
        self.assertAlmostEqual(blocked, 0.15 * 0.15 / 0.66)
        self.assertAlmostEqual(blocked, 0.0341, places=4)

    def test_posteriors_sum_to_one(self):
        for prior in (0.01, 0.15, 0.5, 0.9):
            model = SensorModel(prior, 0.85, 0.25)
            for alert in (True, False):
                _, blocked, clear = bayes_obstacle_probability(model, alert)
                self.assertAlmostEqual(blocked + clear, 1.0)

    def test_alert_raises_and_silence_lowers_risk(self):
        for prior in (0.05, 0.15, 0.5, 0.8):
            model = SensorModel(prior, 0.85, 0.25)
            self.assertGreater(bayes_obstacle_probability(model, True)[1], prior)
            self.assertLess(bayes_obstacle_probability(model, False)[1], prior)

    def test_certain_priors_do_not_change(self):
        for prior in (0.0, 1.0):
            model = SensorModel(prior, 0.85, 0.25)
            for alert in (True, False):
                self.assertAlmostEqual(bayes_obstacle_probability(model, alert)[1], prior)

    def test_impossible_observation_raises(self):
        with self.assertRaises(ValueError):
            bayes_obstacle_probability(SensorModel(0.15, 0.0, 0.0), True)

    def test_posterior_agrees_with_monte_carlo_frequency(self):
        rng = random.Random(7)
        alerts = blocked_and_alert = 0
        for _ in range(200_000):
            blocked = rng.random() < SENSOR.prior_blocked
            likelihood = SENSOR.alert_if_blocked if blocked else SENSOR.alert_if_clear
            if rng.random() < likelihood:
                alerts += 1
                blocked_and_alert += blocked
        self.assertAlmostEqual(blocked_and_alert / alerts, 0.375, delta=0.01)


if __name__ == "__main__":
    unittest.main()
