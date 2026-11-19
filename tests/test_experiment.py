"""Tests for the Monte Carlo experiment in experiment.py."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiment  # noqa: E402
from experiment import (  # noqa: E402
    always_move_policy,
    always_wait_policy,
    bayes_mdp_policy,
    run_episode,
    sensor_reflex_policy,
    simulate,
    summarise,
)

EPISODES = 2_000
SEED = 11


def mean_discounted(policy, false_alarm=experiment.ALERT_IF_CLEAR):
    return summarise("x", simulate(policy, EPISODES, SEED, false_alarm)).mean_discounted


class EpisodeTests(unittest.TestCase):
    def test_same_seed_gives_same_result(self):
        first = simulate(bayes_mdp_policy, 200, SEED)
        second = simulate(bayes_mdp_policy, 200, SEED)
        self.assertEqual([e.reward for e in first], [e.reward for e in second])

    def test_delivered_episode_makes_exactly_route_length_moves(self):
        episode = run_episode(bayes_mdp_policy, 8, random.Random(3))
        self.assertTrue(episode.delivered)
        self.assertEqual(episode.turns - episode.waits - episode.collisions, 8)

    def test_always_wait_never_delivers(self):
        summary = summarise("wait", simulate(always_wait_policy, 200, SEED))
        self.assertEqual(summary.success_rate, 0.0)
        self.assertEqual(summary.mean_waits, experiment.MAX_TURNS)


class PolicyComparisonTests(unittest.TestCase):
    def test_bayes_mdp_beats_ignoring_the_sensor(self):
        ours = summarise("ours", simulate(bayes_mdp_policy, EPISODES, SEED))
        blind = summarise("move", simulate(always_move_policy, EPISODES, SEED))
        self.assertGreater(ours.mean_discounted, blind.mean_discounted)
        self.assertLess(ours.mean_collisions, blind.mean_collisions / 5)
        self.assertEqual(ours.success_rate, 1.0)

    def test_matches_reflex_on_project_sensor(self):
        # With P(alert|clear) = 0.25 the MDP threshold (0.300) sits between the
        # posteriors after "no alert" and after "alert", so both policies agree.
        self.assertAlmostEqual(
            mean_discounted(bayes_mdp_policy), mean_discounted(sensor_reflex_policy)
        )

    def test_beats_reflex_on_noisy_sensor(self):
        self.assertGreater(
            mean_discounted(bayes_mdp_policy, 0.6),
            mean_discounted(sensor_reflex_policy, 0.6) + 3,
        )


class CalibrationTests(unittest.TestCase):
    def test_posterior_matches_observed_frequency(self):
        rows = experiment.calibration(simulate(bayes_mdp_policy, EPISODES, SEED))
        self.assertTrue(rows)
        for label, count, predicted, observed in rows:
            if count > 1_000:
                self.assertAlmostEqual(predicted, observed, delta=0.02, msg=label)


if __name__ == "__main__":
    unittest.main()
