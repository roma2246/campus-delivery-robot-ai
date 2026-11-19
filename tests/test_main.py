"""Unit tests for the three AI components in main.py.

Run from the project root:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import random
import sys
import unittest
from collections import deque
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main  # noqa: E402
from main import (  # noqa: E402
    GOAL,
    GRID_SIZE,
    START,
    WALLS,
    MDPRewards,
    SensorModel,
    a_star,
    bayes_obstacle_probability,
    break_even_probability,
    manhattan_distance,
    mdp_action_values,
    neighbors,
    uniform_cost_search,
)

SENSOR = SensorModel(prior_blocked=0.15, alert_if_blocked=0.85, alert_if_clear=0.25)


def bfs_distances(goal):
    """Independent reference: true shortest distance from every cell to goal."""
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        cell = queue.popleft()
        for nxt in neighbors(cell):
            if nxt not in distances:
                distances[nxt] = distances[cell] + 1
                queue.append(nxt)
    return distances


def free_cells():
    return [
        (r, c)
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if (r, c) not in WALLS
    ]


class AStarTests(unittest.TestCase):
    def test_route_starts_at_start_and_ends_at_goal(self):
        path, _ = a_star(START, GOAL)
        self.assertEqual(path[0], START)
        self.assertEqual(path[-1], GOAL)

    def test_route_is_connected_and_avoids_walls(self):
        path, _ = a_star(START, GOAL)
        for cell in path:
            self.assertNotIn(cell, WALLS)
        for a, b in zip(path, path[1:]):
            self.assertEqual(manhattan_distance(a, b), 1, f"{a} -> {b} is not one step")

    def test_route_is_optimal(self):
        path, _ = a_star(START, GOAL)
        self.assertEqual(len(path) - 1, bfs_distances(GOAL)[START])
        self.assertEqual(len(path) - 1, 8)

    def test_optimal_from_every_free_cell(self):
        truth = bfs_distances(GOAL)
        for cell in free_cells():
            path, _ = a_star(cell, GOAL)
            self.assertEqual(len(path) - 1, truth[cell], f"non-optimal from {cell}")

    def test_heuristic_is_admissible(self):
        truth = bfs_distances(GOAL)
        for cell in free_cells():
            self.assertLessEqual(manhattan_distance(cell, GOAL), truth[cell])

    def test_heuristic_is_consistent(self):
        for cell in free_cells():
            for nxt in neighbors(cell):
                self.assertLessEqual(
                    manhattan_distance(cell, GOAL),
                    1 + manhattan_distance(nxt, GOAL),
                )

    def test_heuristic_reduces_expanded_states(self):
        a_path, a_expanded = a_star(START, GOAL)
        u_path, u_expanded = uniform_cost_search(START, GOAL)
        self.assertEqual(len(a_path), len(u_path))
        self.assertEqual(a_expanded, 9)
        self.assertEqual(u_expanded, 23)
        self.assertLess(a_expanded, u_expanded)

    def test_start_on_wall_raises(self):
        with self.assertRaises(ValueError):
            a_star((1, 1), GOAL)

    def test_goal_outside_grid_raises(self):
        with self.assertRaises(ValueError):
            a_star(START, (5, 5))

    def test_unreachable_goal_raises(self):
        with mock.patch.object(main, "WALLS", {(3, 4), (4, 3)}):
            with self.assertRaises(ValueError):
                a_star(START, GOAL)

    def test_replans_around_new_wall(self):
        with mock.patch.object(main, "WALLS", WALLS | {(4, 1)}):
            path, _ = a_star(START, GOAL)
            self.assertNotIn((4, 1), path)
            self.assertEqual(path[-1], GOAL)


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


class DemoTests(unittest.TestCase):
    def run_demo_output(self, sensor_alert):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main.run_demo(sensor_alert=sensor_alert)
        return buffer.getvalue()

    def test_demo_with_alert_recommends_wait(self):
        output = self.run_demo_output(True)
        self.assertIn("Route length: 8 moves", output)
        self.assertIn("Posterior P(blocked|reading) = 0.375", output)
        self.assertIn("Recommended action: Wait", output)

    def test_demo_without_alert_recommends_move(self):
        output = self.run_demo_output(False)
        self.assertIn("Posterior P(blocked|reading) = 0.034", output)
        self.assertIn("Recommended action: Move", output)


if __name__ == "__main__":
    unittest.main()
