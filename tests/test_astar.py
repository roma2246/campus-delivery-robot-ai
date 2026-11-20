"""A* search tests (owner: Ramazan G.).

Run from the project root:
    python3 -m unittest tests.test_astar -v
"""

from __future__ import annotations

import io
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
    a_star,
    manhattan_distance,
    neighbors,
    uniform_cost_search,
)


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
