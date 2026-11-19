"""Mini Campus Delivery Robot: A*, Bayesian inference, and a one-step MDP."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from typing import Iterable

Position = tuple[int, int]

GRID_SIZE = 5
START: Position = (0, 0)
GOAL: Position = (4, 4)
WALLS: set[Position] = {(1, 1), (2, 1)}


@dataclass(frozen=True)
class SensorModel:
    prior_blocked: float
    alert_if_blocked: float
    alert_if_clear: float


@dataclass(frozen=True)
class MDPRewards:
    move_if_clear: float = 10.0
    move_if_blocked: float = -20.0
    wait_if_clear: float = -2.0
    wait_if_blocked: float = 1.0


TRANSITION_BLOCKED = {
    "Move": {"clear": 0.10, "blocked": 0.80},
    "Wait": {"clear": 0.05, "blocked": 0.40},
}
STATE_VALUE = {"clear": 5.0, "blocked": -10.0}
DISCOUNT = 0.9


def neighbors(position: Position) -> Iterable[Position]:
    row, col = position
    for next_position in (
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    ):
        r, c = next_position
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and next_position not in WALLS:
            yield next_position


def manhattan_distance(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(
    came_from: dict[Position, Position], current: Position
) -> list[Position]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return list(reversed(path))


def a_star(start: Position, goal: Position) -> tuple[list[Position], int]:
    """Return a shortest path and the number of expanded states."""
    if start in WALLS or goal in WALLS:
        raise ValueError("Start and goal must not be walls.")
    if not (0 <= start[0] < GRID_SIZE and 0 <= start[1] < GRID_SIZE):
        raise ValueError("Start must be inside the grid.")
    if not (0 <= goal[0] < GRID_SIZE and 0 <= goal[1] < GRID_SIZE):
        raise ValueError("Goal must be inside the grid.")

    sequence = count()
    frontier: list[tuple[int, int, Position]] = []
    heappush(frontier, (manhattan_distance(start, goal), next(sequence), start))
    came_from: dict[Position, Position] = {}
    cost_so_far = {start: 0}
    expanded = 0

    while frontier:
        _, _, current = heappop(frontier)
        expanded += 1
        if current == goal:
            return reconstruct_path(came_from, current), expanded

        for neighbor in neighbors(current):
            new_cost = cost_so_far[current] + 1
            if new_cost < cost_so_far.get(neighbor, float("inf")):
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                priority = new_cost + manhattan_distance(neighbor, goal)
                heappush(frontier, (priority, next(sequence), neighbor))

    raise ValueError("No route exists between start and goal.")


def bayes_obstacle_probability(
    model: SensorModel, sensor_alert: bool
) -> tuple[float, float, float]:
    """Return P(alert), P(blocked | observation), and P(clear | observation)."""
    p_blocked = model.prior_blocked
    p_clear = 1.0 - p_blocked

    if sensor_alert:
        likelihood_blocked = model.alert_if_blocked
        likelihood_clear = model.alert_if_clear
    else:
        likelihood_blocked = 1.0 - model.alert_if_blocked
        likelihood_clear = 1.0 - model.alert_if_clear

    p_observation = (
        likelihood_blocked * p_blocked + likelihood_clear * p_clear
    )
    if p_observation == 0:
        raise ValueError("The sensor observation has zero probability.")

    posterior_blocked = likelihood_blocked * p_blocked / p_observation
    return p_observation, posterior_blocked, 1.0 - posterior_blocked


def mdp_action_values(
    p_blocked: float, rewards: MDPRewards
) -> dict[str, float]:
    """Calculate one Bellman update for each action from the current belief."""
    p_clear = 1.0 - p_blocked
    values = {}
    for action in ("Move", "Wait"):
        if action == "Move":
            reward_by_state = {
                "clear": rewards.move_if_clear,
                "blocked": rewards.move_if_blocked,
            }
        else:
            reward_by_state = {
                "clear": rewards.wait_if_clear,
                "blocked": rewards.wait_if_blocked,
            }

        value = 0.0
        for state, state_probability in (
            ("clear", p_clear),
            ("blocked", p_blocked),
        ):
            p_next_blocked = TRANSITION_BLOCKED[action][state]
            expected_next_value = (
                (1.0 - p_next_blocked) * STATE_VALUE["clear"]
                + p_next_blocked * STATE_VALUE["blocked"]
            )
            value += state_probability * (
                reward_by_state[state] + DISCOUNT * expected_next_value
            )
        values[action] = value
    return values


def run_demo(sensor_alert: bool = True) -> None:
    path, expanded = a_star(START, GOAL)
    next_cell = path[1]

    sensor = SensorModel(
        prior_blocked=0.15,
        alert_if_blocked=0.85,
        alert_if_clear=0.25,
    )
    p_observation, p_blocked, p_clear = bayes_obstacle_probability(
        sensor, sensor_alert
    )
    values = mdp_action_values(p_blocked, MDPRewards())
    action = max(values, key=values.get)
    observation = "ALERT" if sensor_alert else "NO ALERT"

    print("Mini Campus Delivery Robot")
    print("==========================")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}; start={START}; goal={GOAL}; walls={sorted(WALLS)}")
    print("\n1) A* route planning")
    print(f"   Heuristic: Manhattan distance; step cost: 1")
    print(f"   Route: {path}")
    print(f"   Route length: {len(path) - 1} moves; expanded states: {expanded}")
    print(f"   Next corridor cell to check: {next_cell}")

    print("\n2) Bayesian sensor update")
    print(f"   Prior P(blocked) = {sensor.prior_blocked:.3f}")
    print(f"   Sensor reading = {observation}")
    print(
        "   P(reading) = "
        f"P(reading|blocked)P(blocked) + "
        f"P(reading|clear)P(clear) = {p_observation:.3f}"
    )
    print(
        f"   Posterior P(blocked|reading) = {p_blocked:.3f}; "
        f"P(clear|reading) = {p_clear:.3f}"
    )

    print("\n3) One-step MDP action choice")
    print("   State: corridor blocked or clear; actions: Move or Wait")
    print(
        "   P(next blocked | clear, Move) = "
        f"{TRANSITION_BLOCKED['Move']['clear']:.2f}; "
        "P(next blocked | blocked, Move) = "
        f"{TRANSITION_BLOCKED['Move']['blocked']:.2f}"
    )
    print(
        "   P(next blocked | clear, Wait) = "
        f"{TRANSITION_BLOCKED['Wait']['clear']:.2f}; "
        "P(next blocked | blocked, Wait) = "
        f"{TRANSITION_BLOCKED['Wait']['blocked']:.2f}"
    )
    print(
        "   Rewards (clear/blocked): Move=+10/-20, Wait=-2/+1; "
        "V(clear)=+5, V(blocked)=-10; gamma=0.90"
    )
    print(
        f"   One-step Bellman value(Move) = {values['Move']:.3f}; "
        f"value(Wait) = {values['Wait']:.3f}"
    )
    print(f"   Recommended action: {action}")
    print(
        "\nExplanation: A* proposes the shortest static route, Bayes updates "
        "the risk of its next corridor from the sensor, and the MDP chooses "
        "the action with the larger expected reward."
    )


def self_test() -> None:
    path, _ = a_star(START, GOAL)
    assert path[0] == START and path[-1] == GOAL
    assert all(cell not in WALLS for cell in path)
    assert len(path) - 1 == 8

    model = SensorModel(0.15, 0.85, 0.25)
    _, p_blocked_after_alert, _ = bayes_obstacle_probability(model, True)
    _, p_blocked_without_alert, _ = bayes_obstacle_probability(model, False)
    assert abs(p_blocked_after_alert - 0.375) < 1e-9
    assert p_blocked_after_alert > model.prior_blocked
    assert p_blocked_without_alert < model.prior_blocked
    values_after_alert = mdp_action_values(p_blocked_after_alert, MDPRewards())
    values_without_alert = mdp_action_values(
        p_blocked_without_alert, MDPRewards()
    )
    assert max(values_after_alert, key=values_after_alert.get) == "Wait"
    assert max(values_without_alert, key=values_without_alert.get) == "Move"
    print("All checks passed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-alert",
        action="store_true",
        help="show the alternative sensor reading and MDP decision",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="run built-in checks instead of the demonstration",
    )
    args = parser.parse_args()

    if args.test:
        self_test()
    else:
        run_demo(sensor_alert=not args.no_alert)
