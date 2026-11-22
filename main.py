"""Mini Campus Delivery Robot: A*, Bayesian inference, and a one-step MDP."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from typing import Callable, Iterable

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


def zero_heuristic(a: Position, b: Position) -> int:
    """Heuristic that ignores the goal: turns A* into uniform-cost search."""
    return 0


def a_star(
    start: Position,
    goal: Position,
    heuristic: Callable[[Position, Position], int] = manhattan_distance,
) -> tuple[list[Position], int]:
    """Return a shortest path and the number of expanded states.

    The frontier is ordered by f = g + h. Ties on f are broken in favour of
    the larger g (the node closer to the goal), then by insertion order, so
    the search follows the most promising branch instead of sweeping the map.
    """
    if start in WALLS or goal in WALLS:
        raise ValueError("Start and goal must not be walls.")
    if not (0 <= start[0] < GRID_SIZE and 0 <= start[1] < GRID_SIZE):
        raise ValueError("Start must be inside the grid.")
    if not (0 <= goal[0] < GRID_SIZE and 0 <= goal[1] < GRID_SIZE):
        raise ValueError("Goal must be inside the grid.")

    sequence = count()
    frontier: list[tuple[int, int, int, Position]] = []
    heappush(frontier, (heuristic(start, goal), 0, next(sequence), start))
    came_from: dict[Position, Position] = {}
    cost_so_far = {start: 0}
    closed: set[Position] = set()

    while frontier:
        _, _, _, current = heappop(frontier)
        if current in closed:
            continue  # stale queue entry for an already expanded cell
        closed.add(current)
        if current == goal:
            return reconstruct_path(came_from, current), len(closed)

        for neighbor in neighbors(current):
            new_cost = cost_so_far[current] + 1
            if new_cost < cost_so_far.get(neighbor, float("inf")):
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                priority = new_cost + heuristic(neighbor, goal)
                heappush(frontier, (priority, -new_cost, next(sequence), neighbor))

    raise ValueError("No route exists between start and goal.")


def uniform_cost_search(start: Position, goal: Position) -> tuple[list[Position], int]:
    """Uninformed baseline (Week 2): A* with h = 0."""
    return a_star(start, goal, heuristic=zero_heuristic)


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


def _immediate_reward(action: str, state: str, rewards: MDPRewards) -> float:
    if action == "Move":
        return rewards.move_if_clear if state == "clear" else rewards.move_if_blocked
    return rewards.wait_if_clear if state == "clear" else rewards.wait_if_blocked


def _expected_next_value(action: str, state: str, values: dict[str, float]) -> float:
    p_next_blocked = TRANSITION_BLOCKED[action][state]
    return (
        (1.0 - p_next_blocked) * values["clear"]
        + p_next_blocked * values["blocked"]
    )


def mdp_action_values(
    p_blocked: float, rewards: MDPRewards
) -> dict[str, float]:
    """Calculate one Bellman update for each action from the current belief."""
    p_clear = 1.0 - p_blocked
    values = {}
    for action in ("Move", "Wait"):
        value = 0.0
        for state, state_probability in (
            ("clear", p_clear),
            ("blocked", p_blocked),
        ):
            value += state_probability * (
                _immediate_reward(action, state, rewards)
                + DISCOUNT * _expected_next_value(action, state, STATE_VALUE)
            )
        values[action] = value
    return values


def value_iteration(
    rewards: MDPRewards = MDPRewards(),
    discount: float = DISCOUNT,
    tolerance: float = 1e-9,
    max_iterations: int = 10_000,
) -> tuple[dict[str, float], dict[str, str], int]:
    """Return optimal state values, the greedy policy, and iterations used."""
    if not 0.0 <= discount < 1.0:
        raise ValueError("Discount must be in [0, 1).")
    if tolerance <= 0.0:
        raise ValueError("Tolerance must be positive.")
    if max_iterations <= 0:
        raise ValueError("Maximum iterations must be positive.")

    states = ("clear", "blocked")
    actions = ("Move", "Wait")
    values = {state: 0.0 for state in states}

    for iteration in range(1, max_iterations + 1):
        updated_values = {
            state: max(
                _immediate_reward(action, state, rewards)
                + discount * _expected_next_value(action, state, values)
                for action in actions
            )
            for state in states
        }
        delta = max(
            abs(updated_values[state] - values[state]) for state in states
        )
        values = updated_values
        if delta < tolerance:
            policy = {
                state: max(
                    actions,
                    key=lambda action: _immediate_reward(action, state, rewards)
                    + discount * _expected_next_value(action, state, values),
                )
                for state in states
            }
            return values, policy, iteration

    raise RuntimeError("Value iteration did not converge within max_iterations.")


def break_even_probability(rewards: MDPRewards = MDPRewards()) -> float:
    """P(blocked) at which Move and Wait have equal one-step value.

    Both Q-values are linear in p, so Q(Move) - Q(Wait) = a + b*p and the
    robot should wait exactly when p > -a / b.
    """
    at_clear = mdp_action_values(0.0, rewards)
    at_blocked = mdp_action_values(1.0, rewards)
    a = at_clear["Move"] - at_clear["Wait"]
    b = (at_blocked["Move"] - at_blocked["Wait"]) - a
    return -a / b


def run_demo(sensor_alert: bool = True) -> None:
    path, expanded = a_star(START, GOAL)
    _, ucs_expanded = uniform_cost_search(START, GOAL)
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
    print(f"   Uninformed baseline (uniform-cost search) expands {ucs_expanded} states")
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
    print(f"   Break-even risk: wait when P(blocked) > {break_even_probability():.3f}")
    print(f"   Recommended action: {action}")
    state_values, policy, iterations = value_iteration()
    print("\n4) MDP value iteration")
    print(f"   Converged in {iterations} iterations")
    print(f"   Optimal state values: {state_values}")
    print(f"   Policy: clear → {policy['clear']}; blocked → {policy['blocked']}")
    print(
        "\nExplanation: A* proposes the shortest static route, Bayes updates "
        "the risk of its next corridor from the sensor, the one-step MDP "
        "compares immediate decisions, and value iteration computes a "
        "longer-horizon policy."
    )


def self_test() -> None:
    path, _ = a_star(START, GOAL)
    assert path[0] == START and path[-1] == GOAL
    assert all(cell not in WALLS for cell in path)
    assert len(path) - 1 == 8
    assert a_star(START, GOAL)[1] < uniform_cost_search(START, GOAL)[1]

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
