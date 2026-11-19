"""Monte Carlo evaluation of the delivery robot's decision policy.

The robot follows the A* route from main.py. Each turn the corridor ahead is
either clear or blocked (hidden state). The noisy sensor gives a reading, a
policy chooses Move or Wait, and the hidden state then evolves with the same
transition model the MDP uses.

Policies compared:
  * Bayes + MDP (ours) - sensor -> Bayes posterior -> larger one-step Q-value
  * Sensor reflex      - Wait on alert, Move otherwise (no probabilities)
  * Always Move        - ignores the sensor
  * Random             - Move or Wait with probability 1/2
  * Always Wait        - never moves

Two scores are reported for every policy:
  * discounted return  - sum of gamma^t * reward with gamma = 0.9, the objective
                         the MDP itself optimises (it rewards fast delivery);
  * total reward       - plain sum of rewards until delivery.

Run:
    python3 experiment.py                 # 10 000 episodes, writes results/
    python3 experiment.py --episodes 2000 --seed 1 --no-plots

Only the standard library is required; charts are drawn when matplotlib is
installed and skipped otherwise.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from main import (
    DISCOUNT,
    GOAL,
    START,
    TRANSITION_BLOCKED,
    MDPRewards,
    SensorModel,
    a_star,
    bayes_obstacle_probability,
    break_even_probability,
    mdp_action_values,
    uniform_cost_search,
)

PRIOR_BLOCKED = 0.15
ALERT_IF_BLOCKED = 0.85
ALERT_IF_CLEAR = 0.25  # false-alarm rate of the project's sensor
MAX_TURNS = 60
FALSE_ALARM_SWEEP = (0.10, 0.25, 0.40, 0.50, 0.60, 0.70)
REWARDS = MDPRewards()
REWARD = {
    "Move": {"clear": REWARDS.move_if_clear, "blocked": REWARDS.move_if_blocked},
    "Wait": {"clear": REWARDS.wait_if_clear, "blocked": REWARDS.wait_if_blocked},
}
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# A policy receives the belief before the reading, the reading, the sensor's
# false-alarm rate and a random generator. It returns (action, belief after
# the reading); policies that ignore the sensor return the belief unchanged.
Policy = Callable[[float, bool, float, random.Random], tuple[str, float]]


def bayes_mdp_policy(belief: float, alert: bool, false_alarm: float, rng: random.Random):
    model = SensorModel(belief, ALERT_IF_BLOCKED, false_alarm)
    _, posterior, _ = bayes_obstacle_probability(model, alert)
    values = mdp_action_values(posterior, REWARDS)
    return max(values, key=values.get), posterior


def sensor_reflex_policy(belief: float, alert: bool, false_alarm: float, rng: random.Random):
    return ("Wait" if alert else "Move"), belief


def always_move_policy(belief: float, alert: bool, false_alarm: float, rng: random.Random):
    return "Move", belief


def random_policy(belief: float, alert: bool, false_alarm: float, rng: random.Random):
    return ("Move" if rng.random() < 0.5 else "Wait"), belief


def always_wait_policy(belief: float, alert: bool, false_alarm: float, rng: random.Random):
    return "Wait", belief


POLICIES: dict[str, Policy] = {
    "Bayes + MDP (ours)": bayes_mdp_policy,
    "Sensor reflex": sensor_reflex_policy,
    "Always Move": always_move_policy,
    "Random": random_policy,
    "Always Wait": always_wait_policy,
}


@dataclass
class Episode:
    delivered: bool
    discounted: float
    reward: float
    turns: int
    collisions: int
    waits: int
    # (posterior used for the decision, corridor actually blocked)
    beliefs: list[tuple[float, bool]] = field(default_factory=list)


def run_episode(
    policy: Policy,
    route_moves: int,
    rng: random.Random,
    false_alarm: float = ALERT_IF_CLEAR,
) -> Episode:
    belief = PRIOR_BLOCKED
    blocked = rng.random() < PRIOR_BLOCKED
    position = collisions = waits = 0
    reward = discounted = 0.0
    beliefs: list[tuple[float, bool]] = []

    for turn in range(MAX_TURNS):
        likelihood = ALERT_IF_BLOCKED if blocked else false_alarm
        alert = rng.random() < likelihood
        action, posterior = policy(belief, alert, false_alarm, rng)
        beliefs.append((posterior, blocked))

        state = "blocked" if blocked else "clear"
        step_reward = REWARD[action][state]
        reward += step_reward
        discounted += DISCOUNT**turn * step_reward
        if action == "Wait":
            waits += 1
        elif blocked:
            collisions += 1
        else:
            position += 1
            if position == route_moves:
                return Episode(True, discounted, reward, turn + 1, collisions, waits, beliefs)

        # Hidden state and belief both follow the MDP transition model.
        transition = TRANSITION_BLOCKED[action]
        blocked = rng.random() < transition[state]
        belief = (1 - posterior) * transition["clear"] + posterior * transition["blocked"]

    return Episode(False, discounted, reward, MAX_TURNS, collisions, waits, beliefs)


@dataclass
class Summary:
    name: str
    episodes: int
    success_rate: float
    mean_discounted: float
    discounted_ci95: float
    mean_reward: float
    mean_turns_delivered: float
    mean_collisions: float
    mean_waits: float


def mean_and_ci(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, 1.96 * math.sqrt(variance / n)


def summarise(name: str, episodes: list[Episode]) -> Summary:
    n = len(episodes)
    delivered = [e for e in episodes if e.delivered]
    mean_discounted, ci = mean_and_ci([e.discounted for e in episodes])
    return Summary(
        name=name,
        episodes=n,
        success_rate=len(delivered) / n,
        mean_discounted=mean_discounted,
        discounted_ci95=ci,
        mean_reward=sum(e.reward for e in episodes) / n,
        mean_turns_delivered=(
            sum(e.turns for e in delivered) / len(delivered) if delivered else float("nan")
        ),
        mean_collisions=sum(e.collisions for e in episodes) / n,
        mean_waits=sum(e.waits for e in episodes) / n,
    )


def simulate(policy: Policy, episodes: int, seed: int, false_alarm: float = ALERT_IF_CLEAR):
    route, _ = a_star(START, GOAL)
    rng = random.Random(seed)  # same seed for every policy: identical start states
    return [run_episode(policy, len(route) - 1, rng, false_alarm) for _ in range(episodes)]


def calibration(episodes: list[Episode]) -> list[tuple[str, int, float, float]]:
    """Group the agent's posteriors into bins and compare with reality."""
    bins = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01)]
    rows = []
    for low, high in bins:
        points = [(p, b) for e in episodes for p, b in e.beliefs if low <= p < high]
        if not points:
            continue
        predicted = sum(p for p, _ in points) / len(points)
        observed = sum(b for _, b in points) / len(points)
        rows.append((f"{low:.1f}–{min(high, 1.0):.1f}", len(points), predicted, observed))
    return rows


@dataclass
class Results:
    episodes: int
    seed: int
    baseline: list[Summary]
    calibration: list[tuple[str, int, float, float]]
    # false-alarm rate -> policy name -> summary
    sweep: dict[float, dict[str, Summary]]


SWEEP_POLICIES = ("Bayes + MDP (ours)", "Sensor reflex", "Always Move")


def run_experiment(episodes: int, seed: int) -> Results:
    baseline, ours = [], []
    for name, policy in POLICIES.items():
        runs = simulate(policy, episodes, seed)
        baseline.append(summarise(name, runs))
        if policy is bayes_mdp_policy:
            ours = runs
    sweep = {
        rate: {
            name: summarise(name, simulate(POLICIES[name], episodes, seed, rate))
            for name in SWEEP_POLICIES
        }
        for rate in FALSE_ALARM_SWEEP
    }
    return Results(episodes, seed, baseline, calibration(ours), sweep)


def format_markdown(results: Results) -> str:
    _, astar_expanded = a_star(START, GOAL)
    _, ucs_expanded = uniform_cost_search(START, GOAL)
    lines = [
        "# Experiment results",
        "",
        f"Generated by `python3 experiment.py --episodes {results.episodes} --seed {results.seed}`.",
        f"Each policy runs {results.episodes:,} simulated deliveries along the 8-move A* "
        f"route; a delivery fails if it is not finished within {MAX_TURNS} turns. "
        "Every policy sees the same random start states.",
        "",
        "## 1. Decision policies (project sensor: P(alert | blocked) = 0.85, "
        "P(alert | clear) = 0.25)",
        "",
        "| Policy | Delivered | Discounted return γ=0.9 (±95% CI) | Total reward | "
        "Turns to deliver | Collisions | Waits |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in results.baseline:
        turns = "—" if math.isnan(s.mean_turns_delivered) else f"{s.mean_turns_delivered:.1f}"
        lines.append(
            f"| {s.name} | {s.success_rate:.1%} | {s.mean_discounted:.1f} ± {s.discounted_ci95:.1f} "
            f"| {s.mean_reward:.1f} | {turns} | {s.mean_collisions:.2f} | {s.mean_waits:.2f} |"
        )
    lines += [
        "",
        "*Discounted return* is the quantity the MDP maximises (γ = 0.9 makes a fast "
        "delivery worth more). *Collision* = Move while the corridor was really "
        "blocked (reward −20, no progress). Columns other than *Delivered* are means "
        "per delivery attempt.",
        "",
        "## 2. How noisy can the sensor get? (false-alarm sweep)",
        "",
        "Same experiment, changing only P(alert | clear). Discounted return, higher is better; "
        "turns to deliver in brackets.",
        "",
        "| P(alert \\| clear) | " + " | ".join(SWEEP_POLICIES) + " |",
        "|---:|" + "---:|" * len(SWEEP_POLICIES),
    ]
    for rate, by_policy in results.sweep.items():
        cells = [
            f"{by_policy[name].mean_discounted:.1f} ({by_policy[name].mean_turns_delivered:.1f})"
            for name in SWEEP_POLICIES
        ]
        marker = " (project)" if math.isclose(rate, ALERT_IF_CLEAR) else ""
        lines.append(f"| {rate:.2f}{marker} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## 3. Is the Bayesian belief calibrated?",
        "",
        "Every decision of the Bayes + MDP agent (project sensor), grouped by the "
        "posterior P(blocked) it used. If Bayes' rule is applied correctly, the "
        "predicted risk matches how often the corridor was really blocked.",
        "",
        "| Posterior bin | Decisions | Mean predicted P(blocked) | Observed blocked frequency |",
        "|---|---:|---:|---:|",
    ]
    for label, count, predicted, observed in results.calibration:
        lines.append(f"| {label} | {count:,} | {predicted:.3f} | {observed:.3f} |")
    lines += [
        "",
        "## 4. Search effort",
        "",
        "| Algorithm | Route length | Expanded states |",
        "|---|---:|---:|",
        f"| A* (Manhattan heuristic) | 8 | {astar_expanded} |",
        f"| Uniform-cost search (h = 0) | 8 | {ucs_expanded} |",
        "",
        "## 5. MDP decision threshold",
        "",
        "Q(Move) = 13.15 − 39.45·p and Q(Wait) = 1.825 − 1.725·p, so the robot "
        f"waits exactly when P(blocked) > {break_even_probability():.3f}.",
        "",
    ]
    return "\n".join(lines)


def draw_charts(results: Results) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - charts skipped.")
        return []

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    series = {  # categorical slots 1-3 of the reference palette, fixed order
        "Bayes + MDP (ours)": "#2a78d6",
        "Sensor reflex": "#eb6834",
        "Always Move": "#1baf7a",
    }
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": muted,
        "xtick.color": muted, "ytick.color": muted, "text.color": ink,
        "figure.facecolor": surface, "axes.facecolor": surface,
    })
    paths = []

    # Chart 1: discounted return as the sensor's false-alarm rate grows.
    rates = list(results.sweep)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    for name, color in series.items():
        values = [results.sweep[r][name].mean_discounted for r in rates]
        ax.plot(rates, values, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=surface, markeredgewidth=2, label=name)
        ax.annotate(name.replace(" (ours)", ""), (rates[-1], values[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=ink, fontsize=9)
    ax.axvline(ALERT_IF_CLEAR, color=muted, linewidth=1, linestyle="--")
    ax.annotate("project sensor", (ALERT_IF_CLEAR, 18), xytext=(5, 0),
                textcoords="offset points", color=muted, fontsize=8.5)
    ax.annotate("identical decisions\non a good sensor", (0.17, 44), xytext=(0, -34),
                textcoords="offset points", ha="center", color=muted, fontsize=8.5)
    ax.set_xlim(rates[0] - 0.03, rates[-1] + 0.17)
    ax.set_xlabel("Sensor false-alarm rate  P(alert | clear)")
    ax.set_ylabel("Discounted return (γ = 0.9)")
    ax.set_title("Bayes + MDP pulls ahead as the sensor gets noisier",
                 loc="left", fontsize=12, color=ink)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.grid(color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    path = RESULTS_DIR / "noise_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    # Chart 2: Q-values against the posterior risk, with the break-even point.
    ps = [i / 100 for i in range(101)]
    q_move = [mdp_action_values(p, REWARDS)["Move"] for p in ps]
    q_wait = [mdp_action_values(p, REWARDS)["Wait"] for p in ps]
    threshold = break_even_probability()
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    ax.plot(ps, q_move, color=series["Bayes + MDP (ours)"], linewidth=2, label="Q(Move)")
    ax.plot(ps, q_wait, color=series["Sensor reflex"], linewidth=2, label="Q(Wait)")
    ax.axvline(threshold, color=muted, linewidth=1, linestyle="--")
    ax.annotate(f"break-even p = {threshold:.3f}", (threshold, -20), xytext=(6, 0),
                textcoords="offset points", color=muted, fontsize=9)
    for p, label, offset in (
        (0.034, "no alert: p = 0.034 → Move", (10, -4)),
        (0.375, "alert: p = 0.375 → Wait", (8, 8)),
    ):
        q = mdp_action_values(p, REWARDS)
        best = max(q.values())
        ax.plot([p], [best], marker="o", markersize=7, color=ink,
                markeredgecolor=surface, markeredgewidth=2, zorder=5)
        ax.annotate(label, (p, best), xytext=offset, textcoords="offset points",
                    fontsize=8.5, color=ink)
    ax.annotate("Q(Move)", (0.62, 13.15 - 39.45 * 0.62), xytext=(6, -2),
                textcoords="offset points", color=ink, fontsize=9)
    ax.annotate("Q(Wait)", (0.62, 1.825 - 1.725 * 0.62), xytext=(6, 5),
                textcoords="offset points", color=ink, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Posterior P(corridor blocked)")
    ax.set_ylabel("One-step value")
    ax.set_title("The robot waits once the risk passes 30%", loc="left", fontsize=12, color=ink)
    ax.legend(frameon=False, loc="lower left")
    ax.grid(color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    path = RESULTS_DIR / "q_values.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)
    return paths


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-plots", action="store_true", help="skip the PNG charts")
    args = parser.parse_args()

    results = run_experiment(args.episodes, args.seed)
    report = format_markdown(results)
    print(report)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "experiment_results.md").write_text(report, encoding="utf-8")
    written = [RESULTS_DIR / "experiment_results.md"]
    if not args.no_plots:
        written += draw_charts(results)
    print("Saved:", ", ".join(str(p.relative_to(RESULTS_DIR.parent)) for p in written))


if __name__ == "__main__":
    main_cli()
