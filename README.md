# Mini Campus Delivery Robot

**Course:** AI Fundamentals  
**Project type:** small, reproducible simulation  
**Student/team:** Ramazan G., Merey S., Ali I.

## Project idea

A campus robot must carry a package from a start cell to a destination on a
small grid. It first plans a route, then checks the next corridor with a noisy
sensor. Because the corridor may be blocked, the robot estimates the risk and
decides whether to move or wait.

The three methods support one common goal: deliver the package while avoiding
a potentially blocked corridor.

## Selected syllabus topics

The two main topics were selected from the first seven weeks of the syllabus:

1. **Week 2 — Uninformed search, A* search and heuristics:** A* finds a
   shortest path on the 5×5 campus map using Manhattan distance.
2. **Week 6 — Probability and Bayesian networks:** Bayes' rule updates the
   probability that the next corridor is blocked after a noisy sensor alert.

To meet the Final Project requirement for at least three algorithms from
different topics, the project also uses **Week 5 — Markov decision processes
and reinforcement learning**. The MDP uses a two-state transition model and
one Bellman update to compare `Move` and `Wait`. All selected material is
before Week 8.

## Algorithms and assumptions

### A* search

- Grid: 5×5; start `(0, 0)`; destination `(4, 4)`.
- Walls: `(1, 1)` and `(2, 1)`.
- Each move costs 1. The heuristic is Manhattan distance.
- Ties on `f = g + h` are broken in favour of the larger `g` (the node closer
  to the goal), so A* expands only 9 states, against 23 for uninformed
  uniform-cost search on the same map.
- Output: a shortest route, route length, and number of expanded states.

### Bayesian inference

The sensor is checking whether the next corridor is blocked.

- Prior probability of blockage: `P(blocked) = 0.15`.
- Sensor alert reliability: `P(alert | blocked) = 0.85`.
- False-alert probability: `P(alert | clear) = 0.25`.
- The default demonstration uses an `ALERT` reading.

For an alert:

```text
P(alert) = 0.85 × 0.15 + 0.25 × 0.85 = 0.34
P(blocked | alert) = (0.85 × 0.15) / 0.34 = 0.375
```

### One-step MDP decision

States are `blocked` and `clear`; actions are `Move` and `Wait`. Bayes' posterior
is the current belief over the states. The MDP uses this transition model:

| Action | P(next blocked \| clear) | P(next blocked \| blocked) |
|---|---:|---:|
| Move | 0.10 | 0.80 |
| Wait | 0.05 | 0.40 |

The immediate rewards are:

| Action | Clear corridor | Blocked corridor |
|---|---:|---:|
| Move | +10 | −20 |
| Wait | −2 | +1 |

For a one-step Bellman update, the fixed next-state values are `V(clear) = 5`
and `V(blocked) = −10`, with discount `γ = 0.9`. For each action, the program
calculates `Q(b, a) = Σ_s b(s)[R(s, a) + γΣ_s' T(s, a, s')V(s')]` and selects
the larger value. With the default alert (`P(blocked) = 0.375`), the values
are `Q(Move) = −1.644` and `Q(Wait) = 1.178`, so the robot waits. With no alert,
the posterior blockage probability falls to about `0.034`, and the robot
chooses to move. This is a deliberately small, hypothetical model, not a
trained or production controller.

## Interactive game website

Open `game.html` in a browser. The page is self-contained and needs no install,
server, or internet connection. It presents the task as a calm warehouse
simulation: the map stands in a warehouse hall with a window and wall shelving,
shelving racks on the map are the walls, a 3D forklift robot carries out the
deliveries, and the destination is a parcel on a pallet.

- **Map.** A* plans the route around the racks; arrows on the floor show it.
  After a new start the cells A* expanded are highlighted in order, and
  **f / g / h** labels every cell with its A* values. **Вид сверху** switches
  between the isometric and the top view.
- **Sensor and decision.** **Проверить датчик** gives a noisy reading;
  **Пример: тревога / чисто** give repeatable readings for a presentation. The
  panel shows the risk before and after Bayes' rule, both Bellman Q-values and
  the advice relative to the 30% threshold; **Выполнить совет** carries it out
  and **▶ Автопилот** repeats "check sensor → follow the advice" until delivery.
- **Robot.** The forklift drives cell by cell, turns at corners, looks around
  when it waits and backs off when the corridor is blocked. At the goal it lifts
  the parcel onto its forks and spins around on the spot.
- **Random maps.** **🎲 Новая карта** (key `N`) places 2–6 racks at random while
  start and goal stay fixed; a breadth-first search rejects any layout without
  a path to the parcel. **Карта из задания** restores the assignment's map
  (5×5, 2 walls), which is the one all numbers in this README refer to.
- **Explanations.** Every **!** button opens a professor robot that explains
  the task, A*, Bayes' rule and the MDP with worked formulas, the numbers of the
  current map and the current turn, plus a how-to-play tab.

- **Day and night.** **🌙 Ночь / ☀️ День** (key `T`) switches the theme. At
  night the sun sets behind the city outside the window, the moon and stars
  come out, windows in the buildings light up, the hall goes dark and only the
  ceiling lamps and the forklift's headlights light it. The choice is
  remembered; on the first visit the page follows the system theme.

Keys: `Enter` follows the advice, `R` restarts, `N` makes a new map, `T`
switches day/night, `Esc` closes the explanation.

## Deploy to Vercel

The website is static: no build step, no server code. `vercel.json` serves
`game.html` at the site root and `.vercelignore` leaves the Python code and
tests out of the deployment. In Vercel choose **Add New → Project**, import this
GitHub repository, keep **Framework Preset: Other** with empty build and output
settings, and press **Deploy**. Every push to `main` redeploys the site.

## Run the Python version

Requires Python 3.10 or later; no third-party packages are needed.

```bash
python3 main.py
python3 main.py --no-alert
python3 main.py --test
python3 -m unittest discover -s tests -v
python3 experiment.py
```

The default run prints each algorithm's input, key calculation, output, and a
short explanation. `--no-alert` demonstrates how the Bayesian update changes
the MDP action. `--test` runs the quick built-in checks; the `tests/` suite
(33 unit tests) covers A* optimality from every cell, heuristic admissibility
and consistency, the Bayes and Bellman numbers, the decision threshold and the
simulation. `experiment.py` runs 10,000 simulated deliveries per policy and
writes `results/experiment_results.md` plus two charts (the charts need
matplotlib; everything else uses only the standard library).

## Project structure

| Path | Purpose |
|---|---|
| `main.py` | A*, uniform-cost baseline, Bayes update, one-step MDP, CLI demo |
| `experiment.py` | Monte Carlo comparison of decision policies and sensor-noise sweep |
| `tests/` | Unit tests for `main.py` and `experiment.py` |
| `results/` | Generated tables and charts used in the report |
| `game.html` | Interactive warehouse demo: 3D forklift, random maps, explanations |
| `REPORT.md` | Technical report |

## Results

Full tables: [`results/experiment_results.md`](results/experiment_results.md).

| Policy (project sensor) | Delivered | Discounted return γ=0.9 | Turns | Collisions |
|---|---:|---:|---:|---:|
| **Bayes + MDP (ours)** | 100% | **40.8** | 12.5 | 0.27 |
| Sensor reflex (alert → wait) | 100% | 40.8 | 12.5 | 0.27 |
| Always Move | 100% | 7.8 | 12.4 | 4.39 |
| Random | 100% | 15.7 | 19.1 | 1.58 |
| Always Wait | 0% | −17.3 | — | 0 |

- Using the sensor cuts collisions with blocked corridors 16-fold compared
  with ignoring it.
- With this fairly reliable sensor the MDP's choices coincide with the simple
  rule "wait on alert" — the break-even risk (0.300) lies between the two
  possible posteriors (0.034 and 0.375).
- When the sensor produces more false alarms, the probabilistic agent pulls
  ahead: at P(alert | clear) = 0.6 it scores 27.6 against 19.4 and delivers in
  12.6 turns instead of 22.8.
- The Bayesian belief is calibrated: decisions made at a predicted risk of
  0.363 met a blocked corridor 36.8% of the time.

![Discounted return as the sensor gets noisier](results/noise_sweep.png)

![Q-values and the break-even risk](results/q_values.png)

## Sample output

```text
Mini Campus Delivery Robot
==========================
Grid: 5x5; start=(0, 0); goal=(4, 4); walls=[(1, 1), (2, 1)]

1) A* route planning
   Heuristic: Manhattan distance; step cost: 1
   Route: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
   Route length: 8 moves; expanded states: 9
   Uninformed baseline (uniform-cost search) expands 23 states
   Next corridor cell to check: (1, 0)

2) Bayesian sensor update
   Prior P(blocked) = 0.150
   Sensor reading = ALERT
   P(reading) = ... = 0.340
   Posterior P(blocked|reading) = 0.375; P(clear|reading) = 0.625

3) One-step MDP action choice
   State: corridor blocked or clear; actions: Move or Wait
   P(next blocked | clear, Move) = 0.10; P(next blocked | blocked, Move) = 0.80
   P(next blocked | clear, Wait) = 0.05; P(next blocked | blocked, Wait) = 0.40
   Rewards (clear/blocked): Move=+10/-20, Wait=-2/+1; V(clear)=+5, V(blocked)=-10; gamma=0.90
   One-step Bellman value(Move) = -1.644; value(Wait) = 1.178
   Break-even risk: wait when P(blocked) > 0.300
   Recommended action: Wait
```

All outputs are deterministic. The expanded-state count depends on the
tie-breaking rule described above.

## Live demonstration outline

1. Run `python3 main.py` and point out the grid, route, and shortest-path cost.
2. Show the prior, sensor likelihoods, evidence probability, and posterior.
3. Explain the two expected-reward calculations and why the robot waits.
4. Run `python3 main.py --no-alert`; compare the updated probability and
   explain why the action changes to `Move`.
5. Run `python3 -m unittest discover -s tests -v` to show the 33 tests passing.
6. Show `results/noise_sweep.png`: why Bayes + MDP matters once the sensor is noisy.

## Limitations and future development

The map is static, the sensor model and rewards are hypothetical, and the MDP
uses a one-step lookahead with hand-chosen next-state values `V(clear) = 5`,
`V(blocked) = −10` rather than values computed by value iteration. A
future version could add multiple sensor readings, dynamic obstacle updates,
replanning, longer-horizon value iteration, and measured sensor probabilities.
No real campus, student, or personal data is used.

## AI-use declaration

See [`AI_DECLARATION.md`](AI_DECLARATION.md).
