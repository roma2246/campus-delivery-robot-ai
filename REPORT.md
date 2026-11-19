# Mini Campus Delivery Robot

## 1. Project title
Mini Campus Delivery Robot: AI route planning, risk estimation, and decision-making

## 2. Problem statement
Campus delivery robots need to move goods from one location to another while avoiding blocked corridors and making safe decisions under uncertainty. A robot must choose a route, estimate whether the next corridor is risky, and decide whether it is better to move or wait.

This project presents a small simulation that combines several AI methods in one mission: the robot plans a path, uses sensor information to estimate uncertainty, and selects an action that balances efficiency and safety.

## 3. Project objective
The main objective is to demonstrate how different AI algorithms can work together to solve one practical task:
- choose an efficient route using A* search,
- estimate whether the next step is likely blocked using Bayesian inference,
- decide whether to move or wait using a small Markov Decision Process (MDP) and Bellman-style evaluation.

## 4. Real-world relevance
Robotics and autonomous systems must combine multiple reasoning methods. A delivery robot cannot simply follow a path without considering uncertainty. It should:
- plan a safe route,
- interpret noisy sensor signals,
- adapt decisions based on risk,
- choose actions that improve performance while reducing failure risk.

## 5. Algorithms used

### 5.1 A* search
A* is used to find the shortest route on a 5x5 campus grid. The algorithm uses Manhattan distance as the heuristic and expands cells in order of cost plus heuristic value.

Why this algorithm:
- it is efficient for grid-based navigation,
- it guarantees an optimal route in this deterministic map,
- it is easy to explain and visualize.

### 5.2 Bayesian inference
A noisy sensor can detect whether the next corridor is blocked. The system starts with a prior probability of blockage and then updates it using a sensor signal (alert or clear). This update is done with Bayes' theorem.

Why this algorithm:
- it handles uncertainty in a principled way,
- it is useful when sensor readings are noisy,
- it enables better decisions under incomplete information.

### 5.3 Markov Decision Process (MDP) and Bellman reasoning
The robot chooses between two actions: Move or Wait. The MDP evaluates each action using expected reward and estimated future value.

Why this algorithm:
- it models decision-making under uncertainty,
- it balances immediate reward and future value,
- it demonstrates how a robot can choose a safe action when risk is high.

## 6. System design
The project is implemented as a lightweight web-based simulator with a 5x5 grid map and decision dashboard.

The workflow is:
1. Generate a shortest path with A*.
2. Inspect the next corridor.
3. Update the blockage probability with Bayes' rule.
4. Evaluate Move vs Wait using MDP values.
5. Display the result and continue the mission.

## 7. Implementation details
### Data model
- Grid size: 5x5
- Start position: (0, 0)
- Goal: (4, 4)
- Walls: blocked cells on the map
- Sensor model: prior risk, alert likelihood, false alarm rate
- Rewards for Move and Wait actions

### Example decision logic
- If the posterior probability of blockage is high, the MDP prefers waiting.
- If the probability of blockage is low, the robot chooses to move.

This matches the realistic idea that autonomous systems should not move blindly when the corridor is risky.

## 8. Results and evaluation
The project successfully demonstrates:
- optimized route selection using A*,
- real-time probability update with Bayes' rule,
- action selection based on expected value with MDP reasoning,
- a clear and understandable user interface for presentation.

The final demo shows the robot can complete the mission while taking sensor uncertainty into account. In practice, the system acts as a proof-of-concept for intelligent movement planning in a small campus environment.

## 9. Limitations
The simulation is simplified and not a full production robotic controller. Some assumptions include:
- a fixed map,
- a simplified sensor model,
- single-step MDP evaluation,
- patched reward values instead of a fully learned policy.

## 10. Future improvements
To improve the project in future work, the following can be added:
- dynamic map changes,
- multi-step MDP or value iteration,
- more realistic sensor noise distributions,
- obstacle recovery and replanning,
- extended mission scenarios with multiple goals and packages.

## 11. Conclusion
This project shows how three different AI paradigms can be integrated into one practical system. A* plans the route, Bayesian inference measures uncertainty, and MDP decides what action is best. Together they create a simple but effective intelligent delivery robot simulation.

## 12. Summary of contribution to the course
The project satisfies the Final Project requirement by combining at least three different AI topics in a single unified problem. The result is a working, interactive simulation that is easy to explain, visually clear, and relevant to real autonomous systems.
