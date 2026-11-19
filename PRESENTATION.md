# Presentation outline for the defense

## Slide 1 — Project title and problem
- Title: Mini Campus Delivery Robot
- Problem: a delivery robot must move across campus safely and efficiently.
- Goal: combine route planning, uncertainty estimation, and decision-making in one system.

## Slide 2 — Why this project matters
- Real-world robotics requires path planning and decision-making.
- The project is based on a practical scenario: package delivery under uncertainty.
- We combine several AI methods to solve a single real mission.

## Slide 3 — A* path planning
- Explain the 5x5 grid map and obstacles.
- Show how A* finds a shortest path using Manhattan distance.
- Mention why this is suitable for route optimization.

## Slide 4 — Bayesian risk estimation
- Explain the sensor and probability of blockage.
- Show how prior probability is updated using a sensor alert.
- Explain the difference between low-risk and high-risk corridor states.

## Slide 5 — MDP and Bellman decision-making
- Present the two actions: Move and Wait.
- Explain how expected reward is calculated.
- Describe why the robot waits when risk is high.

## Slide 6 — Demo and user interface
- Show the interactive website and map.
- Highlight the autopilot-style dashboard and decision panels.
- Show how the website visualizes path planning, risk, and action selection.

## Slide 7 — Results and future work
- Summarize the project outcome.
- Mention that the solution is a working prototype and can be extended.
- List possible improvements: more realistic noise, dynamic obstacles, longer-horizon planning.

## Short oral defense script (about 1–2 minutes)
"Our project is called Mini Campus Delivery Robot. The goal is to help a robot carry a package across a small campus map while avoiding blocked corridors and making safe decisions. We combined three AI methods into one system. First, A* search calculates the shortest route using a grid and Manhattan distance. Second, Bayesian inference updates the probability that the next corridor is blocked after a sensor signal. Third, a Markov Decision Process evaluates whether the robot should move or wait by comparing expected rewards. In the demo, the robot plans a route, checks uncertainty, then chooses the best action. This makes the system useful as a basic example of intelligent autonomous behavior and demonstrates how different AI techniques can work together in one practical task."
