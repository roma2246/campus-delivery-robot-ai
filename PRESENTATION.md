# Presentation and defense plan

**Slides (7):** https://claude.ai/artifact/XVg5fLTTqdMjCFxc8kYnXv — speaker notes
are attached to every slide. The deck can be exported to PDF or PowerPoint
from its Share menu.

**Live demo:** https://campus-delivery-robot-ai.vercel.app/ (backup: open
`game.html` locally, no internet needed).

## Slides

| # | Slide | Key message | Presenter |
|---|---|---|---|
| 1 | Mini Campus Delivery Robot | One delivery task, three AI methods working as one pipeline | Ramazan |
| 2 | Task | Route → sensor reading → risk → decision, every turn; what the repository contains | Ramazan |
| 3 | A* | `f = g + h`, Manhattan heuristic is admissible and consistent; tie-breaking on larger `g` cut expansions from 23 to 9 | Ramazan |
| 4 | Bayes' rule | Alert raises the risk from 15% to 37.5%, not to 85%; predicted 36.3% vs observed 36.8% in 10,000 deliveries | Merey |
| 5 | MDP | `Q(Move) = 13.15 − 39.45p`, `Q(Wait) = 1.825 − 1.725p`; wait when `p > 0.300`; value iteration confirms it (`V*` = 88.9 / 76.6, threshold 0.310) | Ali |
| 6 | Testing | 38 unit tests in four files, 10,000 simulated deliveries per policy, 300 random maps | Ali |
| 7 | Results and lessons | 100% delivered, 16× fewer collisions, 2× better than the reflex rule on a noisy sensor | Ali, then all |

## Defense timeline (as in the assignment)

| Part | Time | Content | Presenter |
|---|---|---|---|
| Problem statement and objectives | 2 min | Slides 1–2 | Ramazan |
| Live demo | 3–5 min | Script below | Merey |
| Three AI topics | 2 min | Slides 3–5, about 40 s each | Ramazan, Merey, Ali |
| Testing and results | — | Slides 6–7 if time allows, otherwise in Q&A | Ali |
| Questions and answers | up to 3 min | Every member answers individually | all |

## Live demo script

1. Open the site on **Карта из задания** (the assignment's 5×5 map with 2 walls).
   Point at the arrows: the A* route, 8 moves; the panel shows 9 expanded cells
   against 23 for search without a heuristic. Toggle **f / g / h** to show the
   numbers on the cells.
2. Press **Пример: тревога**. Risk 15% → 37.5%; the MDP panel shows
   `Q(ехать) = −1.64`, `Q(ждать) = 1.18`, advice: wait. Press **Выполнить совет**:
   the forklift looks around and waits.
3. Press **Пример: чисто**. Risk drops below 30%; advice: move. Execute it: the
   forklift drives one cell.
4. Turn on **▶ Автопилот** and let the forklift deliver the parcel: it lifts the
   box and spins around.
5. Press **🎲 Новая карта** once: different racks, A* re-plans; a path always
   exists because the generator checks it with BFS.
6. Optional, if time is left: open one **!** explanation or switch day/night.

Backup: if the site or the internet fails, run `python3 main.py` and
`python3 main.py --no-alert` in a terminal; they print the same numbers.
