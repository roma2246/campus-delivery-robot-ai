<!-- Before signing: make the "What the team did" list match what each of you
     actually did, and remove anything that is not true. -->

# AI Usage Declaration

**Project:** Mini Campus Delivery Robot
**Course:** AI Fundamentals — Final Project
**Team:** Ramazan G., Merey S., Ali I.

## AI tools used

- **Claude (Anthropic)**, an AI assistant, used through Claude Code.

## What the AI assistant was used for

- Writing and revising the code: the A*, Bayes and MDP implementation in
  `main.py`, the unit tests in `tests/`, the simulation study in
  `experiment.py` and the interactive web demo `game.html`.
- Drafting the documentation: `README.md`, `REPORT.md`, the slide deck and its
  speaker notes, `PRESENTATION.md`.
- Suggesting parts of the analysis: the comparison with baseline policies, the
  sensor-noise sweep and the calibration check. While writing the tests the
  assistant also found that A* expanded every free cell because of its
  tie-breaking rule, and proposed the fix.
- Producing the two demo videos: both were recorded and assembled by scripts
  from the working website; the narration is synthesized speech (a macOS
  system voice) and the background music of the animated video was generated
  by a script.

## What the team did

- Chose the project from the course's list of project ideas and the three
  syllabus topics it covers (A* search, Bayes' rule, MDP).
- Defined the requirements for the demo (warehouse setting, forklift robot,
  random maps, day/night themes, topic explanations) and reviewed every version.
- Ran the program, the 38 unit tests and the experiment, and checked the key
  numbers by hand: `P(alert) = 0.34`, `P(blocked | alert) = 0.375`,
  `Q(Move) = −1.644`, `Q(Wait) = 1.178`, threshold `p* = 0.300`.
- Divided the work by algorithm (see `CONTRIBUTION_STATEMENT.md`); each member
  reviewed and tested their part, committed it to the repository and presents
  it at the defense.
- Specified what the demo videos show (scenes, numbers, English narration) and
  reviewed every version; the recording and the voice-over themselves were
  produced by scripts, as described above.
- Rehearsed the defense.

## Responsibility

We have read and understood all submitted code and documents. We are
responsible for their correctness, and each of us can explain every algorithm
and every result in the project during the oral defense.

| Name | Date | Signature |
|---|---|---|
| Ramazan G. | | |
| Merey S. | | |
| Ali I. | | |
