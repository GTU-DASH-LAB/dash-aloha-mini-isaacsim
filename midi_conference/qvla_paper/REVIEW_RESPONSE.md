# MIDI 2026, submission 3628: changes made in response to the reviews

QVLA: Training-Free Language-Driven Robot Navigation by Choosing Among Arcs Drawn on the
Camera Image. F. Aladhami, O. B. Kaya, S. Sarp.

The camera-ready version is 9 pages (the reviews allowed up to 3 extra pages). Below, each
review point is followed by what changed in the paper.

## Reviewer 1

**The TIC-VLA baseline ran on an older version of the infrastructure than QVLA.**
We re-ran the whole closed-loop comparison on one code version (Sect. 4, 5.3, Table 2):
- TIC-VLA, QVLA and a new no-model baseline use the same robot, camera, controller,
  collision guard, contact layer and recovery reverse.
- The three policies alternate within each round, so any slow drift in the machine affects
  all of them equally.
- The re-run also uses fixes to the shared simulator made after submission (controller,
  physics, and a contact layer so that no policy can drive through furniture).

**Few scenarios, high variance, TIC-VLA run only once.**
- Every policy now runs the 13 indoor episodes three times.
- Table 2 reports mean ± s.d. and the range over rounds.
- Fig. 3b shows successes per episode.
- The comparison is paired by episode, with an exact sign-flip test and a bootstrap
  interval.
- The outdoor column was removed. Its numbers were affected by simulation defects found
  after submission, and the paper says so (Sect. 4).

**How was the reported 9/13 selected?**
- The submitted table showed the best of seven ladders.
- The camera-ready reports the mean of three rounds (7.7 of 13).
- It also states that seven earlier repeats scored 6–9 of 13, with the same mean of 7.7
  (Sect. 5.3, "Ablations and variance").

**Compare with a classical arc selection using the same controller.**
- We added a no-model baseline that shares the menu, controller, guard and recovery and
  always takes the straight arc. It reaches 1 of 13 goals, only on the episode whose
  instruction is to go straight. Both learned policies beat it (p = 0.004 and p = 0.041).
- The earlier lidar clearance filter on the arcs (10/19 against 10/19) is reported in
  "Ablations and variance".

**Physical applicability: robot dynamics and collisions.**
- The method now describes the contact layer (Sect. 3.3).
- The limitations name simulation, the kinematic base and the contact layer (Sect. 6).

**Will the code and scenarios be public?**
- Yes. A footnote in the introduction links the repository: code, configurations, per-run
  logs and the campaign summary.

**Present the comparative claims more cautiously.**
- The abstract, results and conclusion now say that the two policies reach a similar
  number of goals: 7.7 against 6.3 of 13, not significant (p = 0.69).
- The earlier claims about closest approach and guard interventions are withdrawn. With
  repeats on one stack they reverse: QVLA's failures end farther from the goal, while the
  medians are equal.
- The paper's remaining claim is the instruction's authority over the plan (102° against
  at most 36°), obtained without training.

## Reviewer 2

**Methodological novelty is moderate.**
- The related-work section now says plainly that QVLA is built from known parts.
- It states what is new: the drawn arcs are the controller's actual plans, labels are
  shuffled at every decision, and perception and choice are split into two calls (wall
  avoidance 22% → 100%).
- It also names the matched-control measurement of instruction following.

**The comparison is not fully fair (model size, controller, recovery).**
- Controller, guard and recovery reverse are now identical. QVLA's prompt-side recovery
  note does not apply to TIC-VLA, and the paper says so (Sect. 4).
- Model size is discussed in Sect. 5.3: TIC-VLA is 27× smaller by design, and QVLA with a
  4B model fell to 3 of 13.

**9/13 vs 6/13 is not statistically convincing; the baseline was not repeated.**
- Both policies are repeated three times.
- The per-episode paired test gives p = 0.69, and the paper reports that the difference is
  not significant.

**Ablations separating the VLM, deterministic selection and collision avoidance.**
- The no-model baseline isolates the VLM's contribution: same deterministic selection,
  controller and collision handling, with no model. That is the gap from 1 to 7.7 of 13.
- The earlier ablations of the menu are kept and labelled with the stack version they
  were measured on.

## Other corrections found by the authors

- The explanation for non-reproducible repeats was corrected. The cause is rendering noise
  in the first frame, not timing; those runs were synchronous.
- The author list, affiliations, project number and the Disclosure of Interests were
  updated.
