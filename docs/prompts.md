# Writing and validating grasp prompts

Describe the **current phase, right hand, motion direction, and stable stance**.
Provide precise coordinates through simulator-state grounding and spatial
constraints. Text influences the motion prior; SONIC tracking, finger contacts,
table clearance, and physical acceptance still require validation.

## Sources and project choices

The pinned ARDY [README](https://github.com/nv-tlabs/ardy/blob/693f74d13b3d04a0a22ce127ee79c929dd89756b/README.md)
and [generation script](https://github.com/nv-tlabs/ardy/blob/693f74d13b3d04a0a22ce127ee79c929dd89756b/scripts/generate.py)
use third-person action descriptions and separate language from spatial
conditions. The README also discusses how history length affects responses to
new prompts. The wording below is a project experiment based on those interfaces,
not an upstream guarantee of grasp success. The original four-run prompt comparison kept history length,
ARDY weights, SONIC, spatial trajectories, finger torques, and failure criteria
fixed to isolate the prompt change. The newer approach protocol changes scene
initialization and adds walking/settling constraints; its outcomes must be
reported separately from that earlier prompt-only comparison.

The old profile repeats the complete reach/grasp/lift mission in each phase,
then appends the current step. Our hypothesis is that later goals can interfere
with the current action. The `focused` profile therefore describes only the
current phase in simple English. This explanation remains a hypothesis.

| Phase | Emphasis | Conflicting instructions to avoid |
| --- | --- | --- |
| approach | Take small steps along the supplied path and stop | Keeping both feet fixed throughout walking |
| settle | Stop and stand still at the measured arrival position | Starting the arm reach before balance is checked |
| prepare | Gently incline the upper body and pause with planted feet | Assuming a movable neck or reaching before the second stability check |
| reach | Bend the right elbow, raise the open hand, then extend above the table | Grasping or lifting the object immediately |
| lower | Slowly lower the right hand beside the block | Restarting the reach from beside the body |
| close | Hold the wrist and elbow position while closing the fingers | Pulling upward or leaning the whole body forward |
| lift | Raise the right hand vertically while maintaining grip and planted feet | Pulling sharply toward the body or walking |
| hold | Maintain the raised pose and grip | Repeating the entire reach/grasp sequence |

The default text is in [configs/grasp-prompts.json](../configs/grasp-prompts.json),
matching `FOCUSED_PROMPTS` in `baseline/grasp.py`. Use English for all project
prompts and reports, consistent with `AGENTS.md` and the existing experiments.

## Editing and comparing prompts

Copy the JSON configuration and change one phase at a time:

```bash
./run.sh batch --grasp --batch 2 --seed 42 --cube-xy .40 -.22 \
  --phase-prompts configs/grasp-prompts.json
# Compare the previous wording with matching seeds and positions
./run.sh batch --grasp --batch 2 --seed 42 --cube-xy .40 -.22 --prompt-profile legacy
```

The actual text submitted to ARDY is recorded in `ardy/PHASE/report.json` and
events.jsonl. Inspect those records rather than inferring the model input from
the shared command-line prefix. Manual JSON also supports partial overrides
through `phase_prompts`. Avoid repeating the whole mission in `--prompt`, since
that field is prepended to every phase.

First fix the block position for diagnosis, then compare multiple paired seeds
and positions. Change only the prompt and retain every failure. Inspect actual
wrist trajectories, lowest-point clearance, contact pairs, and failure times in
task.csv. Render failed attempts to locate table contacts during reach or lower.
Only physical `task_success=true` counts as success; plausible text, convincing
motion, or longer execution does not meet the task criterion.

The paired pilot results and remaining issues are recorded in
[verification.md](verification.md). Calibrate the chosen prompts and spatial
constraints on validation data, then freeze them consistently across methods.
Do not tune prompts per test episode and select only successful outcomes.

## Locating the table and block without vision

Language describes what to do. Current MuJoCo state supplies where to do it.
The sequencer computes a root approach path from table geometry and block
position, then computes wrist goals after physical arrival. ARDY supports
[root paths and waypoints](https://github.com/nv-tlabs/ardy/blob/693f74d13b3d04a0a22ce127ee79c929dd89756b/ardy/constraints.py)
as separate conditions; coordinates do not need to be written into the prompt.
The model is not being asked to infer an unseen scene from text.

The preparation phases can be overridden using the `approach`, `settle`, and `prepare` keys
in the same prompt JSON. Keep spatial settings and safety checks fixed when
comparing wording. More initial distance does not by itself prove that a later
hand trajectory clears the table; inspect actual contact and approach results.
