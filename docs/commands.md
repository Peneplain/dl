# Running the project

Run `./run.sh` from `~/dl` on the S4000 **host**. The launcher uses the existing
container and `.venv-baseline-musa`; do not invoke it from inside a container.
Execution has two modes, `batch` and `manual`. Export videos separately with
`render`.

## 1. Batch generation and video selection

```bash
cd ~/dl
# Default: one standing recording, no task, text, table/block, or GUI
./run.sh batch

# Enable the right-hand tabletop block lift task; default batch size is one
./run.sh batch --grasp
./run.sh batch --grasp --batch 20 --seed 42

# Start at the grounded table approach target (the default; no walking)
./run.sh batch --grasp --seed 42 --table-standoff .22

# Opt into starting farther back and walking to the target
./run.sh batch --grasp --walk --seed 42 --start-back .45 --table-standoff .22

# Skip locomotion and calibrate manipulation from the shared approach target
./run.sh batch --grasp --direct-start --batch 2 --seed 42 --cube-xy .40 -.22 --table-standoff .20

# Fixed position for paired prompt comparisons
./run.sh batch --grasp --batch 2 --seed 42 --cube-xy .40 -.22

# General text-driven motion without a grasp task
./run.sh batch --prompt 'A person stands upright and slowly raises the right hand.'
```

Results are saved automatically to `output/batch-YYMMDD-HHMMSS/`, using
Asia/Shanghai time, for example `batch-261003-031351`. If another launch has
already reserved the same second, allocation waits for the next available second.
Existing sessions are never overwritten. Attempts start at `attempt-00001` and reset independently.
On the S4000 host, `output` may be a symlink to a writable directory on the
larger `/data` filesystem. `docker/run-musa.sh` detects an external symlink
target and bind-mounts it at the same absolute path in the container. Check
`df -h output` before a large batch; no special `--output-root` argument is
needed when the symlink and mount are present. Keep any old output backup until
its copy has been checked file by file.
`--seed 42 --batch 20` uses seeds 42 through 61. Each attempt samples block XY
with its own seed; ARDY phase seeds add the phase index to that seed.
Repeating the same arguments reproduces the sampling plan; folder timestamps
do not change seeds. Default XY bounds are X=[.36,.46], Y=[-.30,-.16] metres.
Use `--cube-xy X Y` for a fixed position or
`--xy-range XMIN XMAX YMIN YMAX` to change the range. See [task details](grasp.md)
for supported bounds. The default scene places the robot at a target 0.20 m from
the front table edge. With `--walk`, `--start-back` (.05-.6 m) adds the initial
separation before walking to that target; `--table-standoff` (.20-.55 m)
configures the target distance. These values measure root position, not fingertip
clearance. The default root is the grounded approach target; the walking default
starts at X=-.34 m and approaches X=.11 m for the current table. Both hands
may touch the table; forearm, torso, and leg contacts still stop the trial.
These are pilot settings.

The default task runs the two-second stand, reach, lower, close, lift, and hold.
Pass `--walk` to add approach, settle, and preparation before the hand phases.
`--direct-start` remains a compatibility alias for the default no-walk mode. The
no-walk path is a manipulation calibration, not a full walk-and-grasp trial.
MuJoCo state provides table/block locations and the root target; images are not
model inputs. A walking attempt must reach the target and settle on both feet
before preparation, then pass the same continuous stability check before reaching.
The G1 head is rigid; `prepare` requests a gentle 8-degree waist inclination
through ARDY.
`prepare_not_settled` identifies an unstable preparation; `prepare_target_drift`
identifies excessive position drift during preparation. Failure
reasons `approach_target_missed`, `approach_not_settled`, and `approach_too_close`
identify walking failures; they count as failed trials. The selected path must
still finish within the same 30-second simulation budget.

Hand/contact calibration options are recorded in the plan and scene settings:

```bash
# Open the fingers during reach/lower; close only once measured alignment passes
./run.sh batch --grasp --hand-approach open
# Increase observation time without extending the 30-second trial timeout
./run.sh batch --grasp --hold-seconds 10 --lift-seconds 4.8
# Current stronger, damped hand controller (original motor limits remain active)
./run.sh batch --grasp --finger-kp 6 --finger-kd .4
# Historical hand/contact settings for an explicit calibration control
./run.sh batch --grasp --finger-kp 4 --finger-kd .2 --contact-profile legacy --table-standoff .22 --hand-approach preshaped --lift-seconds 3.2 --hold-seconds 3
```

`--wrist-offset X Y Z` overrides the nominal wrist-frame offset used to condition
ARDY. By default it is derived from the measured `task_grasp_center` site in the
MuJoCo scene. The physical acquisition gate is derived at runtime from that
same site and the current block dimensions; `--acquisition-z-min` optionally
tightens its lower wrist-frame bound. A missed lower gate can trigger up to two
short measured-state replans by default; tune them with
`--alignment-replans` and `--alignment-replan-seconds`. The measured-pose
acquisition hold uses a .3-second transition by default; configure it with
`--acquisition-transition` when auditing wrist settling. These controls use
current simulator state and remain valid for seeded random block positions.
A historical success followed by a drop is no longer treated as retained success
in new reports. `success_threshold_reached`, `retained_at_end`, and
`post_success_loss_samples` expose those stages without rewriting old evidence.

Repository diagnostic tools are separate from the baseline upload archive:

```bash
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh python scripts/analyze_grasp.py --run output/SESSION --out output/FRESH-AUDIT
MUSA_IMAGE=dl-musa-render:latest ./docker/run-musa.sh python scripts/calibrate_grasp.py --source output/SESSION --out output/FRESH-CALIBRATION --seeds 2 12 16 18 --variants strong open --fresh-motion
```

The calibration tool reruns from reset under frozen SONIC. Without
`--fresh-motion`, it reuses available nominal ARDY references and is a controller
diagnostic; with that flag, ARDY generates every moving phase from current
executed history. All calibration results remain exploratory.
Defaults use fully open fingers, a 4.8-second generated lift, and a five-second
final hold. The acquisition threshold and closure targets retain their previous
values; finger and contact gains are independent of the hand-approach option.

The terminal prints `[SUCCESS]`, `[FAILED]`, or `[COMPLETED]` for each attempt
and lists all results at the end. Open **`summary.txt` or `summary.md`** in the
output directory for counts, success rate, individual outcomes, seeds, phases,
simulation times, and reasons. `results.csv` is suitable for spreadsheets;
`successes.txt` and `failures.txt` list matching attempt directories. Execution
without a task is `COMPLETED` and does not count as grasp success. A failed grasp
does not prevent the next attempt; the overall command returns a nonzero exit
code if any attempt fails.

Both batch and manual sessions load SONIC, ARDY, and the 8B text model before
accepting or executing prompts, including sessions without a default task or
prompt. A prominent `ALL MODELS LOADED: ARDY + text encoder + SONIC` banner
confirms completion. Batch execution then starts its planned attempts. The
terminal prints a loading/generation heartbeat every 15 seconds. Startup status
and loading wall time are saved separately in `startup.json`. Simulation time pauses during
loading and generation. Batch collection **does not render automatically**;
inspect the summary first, then select videos:

```bash
# Replace the timestamp below with the actual output directory name
./run.sh render --run output/batch-YYMMDD-HHMMSS --attempts 1 3 5
./run.sh render --run output/batch-YYMMDD-HHMMSS --all
./run.sh render --run output/batch-YYMMDD-HHMMSS --successes

# A single attempt path also works
./run.sh render --run output/batch-YYMMDD-HHMMSS/attempt-00001
```

Each selected attempt produces `vision/video.mp4` and `vision/images.npz`.
Repeated rendering uses a fresh `vision-TIME/` directory without overwriting
videos or rerunning physics. Failed attempts can be rendered if their rollout
states were saved. `--all` includes failed and interrupted directories; missing
or incomplete states produce an individual error while other attempts continue.
`render-TIME.json` records every result from that rendering command.

Defaults are third-person, 640x480, 25 FPS, with H.264 CRF 18 encoding. The
complete configured hold phase is retained after the two-second success
threshold, so the final holding motion is present in the rollout and rendered
video, unless a failure or the 30-second timeout stops the trial. There is no
14-second video cap. Missing motion in an old rollout requires recollection;
rendering alone cannot add it. This is twice the previous width and height; additional pixels increase rendering
cost and RGB storage, not video duration. For 720p, use `--width 1280 --height 720`. Rendering displays progress and
estimated remaining time; software rendering can be much slower than simulation.
Use `--fps 50` to render every 50 Hz control frame, or `--no-video` for RGB only.
The grasp scene supports multiple cameras; the first camera supplies the video:

```bash
./run.sh render --run output/batch-YYMMDD-HHMMSS --attempts 1 \
  --camera third_person --camera head_camera --camera wrist_camera \
  --width 640 --height 480 --fps 25
```

Adjust the view with `--azimuth 90 --elevation -10 --distance 2.5`.
A 30-second, 640x480, 25 FPS raw RGB stream requires about 0.69 GB per camera.
RGB is saved before MP4 encoding; encoder failures preserve completed RGB and
write an error in the rendering report.

## Selecting Kimodo instead of ARDY

The same batch/manual commands can select the optional Kimodo baseline:

```bash
./run.sh batch --kimodo --prompt "A person slowly raises the right hand." --duration 2
./run.sh batch --kimodo --grasp --batch 20 --seed 0 --device musa:0 --text-device musa:0
./run.sh manual --kimodo --grasp --gui
```

`--kimodo` is the only method-selection flag. All task, grounding, finger,
SONIC, MuJoCo and evaluation options remain shared. Outputs place generated
references under `kimodo/` instead of `ardy/`, and reports identify the method
as `KIMODO`. Omitting the flag preserves the existing ARDY/B0 behavior.
For new Kimodo grasp sessions, the default wrist offset is
`0.125 0.035 0.080` metres, and raw nominal output is used. This calibration
is explicit in the saved plan and does not change the ARDY default. Supply
`--wrist-offset X Y Z` for an override; resumed plans keep their saved values.
`--kimodo-diffusion-steps` defaults to 100 and
`--kimodo-constraint-guidance` to 2. The unvalidated
`--kimodo-project-constraints` nominal projection is opt-in;
`--kimodo-no-projection` remains a compatibility alias for the raw default.
Do not interpret the tuned batch as held-out or paired B0/P evidence.
Individual Kimodo clips are capped at 300 frames (10 seconds at 30 Hz);
longer single generation requests fail explicitly before model inference.
The shared batch CLI returns exit 1 if any attempt fails; inspect
`summary.json` for completion, success counts and failure reasons.

Kimodo generation requires the pinned source checkout and checkpoint:

```bash
python scripts/fetch_kimodo.py --only all
# or after a manual transfer from a connected machine:
python scripts/fetch_kimodo.py --offline
```

The fetch operation is intentionally separate from `fetch_baseline.py`. If the
remote host cannot reach GitHub or Hugging Face, run it elsewhere and transfer:

```bash
rsync -a --progress third_party/kimodo/ user@host:/path/to/dl/third_party/kimodo/
rsync -a --progress checkpoints/kimodo/ user@host:/path/to/dl/checkpoints/kimodo/
```

## 2. Manual execution: one JSON request at a time

```bash
./run.sh manual
./run.sh manual --gui
./run.sh manual --grasp --gui
```

Results are saved to `output/manual-YYMMDD-HHMMSS/attempt-00001/` and
subsequent attempt directories. Startup loads all models before showing
`READY >`, without generating motion or advancing physics. Input entered during
loading is discarded. If loading fails, the session exits without accepting a
prompt; details are recorded in `startup.json`. Each JSON request is a **new independent attempt**: the scene resets,
while models and the GUI window stay loaded. Requests do not append commands
to the previous trajectory. There is no default task or prompt. Without a task,
enter a motion request such as:

```json
{"prompt":"A person stands upright and slowly raises the right hand.","duration":2,"seed":42}
```

With `--grasp`, `{}` uses the default eight-phase prompts. Positions and individual
phase prompts can also be specified:

```json
{"seed":42,"cube_xy":[0.40,-0.22],"phase_prompts":{"reach":"A person stands upright, bends the right elbow and raises the open right hand above the table."}}
```

Supported keys: `prompt`, `duration`, `seed`, `cube_xy`, `phase_prompts`, and
`reference`. `duration` controls general motion length (.08-25 seconds); grasp
phase durations are fixed. `reference` accepts a saved reference.npz for tracking
diagnostics without a task or prompt. An omitted seed increases with the attempt
index; an explicitly supplied seed overrides only that request.

While `BUSY`, the terminal disables echo and **discards** additional input.
Commands are not queued for the next attempt. During generation, the GUI holds
the current pose; during execution, physics is paced to simulation time.
Submit another request only after `READY >` returns. Waiting for input does not
advance physics. Enter `quit` or `exit` at READY to exit, or use Ctrl-C to
interrupt; an in-flight model computation must finish before shutdown.
Do not paste multiple JSON lines at once. Use batch mode for batch execution.

The GUI is displayed through the server's VNC service. From a Mac terminal:

```bash
ssh -tt -o ExitOnForwardFailure=yes \
  -L 5901:127.0.0.1:5900 group3@10.123.0.39 \
  'cd ~/dl && ./run.sh manual --grasp --gui'
# In another Mac terminal
open vnc://localhost:5901
```

Leave the VNC username blank; the current password is `group3`. Enter JSON in
the first terminal and watch motion in VNC. Manual sessions do not render
videos automatically. Afterward, pass the manual directory to the same `render`
command used for batches.

## Resume and prompt configuration

```bash
./run.sh batch --grasp --batch 20 --seed 42 --plan-only
./run.sh batch --resume output/batch-YYMMDD-HHMMSS
./run.sh batch --grasp --phase-prompts configs/grasp-prompts.json
```

`--plan-only` writes a plan without loading models. `--resume` is batch-only:
**supply no other options**, because all settings come from `plan.json`.
Completed successes, failures, and timeouts are skipped. Interrupted attempts
reuse the original seed and write to a sibling such as
`attempt-00001-retry-02`; use that full name when selecting it for rendering.
Changes to code, configuration, model locks, or saved evidence hashes prevent
resume and require a fresh batch. Completed failures are never retried until
success; failures remain in the success-rate denominator.

The default `focused` profile describes only the current phase. `--prompt`
adds a shared prefix to every phase. `--phase-prompts` accepts keys
approach/settle/prepare/reach/lower/close/lift/hold and replaces the entire text for each specified phase.
`--prompt-profile legacy` retains the previous wording for comparisons with
matched seeds and positions. See [prompt guidance](prompts.md) for rationale
and evidence.

## Output structure

```text
output/batch-TIME/                 # Or manual-TIME
  plan.json                       # Source, configuration, seeds, and position plan
  startup.json                    # Model loading status, wall time, and errors
  requests.json                   # Accepted manual requests
  summary.txt / summary.md         # Readable overview
  summary.json / results.csv       # Structured results and spreadsheet
  successes.txt / failures.txt     # Matching attempt names
  attempt-00001/
    request.json / report.json     # Request, outcome, clocks, hashes, and errors
    scene.xml / settings.json      # Grasp scene and task parameters
    events.jsonl / trajectory.csv  # Events, states, references, and torques
    nominal_context.csv           # Causal SONIC/Risk input context at 50 Hz
    task.csv                      # 200 Hz grasp contact, height, and wrist log
    ardy/PHASE/                   # Text, measured history, constraints, references
    grounding/PHASE.json           # Current table, block, root, and approach target
    rollout/                      # Compiled scene and initial + full 50 Hz states
    vision/                       # Created only by a later render command
      images.npz / video.mp4 / report.json
```

State frame -1 is the initial state; subsequent frame_index values match
trajectory.csv. RGB preserves original state_index/frame_index/sim_time and
synchronizes multiple cameras. Visual replay does not rerun SONIC or establish
independent physical acceptance. Task success, rendering success, and training
eligibility are separate fields. Currently `expert_valid=false`; the full
learning-data pipeline remains planned.

## Installation and diagnostics

Reuse an existing environment. For initial setup:

```bash
./run.sh build
./run.sh build-gui
./run.sh shell
# Inside the container; create the environment only if it does not exist
python -m virtualenv --system-site-packages .venv-baseline-musa
source .venv-baseline-musa/bin/activate
bash scripts/install_baseline.sh
exit
# Back on the host
./run.sh fetch --only all
./run.sh check
```

Install virtualenv first if it is missing. Preserve the matched vendor
torch/torch_musa stack; do not replace it with CUDA builds. Llama downloads
require an approved Hugging Face account; run `hf auth login` inside the
container and keep credentials outside Git. Select components with
`fetch --only sources|sonic|ardy|text|llama`. Offline registration uses
`--offline` and requires complete original download metadata.

```bash
./run.sh tests
./run.sh smoke --out output/smoke-NEW-NAME
./run.sh sonic --out output/sonic-NEW-NAME
./run.sh ardy --prompt 'A person stands still.' --duration 2 --seed 42 --out output/ardy-NEW-NAME
./run.sh package --out output/b0-source-NEW-NAME.tar.gz
./run.sh batch --help
./run.sh manual --help
./run.sh render --help
```

`check` verifies complete assets and MUSA primitives. Smoke uses synthetic
references and establishes neither model compatibility nor grasp success.
`ardy`, `sonic`, `convert`, `deploy`, and `service` are module diagnostics or
conversion utilities. `MUSA_IMAGE` overrides the container image;
`MTHREADS_VISIBLE_DEVICES` selects devices. See [baseline](baseline.md) and
[integration](integration.md) for interface details.
