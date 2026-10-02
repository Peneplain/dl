# Running the project

All commands below use the same entry point: **run `./run.sh` from `~/dl` on
the S4000 server host**. The launcher selects the container and existing Python
environment automatically. Do not run this wrapper inside a container.

The currently supported pipeline is frozen ARDY -> frozen SONIC -> free-base
MuJoCo execution -> saved states -> offline camera RGB/video. Tabletop grasping
and trained Risk/Residual controllers remain under development.

## Quick start

```bash
cd ~/dl
./run.sh check
./run.sh live --prompt "Raise both arms slowly, then lower them." \
  --duration 2 --sim-seconds 5 --fast \
  --out artifacts/demo-01 --video artifacts/demo-01/baseline.mp4
```

The run records full states during execution, then renders `vision/images.npz`
and `baseline.mp4`. The first text request loads ARDY and its text encoder.
Rendering is offline and may take longer than the simulated motion.

Use a **new run name** each time (`demo-02`, `demo-03`, ...). Existing run,
render and video outputs are never reused. Default devices are MUSA for ARDY
and text encoding (bfloat16 text weights), CPU ONNX Runtime for SONIC, and
OSMesa software rendering for RGB. The default image is `dl-musa-render:latest`.

## Record first, render later

Save states without starting a renderer:

```bash
./run.sh live --prompt "Raise both arms slowly, then lower them." \
  --duration 2 --sim-seconds 5 --fast --out artifacts/record-01
```

Render those states once, using the default fixed third-person camera:

```bash
./run.sh render --run artifacts/record-01 \
  --video artifacts/record-01/baseline.mp4
```

For RGB only, omit `--video` from the rendering command. Choose either command
for the first render; both default to the same fresh `vision/` directory.

Change the view without rerunning the robot:

```bash
./run.sh render --run artifacts/record-01 \
  --out artifacts/record-01/vision-side \
  --azimuth 90 --elevation -10 --distance 2.5 \
  --video artifacts/record-01/side.mp4
```

| Option | Meaning | Default |
| --- | --- | --- |
| `--camera` | `third_person` or a camera defined in the recorded MJCF | `third_person` |
| `--lookat X Y Z` | Fixed world-space camera target, metres | `0 0 0.75` |
| `--distance` | Distance from camera to target, metres | `3` |
| `--azimuth` | Horizontal camera angle, degrees | `135` |
| `--elevation` | Vertical camera angle, degrees | `-15` |
| `--width`, `--height` | RGB resolution; MP4 requires even dimensions | `640`, `480` |
| `--video-fps` | MP4 simulation-time sampling rate | `25` |
| `--out` | Fresh rendering output directory | `RUN/vision` |

Multiple `--camera` options produce synchronized RGB views; the MP4 uses the
first camera. Head/wrist camera names work only after those cameras are added
to the scene **before recording**. They are not currently defined in the G1
bring-up scene. Old CSV-only runs lack the complete state needed for this replay.

## Interactive prompts

Keep ARDY and SONIC loaded and record a separate episode for each prompt:

```bash
./run.sh live --keep-alive --fast --sim-seconds 5 \
  --out artifacts/session-01 --video artifacts/session-01/baseline.mp4
```

After loading finishes, type one request per line:

```text
{"prompt":"A person stands still.","duration":2,"seed":0}
{"prompt":"Raise both arms slowly, then lower them.","duration":3,"seed":1}
quit
```

Plain text also works, using duration 2 s and seed 0. The robot resets between
episodes. Outputs go to `session-01/run-0001/`, `run-0002/`, ...; rendering
finishes before the next prompt is executed. `session.json` records each run's
status. A failed/stopped episode makes the completed session exit nonzero.
These are bring-up interactions, not the proposal's evaluated correction subset.

| Execution option | Behavior |
| --- | --- |
| `--fast` | Pause simulation during ARDY generation; remove wall-clock pacing |
| No `--fast` | Pace physics at 50 Hz; a single-run request generates while SONIC holds |
| `--keep-alive` | Independent prompt episodes; pause simulation during generation/waiting |
| `--duration` | Requested ARDY motion duration, seconds |
| `--sim-seconds` | Minimum episode duration; successful references also get transition + 2 s hold |
| `--seed` | ARDY generation seed |
| `--video PATH.mp4` | Render saved states to RGB and MP4 after execution |

`--fast` does not establish real-time performance. Safety stops can end an
episode early. A standing/arm-motion run reporting `passed` is not grasp success.

## Live GUI from a Mac

Run this in the **Mac terminal**, not inside Docker:

```bash
ssh -tt -o ExitOnForwardFailure=yes \
  -L 5901:127.0.0.1:5900 group3@10.123.0.39 \
  'cd ~/dl && ./run.sh gui'
```

After the models load and the viewer is ready, run this in another Mac terminal:

```bash
open vnc://localhost:5901
```

Leave the VNC username blank; the current password is `group3`. Enter the same
JSONL prompts in the first terminal. `quit` closes the session. Each episode's
states, RGB and MP4 are saved under `artifacts/interactive-gui-*/run-NNNN/`.
The GUI uses `dl-musa-gui:latest` and its virtual GLX display. Omit `--fast` when
watching motion. Offline rendering between prompts can temporarily leave the
viewer at the final pose.

To start the same GUI on the server host:

```bash
./run.sh gui
```

## Separate model checks and execution

Check standing without loading ARDY:

```bash
./run.sh live --sim-seconds 5 --fast --out artifacts/standing-01 \
  --video artifacts/standing-01/baseline.mp4 < /dev/null
```

Generate an ARDY reference without robot execution:

```bash
./run.sh ardy --prompt "A person stands still." --duration 2 --seed 0 \
  --out artifacts/ardy-01
```

Execute the generated reference through SONIC:

```bash
./run.sh live --reference artifacts/ardy-01/reference.npz \
  --sim-seconds 5 --fast --out artifacts/reference-01 \
  --video artifacts/reference-01/baseline.mp4 < /dev/null
```

Probe the two frozen SONIC ONNX graphs:

```bash
./run.sh sonic --out artifacts/sonic-check-01
```

This is a graph/operator check with synthetic inputs, not robot control.

Keep ARDY loaded for repeated reference generation only:

```bash
./run.sh service --out-root artifacts/ardy-service-01
```

Then enter:

```text
{"name":"stand","prompt":"A person stands still.","duration":2,"seed":0}
{"name":"arms","prompt":"Raise both arms slowly.","duration":2,"seed":1}
quit
```

Convert a motion CSV into a named 50 Hz reference and offline packet:

```bash
./run.sh convert --qpos-csv artifacts/ardy-01/motion.csv \
  --joint-names artifacts/ardy-01/joint_names.json --source-fps 25 \
  --out artifacts/converted-01.npz --packet
```

Export the reference as upstream SONIC deploy CSVs:

```bash
./run.sh deploy --reference artifacts/ardy-01/reference.npz \
  --motion-csv artifacts/ardy-01/motion.csv \
  --out-dir artifacts/deploy-01 --name stand
```

Neither converter starts a simulator, network publisher or C++ deploy process.

## Validation and packaging

```bash
./run.sh check
./run.sh tests
./run.sh smoke --out artifacts/smoke-01
./run.sh package --out artifacts/b0-source-01.tar.gz
```

`check` verifies dependencies, pinned sources/assets and MUSA primitives.
`tests` includes actual RGB/MP4 replay fixtures. `smoke` checks synthetic
reference conversion and packet fields. These do not prove physical grasp
success. `package` includes only declared baseline source files, including
this guide and launcher; model weights and run outputs stay excluded.

## First setup or image rebuild

The current server already has the environment, weights and images. Use these
commands only when setting up another machine or rebuilding an image:

```bash
cd ~/dl
./run.sh build
./run.sh build-gui
./run.sh build-base
```

The host needs the MUSA Docker runtime registered. On a new host with the
toolkit already installed, an administrator can run:

```bash
sudo /usr/bin/musa/docker setup /usr/bin/musa
sudo systemctl restart docker
```

Only the first image is needed for ordinary headless execution and RGB/video.
`build-gui` adds the viewer/VNC environment. `build-base` is control-only and
does not include a graphics backend.

On first setup, enter the image and create the environment **inside Docker**:

```bash
./run.sh shell
```

Then, in that container shell:

```bash
python -m virtualenv --system-site-packages .venv-baseline-musa
source .venv-baseline-musa/bin/activate
bash scripts/install_baseline.sh
exit
```

Reuse an existing environment; do not recreate it. If the vendor image lacks
`virtualenv`, install that tool with `python -m pip install virtualenv` first.
The installer preserves the matched vendor `torch`/`torch_musa` packages.

Back on the host, download pinned sources and weights:

```bash
./run.sh fetch --only all
./run.sh check
```

`--only sources`, `sonic`, `ardy`, `text`, or `llama` selects a subset. Llama
access requires the approved Hugging Face account. To log in using the project's
installed HF client, use `./run.sh shell`, then `hf auth login`; keep credentials
outside Git. Offline registration is `./run.sh fetch --only all --offline` and
requires complete source checkouts plus original download provenance.

For custom Python work, `./run.sh shell` opens the configured environment.
If its shell resets PATH, use `source .venv-baseline-musa/bin/activate` there.

## Outputs and failure handling

```text
artifacts/<run>/
  report.json             Execution status, failures, model hashes and timings
  events.jsonl            Reference, hold and stop events
  trajectory.csv          Control-frame robot state/reference/torque log
  ardy/                   Generated motion and references, when requested
  rollout/
    scene.mjb             Compiled scene with assets and physics settings
    metadata.json         Version, state schema, joint addresses and hashes
    states.npz            Initial state and every completed control-frame state
  vision/
    images.npz            RGB [N,C,H,W,3] uint8 and source frame/time mappings
    report.json           Camera settings, hashes, rendering/encoding status
  baseline.mp4            Optional first-camera H.264 video
  baseline.mp4.ffmpeg.log  Encoder diagnostics
```

State/RGB frame `-1` is the initial state; frames `0..` match `trajectory.csv`.
Recording and RGB use 50 Hz control frames, not the 200 Hz physics substeps.
Full state includes free root, articulated fingers and any dynamic objects
present in the scene. Replay restores each saved state and calls `mj_forward`
to refresh geometry; it never steps physics or runs a teacher/policy.

RGB is saved before MP4 encoding. If encoding fails, inspect `video_status`
and the ffmpeg log; successfully saved RGB and rollout states remain available.
Partial MP4s are not published. MP4 frames use a uniform simulation-time grid,
mapped to the nearest saved states. Video metadata includes source state indices,
frame indices, simulation times and presentation times. The initial/final samples
can make encoded duration differ from the recorded time span by at most one
video-frame interval. RGB always includes every saved state, irrespective of FPS.

If rendering fails, retry from the saved rollout with a new `--out` and video
path. `execution_status` distinguishes execution from automatic rendering failure.
Automatic single runs and completed interactive sessions return nonzero on
failure or safety stop. Read the reports even when a video was produced.

Keep enough disk space for the temporary RGB array: 30 s at 50 Hz, 640x480,
one camera needs about 1.38 GB before compression; each additional camera adds
the same amount. OSMesa rendering is offline software work, not MUSA inference.
Keep simulation time, model latency and rendering wall time separate.

The current real G1 scene has no tabletop grasp task. Saved state is sufficient
for visual replay but not counterfactual learning branches: controller history,
reference buffers and disturbance RNG need separate restoration. See
[verification.md](verification.md) for actual evidence and remaining gaps.

## Help

```bash
./run.sh help
./run.sh live --help
./run.sh render --help
./run.sh ardy --help
```

Advanced overrides on the host:

```bash
MTHREADS_VISIBLE_DEVICES=0 ./run.sh live --sim-seconds 2 --fast \
  --out artifacts/device-01 < /dev/null
MUSA_IMAGE=dl-musa:latest ./run.sh live --sim-seconds 2 --fast \
  --out artifacts/control-only-01 < /dev/null
```

The control-only image can save states; run `./run.sh render` afterward in the
default rendering image. The launcher uses private IPC with 16 GiB shared-memory
capacity. Prefer this default when running multiple containers.
