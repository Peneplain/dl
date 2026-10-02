"""Model-independent MuJoCo state recording for offline visual replay.

This is not a counterfactual branch checkpoint: SONIC history, reference buffers
and disturbance RNG state must also be restored for learning-data branches.
"""

import json
from pathlib import Path
import sys

import mujoco
import numpy as np

from baseline.common import sha256, write_json


class RolloutRecorder:
    """Capture the initial state and every completed control frame, including objects."""

    def __init__(self, model, output, *, scene=None, control_hz=50):
        self.model = model
        self.output = Path(output) / "rollout"
        self.output.mkdir(exist_ok=False)
        self.state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.state_size = mujoco.mj_stateSize(model, self.state_spec)
        self.rows = []
        self.closed = False
        model_path = self.output / "scene.mjb"
        mujoco.mj_saveModel(model, str(model_path))
        self.metadata = {
            "schema_version": 1, "mujoco_version": mujoco.__version__,
            "numpy_version": np.__version__, "state_spec": self.state_spec,
            "command": sys.argv,
            "recording_source_sha256": sha256(__file__),
            "state_fields": "mjSTATE_INTEGRATION", "state_size": self.state_size,
            "model_sha256": sha256(model_path), "control_hz": control_hz,
            "physics_timestep": float(model.opt.timestep),
            "source_scene": str(scene) if scene is not None else None,
            "source_scene_sha256": sha256(scene) if scene is not None else None,
            "joint_names": [model.joint(i).name or "" for i in range(model.njnt)],
            "joint_qpos_address": model.jnt_qposadr.tolist(),
            "joint_dof_address": model.jnt_dofadr.tolist(),
            "joint_types": model.jnt_type.tolist(),
            "body_names": [model.body(i).name or "" for i in range(model.nbody)],
            "actuator_names": [model.actuator(i).name or "" for i in range(model.nu)],
            "quaternion_order": "wxyz", "units": "SI; joint angles in radians",
            "frame_index": "-1 is initial state; >=0 matches trajectory.csv control frames",
            "scope": "visual replay; excludes controller/reference/RNG branch state",
            "status": "recording",
        }
        write_json(self.output / "metadata.json", self.metadata)

    def record(self, data, frame_index, wall_time):
        if self.closed:
            raise RuntimeError("Rollout is already closed")
        if self.rows and (data.time <= self.rows[-1][0] or frame_index <= self.rows[-1][2]):
            raise ValueError("Rollout timestamps and frame indices must increase; reset per episode")
        state = np.empty(self.state_size, dtype=np.float64)
        mujoco.mj_getState(self.model, data, state, self.state_spec)
        self.rows.append((float(data.time), float(wall_time), int(frame_index), state,
                          data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()))

    def close(self):
        if self.closed:
            return
        # Exclusive creation protects the write-once episode boundary.
        path = self.output / "states.npz"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite recorded states: {path}")
        temporary = self.output / ".states.npz.tmp"
        try:
            with temporary.open("xb") as handle:
                np.savez_compressed(
                    handle, sim_time=np.array([r[0] for r in self.rows]),
                    wall_time=np.array([r[1] for r in self.rows]),
                    frame_index=np.array([r[2] for r in self.rows], dtype=np.int64),
                    state=np.array([r[3] for r in self.rows]).reshape(-1, self.state_size),
                    qpos=np.array([r[4] for r in self.rows]).reshape(len(self.rows), self.model.nq),
                    qvel=np.array([r[5] for r in self.rows]).reshape(len(self.rows), self.model.nv),
                    ctrl=np.array([r[6] for r in self.rows]).reshape(len(self.rows), self.model.nu),
                )
            temporary.rename(path)
        finally:
            temporary.unlink(missing_ok=True)
        self.metadata.update(status="complete", frames=len(self.rows),
                             states_sha256=sha256(path))
        write_json(self.output / "metadata.json", self.metadata)
        self.closed = True

    def summary(self):
        self.close()
        return {"path": str(self.output), "frames": len(self.rows),
                "state_fields": "mjSTATE_INTEGRATION",
                "model_sha256": self.metadata["model_sha256"],
                "states_sha256": self.metadata["states_sha256"], "status": "passed"}


class SavedRollout:
    """Load an immutable compiled scene and restore frames without stepping physics."""

    def __init__(self, run):
        self.directory = Path(run) / "rollout"
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if self.metadata.get("schema_version") != 1 or self.metadata.get("status") != "complete":
            raise ValueError("Rollout is incomplete or has an unsupported schema")
        if self.metadata["mujoco_version"] != mujoco.__version__:
            raise ValueError("Compiled scene replay requires the recorded MuJoCo version: "
                             + self.metadata["mujoco_version"])
        for name, key in (("scene.mjb", "model_sha256"), ("states.npz", "states_sha256")):
            if sha256(self.directory / name) != self.metadata[key]:
                raise ValueError(f"Recorded {name} hash mismatch")
        self.model = mujoco.MjModel.from_binary_path(str(self.directory / "scene.mjb"))
        self.data = mujoco.MjData(self.model)
        self.state_spec = self.metadata["state_spec"]
        if self.state_spec != int(mujoco.mjtState.mjSTATE_INTEGRATION):
            raise ValueError("Unsupported recorded state specification")
        with np.load(self.directory / "states.npz", allow_pickle=False) as arrays:
            self.states = arrays["state"]
            self.sim_time = arrays["sim_time"]
            self.frame_index = arrays["frame_index"]
        n = len(self.sim_time)
        if (n == 0 or n != self.metadata.get("frames")
                or self.states.shape != (n, mujoco.mj_stateSize(self.model, self.state_spec))
                or self.sim_time.shape != (n,) or self.frame_index.shape != (n,)
                or self.frame_index.dtype.kind not in "iu"
                or not np.isfinite(self.states).all() or not np.isfinite(self.sim_time).all()
                or (np.diff(self.sim_time) <= 0).any()
                or (np.diff(self.frame_index) <= 0).any()
                or not np.array_equal(self.states[:, 0], self.sim_time)):
            raise ValueError("Invalid recorded state dimensions, timestamps or frame indices")

    def restore(self, index):
        mujoco.mj_setState(self.model, self.data, self.states[index], self.state_spec)
        # Derived body/geom/camera transforms must reflect this qpos, not the
        # previous frame's mj_step internals. Never re-execute the rollout.
        mujoco.mj_forward(self.model, self.data)
        return self.data
