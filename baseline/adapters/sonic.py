"""Protocol-v1 serialization matching SONIC b042411f's C++ receiver.

No socket is opened here; the initial integration emits reviewable packet files.
Wire layout: topic + 1280-byte null-padded JSON + little-endian array bytes.
"""

import json

import numpy as np

HEADER_SIZE = 1280


def pack_joint_reference(joint_pos, joint_vel, body_quat, frame_index, *,
                         left_hand=None, right_hand=None, catch_up=False, topic="pose"):
    q, v, quat = (np.asarray(x, dtype="<f4") for x in (joint_pos, joint_vel, body_quat))
    indices = np.asarray(frame_index)
    n = len(q)
    if n < 1 or q.shape != (n, 29) or v.shape != q.shape or quat.shape != (n, 4):
        raise ValueError("Protocol v1 needs q/v [N,29] and root wxyz [N,4]")
    if indices.shape != (n,) or indices.dtype.kind not in "iu":
        raise ValueError("frame_index must be an integer [N] array")
    if indices.max() > np.iinfo(np.int64).max:
        raise ValueError("Frame indices exceed signed 64-bit range")
    if (indices < 0).any() or (np.diff(indices.astype(np.int64)) <= 0).any():
        raise ValueError("Frame indices must be nonnegative and strictly increasing")
    if not all(np.isfinite(x).all() for x in (q, v, quat)):
        raise ValueError("Nonfinite motion fields")
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-3):
        raise ValueError("Expected unit wxyz quaternion")
    fields = [("joint_pos", "f32", q), ("joint_vel", "f32", v),
              ("body_quat", "f32", quat), ("frame_index", "i64", indices.astype("<i8")),
              ("catch_up", "u8", np.array([catch_up], dtype="u1"))]
    for name, hand in (("left_hand_joints", left_hand), ("right_hand_joints", right_hand)):
        if hand is not None:
            hand = np.asarray(hand, dtype="<f4")
            if hand.shape != (7,) or not np.isfinite(hand).all():
                raise ValueError("Separate Dex3 hand commands must have seven finite joints")
            fields.append((name, "f32", hand))
    header = {"v": 1, "endian": "le", "count": n,
              "fields": [{"name": name, "dtype": dtype, "shape": list(array.shape)}
                         for name, dtype, array in fields]}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(encoded) >= HEADER_SIZE:
        raise ValueError("SONIC header exceeds fixed byte budget")
    return (topic.encode("utf-8") + encoded.ljust(HEADER_SIZE, b"\0") +
            b"".join(array.tobytes(order="C") for _, _, array in fields))


class JointStreamEncoder:
    """Own frame indices across packets; reset only with a fresh SONIC session."""

    def __init__(self):
        self.next_frame = 0
        self.next_time = None

    def encode(self, reference, **kwargs):
        n = len(reference.times)
        if not np.allclose(np.diff(reference.times), 0.02, atol=1e-6, rtol=0):
            raise ValueError("SONIC packet reference must first be resampled to 50 Hz")
        if self.next_time is not None and not np.isclose(reference.times[0], self.next_time, atol=1e-6, rtol=0):
            raise ValueError("Append-only encoder requires contiguous non-overlapping chunks")
        packet = pack_joint_reference(reference.joint_pos, reference.velocities(),
                                      reference.body_quat, np.arange(self.next_frame, self.next_frame + n),
                                      **kwargs)
        self.next_frame += n
        self.next_time = reference.times[-1] + 0.02
        return packet
