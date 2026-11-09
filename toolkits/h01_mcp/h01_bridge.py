#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ROS 2 bridge between the H01 upper body and a plain JSON control surface.

Covers what the vendor calls the upper body: both arms, both grippers, the
two-joint head, and the four-joint waist. The chassis is deliberately absent --
this process never publishes to it.

The motor driver drops a joint command 0.2 s after the last one, so once a
client takes control the last targets are republished at 15 Hz.
"""

import threading
import time
from typing import Any, Optional

import numpy as np
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from realsense2_camera_msgs.msg import FixedImageRGB640x480
from sensor_msgs.msg import JointState

LEFT_ARM_JOINTS = [
    "L_SHOULDER_P",
    "L_SHOULDER_R",
    "L_SHOULDER_Y",
    "L_ELBOW_Y",
    "L_WRIST_P",
    "L_WRIST_Y",
    "L_WRIST_R",
]
RIGHT_ARM_JOINTS = [
    "R_SHOULDER_P",
    "R_SHOULDER_R",
    "R_SHOULDER_Y",
    "R_ELBOW_Y",
    "R_WRIST_P",
    "R_WRIST_Y",
    "R_WRIST_R",
]
HEAD_JOINTS = ["HEAD_YAW", "HEAD_PITCH"]
WAIST_JOINTS = ["WAIST_1", "WAIST_2", "WAIST_3", "WAIST_4"]

#: Travel of each group in radians, from the vendor's ``JointLimits``.
ARM_LIMITS = {
    "left": np.array(
        [
            [-2.356, 0.698],
            [-1.693, 0.960],
            [-3.054, 3.054],
            [-0.175, 2.094],
            [-3.054, 3.054],
            [-1.100, 0.314],
            [-1.5708, 1.309],
        ],
        dtype=float,
    ),
    "right": np.array(
        [
            [-0.698, 2.356],
            [-0.960, 1.693],
            [-3.054, 3.054],
            [-2.094, 0.175],
            [-3.054, 3.054],
            [-0.314, 1.100],
            [-1.309, 1.5708],
        ],
        dtype=float,
    ),
}

HEAD_LIMITS = np.array([[-0.524, 0.524], [-0.349, 0.349]], dtype=float)

WAIST_LIMITS = np.array(
    [[-0.995, 0.995], [-1.24, 0.0], [0.0, 2.08], [0.0, 1.34]], dtype=float
)

#: Gripper opening range in metres.
GRIPPER_LIMITS = (0.0, 0.06)

#: Command stream period. The motor driver faults 0.2 s after the last command.
COMMAND_PERIOD = 1.0 / 15.0


def _qos(reliable: bool, depth: int = 1) -> QoSProfile:
    return QoSProfile(
        reliability=(
            ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT
        ),
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
    )


class H01Bridge(Node):
    """One ROS node holding the whole upper-body control surface."""

    def __init__(self) -> None:
        super().__init__("h01_mcp_bridge")
        self._lock = threading.Lock()

        self._arm: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._arm_vel: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._gripper: dict[str, Optional[float]] = {"left": None, "right": None}
        self._head: Optional[np.ndarray] = None
        self._waist: Optional[np.ndarray] = None
        self._stamp = {"arm": 0.0, "gripper": 0.0, "head": 0.0, "waist": 0.0}
        self._frame: Optional[np.ndarray] = None
        self._frame_seq = 0

        self._cmd_arm: dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
        self._cmd_gripper: dict[str, Optional[float]] = {"left": None, "right": None}
        self._cmd_head: Optional[np.ndarray] = None
        self._cmd_waist: Optional[np.ndarray] = None
        self._engaged = False

        self._pub_arm = {
            "left": self.create_publisher(
                JointState, "/left_arm/joint_command", _qos(True)
            ),
            "right": self.create_publisher(
                JointState, "/right_arm/joint_command", _qos(True)
            ),
            "left_gripper": self.create_publisher(
                JointState, "/left_gripper/joint_command", _qos(True)
            ),
            "right_gripper": self.create_publisher(
                JointState, "/right_gripper/joint_command", _qos(True)
            ),
            "head": self.create_publisher(
                JointState, "/head/joint_command", _qos(True)
            ),
            "waist": self.create_publisher(
                JointState, "/waist/joint_command", _qos(True)
            ),
        }

        for side in ("left", "right"):
            self.create_subscription(
                JointState,
                f"/{side}_arm/joint_state",
                lambda msg, side=side: self._on_arm(side, msg),
                _qos(True, 10),
            )
            self.create_subscription(
                JointState,
                f"/{side}_gripper/joint_state",
                lambda msg, side=side: self._on_gripper(side, msg),
                _qos(True, 10),
            )
        self.create_subscription(
            JointState, "/head/joint_state", self._on_head, _qos(True, 10)
        )
        self.create_subscription(
            JointState, "/waist/joint_state", self._on_waist, _qos(True, 10)
        )
        self.create_subscription(
            FixedImageRGB640x480,
            "/camera_head/color/image",
            self._on_frame,
            _qos(False, 1),
        )

        self._running = True
        self._thread = threading.Thread(target=self._command_loop, daemon=True)
        self._thread.start()

    # -- subscriptions -----------------------------------------------------

    def _on_arm(self, side: str, msg: JointState) -> None:
        with self._lock:
            self._arm[side] = np.asarray(msg.position[:7], dtype=float)
            self._arm_vel[side] = (
                np.asarray(msg.velocity[:7], dtype=float)
                if msg.velocity
                else np.zeros(7)
            )
            self._stamp["arm"] = time.monotonic()

    def _on_gripper(self, side: str, msg: JointState) -> None:
        with self._lock:
            if msg.position:
                self._gripper[side] = float(msg.position[0])
            self._stamp["gripper"] = time.monotonic()

    def _on_head(self, msg: JointState) -> None:
        with self._lock:
            self._head = np.asarray(msg.position[:2], dtype=float)
            self._stamp["head"] = time.monotonic()

    def _on_waist(self, msg: JointState) -> None:
        with self._lock:
            self._waist = np.asarray(msg.position[:4], dtype=float)
            self._stamp["waist"] = time.monotonic()

    def _on_frame(self, msg: FixedImageRGB640x480) -> None:
        raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        if raw.size == 640 * 480 * 3:
            with self._lock:
                self._frame = raw.reshape(480, 640, 3)
                self._frame_seq += 1

    # -- command loop ------------------------------------------------------

    def _publish(self, topic: str, names: list[str], positions: list[float]) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions
        self._pub_arm[topic].publish(msg)

    def _command_loop(self) -> None:
        """Repeat the last targets, so the 0.2 s motor watchdog never expires."""
        while self._running:
            time.sleep(COMMAND_PERIOD)
            with self._lock:
                if not self._engaged:
                    continue
                arm = {
                    k: (None if v is None else v.copy())
                    for k, v in self._cmd_arm.items()
                }
                gripper = dict(self._cmd_gripper)
                head = None if self._cmd_head is None else self._cmd_head.copy()
                waist = None if self._cmd_waist is None else self._cmd_waist.copy()
            for side, names in (("left", LEFT_ARM_JOINTS), ("right", RIGHT_ARM_JOINTS)):
                if arm[side] is not None:
                    self._publish(side, names, [float(v) for v in arm[side]])
                if gripper[side] is not None:
                    self._publish(
                        f"{side}_gripper", [f"{side}_gripper"], [float(gripper[side])]
                    )
            if head is not None:
                self._publish("head", HEAD_JOINTS, [float(v) for v in head])
            if waist is not None:
                self._publish("waist", WAIST_JOINTS, [float(v) for v in waist])

    # -- public surface ----------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return the upper body state as JSON-ready data."""
        with self._lock:
            now = time.monotonic()
            arm = {k: (None if v is None else v.tolist()) for k, v in self._arm.items()}
            arm_vel = {
                k: (None if v is None else v.tolist()) for k, v in self._arm_vel.items()
            }
            snapshot = {
                "left_arm": {"position": arm["left"], "velocity": arm_vel["left"]},
                "right_arm": {"position": arm["right"], "velocity": arm_vel["right"]},
                "left_gripper": {"position": self._gripper["left"]},
                "right_gripper": {"position": self._gripper["right"]},
                "head": {
                    "position": None if self._head is None else self._head.tolist()
                },
                "waist": {
                    "position": None if self._waist is None else self._waist.tolist()
                },
                "ages_s": {
                    key: round(now - value, 4) if value else None
                    for key, value in self._stamp.items()
                },
                "engaged": self._engaged,
            }
        return snapshot

    def command(self, **kwargs: Any) -> dict[str, Any]:
        """Clamp, store, and publish the given targets."""
        with self._lock:
            self._engaged = True
            current = {
                "left": None if self._arm["left"] is None else self._arm["left"].copy(),
                "right": None
                if self._arm["right"] is None
                else self._arm["right"].copy(),
            }
            head_now = None if self._head is None else self._head.copy()
            waist_now = None if self._waist is None else self._waist.copy()

        applied: dict[str, Any] = {}
        for side in ("left", "right"):
            target = kwargs.get(f"{side}_arm")
            if target is None:
                continue
            applied[f"{side}_arm"] = self._clip_group(
                target, ARM_LIMITS[side], current[side], kwargs.get("max_delta_rad")
            ).tolist()

        for side in ("left", "right"):
            position = kwargs.get(f"{side}_gripper")
            if position is None:
                continue
            applied[f"{side}_gripper"] = float(
                np.clip(float(position), *GRIPPER_LIMITS)
            )

        if kwargs.get("head") is not None:
            applied["head"] = self._clip_group(
                kwargs["head"], HEAD_LIMITS, head_now, kwargs.get("max_delta_rad")
            ).tolist()
        if kwargs.get("waist") is not None:
            applied["waist"] = self._clip_group(
                kwargs["waist"], WAIST_LIMITS, waist_now, kwargs.get("max_delta_rad")
            ).tolist()

        with self._lock:
            for key, value in applied.items():
                if key.endswith("_arm"):
                    self._cmd_arm[key[: -len("_arm")]] = np.asarray(value, dtype=float)
                elif key.endswith("_gripper"):
                    self._cmd_gripper[key[: -len("_gripper")]] = value
                elif key == "head":
                    self._cmd_head = np.asarray(value, dtype=float)
                elif key == "waist":
                    self._cmd_waist = np.asarray(value, dtype=float)
        return applied

    @staticmethod
    def _clip_group(
        target: Any,
        limits: np.ndarray,
        measured: Optional[np.ndarray],
        max_delta: Optional[float],
    ) -> np.ndarray:
        """Clip a joint target to its travel and to one step of motion."""
        values = np.asarray(target, dtype=float).reshape(-1)
        dof = limits.shape[0]
        if values.shape != (dof,):
            raise ValueError(f"Expected {dof} joint values, got {values.shape}.")
        values = np.clip(values, limits[:, 0], limits[:, 1])
        if max_delta is not None and measured is not None:
            values = measured + np.clip(values - measured, -max_delta, max_delta)
        return values

    def stop(self) -> None:
        """Hold every commanded group where it is."""
        with self._lock:
            self._engaged = True
            self._cmd_arm = {
                side: None if self._arm[side] is None else self._arm[side].copy()
                for side in ("left", "right")
            }
            self._cmd_head = None if self._head is None else self._head.copy()
            self._cmd_waist = None if self._waist is None else self._waist.copy()

    def release(self) -> None:
        """Stop repeating commands."""
        with self._lock:
            self._engaged = False

    def frame(self) -> tuple[int, Optional[np.ndarray]]:
        """Return the latest head frame and the counter that identifies it."""
        with self._lock:
            seq = self._frame_seq
            frame = None if self._frame is None else self._frame.copy()
        return seq, frame

    def close(self) -> None:
        self._running = False
        self.release()
        self._thread.join(timeout=1.0)
