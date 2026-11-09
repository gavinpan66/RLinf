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

"""Maker H01 upper-body environment: absolute joint targets.

Both arms, both grippers, the head, and the waist are controlled in joint space
because the vendor publishes no tool pose for the arms. The head and the waist
are observed always and commanded only when the task asks for them; the chassis
is not part of this environment.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.real.utils.config import get_hardware_config
from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.robotics import H01Config, H01Robot, RobotInfo
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.robotics.parts.h01 import ARM_DOF
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

#: Joints across both arms, the way the observation vector lays them out.
BOTH_ARMS_DOF = 2 * ARM_DOF

#: Sides in the order their joints and grippers appear in an action.
SIDES = ("left", "right")


@dataclass
class H01EnvConfig:
    """Task, control, and observation settings for an H01 environment."""

    is_dummy: bool = False
    """Run without hardware, sampling observations from the space."""

    step_frequency: float = 10.0
    """Control rate in Hz. A step sleeps for the remainder of its period."""

    reset_joint_qpos: Optional[list[float]] = None
    """Rest configuration in radians, both arms in action order.

    ``None`` holds whatever pose the arms were in when the environment was
    built, which is the safe default for a robot with no calibrated home."""

    target_joint_qpos: Optional[list[float]] = None
    """Goal configuration in radians. ``None`` reuses the rest configuration."""

    reward_threshold: float = 0.05
    """Per-joint tolerance in radians. Every joint must be within it."""

    use_dense_reward: bool = False
    """Report the negative joint distance instead of a sparse hit."""

    success_hold_steps: int = 1
    """Consecutive successful steps before the episode terminates."""

    max_num_steps: int = 100
    """Steps before an episode is truncated."""

    control_head: bool = False
    """Append the two head joints to the action vector instead of holding them."""

    control_waist: bool = False
    """Append the four waist joints to the action vector instead of holding them."""

    reset_timeout_s: float = 10.0
    """Seconds the reset motion may take before it stops waiting."""

    reset_tolerance: float = 0.05
    """Joint distance in radians at which a reset counts as arrived."""

    frame_size: int = 128
    """Square size the head camera frame is resized to."""


class H01Env(gym.Env):
    """The H01 upper body: both arms, their grippers, head, and waist.

    An action is ``[left 7, left gripper, right 7, right gripper]``, followed by
    ``[head yaw, head pitch]`` and the four waist joints when the matching
    ``control_*`` flag is set.
    """

    CONFIG_CLS: type[H01EnvConfig] = H01EnvConfig
    TELEOP = ()
    TELEOP_DEFAULT = "none"
    ACTION_WRAPPERS = ()
    TRANSFORMS = ()

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Optional[WorkerInfo] = None,
        robot_info: Optional[RobotInfo[H01Config]] = None,
        env_idx: int = 0,
    ) -> None:
        self.config = self.CONFIG_CLS(**override_cfg)
        self._logger = get_logger()
        self.hardware = get_hardware_config(
            H01Config, robot_info, is_dummy=self.config.is_dummy
        )
        self.robot_info = robot_info
        self._placed = worker_info is not None
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        self._num_steps = 0
        self._success_hold_counter = 0
        self._joints = np.zeros(BOTH_ARMS_DOF)
        self.robot: Optional[H01Robot] = None

        if not self.config.is_dummy:
            self._setup_hardware()
            self._reset_joints = self._resolve_reset_pose()
        else:
            self._reset_joints = np.asarray(
                self.config.reset_joint_qpos or np.zeros(BOTH_ARMS_DOF), dtype=float
            )
        target = self.config.target_joint_qpos
        self._target_joints = (
            self._reset_joints.copy()
            if target is None
            else np.asarray(target, dtype=float)
        )
        self._init_spaces()

    # Hardware setup.

    def _setup_hardware(self) -> None:
        """Compose and connect the robot through its MCP server."""
        hardware = self.hardware
        if not hardware.endpoint:
            raise ValueError(
                "H01 needs the MCP endpoint of its server; set "
                "cluster.node_groups[].hardware.configs[].endpoint."
            )
        # Without a scheduler worker there is no node to place onto, so the
        # connection opens in this process; a bench script needs no cluster.
        controller_rank = hardware.controller_node_rank
        if controller_rank is None:
            controller_rank = hardware.node_rank if self._placed else None
        self.robot = H01Robot.build(
            endpoint=hardware.endpoint,
            node_rank=controller_rank,
            camera=hardware.camera,
            camera_width=hardware.camera_width,
            camera_quality=hardware.camera_quality,
            max_step_delta=hardware.max_step_delta,
            worker_rank=self.env_worker_rank,
            env_idx=self.env_idx,
        )
        self.robot.connect()

    def _resolve_reset_pose(self) -> np.ndarray:
        """Return the configured rest pose, or the pose the arms hold now."""
        if self.config.reset_joint_qpos is not None:
            return np.asarray(self.config.reset_joint_qpos, dtype=float)
        joints = self._read_joints()
        self._logger.info(
            "H01 has no configured rest pose; holding the measured pose %s.",
            np.round(joints, 3).tolist(),
        )
        return joints

    def _read_joints(self) -> np.ndarray:
        """Read both arms as one vector, in the action order."""
        assert self.robot is not None
        reading = self.robot.get_observation()
        return np.concatenate(
            [
                np.asarray(reading[side]["arm"]["arm_joint_position"], dtype=float)
                for side in SIDES
            ]
        )

    def _init_spaces(self) -> None:
        """Build the action and observation spaces from the declared layout."""
        from rlinf.robotics.parts.h01 import (
            ARM_LIMITS,
            GRIPPER_LIMITS,
            HEAD_LIMITS,
            WAIST_LIMITS,
        )

        low: list[float] = []
        high: list[float] = []
        for side in SIDES:
            low += ARM_LIMITS[side][:, 0].tolist()
            high += ARM_LIMITS[side][:, 1].tolist()
            low.append(GRIPPER_LIMITS[0])
            high.append(GRIPPER_LIMITS[1])
        if self.config.control_head:
            low += HEAD_LIMITS[:, 0].tolist()
            high += HEAD_LIMITS[:, 1].tolist()
        if self.config.control_waist:
            low += WAIST_LIMITS[:, 0].tolist()
            high += WAIST_LIMITS[:, 1].tolist()
        self.action_space = gym.spaces.Box(
            np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)
        )

        state = {
            "arm_joint_position": gym.spaces.Box(
                -np.inf, np.inf, shape=(BOTH_ARMS_DOF,), dtype=np.float64
            ),
            "gripper_position": gym.spaces.Box(
                0.0, 1.0, shape=(len(SIDES),), dtype=np.float64
            ),
            "head_joint_position": gym.spaces.Box(
                -np.inf, np.inf, shape=(2,), dtype=np.float64
            ),
            "waist_joint_position": gym.spaces.Box(
                -np.inf, np.inf, shape=(4,), dtype=np.float64
            ),
        }
        spaces: dict[str, gym.Space] = {"state": gym.spaces.Dict(state)}
        if self.hardware.camera:
            spaces["frames"] = gym.spaces.Dict(
                {
                    self.hardware.camera: gym.spaces.Box(
                        0,
                        255,
                        shape=(self.config.frame_size, self.config.frame_size, 3),
                        dtype=np.uint8,
                    )
                }
            )
        self.observation_space = gym.spaces.Dict(spaces)
        self._base_observation_space = copy.deepcopy(self.observation_space)

    # Gymnasium API.

    def action_parts(self) -> tuple[ActionPart, ...]:
        """Return the action layout, in the order :pyattr:`action_space` uses."""
        parts = [
            ActionPart(f"{side}.{name}", width, kind)
            for side in SIDES
            for name, width, kind in (
                ("arm", ARM_DOF, ActionKind.JOINT_POSITION),
                ("end_effector", 1, ActionKind.GRIPPER),
            )
        ]
        if self.config.control_head:
            parts.append(ActionPart("head", 2, ActionKind.JOINT_POSITION))
        if self.config.control_waist:
            parts.append(ActionPart("waist", 4, ActionKind.JOINT_POSITION))
        return tuple(parts)

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Apply one joint-space action and return the next observation."""
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if not self.config.is_dummy:
            self.robot.send_action(self._split_action(action))

        self._num_steps += 1
        observation = self._get_observation()
        # Read first, then wait out the rest of the period: the observation is
        # part of the step, and pacing from the command alone runs slow.
        time.sleep(
            max(0.0, (1.0 / self.config.step_frequency) - (time.time() - start_time))
        )
        reward = self._calc_step_reward(observation)
        terminated = (
            reward >= 1.0
            and self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, reward, terminated, truncated, {}

    def _split_action(self, action: np.ndarray) -> dict[str, Any]:
        """Turn the flat action into the robot tree's nested command."""
        assert self.robot is not None
        offset = 0
        command: dict[str, Any] = {}
        for side in SIDES:
            joints = action[offset : offset + ARM_DOF]
            gripper = float(action[offset + ARM_DOF])
            offset += ARM_DOF + 1
            command[side] = {
                "arm": {"joint_position": joints},
                "end_effector": {"target": np.array([gripper], dtype=np.float32)},
            }
        if self.config.control_head:
            command["head"] = {
                "joint_position": np.asarray(action[offset : offset + 2])
            }
            offset += 2
        if self.config.control_waist:
            command["waist"] = {
                "joint_position": np.asarray(action[offset : offset + 4])
            }
        return command

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Hold the rest configuration and return the first observation."""
        seed_sampled_spaces(seed, self._base_observation_space)
        self._num_steps = 0
        self._success_hold_counter = 0
        if self.config.is_dummy:
            return self._get_observation(), {}
        self.go_to_rest()
        return self._get_observation(), {}

    def go_to_rest(self) -> None:
        """Command the rest pose and wait until the arms arrive or time out."""
        assert self.robot is not None
        deadline = time.monotonic() + self.config.reset_timeout_s
        period = 1.0 / self.config.step_frequency
        while time.monotonic() < deadline:
            self.robot.send_action(
                {
                    side: {
                        "arm": {
                            "joint_position": self._reset_joints[
                                index * ARM_DOF : (index + 1) * ARM_DOF
                            ]
                        }
                    }
                    for index, side in enumerate(SIDES)
                }
            )
            if np.all(
                np.abs(self._read_joints() - self._reset_joints)
                < self.config.reset_tolerance
            ):
                return
            time.sleep(period)
        self._logger.warning(
            "H01 did not reach its rest pose within %.1f s; continuing from where "
            "the arms are.",
            self.config.reset_timeout_s,
        )

    def close(self) -> None:
        """Disconnect the robot, which releases the command stream."""
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None

    # Reward.

    def _calc_step_reward(self, observation: dict[str, Any]) -> float:
        """Score the joint distance to the target configuration."""
        measured = np.asarray(observation["state"]["arm_joint_position"], dtype=float)
        distance = np.abs(measured - self._target_joints)
        if self.config.use_dense_reward:
            return float(-np.linalg.norm(distance))
        hit = bool(np.all(distance < self.config.reward_threshold))
        self._success_hold_counter = self._success_hold_counter + 1 if hit else 0
        return 1.0 if hit else 0.0

    # Observation.

    def _get_observation(self) -> dict[str, Any]:
        """Return joint state, gripper openings, and the head frame."""
        if self.config.is_dummy:
            return self._base_observation_space.sample()

        assert self.robot is not None
        reading = self.robot.get_observation()
        self._joints = np.concatenate(
            [
                np.asarray(reading[side]["arm"]["arm_joint_position"], dtype=float)
                for side in SIDES
            ]
        )
        state = {
            "arm_joint_position": self._joints.copy(),
            "gripper_position": np.asarray(
                [reading[side]["end_effector"]["state"][0] for side in SIDES],
                dtype=float,
            ),
            "head_joint_position": np.asarray(
                reading["head"]["joint_position"], dtype=float
            ),
            "waist_joint_position": np.asarray(
                reading["waist"]["joint_position"], dtype=float
            ),
        }
        observation: dict[str, Any] = {"state": state}
        if "frames" in self.observation_space.spaces:
            observation["frames"] = {
                self.hardware.camera: self._resize_frame(
                    reading[f"camera_{self.hardware.camera}"]["frame"]
                )
            }
        return copy.deepcopy(observation)

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Centre-crop the camera frame to a square and resize it."""
        import cv2

        size = (self.config.frame_size, self.config.frame_size)
        height, width, _ = frame.shape
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        return cv2.resize(frame[top : top + side, left : left + side], size)
