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

"""H01 upper body backed by the robot's own MCP server.

The vendor exposes ROS 2 topics rather than a Python SDK, so a small server on
the robot wraps them and this connection speaks Model Context Protocol to it
over streamable HTTP. MCP is transport only: the part tree, the observation
contract, and the placement are the same as for any other robot, and nothing
above this module knows the difference.

The tree is the upper body the vendor drives: ``left``/``right`` groups holding
a seven-joint arm and its gripper, ``head`` with its two joints, and ``waist``
with its four. The chassis is not part of it.

One ``h01_step`` call carries every staged target and returns the whole body
state, so an environment step costs one round trip rather than one per part.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
from typing import Any, Optional

import numpy as np

from rlinf.robotics.parts.base import Connection, PartGroup
from rlinf.robotics.parts.views import (
    MethodArm,
    MethodCamera,
    MethodEndEffector,
    MethodJointGroup,
)
from rlinf.utils.logging import get_logger

#: Arm travel in radians, from the vendor's ``JointLimits``.
ARM_LIMITS: dict[str, np.ndarray] = {
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

#: Head travel in radians, ``[yaw, pitch]``.
HEAD_LIMITS = np.array([[-0.524, 0.524], [-0.349, 0.349]], dtype=float)

#: Waist travel in radians, four joints.
WAIST_LIMITS = np.array(
    [[-0.995, 0.995], [-1.24, 0.0], [0.0, 2.08], [0.0, 1.34]], dtype=float
)

#: Gripper opening in metres.
GRIPPER_LIMITS = (0.0, 0.06)

#: Joints per arm on a Maker H01.
ARM_DOF = 7

#: State topic behind each group, named in the error a missing group raises.
STATE_TOPICS = {
    "left_arm": "/left_arm/joint_state",
    "right_arm": "/right_arm/joint_state",
    "head": "/head/joint_state",
    "waist": "/waist/joint_state",
}


class H01Arm(MethodArm):
    """One seven-joint H01 arm viewed through the shared MCP connection."""

    @property
    def observation_features(self) -> dict[str, Any]:
        """Describe the joint state this arm reports."""
        return {
            "arm_joint_position": {"shape": (ARM_DOF,), "dtype": "float32"},
            "arm_joint_velocity": {"shape": (ARM_DOF,), "dtype": "float32"},
        }


class H01MCPConnection(Connection):
    """One MCP session backing the whole H01 upper body.

    Args:
        endpoint: URL of the robot's MCP server, ending in ``/mcp``.
        camera: Camera name the server exposes, or ``None`` for no camera.
        camera_width: Frame width requested from the server, in pixels.
        camera_quality: JPEG quality requested from the server.
        request_timeout: Seconds to wait for one tool call.
        connect_timeout: Seconds to wait for the session handshake.
        max_step_delta: Largest joint change, in radians, one step may command.
            ``None`` passes targets through unchanged.
    """

    #: A second read inside this window reuses the snapshot instead of asking
    #: again, so one environment step costs one round trip.
    CACHE_WINDOW_S = 0.02

    def __init__(
        self,
        endpoint: str,
        *,
        camera: Optional[str] = "head",
        camera_width: int = 320,
        camera_quality: int = 75,
        request_timeout: float = 10.0,
        connect_timeout: float = 15.0,
        max_step_delta: Optional[float] = 0.1,
    ) -> None:
        self._logger = get_logger()
        self._endpoint = endpoint
        self._camera = camera
        self._camera_width = camera_width
        self._camera_quality = camera_quality
        self._request_timeout = request_timeout
        self._connect_timeout = connect_timeout
        self._max_step_delta = max_step_delta

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Any = None
        self._stop: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._error: Optional[BaseException] = None
        self._call_lock = threading.Lock()

        self._state: dict[str, Any] = {}
        self._read_stamp = 0.0
        self._pending: dict[str, Any] = {}
        self._frame: Optional[np.ndarray] = None

    # -- composition -------------------------------------------------------

    @property
    def parts(self) -> dict[str, Any]:
        """Decompose the session into the upper body's addressable parts."""
        parts: dict[str, Any] = {
            "left": PartGroup(
                arm=H01Arm(
                    self,
                    commands={"joint_position": "move_left_arm"},
                    state_fields={
                        "arm_joint_position": "left_arm_joints",
                        "arm_joint_velocity": "left_arm_velocity",
                    },
                ),
                end_effector=MethodEndEffector(
                    self,
                    state_field="left_gripper_position",
                    command="set_left_gripper",
                    dims=1,
                    is_gripper=True,
                ),
            ),
            "right": PartGroup(
                arm=H01Arm(
                    self,
                    commands={"joint_position": "move_right_arm"},
                    state_fields={
                        "arm_joint_position": "right_arm_joints",
                        "arm_joint_velocity": "right_arm_velocity",
                    },
                ),
                end_effector=MethodEndEffector(
                    self,
                    state_field="right_gripper_position",
                    command="set_right_gripper",
                    dims=1,
                    is_gripper=True,
                ),
            ),
            "head": MethodJointGroup(self, "head", HEAD_LIMITS),
            "waist": MethodJointGroup(self, "waist", WAIST_LIMITS),
        }
        if self._camera:
            parts[f"camera_{self._camera}"] = MethodCamera(
                self, "get_camera", self._camera, check_method="check_cameras"
            )
        return parts

    # -- lifecycle ---------------------------------------------------------

    def _open(self) -> Any:
        """Start the session thread and wait for the MCP handshake."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="h01-mcp", daemon=True
        )
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._session_lifetime(), self._loop)
        if not self._ready.wait(self._connect_timeout):
            raise TimeoutError(
                f"No MCP session with {self._endpoint} within "
                f"{self._connect_timeout} s."
            )
        if self._error is not None:
            raise RuntimeError(
                f"Connecting to the H01 MCP server at {self._endpoint} failed."
            ) from self._error
        return self._session

    def _release(self, device: Any) -> None:
        """Stop the command stream, then close the session."""
        try:
            self._call("h01_stop", {})
            self._call("h01_release", {})
        except Exception as error:  # noqa: BLE001 - teardown reports and continues
            self._logger.warning("h01: teardown command failed: %s", error)
        if self._loop is None:
            return
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._closed.wait(5.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._session = None

    async def _session_lifetime(self) -> None:
        """Hold the session open for as long as the connection is connected."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        def client_factory(
            headers: Any = None, timeout: Any = None, auth: Any = None
        ) -> Any:
            # Ignore proxy settings: the robot sits on the lab network, and an
            # environment-level proxy would silently intercept the session.
            import httpx

            return httpx.AsyncClient(
                headers=headers, timeout=timeout, auth=auth, trust_env=False
            )

        try:
            async with streamablehttp_client(
                self._endpoint,
                timeout=self._request_timeout,
                sse_read_timeout=max(60.0, self._request_timeout * 6),
                httpx_client_factory=client_factory,
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._stop = asyncio.Event()
                    self._ready.set()
                    await self._stop.wait()
        except BaseException as error:  # noqa: BLE001 - reported by _open
            self._error = error
            self._ready.set()
        finally:
            self._session = None
            self._closed.set()

    # -- tool calls --------------------------------------------------------

    def _call(self, tool: str, arguments: Optional[dict[str, Any]] = None) -> Any:
        """Run one MCP tool call and return its JSON result."""
        session = self._session
        loop = self._loop
        if session is None or loop is None:
            raise RuntimeError(f"The H01 MCP session is closed; cannot call {tool}.")
        with self._call_lock:
            future = asyncio.run_coroutine_threadsafe(
                session.call_tool(tool, arguments or {}), loop
            )
            result = future.result(self._request_timeout)
        if result.isError:
            text = " ".join(getattr(item, "text", "") for item in result.content)
            raise RuntimeError(f"H01 tool {tool} failed: {text.strip()}")
        return json.loads(result.content[0].text)

    # -- device surface ----------------------------------------------------

    def read_state(self) -> dict[str, Any]:
        """Return the body state, flushing staged targets in the same call."""
        pending, self._pending = self._pending, {}
        if not pending and (time.monotonic() - self._read_stamp) < self.CACHE_WINDOW_S:
            return self._state

        if pending or self._camera:
            arguments: dict[str, Any] = dict(pending)
            if self._camera:
                arguments["frames"] = 1
                arguments["width"] = self._camera_width
                arguments["quality"] = self._camera_quality
            reply = self._call("h01_step", arguments)
            state = reply["state"]
            self._store_frame(reply.get("frames", {}))
        else:
            state = self._call("h01_get_state", {})

        try:
            self._state = self._flatten(state)
        except RuntimeError:
            # A server that just started may not have received every group
            # yet; one retry covers the race without hiding a dead topic.
            time.sleep(0.05)
            state = self._call("h01_get_state", {})
            self._state = self._flatten(state)
        self._read_stamp = time.monotonic()
        return self._state

    def move_left_arm(self, target: np.ndarray) -> np.ndarray:
        """Stage absolute left-arm joint targets in radians."""
        return self._stage("left_arm", target, ARM_LIMITS["left"])

    def move_right_arm(self, target: np.ndarray) -> np.ndarray:
        """Stage absolute right-arm joint targets in radians."""
        return self._stage("right_arm", target, ARM_LIMITS["right"])

    def move_head(self, target: np.ndarray) -> np.ndarray:
        """Stage absolute head joint targets in radians."""
        return self._stage("head", target, HEAD_LIMITS)

    def move_waist(self, target: np.ndarray) -> np.ndarray:
        """Stage absolute waist joint targets in radians."""
        return self._stage("waist", target, WAIST_LIMITS)

    def set_left_gripper(self, target: np.ndarray) -> float:
        """Stage a left gripper opening in metres."""
        return self._stage_gripper("left", target)

    def set_right_gripper(self, target: np.ndarray) -> float:
        """Stage a right gripper opening in metres."""
        return self._stage_gripper("right", target)

    def get_camera(self, name: str = "head") -> np.ndarray:
        """Return the last frame the server sent for ``name``."""
        if name != self._camera:
            raise KeyError(f"No camera {name!r}; this connection has {self._camera!r}.")
        if self._frame is None:
            reply = self._call("h01_get_frame", {"width": self._camera_width})
            self._store_frame({"head": reply})
        if self._frame is None:
            raise RuntimeError("The H01 MCP server returned no camera frame.")
        return self._frame.copy()

    def check_cameras(self, timeout: float = 0.5) -> dict[str, bool]:
        """Report whether the camera produced a frame."""
        if self._frame is None:
            try:
                self.get_camera(self._camera)
            except Exception:  # noqa: BLE001 - a health check reports, not raises
                return {self._camera: False}
        return {self._camera: self._frame is not None}

    def stop(self) -> None:
        """Hold every upper-body group where it is."""
        reply = self._call("h01_stop", {})
        self._state = self._flatten(reply)
        self._read_stamp = time.monotonic()

    # -- internals ---------------------------------------------------------

    def _stage(self, group: str, target: np.ndarray, limits: np.ndarray) -> np.ndarray:
        values = np.asarray(target, dtype=float).reshape(-1)
        if values.shape != (limits.shape[0],):
            raise ValueError(
                f"Expected {limits.shape[0]} joint targets for {group}, got "
                f"{values.shape}."
            )
        values = np.clip(values, limits[:, 0], limits[:, 1])
        measured = self._state.get(f"{group}_joints")
        if self._max_step_delta is not None and measured is not None:
            values = measured + np.clip(
                values - measured, -self._max_step_delta, self._max_step_delta
            )
        self._pending[group] = values.tolist()
        return values

    def _stage_gripper(self, side: str, target: np.ndarray) -> float:
        value = float(
            np.clip(float(np.asarray(target).reshape(-1)[0]), *GRIPPER_LIMITS)
        )
        self._pending[f"{side}_gripper"] = value
        return value

    def _store_frame(self, frames: dict[str, Any]) -> None:
        """Decode the JPEG the server sent, if any."""
        payload = frames.get(self._camera or "")
        if not payload:
            return
        import cv2

        buffer = np.frombuffer(base64.b64decode(payload["base64"]), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("The H01 MCP server sent a frame that is not a JPEG.")
        self._frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _flatten(self, state: dict[str, Any]) -> dict[str, Any]:
        """Turn the server's nested state into the names the views select."""
        flat: dict[str, Any] = {"stamp": float(state.get("stamp", time.time()))}
        for group, dof in (
            ("left_arm", ARM_DOF),
            ("right_arm", ARM_DOF),
            ("head", 2),
            ("waist", 4),
        ):
            joints = state[group]["position"]
            if joints is None:
                raise RuntimeError(
                    f"The H01 MCP server has no {group} state yet; is "
                    f"{STATE_TOPICS[group]} publishing?"
                )
            flat[f"{group}_joints"] = np.asarray(joints, dtype=np.float32)
            if group.endswith("_arm"):
                velocity = state[group].get("velocity")
                flat[f"{group}_velocity"] = np.asarray(
                    velocity if velocity is not None else np.zeros(dof),
                    dtype=np.float32,
                )
        for side in ("left", "right"):
            flat[f"{side}_gripper_position"] = np.asarray(
                [state[f"{side}_gripper"]["position"]], dtype=np.float32
            )
        return flat
