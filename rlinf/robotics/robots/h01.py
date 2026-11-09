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

"""GigaAI Maker H01: a wheeled humanoid behind its own MCP server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rlinf.robotics.discovery import RobotConfig
from rlinf.robotics.parts.h01 import H01MCPConnection
from rlinf.robotics.robot import Robot


class H01Robot(Robot):
    """Composable Maker H01 driven over Model Context Protocol.

    The tree is the vendor's upper body: a ``left``/``right`` group holding a
    seven-joint arm and its gripper, a two-joint ``head``, a four-joint
    ``waist``, and a camera per configured name. The chassis is not included.
    """

    ROBOT_TYPE = "H01"

    @classmethod
    def build(
        cls,
        *,
        endpoint: str,
        node_rank: int,
        worker_rank: int = 0,
        env_idx: int = 0,
        camera: Optional[str] = "head",
        camera_width: int = 320,
        camera_quality: int = 75,
        max_step_delta: Optional[float] = 0.1,
    ) -> H01Robot:
        """Compose an unconnected H01 from its MCP endpoint."""
        if not endpoint:
            raise ValueError("H01 requires the MCP endpoint of its server.")
        connection = H01MCPConnection(
            endpoint,
            camera=camera,
            camera_width=camera_width,
            camera_quality=camera_quality,
            max_step_delta=max_step_delta,
            node_rank=node_rank,
            worker_name=f"H01MCP-{worker_rank}-{env_idx}",
        )
        return cls(**connection.parts)


@dataclass
class H01Config(RobotConfig):
    """Hardware and placement configuration for a Maker H01."""

    endpoint: str = ""
    """URL of the robot's MCP server, ending in ``/mcp``.

    Unset values come from the ``ENDPOINT`` environment variable on the node
    that enumerates the robot."""

    camera: Optional[str] = "head"
    """Camera the server exposes, or ``None`` for a body without one."""

    camera_width: int = 320
    """Frame width requested from the server, in pixels."""

    camera_quality: int = 75
    """JPEG quality requested from the server."""

    max_step_delta: Optional[float] = 0.1
    """Largest joint change, in radians, that one step may command."""

    controller_node_rank: Optional[int] = None
    """Node that opens the MCP session, when it differs from the robot record."""

    def hardware_model(self, robot_type: str) -> str:
        """Report the body rather than the deployment as its model."""
        return f"{robot_type}-Maker"


H01Robot.register_type(H01Config)
