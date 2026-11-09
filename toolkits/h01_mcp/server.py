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
"""MCP server for the GigaAI Maker H01 upper body.

Runs on the robot next to the ROS 2 driver stack and exposes both arms, both
grippers, the head, the waist, and the head camera over Model Context Protocol.
The chassis is not part of this server and is never commanded by it.

One ``h01_step`` call is one environment step: it applies every target and
returns the fresh state, so the training side pays a single round trip.

    python3 server.py --host 0.0.0.0 --port 8100
    python3 server.py --stdio            # local smoke test
"""

import argparse
import base64
import threading
import time
from typing import Any, Optional

import cv2
import rclpy
from h01_bridge import (
    ARM_LIMITS,
    GRIPPER_LIMITS,
    HEAD_LIMITS,
    WAIST_LIMITS,
    H01Bridge,
)
from mcp.server.fastmcp import FastMCP

BRIDGE: Optional[H01Bridge] = None


def bridge() -> H01Bridge:
    """Return the live bridge, or fail loudly."""
    if BRIDGE is None:
        raise RuntimeError("The ROS bridge is not running.")
    return BRIDGE


def _encode(frame: Any, width: int, quality: int) -> dict[str, Any]:
    """Return a JPEG frame as JSON-ready data."""
    image = frame
    if width and image.shape[1] != width:
        scale = width / image.shape[1]
        image = cv2.resize(
            image,
            (width, int(round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed.")
    return {
        "format": "jpeg",
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "base64": base64.b64encode(buffer.tobytes()).decode("ascii"),
    }


def _frames(width: int, quality: int, wait: bool) -> dict[str, Any]:
    """Capture the head camera, optionally waiting for the next frame."""
    seq, frame = bridge().frame()
    if wait and frame is not None:
        deadline = time.monotonic() + 0.05
        while time.monotonic() < deadline:
            newer, candidate = bridge().frame()
            if newer != seq:
                frame = candidate
                break
            time.sleep(0.004)
    if frame is None:
        return {}
    return {"head": _encode(frame, width, quality)}


def build(name: str, host: str, port: int) -> FastMCP:
    """Create the MCP server and register the H01 tools."""
    mcp = FastMCP(name, host=host, port=port)

    @mcp.tool()
    def h01_describe() -> dict[str, Any]:
        """Describe the upper-body groups, limits, and control rates."""
        return {
            "robot": "GigaAI Maker H01 upper body",
            "joints": {
                "left_arm": [
                    "L_SHOULDER_P",
                    "L_SHOULDER_R",
                    "L_SHOULDER_Y",
                    "L_ELBOW_Y",
                    "L_WRIST_P",
                    "L_WRIST_Y",
                    "L_WRIST_R",
                ],
                "right_arm": [
                    "R_SHOULDER_P",
                    "R_SHOULDER_R",
                    "R_SHOULDER_Y",
                    "R_ELBOW_Y",
                    "R_WRIST_P",
                    "R_WRIST_Y",
                    "R_WRIST_R",
                ],
                "head": ["HEAD_YAW", "HEAD_PITCH"],
                "waist": ["WAIST_1", "WAIST_2", "WAIST_3", "WAIST_4"],
            },
            "limits_rad": {
                "left_arm": ARM_LIMITS["left"].tolist(),
                "right_arm": ARM_LIMITS["right"].tolist(),
                "head": HEAD_LIMITS.tolist(),
                "waist": WAIST_LIMITS.tolist(),
            },
            "gripper_limits_m": list(GRIPPER_LIMITS),
            "camera": {"head": "realsense 640x480 rgb, 30 Hz, jpeg over MCP"},
            "command_rate_hz": 15.0,
            "excluded": "chassis; this server never publishes to it",
            "note": "Targets are held and republished at 15 Hz; the motor driver "
            "stops a joint 0.2 s after its last command.",
        }

    @mcp.tool()
    def h01_get_state() -> dict[str, Any]:
        """Return both arms, both grippers, the head, and the waist."""
        return bridge().snapshot()

    @mcp.tool()
    def h01_step(
        left_arm: Optional[list[float]] = None,
        right_arm: Optional[list[float]] = None,
        left_gripper: Optional[float] = None,
        right_gripper: Optional[float] = None,
        head: Optional[list[float]] = None,
        waist: Optional[list[float]] = None,
        max_delta_rad: float = 0.1,
        frames: int = 0,
        width: int = 320,
        quality: int = 75,
    ) -> dict[str, Any]:
        """Apply one action and return the resulting state.

        ``max_delta_rad`` caps how far each joint may move in this step, which
        keeps a freshly initialised policy from snapping the arms.
        """
        applied = bridge().command(
            left_arm=left_arm,
            right_arm=right_arm,
            left_gripper=left_gripper,
            right_gripper=right_gripper,
            head=head,
            waist=waist,
            max_delta_rad=max_delta_rad,
        )
        result: dict[str, Any] = {"applied": applied, "state": bridge().snapshot()}
        if frames:
            result["frames"] = _frames(width, quality, wait=True)
        return result

    @mcp.tool()
    def h01_get_frame(width: int = 320, quality: int = 75) -> dict[str, Any]:
        """Return the current head camera frame as JPEG."""
        frames = _frames(width, quality, wait=False)
        if not frames:
            return {"error": "no frame received from /camera_head/color/image"}
        return frames["head"]

    @mcp.tool()
    def h01_stop() -> dict[str, Any]:
        """Hold every upper-body group at its measured position."""
        bridge().stop()
        return bridge().snapshot()

    @mcp.tool()
    def h01_release() -> dict[str, Any]:
        """Stop the command stream; the motors keep their own watchdog."""
        bridge().release()
        return {"released": True, "engaged": False}

    return mcp


def main() -> int:
    global BRIDGE
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument(
        "--transport",
        default="streamable-http",
        choices=("streamable-http", "stdio"),
    )
    parser.add_argument("--name", default="h01")
    args = parser.parse_args()

    rclpy.init()
    BRIDGE = H01Bridge()
    threading.Thread(target=rclpy.spin, args=(BRIDGE,), daemon=True).start()
    server = build(args.name, args.host, args.port)
    try:
        server.run(transport=args.transport)
    finally:
        BRIDGE.close()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
