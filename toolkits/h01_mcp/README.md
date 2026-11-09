# H01 upper-body MCP server

The GigaAI Maker H01 ships ROS 2 topics, not a Python SDK. This server runs on
the robot, wraps those topics, and exposes the **upper body** over Model Context
Protocol so a training host can drive it with one transport and no ROS in the
loop. The chassis is out of scope: this process never publishes to it.

RLinf talks to it through `rlinf/robotics/parts/h01.py`, which is
transport-agnostic: MCP is only how the connection opens, so the part tree, the
observation contract, and the placement are the same as for any other robot.

## The part tree it serves

| Path | Joints | Command |
|---|---|---|
| `left.arm` | `L_SHOULDER_P/R/Y`, `L_ELBOW_Y`, `L_WRIST_P/Y/R` | absolute radians |
| `left.end_effector` | `left_gripper` | opening in metres, `0 .. 0.06` |
| `right.arm` | `R_SHOULDER_P/R/Y`, `R_ELBOW_Y`, `R_WRIST_P/Y/R` | absolute radians |
| `right.end_effector` | `right_gripper` | opening in metres, `0 .. 0.06` |
| `head` | `HEAD_YAW`, `HEAD_PITCH` | absolute radians |
| `waist` | `WAIST_1 .. WAIST_4` | absolute radians |
| `camera_head` | RealSense D435 colour | read-only |

## Deploy

The robot needs `mcp` in a virtual environment that can also see ROS 2:

```bash
mkdir -p ~/h01_mcp && cd ~/h01_mcp
python3 -m venv --system-site-packages .venv   # rclpy, cv2, numpy come from the system
.venv/bin/pip install "mcp<2"                  # FastMCP v1 lives in 1.x
```

Copy `server.py`, `h01_bridge.py`, and `restart.sh` into `~/h01_mcp`, then:

```bash
bash ~/h01_mcp/restart.sh    # source ROS, start detached, print the listener
```

The server listens on `0.0.0.0:8100` at path `/mcp` by default.

## Tools

| Tool | Purpose |
|---|---|
| `h01_describe` | Joint names, limits, command rate, camera, and what is excluded |
| `h01_get_state` | Both arms, both grippers, head, waist |
| `h01_step` | Apply one action and return the fresh state, with an optional JPEG frame |
| `h01_get_frame` | Head camera frame as base64 JPEG |
| `h01_stop` | Hold every upper-body group where it is |
| `h01_release` | End the command stream |

`h01_step` is the one an environment uses: it carries every staged target in a
single round trip and returns the state that resulted.

## Why the bridge holds a command stream

The motor driver drops a joint command 0.2 s after the last one
(`command_timeout_s` in `left_arm_motors.yaml`), so once a client takes control
the bridge republishes the last targets at 15 Hz. A wedged client therefore
leaves the groups holding their pose rather than going idle mid-motion.

## Safety limits

- Targets are clipped to the vendor's travel, and `max_delta_rad` (default
  0.1 rad) caps how far one step may move a joint, so a randomly initialised
  policy cannot snap the arms, the head, or the waist.
- `h01_stop` is what teardown calls: every commanded group is held in place.
- Nothing is published before a client sends a target, so a freshly started
  server leaves the robot exactly as it found it.
