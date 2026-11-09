#!/bin/bash
# Restart the H01 MCP server. Kill by listening port: a command-line pattern
# would also match this script, whose text mentions server.py.
PORT="${PORT:-8100}"
if command -v fuser >/dev/null 2>&1; then
    fuser -k "${PORT}/tcp" 2>/dev/null
fi
sleep 2
cd "$HOME/h01_mcp" || exit 1
source /opt/ros/humble/setup.bash
setsid nohup .venv/bin/python server.py --host 0.0.0.0 --port "${PORT}" \
    >> "$HOME/h01_mcp/server.log" 2>&1 < /dev/null &
sleep 5
ss -lntp 2>/dev/null | grep ":${PORT}" || echo "NOT LISTENING"
