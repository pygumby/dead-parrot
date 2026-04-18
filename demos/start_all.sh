#!/bin/bash

set -e

PIDS=()

section() {
    local bar
    bar=$(printf '=%.0s' {1..88})
    echo ""
    echo "$bar"
    echo "$1"
    echo "$bar"
}

subsection() {
    local bar
    bar=$(printf -- '-%.0s' {1..88})
    echo ""
    echo "$bar"
    echo "$1"
    echo "$bar"
    echo ""
}


cleanup() {
    section "Shutting down all processes"
    echo ""
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    section "All processes stopped"
    echo ""
    exit 0
}

trap cleanup SIGINT SIGTERM

# Setting up environment
section "Setting up environment"

subsection "Installing dependencies"
uv sync --all-packages

subsection "Starting Temporal Service"
temporal server start-dev &
PIDS+=($!)
sleep 3

# Starting ecb_bs_ai_assistant
section "Starting ecb_bs_ai_assistant"

subsection "Starting ecb_bs_ai_assistant - Temporal worker"
uv run --directory demos/ecb_bs_ai_assistant python -m ecb_bs_ai_assistant.temporal_worker &
PIDS+=($!)
sleep 3

subsection "Starting ecb_bs_ai_assistant - REST server"
uv run --directory demos/ecb_bs_ai_assistant python -m ecb_bs_ai_assistant.rest_server &
PIDS+=($!)
sleep 3

subsection "Starting ecb_bs_ai_assistant - MCP server"
uv run --directory demos/ecb_bs_ai_assistant python -m ecb_bs_ai_assistant.mcp_server &
PIDS+=($!)
sleep 3

# Starting ecb_hr_ai_assistant
section "Starting ecb_hr_ai_assistant"

subsection "Starting ecb_hr_ai_assistant - Temporal worker"
uv run --directory demos/ecb_hr_ai_assistant python -m ecb_hr_ai_assistant.temporal_worker &
PIDS+=($!)
sleep 3

subsection "Starting ecb_hr_ai_assistant - REST server"
uv run --directory demos/ecb_hr_ai_assistant python -m ecb_hr_ai_assistant.rest_server &
PIDS+=($!)
sleep 3

subsection "Starting ecb_hr_ai_assistant - MCP server"
uv run --directory demos/ecb_hr_ai_assistant python -m ecb_hr_ai_assistant.mcp_server &
PIDS+=($!)
sleep 3

# Starting ecb_ai_agent
section "Starting ecb_ai_agent"

subsection "Starting ecb_ai_agent - Temporal worker"
uv run --directory demos/ecb_ai_agent python -m ecb_ai_agent.temporal_worker &
PIDS+=($!)
sleep 3

subsection "Starting ecb_ai_agent - REST server"
uv run --directory demos/ecb_ai_agent python -m ecb_ai_agent.rest_server &
PIDS+=($!)
sleep 3

subsection "Starting ecb_ai_agent - MCP server"
uv run --directory demos/ecb_ai_agent python -m ecb_ai_agent.mcp_server &
PIDS+=($!)
sleep 3

# Summary
section "${#PIDS[@]} processes started"
echo ""

wait
