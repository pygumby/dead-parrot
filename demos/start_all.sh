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

# Starting ecb_bs_expert_agent
section "Starting ecb_bs_expert_agent"

subsection "Starting ecb_bs_expert_agent - Temporal worker"
uv run --directory demos/ecb_bs_expert_agent python -m ecb_bs_expert_agent.temporal_worker &
PIDS+=($!)
sleep 3

subsection "Starting ecb_bs_expert_agent - REST server"
uv run --directory demos/ecb_bs_expert_agent python -m ecb_bs_expert_agent.rest_server &
PIDS+=($!)
sleep 3

subsection "Starting ecb_bs_expert_agent - MCP server"
uv run --directory demos/ecb_bs_expert_agent python -m ecb_bs_expert_agent.mcp_server &
PIDS+=($!)
sleep 3

# Starting ecb_hr_expert_agent
section "Starting ecb_hr_expert_agent"

subsection "Starting ecb_hr_expert_agent - Temporal worker"
uv run --directory demos/ecb_hr_expert_agent python -m ecb_hr_expert_agent.temporal_worker &
PIDS+=($!)
sleep 3

subsection "Starting ecb_hr_expert_agent - REST server"
uv run --directory demos/ecb_hr_expert_agent python -m ecb_hr_expert_agent.rest_server &
PIDS+=($!)
sleep 3

subsection "Starting ecb_hr_expert_agent - MCP server"
uv run --directory demos/ecb_hr_expert_agent python -m ecb_hr_expert_agent.mcp_server &
PIDS+=($!)
sleep 3

# Starting ecb_triage_agent
section "Starting ecb_triage_agent"

subsection "Starting ecb_triage_agent - Temporal worker"
uv run --directory demos/ecb_triage_agent python -m ecb_triage_agent.temporal_worker &
PIDS+=($!)
sleep 3

subsection "Starting ecb_triage_agent - REST server"
uv run --directory demos/ecb_triage_agent python -m ecb_triage_agent.rest_server &
PIDS+=($!)
sleep 3

subsection "Starting ecb_triage_agent - MCP server"
uv run --directory demos/ecb_triage_agent python -m ecb_triage_agent.mcp_server &
PIDS+=($!)
sleep 3

# Summary
section "${#PIDS[@]} processes started"
echo ""

wait
