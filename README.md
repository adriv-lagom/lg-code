# lg-code — Multi-Agent LangGraph Coding Platform

Four specialized coding agents sharing a common two-LLM architecture (local vLLM + DeepSeek), designed to run anywhere on a Tailscale network.

## Architecture

```
                  ┌──────────────────────────┐
                  │     DeepSeek API          │
                  │  (orchestration + diag)   │
                  │ 🔵 1 call per run (typ.)  │
                  └──────────┬───────────────┘
                             │ HTTPS
    ┌────────────────────────┼───────────────────────┐
    │  Tailscale tailnet     │                       │
    │                        │                       │
    │  ┌──────────────┐      │      ┌──────────────┐ │
    │  │ les (GPU)    │      │      │ les-hub      │ │
    │  │ vLLM :8002   │◄─────┼──────│ langgraph    │ │
    │  │ Qwen2.5 3B   │    Magic    │ :2024        │ │
    │  │ 🟢 worker    │    DNS     │ 4 agents     │ │
    │  └──────────────┘      │      └──────────────┘ │
    │                        │                       │
    └────────────────────────┴───────────────────────┘
```

- **🟢 Worker (vLLM)**: `Qwen2.5-Coder-3B-Instruct-AWQ` on RTX 3060 12 GB, serves via OpenAI-compatible API at `les.tail9c372e.ts.net:8002`. Fast, free, local. Does the heavy lifting: code editing, self-review.
- **🔵 Supervisor (DeepSeek)**: `deepseek-v4-pro` via HTTPS. Called sparingly — only for orchestration (planning subtasks) and diagnosis (on test failure). Expensive but smart.

Every agent follows the same pattern: orchestrate → edit → self-review → test → commit, with diagnosis + retry on failure.

## Agents

| Agent | Graph ID | Purpose | Key feature |
|-------|----------|---------|-------------|
| **supervisor_worker** | `supervisor_worker` | Python coding with delegated editing | Single DeepSeek call per run |
| **julia_agent** | `julia_agent` | Multi-repo Julia package editor | Git commit + API sync across packages |
| **node_red_agent** | `node_red_agent` | Node-RED flow JSON editor | MQTT-aware, validates wires + IDs |
| **coding_agent** | `coding_agent` | Original plan→edit→test→review loop | MWE / legacy |

### julia_agent pipeline

```
LOAD → ORCHESTRATE → PREP → EDIT → SELF-REVIEW → API-SYNC → TEST → COMMIT
  ↑                                                              ↓
  └───────────── DIAGNOSE ← (retry ≤1 on failure) ───────────────┘
```

Reads all Julia repos, snapshots source files, plans cross-package subtasks, edits with full context, auto-detects changed exports and fixes downstream importers, runs `Pkg.test()` per repo, and `git commit`s each successful subtask.

### node_red_agent pipeline

```
LOAD → ORCHESTRATE → PREP → EDIT → SELF-REVIEW → TEST → COMMIT
  ↑                                                      ↓
  └──────────── DIAGNOSE ← (retry ≤1 on failure) ────────┘
```

Parses `flows.json`, operates with `<add>` / `<remove>` / `<modify>` JSON blocks, validates unique IDs, wire targets, required node properties (MQTT broker, function code, etc.), and git commits the modified flow.

### All agents share

- **Token tracking**: every LLM call logged (input/output/total tokens, call name, model)
- **Notes system**: `.md` scratchpads per role (`plan`, `edit`, `review`, `test`, `diagnose`) persisted to `.agent_notes/`
- **Two retries max**: test failure → diagnose → retry edit once → skip subtask on second failure
- **Git commit per subtask**: clean commit messages with subtask description + files changed

## File structure

```
lg-code/
├── .env.example              # template — committed, no secrets
├── .env                      # your secrets — gitignored
├── .gitignore
├── pyproject.toml            # pip install -e .
├── langgraph.json            # graph registry
├── setup.sh                  # one-command bootstrap
├── README.md
├── sample_flows/
│   └── temperature_mqtt.json # example Node-RED flow
├── src/code_graph/
│   ├── agent.py              # coding_agent (MWE)
│   ├── sw_agent.py           # supervisor_worker
│   ├── julia_agent.py        # julia_agent
│   └── node_red_agent.py     # node_red_agent
└── tests/
    └── test_math_utils.py    # MWE tests
```

## Quick start (any machine)

```bash
git clone https://github.com/adriv-lagom/lg-code.git
cd lg-code
bash setup.sh
```

This creates a `.venv`, installs all dependencies (editable mode), and copies `.env.example` → `.env` if it doesn't exist.

### Per-machine config

Edit `.env` — only two things are machine-specific:

| Variable | What to set |
|----------|------------|
| `SUPERVISOR_API_KEY` | DeepSeek API key (same everywhere) |
| `LOCAL_BASE_URL` | vLLM endpoint — Magic DNS works from anywhere on the tailnet |
| `JULIA_REPOS` | Absolute paths to Julia packages **on this machine** |
| `NODERED_FLOW_PATH` | Absolute path to `flows.json` **on this machine** |

Everything else is in git and shared.

### Start the dev server

```bash
source .venv/bin/activate
# Use your Tailscale hostname (not 0.0.0.0!) so the Studio URL is correct:
langgraph dev --host $(hostname).tail9c372e.ts.net --port 2024 --no-reload

# Detached (survives terminal close):
nohup .venv/bin/langgraph dev --host $(hostname).tail9c372e.ts.net --port 2024 --no-reload > langgraph.log 2>&1 &
```

> **Important**: Do NOT use `--host 0.0.0.0`. The LangGraph Studio browser UI uses the `--host` value as the `baseUrl` to connect back to the server. Your browser can't resolve `0.0.0.0`. Use your machine's Tailscale Magic DNS name instead.

## Accessing the Studio UI

The LangGraph Studio runs in your browser and connects via the tailnet. Replace `<hostname>` with your agent host's Tailscale name:

```
https://smith.langchain.com/studio/?baseUrl=http://<hostname>.tail9c372e.ts.net:2024
```

You can also reach the API directly:

```bash
curl http://<hostname>.tail9c372e.ts.net:2024/ok
```

## vLLM management

vLLM is managed via `vllmctl` on `les`:

```bash
vllmctl status          # check if running
vllmctl restart         # restart current profile
vllmctl switch coder3b  # switch model profile
vllmctl test            # send test prompt
```

Key config files (on `les`):
- `/etc/vllm-active.env` — active profile, model, context length, `VLLM_HOST=0.0.0.0`
- `/usr/local/bin/vllm-active-run` — uses `${VLLM_HOST:-127.0.0.1}` for `--host`
- `/usr/local/bin/vllmctl` — reads `VLLM_HOST` from env file, preserves across profile switches

Critical: `VLLM_HOST=0.0.0.0` in `/etc/vllm-active.env` is what makes vLLM accept connections from other machines on the tailnet. `vllmctl switch` preserves this setting.

## Updating the code

```bash
# On les (dev machine):
git pull
# make changes, then:
git add -A && git commit -m "..." && git push

# On les-hub (agent host):
cd ~/lg-code
git pull
# Kill old process and restart:
pkill -f '.venv/bin/langgraph' 2>/dev/null
nohup .venv/bin/langgraph dev --host les-hub.tail9c372e.ts.net --port 2024 --no-reload > langgraph.log 2>&1 &
```

## Running tests

```bash
source .venv/bin/activate
pytest
```

Python syntax check only (agents themselves need live LLM for integration testing).

## Design decisions

1. **One DeepSeek call per run**. Orchestration plans all subtasks upfront. Self-review and editing use the free local vLLM. DeepSeek is only called again on test failure.

2. **Self-review pass-through**. If the vLLM reviewer doesn't produce valid file blocks, the edit is silently accepted. This prevents the reviewer from blocking progress when it echoes commentary instead of code.

3. **Tailscale Magic DNS everywhere**. `LOCAL_BASE_URL` uses `les.tail9c372e.ts.net` so the `.env` is portable across all machines on the tailnet. No per-machine URL editing.

4. **nohup on the agent host**. LangGraph runs detached so it survives SSH disconnects. Logs go to `langgraph.log`.

5. **Subtask-level git commits**. Each successful subtask gets its own commit with the subtask description. Failed subtasks are skipped after one retry.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `langgraph: command not found` | Missing `langgraph-cli[inmem]` | `pip install -U "langgraph-cli[inmem]"` |
| vLLM unreachable from other machine | `--host 127.0.0.1` | Set `VLLM_HOST=0.0.0.0` in `/etc/vllm-active.env` and `vllmctl restart` |
| Empty worker output | vLLM model not following format | Check prompt has FORMAT RULES with CORRECT/WRONG examples |
| `.agent_notes/` not appearing | No commits yet | Notes persist to disk only on `commit_node` |
| Julia tests fail on les-hub | Julia not installed or wrong paths | `JULIA_REPOS` must use absolute paths valid on that machine |
| Git push denied | Wrong GitHub account | `gh auth switch --user adriv-lagom && gh auth setup-git` |
