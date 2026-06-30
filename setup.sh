#!/usr/bin/env bash
set -euo pipefail

# ── lg-code setup ──────────────────────────────────────────────────────────
# Usage: bash setup.sh
#
# What this does:
#   1. Creates a Python venv
#   2. Installs all dependencies (editable mode)
#   3. Copies .env.example → .env (if it doesn't exist)
#   4. Prints what to do next
#
# Prerequisites:
#   - Python ≥ 3.10
#   - git (for commit functionality)
#   - Julia (optional — only for julia_agent)
#   - vLLM or any OpenAI-compatible LLM endpoint (see LOCAL_BASE_URL in .env)
# ────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "═══ lg-code setup ═══"

# 1. Python venv
if [[ ! -d .venv ]]; then
    echo "→ Creating virtual environment..."
    python3 -m venv .venv
else
    echo "✓ .venv already exists"
fi

# Activate
source .venv/bin/activate

# 2. Install dependencies
echo "→ Installing dependencies..."
pip install --upgrade pip -q
pip install -e . -q
echo "✓ Package installed (editable mode)"

# 3. .env
if [[ ! -f .env ]]; then
    cp .env.example .env
    echo ""
    echo "⚠️  .env created from .env.example — EDIT IT NOW:"
    echo "   → SUPERVISOR_API_KEY  (DeepSeek API key)"
    echo "   → LOCAL_BASE_URL      (your LLM endpoint)"
    echo "   → JULIA_REPOS         (paths to your Julia packages, or remove)"
    echo "   → NODERED_FLOW_PATH   (path to your flows.json, or remove)"
    echo ""
else
    echo "✓ .env already exists"
fi

# 4. Verify langgraph CLI
if ! command -v langgraph &>/dev/null; then
    echo ""
    echo "⚠️  'langgraph' CLI not found on PATH."
    echo "   It should be installed via pip as a dependency."
    echo "   Try: source .venv/bin/activate && langgraph --version"
else
    echo "✓ langgraph CLI available ($(langgraph --version 2>&1 || echo 'ok'))"
fi

# ── Done ───────────────────────────────────────────────────────────────────
echo ""
echo "═══ Setup complete ═══"
echo ""
echo "Next steps:"
echo "  1. Edit .env — set your API keys and paths"
echo "  2. Start the LangGraph dev server:"
echo "       source .venv/bin/activate"
echo "       langgraph dev --port 2024"
echo "  3. Open http://localhost:2024 in your browser"
echo ""
echo "For remote access via SSH:"
echo "  On the remote machine:  langgraph dev --port 2024 --host 0.0.0.0"
echo "  On your local machine:  ssh -L 2024:localhost:2024 user@remote"
echo ""
echo "Optional — run tests:"
echo "  source .venv/bin/activate && pytest"
