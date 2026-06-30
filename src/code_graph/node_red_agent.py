"""
Node-RED Flow Editing Agent.

Architecture:
    LOAD (read flows.json, parse JSON nodes)
      → ORCHESTRATE (DeepSeek: ONE call — plans subtasks for flow edits)
      → PREP (prepare context for current subtask, no LLM)
      → EDIT (vLLM: edits/adds/removes nodes in the flow)
      → SELF-REVIEW (vLLM: checks own output, fixes mistakes — free)
      → TEST (validate JSON structure, unique IDs, valid wire targets)
        ├─ pass → COMMIT (git commit on flows.json) → advance or DONE
        └─ fail → DIAGNOSE (DeepSeek, only on failure) → EDIT (retry ≤1)

Key properties:
- 🔵 DeepSeek: 1 orchestrate call + diagnose only on failure
- 🟢 vLLM: edit + self-review — runs on local GPU, free
- 🔧 Flow-aware: understands Node-RED node types (mqtt, function, switch, change, etc.)
- 📦 Git: auto-commit after each successful subtask
- 📝 Notes: persisted to .agent_notes/ for self-awareness across runs

Config via .env:
    NODERED_FLOW_PATH — absolute path to flows.json (default: ./workspace/flows.json)
    NODERED_FLOWS_DIR — absolute path to directory with multiple flow files (optional)
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]

# ── Flow configuration ─────────────────────────────────────────────────────
# Override via NODERED_FLOW_PATH or NODERED_FLOWS_DIR env vars.
# Default: workspace/flows.json

_DEFAULT_FLOW_PATH = os.getenv(
    "NODERED_FLOW_PATH",
    str(ROOT / "workspace" / "flows.json"),
)
_DEFAULT_FLOWS_DIR = os.getenv("NODERED_FLOWS_DIR", "")


# ── LLM factories ──────────────────────────────────────────────────────────


def _worker_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LOCAL_MODEL", "local-active-agent"),
        base_url=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8002/v1"),
        api_key=os.getenv("LOCAL_API_KEY", "dummy"),
        temperature=0,
        timeout=180,
    )


def _supervisor_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("SUPERVISOR_MODEL", "deepseek-v4-pro"),
        base_url=os.getenv("SUPERVISOR_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.getenv("SUPERVISOR_API_KEY", "dummy"),
        temperature=0,
        timeout=180,
    )


# ── Token tracking ─────────────────────────────────────────────────────────


def _tracked_invoke(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    call_name: str,
    state: dict,
) -> Any:
    response = llm.invoke(messages)
    usage = {}
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        um = response.usage_metadata
        usage = {
            "call": call_name,
            "input_tokens": um.get("input_tokens", 0),
            "output_tokens": um.get("output_tokens", 0),
            "total_tokens": um.get("total_tokens", 0),
        }
    elif hasattr(response, "response_metadata") and response.response_metadata:
        rm = response.response_metadata
        usage = {
            "call": call_name,
            "token_usage": rm.get("token_usage", {}),
        }
    if usage:
        state.setdefault("token_usage", []).append(usage)
    return response


# ── State ──────────────────────────────────────────────────────────────────


class Subtask(TypedDict, total=False):
    desc: str
    instruction: str
    node_types: str  # comma-separated node types involved


class CommitRecord(TypedDict, total=False):
    repo: str
    files: str
    hash: str
    message: str


class NodeRedState(TypedDict, total=False):
    task: str
    flow_path: str
    flow_data: Annotated[list[dict], "parsed JSON flow — list of node objects"]
    flow_snapshot: Annotated[
        dict[str, dict],
        "node_id → full node dict, for quick lookup",
    ]
    subtasks: Annotated[list[Subtask], "subtasks with instructions baked in"]
    subtask_index: int
    fix_instruction: str
    test_output: str
    attempt: int
    status: str
    notes: Annotated[dict[str, str], "role → markdown notes, persisted to .agent_notes/"]
    commits: Annotated[list[CommitRecord], "git commits made during run"]
    token_usage: Annotated[list[dict[str, Any]], "per-call token records"]


# ── Notes system ───────────────────────────────────────────────────────────

NOTES_DIR = ROOT / ".agent_notes"


def _note_path(role: str) -> Path:
    return NOTES_DIR / f"nodered_{role}.md"


def _read_notes_summary(state: NodeRedState) -> str:
    """Compact summary of all notes for injecting into prompts."""
    notes = state.get("notes") or {}
    if not notes:
        return ""
    parts = []
    for role in ["plan", "edit", "review", "test", "diagnose"]:
        if role in notes:
            content = notes[role].strip()
            if len(content) > 500:
                content = content[:500] + "\n... (truncated)"
            parts.append(f"## {role} notes\n{content}")
    return "\n\n".join(parts)


def _write_note(state: NodeRedState, role: str, content: str) -> None:
    """Append to a role's notes. Creates `notes` key if needed."""
    notes = state.setdefault("notes", {})
    existing = notes.get(role, "")
    timestamped = (
        f"### {role}\n{content.strip()}\n"
        if not existing
        else f"{existing}\n\n### {role}\n{content.strip()}\n"
    )
    notes[role] = timestamped


def _persist_notes(state: NodeRedState) -> None:
    """Write all notes to .agent_notes/ on disk."""
    notes = state.get("notes") or {}
    if not notes:
        return
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    for role, content in notes.items():
        _note_path(role).write_text(content.strip() + "\n")


# ── Flow helpers ───────────────────────────────────────────────────────────


# Known Node-RED node type categories — used to give the LLM context
_NODE_CATEGORIES = {
    "mqtt in": "input",
    "mqtt out": "output",
    "http in": "input",
    "http response": "output",
    "http request": "function",
    "inject": "input",
    "debug": "output",
    "function": "function",
    "switch": "function",
    "change": "function",
    "template": "function",
    "delay": "function",
    "trigger": "function",
    "rbe": "function",
    "link in": "input",
    "link out": "output",
    "link call": "function",
    "comment": "comment",
    "catch": "status",
    "status": "status",
    "complete": "status",
    "join": "function",
    "split": "function",
    "sort": "function",
    "batch": "function",
    "csv": "parser",
    "html": "parser",
    "json": "parser",
    "xml": "parser",
    "yaml": "parser",
    "tcp in": "input",
    "tcp out": "output",
    "tcp request": "function",
    "udp in": "input",
    "udp out": "output",
    "websocket in": "input",
    "websocket out": "output",
    "serial in": "input",
    "serial out": "output",
    "exec": "function",
}


def _summarize_flow(flow_data: list[dict]) -> str:
    """Build a human-readable summary of the flow for LLM context."""
    if not flow_data:
        return "(empty flow)"
    lines = []
    for node in flow_data:
        nid = node.get("id", "?")
        ntype = node.get("type", "?")
        name = node.get("name", "") or ntype
        wires = node.get("wires", [])
        if wires:
            targets = [w[0] if w else "?" for w in wires]
            wire_str = " → " + ", ".join(targets)
        else:
            wire_str = " (end)"
        category = _NODE_CATEGORIES.get(ntype, "?")
        lines.append(
            f"  [{category}] {ntype} \"{name}\" (id={nid}){wire_str}"
        )
    return "\n".join(lines)


def _summarize_flow_compact(flow_data: list[dict]) -> str:
    """Compact single-line-per-node summary for tight context."""
    if not flow_data:
        return "(empty flow)"
    return "\n".join(
        f"  {n.get('id','?')} | {n.get('type','?')} | {n.get('name','') or n.get('type','?')}"
        for n in flow_data
    )


def _get_node_by_id(flow_data: list[dict], node_id: str) -> dict | None:
    for node in flow_data:
        if node.get("id") == node_id:
            return node
    return None


def _validate_flow(flow_data: list[dict]) -> tuple[bool, list[str]]:
    """Validate a Node-RED flow JSON structure. Returns (is_valid, errors)."""
    errors = []
    ids = set()

    if not isinstance(flow_data, list):
        return False, ["Flow data is not a JSON array"]

    for i, node in enumerate(flow_data):
        if not isinstance(node, dict):
            errors.append(f"Node {i}: not a JSON object")
            continue

        nid = node.get("id")
        if not nid:
            errors.append(f"Node at index {i}: missing 'id' field")
        elif nid in ids:
            errors.append(f"Node {nid}: duplicate ID")
        else:
            ids.add(nid)

        ntype = node.get("type")
        if not ntype:
            errors.append(f"Node {nid or i}: missing 'type' field")

        # Check config nodes (type ends with -config) don't have wires
        if ntype and ntype.endswith("-config"):
            if node.get("wires"):
                errors.append(f"Config node {nid}: should not have wires")

    # Validate all wire targets reference existing nodes
    all_ids = {n.get("id") for n in flow_data if n.get("id")}
    for node in flow_data:
        nid = node.get("id", "?")
        for i, wire_list in enumerate(node.get("wires", [])):
            for target in wire_list:
                if target and target not in all_ids:
                    errors.append(
                        f"Node {nid}: wire[{i}] targets nonexistent node '{target}'"
                    )

    return len(errors) == 0, errors


def _rebuild_flow_snapshot(flow_data: list[dict]) -> dict[str, dict]:
    """Rebuild node_id → node lookup."""
    return {n["id"]: n for n in flow_data if "id" in n}


# ── Git helpers ────────────────────────────────────────────────────────────


def _run_git(flow_dir: str, *args: str) -> tuple[int, str, str]:
    """Run a git command. Returns (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=flow_dir,
            text=True,
            capture_output=True,
            timeout=30,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return 1, "", str(e)


# ── Parsing helpers ────────────────────────────────────────────────────────


def _strip_fences(text: str) -> str:
    """Remove outermost ``` fences if present."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    return t.strip()


def _extract_field(text: str, field: str) -> str:
    """Extract `Field: value` from orchestrate output."""
    m = re.search(rf"^{field}:\s*(.+?)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _extract_node_blocks(text: str) -> list[dict]:
    """Extract <node>...</node> blocks from LLM output, returns list of node dicts."""
    # Match <node>...JSON content...</node>
    nodes = []
    pattern = r"<node>\s*(.*?)\s*</node>"
    for match in re.finditer(pattern, text, re.DOTALL):
        raw = match.group(1).strip()
        raw = _strip_fences(raw)
        try:
            node = json.loads(raw)
            if isinstance(node, dict):
                nodes.append(node)
        except json.JSONDecodeError:
            # Try to salvage — might be missing commas or have trailing commas
            try:
                # Remove trailing commas before } or ]
                fixed = re.sub(r",\s*([}\]])", r"\1", raw)
                node = json.loads(fixed)
                if isinstance(node, dict):
                    nodes.append(node)
            except json.JSONDecodeError:
                continue
    return nodes


def _extract_instructions(text: str) -> list[dict]:
    """Extract add/remove/modify instructions from LLM output.
    Returns list of {action: 'add'|'remove'|'modify', node: dict, target_id: str}.
    """
    actions = []
    # Parse <add>, <remove>, <modify> blocks
    for action_type in ["add", "remove", "modify"]:
        pattern = rf"<{action_type}>\s*(.*?)\s*</{action_type}>"
        for match in re.finditer(pattern, text, re.DOTALL):
            raw = match.group(1).strip()
            if action_type == "remove":
                # Just a node ID
                nid = raw.strip().strip('"')
                actions.append({"action": "remove", "target_id": nid})
            else:
                raw = _strip_fences(raw)
                try:
                    node = json.loads(raw)
                    if isinstance(node, dict):
                        actions.append({"action": action_type, "node": node})
                except json.JSONDecodeError:
                    pass
    return actions


def _reconstruct_flow(
    flow_data: list[dict],
    node_blocks: list[dict],
    actions: list[dict],
) -> list[dict]:
    """Apply node blocks and actions to the flow. Returns new flow list."""
    flow = [dict(n) for n in flow_data]  # deep-ish copy

    # First process remove actions
    remove_ids = set()
    for action in actions:
        if action.get("action") == "remove":
            tid = action.get("target_id", "")
            if tid:
                remove_ids.add(tid)

    # Also check node_blocks for nodes with _action: "remove"
    for node in node_blocks:
        if node.get("_action") == "remove":
            remove_ids.add(node.get("id", ""))

    if remove_ids:
        flow = [n for n in flow if n.get("id") not in remove_ids]
        # Clean up wires referencing removed nodes
        for node in flow:
            new_wires = []
            for wire_list in node.get("wires", []):
                cleaned = [w for w in wire_list if w not in remove_ids]
                if cleaned:
                    new_wires.append(cleaned)
                else:
                    new_wires.append([])
            node["wires"] = new_wires

    # Process modify and add from node_blocks
    for node in node_blocks:
        nid = node.get("id", "")
        if node.get("_action") == "remove":
            continue
        # Check if this node already exists → modify
        existing = _get_node_by_id(flow, nid)
        if existing:
            existing.update(node)
        else:
            flow.append(dict(node))

    # Process explicit add actions
    for action in actions:
        if action.get("action") == "add":
            new_node = dict(action["node"])
            nid = new_node.get("id", "")
            existing = _get_node_by_id(flow, nid)
            if existing:
                existing.update(new_node)
            else:
                flow.append(new_node)
        elif action.get("action") == "modify":
            mod_node = action.get("node", {})
            nid = mod_node.get("id", "")
            existing = _get_node_by_id(flow, nid)
            if existing:
                existing.update(mod_node)

    return flow


# ── Nodes ──────────────────────────────────────────────────────────────────


def load_node(state: NodeRedState) -> NodeRedState:
    """Read flows.json, parse, validate, build snapshot."""
    flow_path = state.get("flow_path", _DEFAULT_FLOW_PATH)
    path = Path(flow_path)

    if not path.exists():
        # Create empty flow file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]")
        flow_data = []
    else:
        raw = path.read_text()
        try:
            flow_data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid flows.json: {e}")

    if not isinstance(flow_data, list):
        raise ValueError("flows.json must be a JSON array of node objects")

    return {
        **state,
        "flow_path": str(path),
        "flow_data": flow_data,
        "flow_snapshot": _rebuild_flow_snapshot(flow_data),
        "subtask_index": 0,
        "attempt": 0,
        "status": "loaded",
    }


def orchestrate_node(state: NodeRedState) -> NodeRedState:
    """🔵 DeepSeek: plans subtasks for flow edits. ONE call."""
    flow_data = state["flow_data"]
    llm = _supervisor_llm()

    flow_summary = _summarize_flow(flow_data)
    flow_dir = str(Path(state["flow_path"]).parent)

    response = _tracked_invoke(
        llm,
        [
            SystemMessage(
                content=(
                    "You are a Node-RED flow architect. You plan edits to a flow JSON.\n\n"
                    "Break a task into SUBTASKS. Each subtask should be a self-contained\n"
                    "set of node additions, removals, or modifications.\n\n"
                    "Known node types and their categories:\n"
                    + "\n".join(
                        f"  {t} ({c})" for t, c in sorted(_NODE_CATEGORIES.items())
                    )
                    + "\n\n"
                    "FORMAT — separate subtasks with ---SUBTASK---:\n"
                    "Description: what this subtask does (1 sentence)\n"
                    "Instruction: detailed step-by-step for the code model\n"
                    "NodeTypes: comma-separated node types involved\n"
                    "---SUBTASK---\n"
                    "Description: ...\n"
                    "Instruction: ...\n"
                    "NodeTypes: ...\n\n"
                    "RULES:\n"
                    "1. First subtask should set up infrastructure (broker config, etc.)\n"
                    "2. Group related MQTT in/out + processing logic together\n"
                    "3. Ensure wire connections are explicit\n"
                    "4. Each subtask must be independently testable\n"
                    "5. Keep to 1-4 subtasks total\n"
                    "6. Always specify what node IDs to connect with wires\n"
                )
            ),
            HumanMessage(
                content=(
                    f"**Task:** {state['task']}\n\n"
                    f"**Current flow ({len(flow_data)} nodes):**\n{flow_summary}\n\n"
                    f"**Flow directory:** {flow_dir}\n\n"
                    "Plan the subtasks:"
                )
            ),
        ],
        "orchestrate",
        state,
    )

    subtasks = _parse_orchestration(response.content.strip())
    if not subtasks:
        subtasks = [
            Subtask(
                desc=state["task"],
                instruction=state["task"],
                node_types="function",
            )
        ]

    _write_note(
        state,
        "plan",
        f"**Task:** {state['task']}\n\n"
        f"**Flow:** {state['flow_path']} ({len(flow_data)} nodes)\n\n"
        f"**Subtasks planned:**\n"
        + "\n".join(
            f"{i+1}. [{s.get('node_types','?')}] {s.get('desc','?')}"
            for i, s in enumerate(subtasks)
        ),
    )

    return {
        **state,
        "subtasks": subtasks,
        "subtask_index": 0,
        "attempt": 0,
        "status": "orchestrated",
    }


def _parse_orchestration(text: str) -> list[Subtask]:
    blocks = re.split(r"\n?---SUBTASK---\n?", text)
    subtasks = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        desc = _extract_field(block, "Description")
        instruction = _extract_field(block, "Instruction")
        node_types = _extract_field(block, "NodeTypes")
        if desc:
            subtasks.append(
                Subtask(
                    desc=desc,
                    instruction=instruction or desc,
                    node_types=node_types or "",
                )
            )
    return subtasks


def prep_node(state: NodeRedState) -> NodeRedState:
    """Prepare context for current subtask. No LLM call."""
    idx = state["subtask_index"]
    st = state["subtasks"][idx]

    # Re-snapshot flow
    state["flow_snapshot"] = _rebuild_flow_snapshot(state["flow_data"])

    return {
        **state,
        "attempt": 0,
        "fix_instruction": "",
        "status": "prepped",
    }


def edit_node(state: NodeRedState) -> NodeRedState:
    """🟢 vLLM: edits flow nodes — add, remove, or modify. Produces JSON blocks."""
    idx = state["subtask_index"]
    st = state["subtasks"][idx]
    llm = _worker_llm()

    flow_data = state["flow_data"]
    flow_summary = _summarize_flow(flow_data)
    flow_json = json.dumps(flow_data, indent=2)

    fix_context = ""
    if state.get("fix_instruction"):
        fix_context = (
            f"\n\n⚠️ PREVIOUS ATTEMPT FAILED. Fix:\n{state['fix_instruction']}\n"
        )

    notes_context = _read_notes_summary(state)
    if notes_context:
        notes_context = f"\n\n**Agent notes:**\n{notes_context}\n"

    # Build known node IDs for wire references
    known_ids = [n.get("id", "") for n in flow_data if n.get("id")]
    id_list = ", ".join(known_ids) if known_ids else "(none — flow is empty)"

    system = (
        "You are a Node-RED flow editor. Output COMPLETE node JSON objects.\n\n"
        "ACTIONS — use these EXACT tags:\n"
        "<add>\n"
        '  {"id": "...", "type": "...", "name": "...", ...}\n'
        "</add>\n"
        "<remove>\n"
        "  node-id-to-remove\n"
        "</remove>\n"
        "<modify>\n"
        '  {"id": "existing-node-id", "type": "...", "name": "...", ...}\n'
        "</modify>\n\n"
        "NODE PROPERTIES every node MUST have:\n"
        '- "id": unique string (e.g. "n1", "n2", ... or UUID-style)\n'
        '- "type": valid Node-RED type (mqtt in, mqtt out, function, switch, change, inject, debug, template, delay, http in, http response, etc.)\n'
        '- "wires": [[target_id, ...], ...] — array of output wire arrays\n'
        '- "name": human-readable label (optional but recommended)\n\n'
        "TYPE-SPECIFIC required properties:\n"
        '- mqtt in/out: "topic", "broker"\n'
        '- function: "func" (JavaScript string)\n'
        '- switch: "property", "rules" (array of {t, v, vt})\n'
        '- change: "rules" (array of {t, from, to, tot})\n'
        '- template: "template", "syntax" (mustache|plain)\n'
        '- inject: "props", "repeat", "once"\n'
        '- debug: "complete" (true|false|"payload")\n\n'
        "RULES:\n"
        "1. PRESERVE ALL EXISTING NODES unless you are modifying or removing them\n"
        "2. Always output COMPLETE node objects, not partial updates\n"
        "3. Wire IDs must reference existing nodes OR new nodes you are adding\n"
        "4. Each node needs unique ID — use descriptive prefixes like 'mqtt_in_', 'func_', 'debug_'\n"
        "5. No markdown fences, no explanations outside the tags\n"
        "6. Use valid JSON — no trailing commas, double-quoted strings only"
    )

    human = (
        f"**Task:** {state['task']}\n\n"
        f"**Subtask {idx+1}/{len(state['subtasks'])}:** {st.get('desc', '')}\n\n"
        f"**Instructions:** {st.get('instruction', '')}{fix_context}{notes_context}\n"
        f"**Existing node IDs:** {id_list}\n\n"
        f"**Current flow summary:**\n{flow_summary}\n\n"
        f"**Full flow JSON for reference:**\n```json\n{flow_json[:6000]}\n```\n\n"
        "Output add/remove/modify actions:"
    )

    response = _tracked_invoke(
        llm,
        [SystemMessage(content=system), HumanMessage(content=human)],
        "edit",
        state,
    )

    raw = (response.content or "").strip()
    node_blocks = _extract_node_blocks(raw)
    actions = _extract_instructions(raw)

    if not node_blocks and not actions:
        # Fallback: try to parse the whole response as nodes
        try:
            parsed = json.loads(_strip_fences(raw))
            if isinstance(parsed, list):
                node_blocks = parsed
            elif isinstance(parsed, dict):
                node_blocks = [parsed]
        except json.JSONDecodeError:
            pass

    # Apply changes
    new_flow = _reconstruct_flow(flow_data, node_blocks, actions)
    state["flow_data"] = new_flow
    state["flow_snapshot"] = _rebuild_flow_snapshot(new_flow)

    _write_note(
        state,
        "edit",
        f"**Subtask {idx+1}** attempt {state['attempt']+1}\n"
        f"Added: {len([a for a in actions if a.get('action')=='add'])} nodes, "
        f"Removed: {len([a for a in actions if a.get('action')=='remove'])} nodes, "
        f"Modified: {len([a for a in actions if a.get('action')=='modify'])} nodes\n"
        f"Total flow nodes: {len(new_flow)}",
    )

    return {
        **state,
        "attempt": state["attempt"] + 1,
        "status": "edited",
    }


def self_review_node(state: NodeRedState) -> NodeRedState:
    """🟢 vLLM self-review: checks flow JSON quality."""
    llm = _worker_llm()
    idx = state["subtask_index"]
    st = state["subtasks"][idx]

    flow_json = json.dumps(state["flow_data"], indent=2)
    flow_summary = _summarize_flow(state["flow_data"])

    response = _tracked_invoke(
        llm,
        [
            SystemMessage(
                content=(
                    "You are a Node-RED flow reviewer. Check for:\n"
                    "1. Missing required properties per node type\n"
                    "2. Broken wire connections (targeting nonexistent nodes)\n"
                    "3. Duplicate node IDs\n"
                    "4. Invalid JSON syntax\n"
                    "5. MQTT nodes missing broker or topic\n"
                    "6. Function nodes with JavaScript syntax errors\n"
                    "7. Nodes that don't connect to anything (dead ends that should connect)\n\n"
                    "If correct, output exactly: PASS\n"
                    "If issues found, use <add>/<remove>/<modify> tags to fix."
                )
            ),
            HumanMessage(
                content=(
                    f"**Task:** {state['task']}\n\n"
                    f"**Subtask:** {st.get('desc', '')}\n\n"
                    f"**Flow summary:**\n{flow_summary}\n\n"
                    f"**Full flow JSON:**\n```json\n{flow_json[:6000]}\n```\n\n"
                    "Review:"
                )
            ),
        ],
        "self_review",
        state,
    )

    raw = (response.content or "").strip()
    if raw.upper().startswith("PASS"):
        _write_note(state, "review", f"**Subtask {idx+1}** — Passed review (no issues).")
        return {**state, "status": "self_reviewed"}

    # Try to extract fixes
    node_blocks = _extract_node_blocks(raw)
    actions = _extract_instructions(raw)

    if node_blocks or actions:
        new_flow = _reconstruct_flow(state["flow_data"], node_blocks, actions)
        state["flow_data"] = new_flow
        state["flow_snapshot"] = _rebuild_flow_snapshot(new_flow)
        _write_note(state, "review", f"**Subtask {idx+1}** — Applied corrections from review.")

    return {**state, "status": "self_reviewed"}


def test_node(state: NodeRedState) -> NodeRedState:
    """Validate flow JSON structure and integrity. No LLM call."""
    idx = state["subtask_index"]
    flow_data = state["flow_data"]

    is_valid, errors = _validate_flow(flow_data)

    # Also validate JSON round-trips cleanly
    try:
        json_str = json.dumps(flow_data, indent=2)
        json.loads(json_str)
        json_ok = True
    except (json.JSONDecodeError, TypeError) as e:
        json_ok = False
        errors.append(f"JSON round-trip failed: {e}")

    # Check for common node issues
    for node in flow_data:
        ntype = node.get("type", "")
        nid = node.get("id", "?")
        if ntype == "mqtt in" and "broker" not in node:
            errors.append(f"MQTT in node {nid}: missing 'broker'")
        if ntype == "mqtt out" and "broker" not in node:
            errors.append(f"MQTT out node {nid}: missing 'broker'")
        if ntype == "function" and "func" not in node:
            errors.append(f"Function node {nid}: missing 'func'")

    all_passed = is_valid and json_ok and len(errors) == 0
    output = "\n".join(errors) if errors else "All validations passed."

    _write_note(
        state,
        "test",
        f"**Subtask {idx+1}** — {'✅ PASSED' if all_passed else '❌ FAILED'}\n\n"
        f"```\n{output[:3000]}\n```\n"
        f"Nodes: {len(flow_data)}, Valid JSON: {json_ok}, Valid structure: {is_valid}",
    )

    return {
        **state,
        "test_output": output,
        "status": "tests_passed" if all_passed else "tests_failed",
    }


def commit_node(state: NodeRedState) -> NodeRedState:
    """Git commit the modified flows.json."""
    idx = state["subtask_index"]
    st = state["subtasks"][idx]
    flow_path = Path(state["flow_path"])
    flow_dir = str(flow_path.parent)

    # Write the current flow to disk
    flow_path.write_text(json.dumps(state["flow_data"], indent=2) + "\n")

    # Check if git repo
    rc, stdout, _ = _run_git(flow_dir, "status", "--porcelain")
    if rc != 0:
        _write_note(state, "commit", f"**Subtask {idx+1}** — No git repo at {flow_dir}")
        _persist_notes(state)
        return {**state, "status": "committed"}

    flow_rel = flow_path.name
    changed = [line[3:] for line in stdout.splitlines() if flow_rel in line]

    if not changed:
        # Stage everything new
        _run_git(flow_dir, "add", flow_rel)
    else:
        _run_git(flow_dir, "add", "-A")

    msg = f"{st.get('desc', 'flow edit')}\n\nTask: {state['task']}\nSubtasks: {idx+1}/{len(state['subtasks'])}"
    rc, _, _ = _run_git(flow_dir, "commit", "-m", msg)

    if rc == 0:
        rc2, commit_hash, _ = _run_git(flow_dir, "rev-parse", "--short", "HEAD")
        state.setdefault("commits", []).append(
            CommitRecord(
                repo="nodered",
                files=flow_rel,
                hash=commit_hash if rc2 == 0 else "?",
                message=msg[:200],
            )
        )

    _write_note(
        state,
        "commit",
        f"**Subtask {idx+1}** — Committed\n\n"
        + "\n".join(
            f"- `{c['repo']}` — {c.get('hash','?')}: {c.get('message','')[:80]}"
            for c in state.get("commits", [])[-3:]
        ),
    )
    _persist_notes(state)

    return {**state, "status": "committed"}


def diagnose_node(state: NodeRedState) -> NodeRedState:
    """🔵 DeepSeek: ONLY called on test failure. Diagnoses and writes fix."""
    idx = state["subtask_index"]
    st = state["subtasks"][idx]
    llm = _supervisor_llm()

    flow_summary = _summarize_flow(state["flow_data"])

    notes_context = _read_notes_summary(state)
    if notes_context:
        notes_context = f"\n\n**Agent notes:**\n{notes_context}\n"

    response = _tracked_invoke(
        llm,
        [
            SystemMessage(
                content=(
                    "You are a Node-RED debugging assistant. Flow validation FAILED.\n"
                    "Write a SPECIFIC fix instruction for a code model.\n\n"
                    "Start with: 'FIX: ' then describe exactly what to change.\n"
                    "Be precise: node ID, property, wire target, action needed.\n"
                    "2-5 sentences."
                )
            ),
            HumanMessage(
                content=(
                    f"**Subtask:** {st.get('desc', '')}\n\n"
                    f"**Instructions given:**\n{st.get('instruction', '')}\n\n"
                    f"**Current flow:**\n{flow_summary}\n\n"
                    f"**Validation errors:**\n```\n{state['test_output']}\n```\n\n"
                    f"**Attempt {state['attempt']} of 1.**{notes_context}\n\n"
                    "Write the fix instruction:"
                )
            ),
        ],
        "diagnose",
        state,
    )

    fix = (response.content or "").strip()
    _write_note(
        state,
        "diagnose",
        f"**Subtask {idx+1}** attempt {state['attempt']+1}\n\n**Fix:** {fix}",
    )

    return {
        **state,
        "fix_instruction": fix,
        "status": "diagnosed",
    }


# ── Routing ────────────────────────────────────────────────────────────────


def route_after_test(
    state: NodeRedState,
) -> Literal["commit", "diagnose", "prep", END]:
    """After test: pass→commit→next, fail→diagnose (retry≤1)."""
    if state["status"] == "tests_passed":
        return "commit"

    if state["attempt"] >= 1:
        nxt = state["subtask_index"] + 1
        if nxt < len(state["subtasks"]):
            return "prep"
        return END

    return "diagnose"


def route_after_commit(state: NodeRedState) -> Literal["prep", END]:
    """After commit: advance to next subtask or finish."""
    nxt = state["subtask_index"] + 1
    if nxt < len(state["subtasks"]):
        return "prep"
    return END


def advance_subtask(state: NodeRedState) -> NodeRedState:
    """Advance subtask index."""
    return {
        **state,
        "subtask_index": state["subtask_index"] + 1,
        "attempt": 0,
        "fix_instruction": "",
        "status": "advanced",
    }


# ── Build graph ────────────────────────────────────────────────────────────

_builder = StateGraph(NodeRedState)

_builder.add_node("load", load_node)
_builder.add_node("orchestrate", orchestrate_node)
_builder.add_node("prep", prep_node)
_builder.add_node("edit", edit_node)
_builder.add_node("self_review", self_review_node)
_builder.add_node("test", test_node)
_builder.add_node("commit", commit_node)
_builder.add_node("diagnose", diagnose_node)
_builder.add_node("advance", advance_subtask)

_builder.add_edge(START, "load")
_builder.add_edge("load", "orchestrate")
_builder.add_edge("orchestrate", "prep")
_builder.add_edge("prep", "edit")
_builder.add_edge("edit", "self_review")
_builder.add_edge("self_review", "test")

_builder.add_conditional_edges(
    "test",
    route_after_test,
    {
        "commit": "commit",
        "diagnose": "diagnose",
        "prep": "advance",
        END: END,
    },
)

_builder.add_edge("diagnose", "edit")

_builder.add_conditional_edges(
    "commit",
    route_after_commit,
    {
        "prep": "advance",
        END: END,
    },
)

_builder.add_edge("advance", "prep")

graph = _builder.compile()
