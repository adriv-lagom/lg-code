"""
Supervisor-Worker coding agent — DELEGATED edition.

Architecture (DeepSeek minimized, worker maximized):
    ORCHESTRATE (DeepSeek: ONE call — plans subtasks AND writes instructions)
        → PREP (load files for current subtask, no LLM)
        → EDIT (vLLM: full context, multi-file output)
        → SELF-REVIEW (vLLM: checks own work, fixes obvious mistakes — free)
        → TEST
           ├─ pass → PREP (next subtask) or DONE
           └─ fail → DIAGNOSE (DeepSeek: only on failure) → EDIT (retry)

Key delegation wins:
- 🔵 DeepSeek: 1 orchestrate call + diagnose only on failure (was 3-4 per subtask)
- 🟢 vLLM: edit + self-review + fixes — catches ~50% of mistakes before test
- No per-subtask dispatch or review — worker is trusted to execute broader tasks
"""

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
WORKSPACE = ROOT / "workspace"

# ── LLM factories ──────────────────────────────────────────────────────────


def _worker_llm() -> ChatOpenAI:
    """Fast local vLLM for executing broad, well-specified edits."""
    return ChatOpenAI(
        model=os.getenv("LOCAL_MODEL", "local-active-agent"),
        base_url=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8002/v1"),
        api_key=os.getenv("LOCAL_API_KEY", "dummy"),
        temperature=0,
        timeout=120,
    )


def _supervisor_llm() -> ChatOpenAI:
    """DeepSeek for orchestration and failure diagnosis only."""
    return ChatOpenAI(
        model=os.getenv("SUPERVISOR_MODEL", "deepseek-v4-pro"),
        base_url=os.getenv("SUPERVISOR_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.getenv("SUPERVISOR_API_KEY", "dummy"),
        temperature=0,
        timeout=120,
    )


# ── State ──────────────────────────────────────────────────────────────────


class Subtask(TypedDict, total=False):
    desc: str
    instruction: str
    files: str  # comma-separated file paths the subtask touches


class SWState(TypedDict, total=False):
    task: str
    target_file: str
    file_snapshots: Annotated[dict[str, str], "current content of all relevant files"]
    subtasks: Annotated[list[Subtask], "subtasks with instructions baked in"]
    subtask_index: int
    fix_instruction: str  # DeepSeek feedback on test failure (retry)
    test_output: str
    attempt: int
    status: str
    done: bool
    token_usage: Annotated[list[dict[str, Any]], "per-call token records"]


# ── Helpers ────────────────────────────────────────────────────────────────


def _resolve_path(p: str) -> Path:
    """Resolve a relative path, ensuring it is under ROOT (not just workspace/)."""
    path = (ROOT / p).resolve()
    if not str(path).startswith(str(ROOT.resolve())):
        raise ValueError(f"Refusing to edit outside project: {p}")
    return path


def _snapshot_files(file_list: list[str]) -> dict[str, str]:
    """Read current content of each file, return {path: content}. Empty for nonexistent files."""
    snap = {}
    for f in file_list:
        f = f.strip()
        if not f:
            continue
        p = _resolve_path(f)
        snap[f] = p.read_text() if p.exists() else ""
    return snap


def _extract_multi_files(text: str) -> dict[str, str]:
    """Parse multi-file worker output:
    <file path="workspace/X.py">
    ...code...
    </file>
    Returns {path: code}. Falls back to bare <file>...</file> for single-file output.
    """
    results = {}

    # Try multi-file format: <file path="X">...</file>
    pattern = r'<file\s+path="([^"]+)"\s*>(.*?)</file>'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        for path, code in matches:
            code = _strip_fences(code.strip())
            if code:
                results[path.strip()] = code
        if results:
            return results

    # Fallback: single bare <file>...</file> — use target_file from state
    match = re.search(r"<file>\s*(.*?)\s*</file>", text, flags=re.DOTALL)
    if not match:
        match = re.search(r"<file>\s*(.*)", text, flags=re.DOTALL)
    if match:
        code = _strip_fences(match.group(1).strip())
        if code:
            results["__single__"] = code  # caller resolves path
            return results

    raise ValueError(f"Worker did not return valid <file> blocks.\nRaw:\n{text}")


def _strip_fences(code: str) -> str:
    """Strip ```python or ``` fences from code."""
    fm = re.match(r"```(?:python|py)?\s*\n(.*)\n?\s*```", code, flags=re.DOTALL)
    if fm:
        return fm.group(1).strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:python|py)?\s*\n?", "", code, count=1)
        code = re.sub(r"\n?\s*```\s*$", "", code)
    return code.strip()


def _tracked_invoke(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    role: str,
    state: SWState,
) -> Any:
    """Invoke an LLM and record token usage."""
    response = llm.invoke(messages)
    try:
        meta = getattr(response, "response_metadata", {}) or {}
        tu = meta.get("token_usage", {})
        usage = {
            "role": role,
            "model": llm.model_name,
            "prompt_tokens": tu.get("prompt_tokens", 0),
            "completion_tokens": tu.get("completion_tokens", 0),
            "total_tokens": tu.get("total_tokens", 0),
        }
    except Exception:
        usage = {"role": role, "model": llm.model_name, "error": "could not extract"}
    state.setdefault("token_usage", []).append(usage)
    return response


# ── Nodes ──────────────────────────────────────────────────────────────────


def load_node(state: SWState) -> SWState:
    """Load initial file(s) into state."""
    target = state.get("target_file", "workspace/math_utils.py")
    test_file = "tests/test_math_utils.py"
    snaps = _snapshot_files([target, test_file])
    return {
        **state,
        "target_file": target,
        "file_snapshots": snaps,
        "subtask_index": 0,
        "attempt": 0,
        "status": "loaded",
    }


def orchestrate_node(state: SWState) -> SWState:
    """ONE DeepSeek call: plan subtasks AND write worker instructions together."""
    llm = _supervisor_llm()

    files_block = "\n\n".join(
        f"File `{p}`:\n```python\n{content or '(empty/new)'}\n```"
        for p, content in sorted(state["file_snapshots"].items())
    )

    response = _tracked_invoke(llm, [
        SystemMessage(content=(
            "You orchestrate a coding agent. You will produce a plan of subtasks "
            "with instructions, in ONE structured response.\n\n"
            "FORMAT — exactly ONE blank line between subtask blocks:\n"
            "---SUBTASK---\n"
            "Description: what to accomplish (one sentence)\n"
            "Instruction: precise instructions for a 3B code model. Start with "
            "'PRESERVE all existing code.'. Be specific about file paths, "
            "function names, types, and expected behavior. 2-5 sentences.\n"
            "Files: comma-separated paths to edit (e.g. workspace/X.py, tests/test_Y.py)\n\n"
            "RULES:\n"
            "- 1-3 subtasks total\n"
            "- Each subtask can touch 1-2 files\n"
            "- Test files live in tests/, source files in workspace/\n"
            "- The worker is a 3B code model: be SPECIFIC about what to write"
        )),
        HumanMessage(content=(
            f"Task: {state['task']}\n\n"
            f"Existing files:\n{files_block}\n\n"
            "Write the orchestration plan:"
        )),
    ], "orchestrate", state)

    subtasks = _parse_orchestration(response.content.strip())
    if not subtasks:
        subtasks = [Subtask(desc=state["task"], instruction=state["task"],
                            files=state["target_file"])]

    return {
        **state,
        "subtasks": subtasks,
        "subtask_index": 0,
        "attempt": 0,
        "status": "orchestrated",
    }


def _parse_orchestration(text: str) -> list[Subtask]:
    """Parse ---SUBTASK--- blocks into Subtask dicts."""
    blocks = re.split(r"\n?---SUBTASK---\n?", text)
    subtasks = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        desc = _extract_field(block, "Description")
        instruction = _extract_field(block, "Instruction")
        files = _extract_field(block, "Files")
        if desc or instruction:
            subtasks.append(Subtask(
                desc=desc or instruction or block[:80],
                instruction=instruction or desc or block[:200],
                files=files or "",
            ))
    return subtasks


def _extract_field(block: str, field: str) -> str:
    m = re.search(rf"^{field}:\s*(.+?)(?:\n(?:Description|Instruction|Files):|\Z)",
                  block, flags=re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


def prep_node(state: SWState) -> SWState:
    """Load file snapshots for the current subtask. No LLM call."""
    idx = state["subtask_index"]
    st = state["subtasks"][idx]
    files_str = st.get("files", state["target_file"])
    file_list = [f.strip() for f in files_str.split(",") if f.strip()]
    if not file_list:
        file_list = [state["target_file"]]
    snaps = _snapshot_files(file_list)
    # Merge with existing snapshots
    all_snaps = {**state.get("file_snapshots", {}), **snaps}
    return {
        **state,
        "file_snapshots": all_snaps,
        "target_file": file_list[0],
        "attempt": 0,
        "fix_instruction": "",
        "status": "prepped",
    }


def edit_node(state: SWState) -> SWState:
    """vLLM worker: full context, multi-file output. Trusted with broader tasks."""
    idx = state["subtask_index"]
    st = state["subtasks"][idx]
    llm = _worker_llm()

    # Build rich context: task, subtask, instruction, fix feedback, ALL relevant files
    files_block = "\n\n".join(
        f"File `{p}`:\n```\n{content or '(empty/new)'}\n```"
        for p, content in sorted(state["file_snapshots"].items())
    )

    fix_context = ""
    if state.get("fix_instruction"):
        fix_context = (
            f"\n\n⚠️ PREVIOUS ATTEMPT FAILED. Fix instruction from reviewer:\n"
            f"{state['fix_instruction']}\n"
        )

    response = _tracked_invoke(llm, [
        SystemMessage(content=(
            "You are a coding worker. You receive a subtask with full context "
            "and must edit one or more files.\n\n"
            "FORMAT — output file blocks using:\n"
            "<file path=\"workspace/X.py\">\n"
            "RAW Python code here — no markdown, no ``` fences\n"
            "</file>\n"
            "<file path=\"tests/test_Y.py\">\n"
            "RAW Python code here\n"
            "</file>\n\n"
            "RULES:\n"
            "1. PRESERVE ALL EXISTING CODE unless instructed to change it\n"
            "2. Always include complete file contents\n"
            "3. NO markdown fences, NO explanations, NO thinking aloud\n"
            "4. Output ONLY <file path=\"...\">...</file> blocks"
        )),
        HumanMessage(content=(
            f"**Overall task:** {state['task']}\n\n"
            f"**Subtask {idx+1}/{len(state['subtasks'])}:** {st.get('desc', '')}\n\n"
            f"**Instructions:** {st.get('instruction', '')}"
            f"{fix_context}\n\n"
            f"**Existing files:**\n{files_block}\n\n"
            "Return the corrected files using <file path=\"...\">...</file> blocks:"
        )),
    ], "edit", state)

    edits = _extract_multi_files(response.content)

    # Write each file
    for path, code in edits.items():
        if path == "__single__":
            path = state["target_file"]
        if "..." in code:
            raise ValueError(f"Worker returned placeholder in {path}:\n{code}")
        if len(code.strip()) < 5:
            raise ValueError(f"Worker returned near-empty file for {path}:\n{code}")

        p = _resolve_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code)
        state["file_snapshots"][path] = code

    # Update current_code for backward compat
    primary_code = state["file_snapshots"].get(state["target_file"], "")

    return {
        **state,
        "current_code": primary_code,
        "attempt": state["attempt"] + 1,
        "status": "edited",
    }


def self_review_node(state: SWState) -> SWState:
    """🟢 vLLM self-review: checks own output for obvious mistakes. Free, fast."""
    llm = _worker_llm()

    files_block = "\n\n".join(
        f"File `{p}`:\n```\n{content or '(empty/new)'}\n```"
        for p, content in sorted(state["file_snapshots"].items())
    )

    idx = state["subtask_index"]
    st = state["subtasks"][idx]

    response = _tracked_invoke(llm, [
        SystemMessage(content=(
            "You are a code reviewer. You just wrote some code and now you must "
            "CHECK IT for mistakes. Look for:\n"
            "1. Placeholder text (..., TODO, pass, 'your code here')\n"
            "2. Missing functions that were supposed to be added\n"
            "3. Existing functions accidentally deleted or changed\n"
            "4. Syntax errors, missing imports, wrong indentation\n"
            "5. Markdown fences (```) inside file blocks\n\n"
            "If everything looks correct, output the files EXACTLY as-is.\n"
            "If you find issues, output CORRECTED versions.\n\n"
            "FORMAT — same as before:\n"
            "<file path=\"workspace/X.py\">\n"
            "RAW Python code\n"
            "</file>\n"
            "<file path=\"tests/test_Y.py\">\n"
            "RAW Python code\n"
            "</file>\n\n"
            "RULES:\n"
            "1. Always output COMPLETE file contents\n"
            "2. NO markdown fences, NO explanations\n"
            "3. Output ONLY <file path=\"...\">...</file> blocks"
        )),
        HumanMessage(content=(
            f"**Task:** {state['task']}\n\n"
            f"**Subtask:** {st.get('desc', '')}\n\n"
            f"**Instructions I was given:** {st.get('instruction', '')}\n\n"
            f"**My output to review:**\n{files_block}\n\n"
            "Review and output corrected files (or same files if correct):"
        )),
    ], "self_review", state)

    # Parse the review — if the model echoes back valid <file> blocks, use them.
    # If it produces garbage or just commentary, silently pass through.
    try:
        edits = _extract_multi_files(response.content)

        for path, code in edits.items():
            if path == "__single__":
                path = state["target_file"]
            if "..." in code:
                continue
            if len(code.strip()) < 5:
                continue

            p = _resolve_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(code)
            state["file_snapshots"][path] = code
    except ValueError:
        pass  # self-review didn't produce valid file blocks — accept edit as-is

    primary_code = state["file_snapshots"].get(state["target_file"], "")

    return {
        **state,
        "current_code": primary_code,
        "status": "self_reviewed",
    }


def test_node(state: SWState) -> SWState:
    """Run pytest."""
    proc = subprocess.run(
        ["pytest", "-q"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    passed = proc.returncode == 0
    return {
        **state,
        "test_output": proc.stdout,
        "status": "tests_passed" if passed else "tests_failed",
    }


def diagnose_node(state: SWState) -> SWState:
    """DeepSeek: ONLY called when tests fail. Diagnoses and writes a fix instruction."""
    idx = state["subtask_index"]
    st = state["subtasks"][idx]
    llm = _supervisor_llm()

    response = _tracked_invoke(llm, [
        SystemMessage(content=(
            "You are a debugging assistant. Tests have FAILED. Look at the test "
            "output and the current code, then write a SPECIFIC fix instruction "
            "for a 3B code model.\n\n"
            "Start with: 'FIX: ' then describe exactly what to change.\n"
            "2-4 sentences. Be precise about file paths, line changes, and "
            "expected behavior."
        )),
        HumanMessage(content=(
            f"Subtask: {st.get('desc', '')}\n\n"
            f"Instructions given to worker:\n{st.get('instruction', '')}\n\n"
            f"Current code:\n"
            + "\n".join(f"  {p}:\n```\n{c}\n```"
                        for p, c in sorted(state["file_snapshots"].items()))
            + f"\n\nTest output:\n```\n{state['test_output']}\n```\n\n"
            f"Attempt {state['attempt']} of 1.\n\n"
            "Write the fix instruction:"
        )),
    ], "diagnose", state)

    return {
        **state,
        "fix_instruction": (response.content or "").strip(),
        "status": "diagnosed",
    }


def route_after_test(state: SWState) -> Literal["prep", "diagnose", END]:
    """Route after test: pass→next, fail→diagnose (with retry limit)."""
    if state["status"] == "tests_passed":
        # Advance to next subtask
        nxt = state["subtask_index"] + 1
        if nxt >= len(state["subtasks"]):
            return END
        return "prep"

    # Tests failed
    if state["attempt"] >= 1:
        # Max retries — skip this subtask
        nxt = state["subtask_index"] + 1
        if nxt >= len(state["subtasks"]):
            return END
        return "prep"

    return "diagnose"


def advance_subtask(state: SWState) -> SWState:
    """Move to next subtask."""
    return {
        **state,
        "subtask_index": state["subtask_index"] + 1,
        "attempt": 0,
        "fix_instruction": "",
        "status": "advanced",
    }


# ── Build graph ────────────────────────────────────────────────────────────

builder = StateGraph(SWState)

builder.add_node("load", load_node)
builder.add_node("orchestrate", orchestrate_node)
builder.add_node("prep", prep_node)
builder.add_node("edit", edit_node)
builder.add_node("self_review", self_review_node)
builder.add_node("test", test_node)
builder.add_node("diagnose", diagnose_node)
builder.add_node("advance", advance_subtask)

builder.add_edge(START, "load")
builder.add_edge("load", "orchestrate")
builder.add_edge("orchestrate", "prep")  # prep first subtask
builder.add_edge("prep", "edit")
builder.add_edge("edit", "self_review")
builder.add_edge("self_review", "test")

# After test: pass → next subtask or done; fail → diagnose → edit (retry)
builder.add_conditional_edges("test", route_after_test, {
    "prep": "advance",
    "diagnose": "diagnose",
    END: END,
})

builder.add_edge("diagnose", "edit")       # retry with fix instruction
builder.add_edge("advance", "prep")        # load next subtask

graph = builder.compile()
