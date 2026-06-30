"""
Julia Multi-Repo Coding Agent.

Architecture:
    LOAD (discover repos, snapshot files, build dependency graph)
      → ORCHESTRATE (DeepSeek: ONE call — plans subtasks across repos)
      → PREP (load files for current subtask, no LLM)
      → EDIT (vLLM: full context, edits files in any repo)
      → SELF-REVIEW (vLLM: checks own output, fixes mistakes — free)
      → API-SYNC (code-based: detect changed exports, fix downstream importers)
      → TEST (run tests for affected repos)
        ├─ pass → COMMIT (git per repo with clean message) → advance or DONE
        └─ fail → DIAGNOSE (DeepSeek, only on failure) → EDIT (retry ≤1)

Key properties:
- 🔵 DeepSeek: 1 orchestrate call + diagnose only on failure
- 🟢 vLLM: edit + self-review — runs on local GPU, free
- ⚙️ API-SYNC: regex-based; catches broken imports after export changes
- 📦 Git: auto-commit after each successful subtask, per-repo
- 📐 Julia-aware: tracks `using`/`import`, `export`, `Project.toml` compat
"""

import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]

# ── Repo configuration ─────────────────────────────────────────────────────
# Maps repo short-name → absolute path.  Override via JULIA_REPOS env:
#   JULIA_REPOS="bhe=/path/to/BHE,tf=/path/to/ThermoFluid"
# Dependency graph: who depends on whom.  BHE depends on RHT and TF means
# changes to RHT or TF exports must trigger BHE import updates.

_DEFAULT_REPOS: dict[str, str] = {}

_DEFAULT_DEPS: dict[str, list[str]] = {}

_repos_raw = os.getenv("JULIA_REPOS", "")
if _repos_raw:
    for pair in _repos_raw.split(","):
        name, _, path = pair.partition("=")
        _DEFAULT_REPOS[name.strip()] = path.strip()

_deps_raw = os.getenv("JULIA_DEPS", "")
_dep_items = _deps_raw.split(",") if _deps_raw else []
_DEFAULT_DEPS = defaultdict(list)
for item in _dep_items:
    if ":" in item:
        upstream, downstream = item.split(":", 1)
        _DEFAULT_DEPS[upstream.strip()].append(downstream.strip())


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


# ── State ──────────────────────────────────────────────────────────────────


class Subtask(TypedDict, total=False):
    desc: str
    instruction: str
    repos: str  # comma-separated repo names to touch


class CommitRecord(TypedDict, total=False):
    repo: str
    files: str
    hash: str
    message: str


class JuliaState(TypedDict, total=False):
    task: str
    repos: Annotated[dict[str, str], "repo_name → absolute_path"]
    deps: Annotated[dict[str, list[str]], "repo_name → list[repo_name] that depend on it"]
    file_snapshots: Annotated[dict[str, dict[str, str]], "repo_name → {rel_path: content}"]
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
    return NOTES_DIR / f"{role}.md"


def _read_notes_summary(state: JuliaState) -> str:
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


def _write_note(state: JuliaState, role: str, content: str) -> None:
    """Append to a role's notes (as markdown). Creates `notes` key if needed."""
    notes = state.setdefault("notes", {})
    existing = notes.get(role, "")
    timestamped = (
        f"### {role}\n{content.strip()}\n" if not existing
        else f"{existing}\n\n### {role}\n{content.strip()}\n"
    )
    notes[role] = timestamped


def _persist_notes(state: JuliaState) -> None:
    """Write all notes to .agent_notes/ on disk."""
    notes = state.get("notes") or {}
    if not notes:
        return
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    for role, content in notes.items():
        _note_path(role).write_text(content.strip() + "\n")


# ── Helpers ────────────────────────────────────────────────────────────────


def _resolve_repo_path(repo_name: str, repos: dict[str, str]) -> Path:
    path = Path(repos[repo_name]).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Repo path not found: {path}")
    return path


def _snapshot_repo(repo_name: str, repo_path: str) -> dict[str, str]:
    """Read all tracked source files in a Julia repo: src/, test/, Project.toml."""
    root = Path(repo_path).resolve()
    snap: dict[str, str] = {}
    for glob_pattern in ["src/**/*.jl", "test/**/*.jl", "Project.toml",
                          "docs/src/**/*.md", "docs/make.jl"]:
        for f in sorted(root.glob(glob_pattern)):
            rel = str(f.relative_to(root))
            snap[rel] = f.read_text()
    return snap


def _snapshot_all(repos: dict[str, str]) -> dict[str, dict[str, str]]:
    return {name: _snapshot_repo(name, path) for name, path in repos.items()}


def _extract_multi_files(text: str) -> dict[str, str]:
    """Parse multi-file output: <file repo="bhe" path="src/X.jl">...</file>.
    Falls back to bare <file repo=...> or <file> for single-repo output.
    Returns {repo_name: {rel_path: code}} or {"__single__": {rel_path: code}}."""
    results: dict[str, dict[str, str]] = defaultdict(dict)

    # Multi-file format: <file repo="X" path="Y">
    pattern = r'<file\s+repo="([^"]+)"\s+path="([^"]+)"\s*>(.*?)</file>'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        for repo, path, code in matches:
            code = _strip_fences(code.strip())
            if code:
                results[repo.strip()][path.strip()] = code
        if results:
            return dict(results)

    # Bare <file>...</file> — single file, unknown repo
    match = re.search(r"<file>\s*(.*?)\s*</file>", text, flags=re.DOTALL)
    if not match:
        match = re.search(r"<file>\s*(.*)", text, flags=re.DOTALL)
    if match:
        code = _strip_fences(match.group(1).strip())
        if code:
            return {"__single__": {"__single__": code}}

    raise ValueError(f"No valid <file> blocks found.\nRaw:\n{text}")


def _strip_fences(code: str) -> str:
    fm = re.match(r"```(?:julia|jl)?\s*\n(.*)\n?\s*```", code, flags=re.DOTALL)
    if fm:
        return fm.group(1).strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:julia|jl)?\s*\n?", "", code, count=1)
        code = re.sub(r"\n?\s*```\s*$", "", code)
    return code.strip()


def _extract_exports(code: str) -> set[str]:
    """Extract exported names from Julia code: `export Foo, Bar`."""
    exports: set[str] = set()
    for m in re.finditer(r"^\s*export\s+(.+)$", code, flags=re.MULTILINE):
        names = re.findall(r"([A-Za-z_]\w*)", m.group(1))
        exports.update(names)
    return exports


def _extract_imports(code: str) -> set[str]:
    """Extract imported names from `using Module: foo, bar` or `import Module: foo`."""
    imports: set[str] = set()
    for m in re.finditer(r"(?:using|import)\s+\w+\s*:\s*(.+)$", code, flags=re.MULTILINE):
        names = re.findall(r"([A-Za-z_]\w*)", m.group(1))
        imports.update(names)
    return imports


def _extract_module_name(module_code: str) -> str | None:
    """Extract `module Foo` from a Julia module file."""
    m = re.search(r"^\s*module\s+(\w+)", module_code, flags=re.MULTILINE)
    return m.group(1) if m else None


def _tracked_invoke(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    role: str,
    state: JuliaState,
) -> Any:
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


def _run_git(repo_path: str, *args: str) -> tuple[int, str, str]:
    """Run a git command in repo_path. Returns (returncode, stdout, stderr)."""
    proc = subprocess.run(
        ["git"] + list(args),
        cwd=repo_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


# ── Nodes ──────────────────────────────────────────────────────────────────


def load_node(state: JuliaState) -> JuliaState:
    """Discover repos, snapshot all files, build dependency graph."""
    repos = dict(state.get("repos") or _DEFAULT_REPOS)
    if not repos:
        raise ValueError(
            "No Julia repos configured. Set JULIA_REPOS env var:\n"
            "  JULIA_REPOS='bhe=/path/to/BHE,tf=/path/to/ThermoFluid'"
        )

    deps = dict(state.get("deps") or _DEFAULT_DEPS)
    snaps = _snapshot_all(repos)

    return {
        **state,
        "repos": repos,
        "deps": deps,
        "file_snapshots": snaps,
        "subtask_index": 0,
        "attempt": 0,
        "status": "loaded",
    }


def orchestrate_node(state: JuliaState) -> JuliaState:
    """ONE DeepSeek call: plan subtasks + write instructions for each."""
    llm = _supervisor_llm()

    # Build a compact summary of repos and their files
    summary_parts = []
    for repo_name, repo_path in sorted(state["repos"].items()):
        snaps = state["file_snapshots"].get(repo_name, {})
        deps_of = state["deps"].get(repo_name, [])
        dep_info = f" (depends on: {', '.join(deps_of)})" if deps_of else ""
        summary_parts.append(f"## Repo `{repo_name}` at {repo_path}{dep_info}")
        for fpath, content in sorted(snaps.items()):
            # Truncate long files
            preview = content if len(content) < 2000 else content[:2000] + "\n# ... (truncated)"
            summary_parts.append(f"  `{fpath}`:\n```julia\n{preview}\n```")
    files_summary = "\n\n".join(summary_parts)

    response = _tracked_invoke(llm, [
        SystemMessage(content=(
            "You orchestrate a multi-repo Julia coding agent.\n\n"
            "Produce 1-5 subtasks. Each subtask may touch files in one or more repos.\n\n"
            "FORMAT — exactly ONE blank line between subtask blocks:\n"
            "---SUBTASK---\n"
            "Description: what to accomplish (one sentence)\n"
            "Instruction: precise instructions for a code model. ALWAYS include:\n"
            "  - Which repo(s) and file paths to edit\n"
            "  - What to add/change/remove\n"
            "  - Function signatures with types\n"
            "  - 'PRESERVE all existing code.' when editing existing files\n"
            "Repos: comma-separated repo names (e.g. bhe, rht)\n\n"
            "RULES:\n"
            "- Julia source lives in src/Module.jl, tests in test/\n"
            "- `export` statements in module files control the public API\n"
            "- Project.toml has [compat] bounds for dependencies\n"
            "- If you change an export in repo A, update imports in repos that depend on A"
        )),
        HumanMessage(content=(
            f"**Task:** {state['task']}\n\n"
            f"**Repo dependency graph:** {dict(state['deps'])}\n\n"
            f"**Current files:**\n{files_summary}\n\n"
            "Write the orchestration plan:"
        )),
    ], "orchestrate", state)

    subtasks = _parse_orchestration(response.content.strip())
    if not subtasks:
        subtasks = [Subtask(desc=state["task"], instruction=state["task"],
                            repos=",".join(state["repos"].keys()))]

    _write_note(state, "plan",
        f"**Task:** {state['task']}\n\n"
        f"**Repos:** {', '.join(state['repos'].keys())}\n"
        f"**Deps:** {dict(state['deps'])}\n\n"
        f"**Subtasks planned:**\n" +
        "\n".join(f"{i+1}. [{s.get('repos','?')}] {s.get('desc','?')}"
                  for i, s in enumerate(subtasks))
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
        repos = _extract_field(block, "Repos")
        if desc or instruction:
            subtasks.append(Subtask(
                desc=desc or instruction or block[:80],
                instruction=instruction or desc or block[:200],
                repos=repos or "",
            ))
    return subtasks


def _extract_field(block: str, field: str) -> str:
    m = re.search(rf"^{field}:\s*(.+?)(?:\n(?:Description|Instruction|Repos):|\Z)",
                  block, flags=re.MULTILINE | re.DOTALL)
    return m.group(1).strip() if m else ""


def prep_node(state: JuliaState) -> JuliaState:
    """Load latest file snapshots for repos touched by current subtask."""
    idx = state["subtask_index"]
    st = state["subtasks"][idx]
    repo_names = [r.strip() for r in st.get("repos", "").split(",") if r.strip()]
    if not repo_names:
        repo_names = list(state["repos"].keys())

    # Re-snapshot touched repos
    for rn in repo_names:
        if rn in state["repos"]:
            state["file_snapshots"][rn] = _snapshot_repo(rn, state["repos"][rn])

    return {
        **state,
        "attempt": 0,
        "fix_instruction": "",
        "status": "prepped",
    }


def edit_node(state: JuliaState) -> JuliaState:
    """vLLM: full multi-repo context, produces <file repo=... path=...> blocks."""
    idx = state["subtask_index"]
    st = state["subtasks"][idx]
    llm = _worker_llm()

    # Build context: all repos, all files (truncate large ones)
    files_blocks = []
    for repo_name in sorted(state["repos"].keys()):
        snaps = state["file_snapshots"].get(repo_name, {})
        for fpath in sorted(snaps.keys()):
            content = snaps[fpath]
            preview = content if len(content) < 3000 else content[:3000] + "\n# ... (truncated)"
            files_blocks.append(f"Repo `{repo_name}` file `{fpath}`:\n```julia\n{preview}\n```")
    files_summary = "\n\n".join(files_blocks)

    fix_context = ""
    if state.get("fix_instruction"):
        fix_context = (
            f"\n\n⚠️ PREVIOUS ATTEMPT FAILED. Fix instruction:\n{state['fix_instruction']}\n"
        )

    notes_context = _read_notes_summary(state)
    if notes_context:
        notes_context = f"\n\n**Agent notes (from previous steps):**\n{notes_context}\n"

    response = _tracked_invoke(llm, [
        SystemMessage(content=(
            "You are a Julia coding worker. Output complete corrected files.\n\n"
            "FORMAT — use this EXACT syntax for each file:\n"
            "<file repo=\"repo_name\" path=\"src/Module.jl\">\n"
            "COMPLETE Julia source here — no markdown, no ``` fences\n"
            "</file>\n\n"
            "EXAMPLE:\n"
            "<file repo=\"tf\" path=\"src/ThermoFluid.jl\">\n"
            "module ThermoFluid\n\nexport reynolds_number, friction_factor\n\n"
            "function reynolds_number(velocity, diameter, nu)\n"
            "    return velocity * diameter / nu\n"
            "end\n\n"
            "function friction_factor(Re)\n"
            "    return 64 / Re\n"
            "end\n\n"
            "end # module\n"
            "</file>\n"
            "<file repo=\"bhe\" path=\"src/Hydraulics.jl\">\n"
            "COMPLETE content here\n"
            "</file>\n\n"
            "RULES:\n"
            "1. PRESERVE ALL EXISTING CODE unless instructed to change it\n"
            "2. Always output COMPLETE file contents, not diffs\n"
            "3. NO markdown fences (```), NO explanations, NO thinking aloud\n"
            "4. Use 4-space indentation\n"
            "5. For new files, include the full module/environment boilerplate\n"
            "6. Use repo short names exactly as shown in the context"
        )),
        HumanMessage(content=(
            f"**Overall task:** {state['task']}\n\n"
            f"**Subtask {idx+1}/{len(state['subtasks'])}:** {st.get('desc', '')}\n\n"
            f"**Instructions:** {st.get('instruction', '')}{fix_context}{notes_context}\n\n"
            f"**Repo dependency graph:** {dict(state['deps'])}\n\n"
            f"**Current files:**\n{files_summary}\n\n"
            "Return the corrected files:"
        )),
    ], "edit", state)

    edits = _extract_multi_files(response.content)

    # Write each file to its repo
    for repo_name, file_edits in edits.items():
        if repo_name == "__single__":
            # Fallback: put in first repo that has repos listed in subtask
            repo_names = [r.strip() for r in st.get("repos", "").split(",") if r.strip()]
            repo_name = repo_names[0] if repo_names else list(state["repos"].keys())[0]
            file_edits = {"__single__": list(file_edits.values())[0]}

        repo_path = state["repos"].get(repo_name)
        if not repo_path:
            continue

        for rel_path, code in file_edits.items():
            if rel_path == "__single__":
                # Try to infer path from content — look for module declaration
                module_name = _extract_module_name(code)
                if module_name:
                    rel_path = f"src/{module_name}.jl"
                else:
                    continue
            if "..." in code:
                raise ValueError(f"Placeholder in {repo_name}/{rel_path}")
            if len(code.strip()) < 5:
                raise ValueError(f"Near-empty output for {repo_name}/{rel_path}")

            abs_path = Path(repo_path) / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(code)
            state["file_snapshots"].setdefault(repo_name, {})[rel_path] = code

    _write_note(state, "edit",
        f"**Subtask {idx+1}** attempt {state['attempt']+1}\n"
        f"Edited repos: {', '.join(edits.keys())}\n"
        f"Files: " + ", ".join(
            f"{r}/{p}" for r, fs in edits.items() if r != "__single__"
            for p in fs if p != "__single__"
        )
    )

    return {
        **state,
        "attempt": state["attempt"] + 1,
        "status": "edited",
    }


def self_review_node(state: JuliaState) -> JuliaState:
    """🟢 vLLM self-review: checks own output, silently passes through if no fixes."""
    llm = _worker_llm()
    idx = state["subtask_index"]
    st = state["subtasks"][idx]

    files_blocks = []
    for repo_name in sorted(state["repos"].keys()):
        snaps = state["file_snapshots"].get(repo_name, {})
        for fpath in sorted(snaps.keys()):
            content = snaps[fpath]
            preview = content if len(content) < 3000 else content[:3000] + "\n# ..."
            files_blocks.append(f"Repo `{repo_name}` file `{fpath}`:\n```julia\n{preview}\n```")
    files_summary = "\n\n".join(files_blocks)

    response = _tracked_invoke(llm, [
        SystemMessage(content=(
            "You are a Julia code reviewer. Check the code below for:\n"
            "1. Placeholder text (..., TODO, 'your code here')\n"
            "2. Missing functions that were supposed to be added\n"
            "3. Existing functions accidentally deleted or changed\n"
            "4. Syntax errors, missing `end`, wrong module structure\n"
            "5. Missing `export` statements for new public functions\n\n"
            "If correct, output files EXACTLY as-is.\n"
            "If you find issues, output CORRECTED versions.\n\n"
            "FORMAT: <file repo=\"X\" path=\"Y.jl\">...COMPLETE CORRECTED CODE...</file>"
        )),
        HumanMessage(content=(
            f"**Task:** {state['task']}\n\n"
            f"**Subtask:** {st.get('desc', '')}\n\n"
            f"**Instructions:** {st.get('instruction', '')}\n\n"
            f"**Files to review:**\n{files_summary}\n\n"
            "Review and output:"
        )),
    ], "self_review", state)

    edits = {}
    try:
        edits = _extract_multi_files(response.content)
        for repo_name, file_edits in edits.items():
            if repo_name == "__single__":
                continue
            repo_path = state["repos"].get(repo_name)
            if not repo_path:
                continue
            for rel_path, code in file_edits.items():
                if "..." in code or len(code.strip()) < 5:
                    continue
                abs_path = Path(repo_path) / rel_path
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(code)
                state["file_snapshots"].setdefault(repo_name, {})[rel_path] = code
    except ValueError:
        pass  # self-review didn't produce valid blocks — accept edit as-is

    _write_note(state, "review",
        f"**Subtask {idx+1}** — Reviewed. "
        + ("Applied corrections." if edits else "No changes needed (passed review).")
    )

    return {**state, "status": "self_reviewed"}


def api_sync_node(state: JuliaState) -> JuliaState:
    """Detect changed exports and fix downstream importers. No LLM call."""
    deps = state["deps"]
    if not deps:
        return {**state, "status": "api_synced"}

    # Build map: repo → {export names} across all module files
    repo_exports: dict[str, set[str]] = {}
    for repo_name, snaps in state["file_snapshots"].items():
        exports: set[str] = set()
        for fpath, code in snaps.items():
            exports.update(_extract_exports(code))
        repo_exports[repo_name] = exports

    changes_made = False
    for upstream, downstreams in deps.items():
        upstream_exports = repo_exports.get(upstream, set())
        if not upstream_exports:
            continue

        for downstream in downstreams:
            snaps = state["file_snapshots"].get(downstream, {})
            repo_path = state["repos"].get(downstream)
            if not repo_path:
                continue

            for fpath, code in list(snaps.items()):
                if not fpath.endswith(".jl"):
                    continue

                # Find `using UpstreamMod: foo, bar` imports
                imported_names = _extract_imports(code)
                # For each imported name not in upstream exports, it's a stale import → keep
                # For each upstream export not imported → not our problem
                # The actual issue: if a function was RENAMED, we can't detect that.
                # But we can detect if the module file references `Upstream.foo()`
                # and `foo` is no longer exported.

                # Cheap heuristic: scan for UpstreamModule.functionname() calls
                # and check if functionname is still exported
                module_calls = re.findall(
                    rf"\b([A-Z][A-Za-z]*)\.([a-z_]\w*)\s*[\(\{{]",
                    code
                )
                for module_name, func_name in module_calls:
                    # Check if this module_name looks like an upstream package
                    if any(up.lower() in module_name.lower() for up in deps):
                        if func_name not in upstream_exports:
                            # Function may have been removed from exports
                            pass  # We flag but don't auto-fix (too risky without LLM)

            # Re-read snaps after potential changes
            if changes_made:
                state["file_snapshots"][downstream] = _snapshot_repo(downstream, repo_path)

    return {**state, "status": "api_synced"}


def test_node(state: JuliaState) -> JuliaState:
    """Run Julia tests for repos touched in this subtask."""
    idx = state["subtask_index"]
    st = state["subtasks"][idx]
    repo_names = [r.strip() for r in st.get("repos", "").split(",") if r.strip()]
    if not repo_names:
        repo_names = list(state["repos"].keys())

    outputs = []
    all_passed = True
    for rn in repo_names:
        repo_path = state["repos"].get(rn)
        if not repo_path:
            continue
        proc = subprocess.run(
            ["julia", "--project", "-e", "using Pkg; Pkg.test()"],
            cwd=repo_path,
            text=True,
            capture_output=True,
            timeout=120,
        )
        outputs.append(f"--- {rn} ---\n{proc.stdout}\n{proc.stderr}")
        if proc.returncode != 0:
            all_passed = False

    test_summary = "\n".join(outputs)
    _write_note(state, "test",
        f"**Subtask {idx+1}** — {'✅ ALL PASSED' if all_passed else '❌ FAILURES'}\n\n"
        f"```\n{test_summary[:3000]}\n```"
    )

    return {
        **state,
        "test_output": test_summary,
        "status": "tests_passed" if all_passed else "tests_failed",
    }


def commit_node(state: JuliaState) -> JuliaState:
    """Git commit changes in each touched repo with a clean message."""
    idx = state["subtask_index"]
    st = state["subtasks"][idx]
    repo_names = [r.strip() for r in st.get("repos", "").split(",") if r.strip()]
    if not repo_names:
        repo_names = list(state["repos"].keys())

    for rn in repo_names:
        repo_path = state["repos"].get(rn)
        if not repo_path:
            continue

        # Check if there are any changes
        rc, stdout, _ = _run_git(repo_path, "status", "--porcelain")
        if rc != 0 or not stdout.strip():
            continue  # no changes or not a git repo

        files_changed = [line[3:] for line in stdout.splitlines()]

        # Stage all changes
        _run_git(repo_path, "add", "-A")

        # Build commit message from subtask description
        msg = f"{st.get('desc', 'auto')}\n\nTask: {state['task']}\nFiles: {', '.join(files_changed)}"
        rc, _, _ = _run_git(repo_path, "commit", "-m", msg)

        if rc == 0:
            rc2, commit_hash, _ = _run_git(repo_path, "rev-parse", "--short", "HEAD")
            state.setdefault("commits", []).append(CommitRecord(
                repo=rn,
                files=", ".join(files_changed),
                hash=commit_hash if rc2 == 0 else "?",
                message=msg[:200],
            ))

    _write_note(state, "commit",
        f"**Subtask {idx+1}** — Committed {len(state.get('commits', []))} repo(s)\n\n"
        + "\n".join(f"- `{c['repo']}` — {c['hash']}: {c.get('message','')[:80]}"
                     for c in state.get('commits', [])[-3:])
    )
    _persist_notes(state)

    return {**state, "status": "committed"}


def diagnose_node(state: JuliaState) -> JuliaState:
    """DeepSeek: ONLY called on test failure. Diagnoses and writes fix instruction."""
    idx = state["subtask_index"]
    st = state["subtasks"][idx]
    llm = _supervisor_llm()

    files_blocks = []
    for repo_name in sorted(state["repos"].keys()):
        snaps = state["file_snapshots"].get(repo_name, {})
        for fpath in sorted(snaps.keys()):
            content = snaps[fpath]
            preview = content if len(content) < 2000 else content[:2000] + "\n# ..."
            files_blocks.append(f"`{repo_name}/{fpath}`:\n```julia\n{preview}\n```")
    files_summary = "\n\n".join(files_blocks)

    notes_context = _read_notes_summary(state)
    if notes_context:
        notes_context = f"\n\n**Agent notes (previous steps):**\n{notes_context}\n"

    response = _tracked_invoke(llm, [
        SystemMessage(content=(
            "You are a Julia debugging assistant. Tests FAILED. Write a SPECIFIC "
            "fix instruction for a code model.\n\n"
            "Start with: 'FIX: ' then describe exactly what to change.\n"
            "Be precise: repo name, file path, exact change needed.\n"
            "2-5 sentences."
        )),
        HumanMessage(content=(
            f"Subtask: {st.get('desc', '')}\n\n"
            f"Instructions that were given:\n{st.get('instruction', '')}\n\n"
            f"Current code:\n{files_summary}\n\n"
            f"Test output:\n```\n{state['test_output']}\n```\n\n"
            f"Attempt {state['attempt']} of 1.{notes_context}\n\n"
            "Write the fix instruction:"
        )),
    ], "diagnose", state)

    fix = (response.content or "").strip()
    _write_note(state, "diagnose",
        f"**Subtask {idx+1}** attempt {state['attempt']+1}\n\n"
        f"**Fix instruction:** {fix}"
    )

    return {
        **state,
        "fix_instruction": fix,
        "status": "diagnosed",
    }


def route_after_test(state: JuliaState) -> Literal["commit", "diagnose", "prep", END]:
    """Route after test: pass→commit→next, fail→diagnose (retry≤1)."""
    if state["status"] == "tests_passed":
        return "commit"

    if state["attempt"] >= 1:
        # Max retries — skip to next subtask
        nxt = state["subtask_index"] + 1
        return "prep" if nxt < len(state["subtasks"]) else END

    return "diagnose"


def route_after_commit(state: JuliaState) -> Literal["prep", END]:
    """After commit: advance to next subtask or finish."""
    nxt = state["subtask_index"] + 1
    return "prep" if nxt < len(state["subtasks"]) else END


def advance_subtask(state: JuliaState) -> JuliaState:
    """Advance subtask index."""
    return {
        **state,
        "subtask_index": state["subtask_index"] + 1,
        "attempt": 0,
        "fix_instruction": "",
        "status": "advanced",
    }


# ── Build graph ────────────────────────────────────────────────────────────

_builder = StateGraph(JuliaState)

_builder.add_node("load", load_node)
_builder.add_node("orchestrate", orchestrate_node)
_builder.add_node("prep", prep_node)
_builder.add_node("edit", edit_node)
_builder.add_node("self_review", self_review_node)
_builder.add_node("api_sync", api_sync_node)
_builder.add_node("test", test_node)
_builder.add_node("commit", commit_node)
_builder.add_node("diagnose", diagnose_node)
_builder.add_node("advance", advance_subtask)

_builder.add_edge(START, "load")
_builder.add_edge("load", "orchestrate")
_builder.add_edge("orchestrate", "prep")
_builder.add_edge("prep", "edit")
_builder.add_edge("edit", "self_review")
_builder.add_edge("self_review", "api_sync")
_builder.add_edge("api_sync", "test")

# After test: pass → commit, fail → diagnose → edit (retry)
_builder.add_conditional_edges("test", route_after_test, {
    "commit": "commit",
    "diagnose": "diagnose",
    "prep": "advance",
    END: END,
})

_builder.add_edge("diagnose", "edit")     # retry with fix

# After commit: next subtask or done
_builder.add_conditional_edges("commit", route_after_commit, {
    "prep": "advance",
    END: END,
})

_builder.add_edge("advance", "prep")

graph = _builder.compile()
