import os
import re
import subprocess
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "workspace"


class CodingState(TypedDict, total=False):
    task: str
    target_file: str
    original_code: str
    current_code: str
    plan: str
    test_output: str
    review: str
    attempts: int
    status: str


def local_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LOCAL_MODEL", "local-coder"),
        base_url=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.getenv("LOCAL_API_KEY", "dummy"),
        temperature=0,
        timeout=120,
    )


def premium_or_local_llm() -> ChatOpenAI:
    premium_model = os.getenv("PREMIUM_MODEL")
    premium_base_url = os.getenv("PREMIUM_BASE_URL")
    premium_api_key = os.getenv("PREMIUM_API_KEY")

    if premium_model and premium_base_url and premium_api_key:
        return ChatOpenAI(
            model=premium_model,
            base_url=premium_base_url,
            api_key=premium_api_key,
            temperature=0,
            timeout=120,
        )

    return local_llm()


def safe_path(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    workspace_root = WORKSPACE.resolve()

    if not str(path).startswith(str(workspace_root)):
        raise ValueError(f"Refusing to edit outside workspace/: {relative_path}")

    return path


def extract_file_block(text: str) -> str:
    match = re.search(r"<file>\s*(.*?)\s*</file>", text, flags=re.DOTALL)
    if not match:
        raise ValueError(
            "Model did not return a valid <file>...</file> block.\n\n"
            f"Raw response:\n{text}"
        )
    return match.group(1).strip() + "\n"


def load_node(state: CodingState) -> CodingState:
    target_file = state.get("target_file", "workspace/math_utils.py")
    path = safe_path(target_file)
    code = path.read_text()

    return {
        **state,
        "target_file": target_file,
        "original_code": code,
        "current_code": code,
        "attempts": state.get("attempts", 0),
        "status": "loaded",
    }


def plan_node(state: CodingState) -> CodingState:
    llm = premium_or_local_llm()

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a senior software engineer. "
                    "Create a concise repair plan. Do not edit code yet."
                )
            ),
            HumanMessage(
                content=(
                    f"Task:\n{state['task']}\n\n"
                    f"Target file: {state['target_file']}\n\n"
                    f"Current code:\n```python\n{state['current_code']}\n```"
                )
            ),
        ]
    )

    return {**state, "plan": response.content, "status": "planned"}


def edit_node(state: CodingState) -> CodingState:
    llm = local_llm()

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a precise coding agent. "
                    "You must return ONLY one XML block: <file>...</file>. "
                    "Inside that block, put the COMPLETE corrected Python file. "
                    "Do not include explanations, markdown, thinking, placeholders, "
                    "ellipsis, comments about the change, or omitted code. "
                    "The file must define the requested function."
                )
            ),
            HumanMessage(
                content=(
                    f"Task:\n{state['task']}\n\n"
                    f"Plan:\n{state['plan']}\n\n"
                    f"Current file {state['target_file']}:\n"
                    f"```python\n{state['current_code']}\n```\n\n"
                    "Return the complete corrected file in this exact format:\n"
                    "<file>\n"
                    "def add(a: int, b: int) -> int:\n"
                    '    """Return the sum of a and b."""\n'
                    "    return a + b\n"
                    "</file>"
                )
            ),
        ]
    )

    new_code = extract_file_block(response.content)

    if "..." in new_code:
        raise ValueError(f"Model returned placeholder code:\n{new_code}")

    if "def add" not in new_code:
        raise ValueError(f"Model output does not define add():\n{new_code}")

    path = safe_path(state["target_file"])
    path.write_text(new_code)

    return {
        **state,
        "current_code": new_code,
        "attempts": state.get("attempts", 0) + 1,
        "status": "edited",
    }


def test_node(state: CodingState) -> CodingState:
    proc = subprocess.run(
        ["pytest", "-q"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )

    return {
        **state,
        "test_output": proc.stdout,
        "status": "tests_passed" if proc.returncode == 0 else "tests_failed",
    }


def review_node(state: CodingState) -> CodingState:
    llm = premium_or_local_llm()

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a strict code reviewer. "
                    "Respond with exactly one first-line verdict: PASS or RETRY. "
                    "Then give a short reason."
                )
            ),
            HumanMessage(
                content=(
                    f"Task:\n{state['task']}\n\n"
                    f"Plan:\n{state['plan']}\n\n"
                    f"Current code:\n```python\n{state['current_code']}\n```\n\n"
                    f"Test output:\n```text\n{state['test_output']}\n```\n\n"
                    "Should we accept this patch?"
                )
            ),
        ]
    )

    review = response.content.strip()
    first_line = review.splitlines()[0].strip().upper() if review else "RETRY"

    if state["status"] == "tests_passed" and first_line.startswith("PASS"):
        status = "accepted"
    elif state["status"] == "tests_passed":
        # Tests passed but reviewer was unconvinced. Accept for this MWE.
        status = "accepted"
    else:
        status = "needs_retry"

    return {**state, "review": review, "status": status}


def route_after_review(state: CodingState) -> Literal["edit", "__end__"]:
    if state["status"] == "accepted":
        return END

    if state.get("attempts", 0) >= 2:
        return END

    return "edit"


builder = StateGraph(CodingState)

builder.add_node("load", load_node)
builder.add_node("plan", plan_node)
builder.add_node("edit", edit_node)
builder.add_node("test", test_node)
builder.add_node("review", review_node)

builder.add_edge(START, "load")
builder.add_edge("load", "plan")
builder.add_edge("plan", "edit")
builder.add_edge("edit", "test")
builder.add_edge("test", "review")
builder.add_conditional_edges("review", route_after_review, ["edit", END])

graph = builder.compile()
