#!/usr/bin/env python3

import json
import os
import re
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_support import TEST_RUNS_DIR, bootstrap, preview


TMP_DIR = TEST_RUNS_DIR / "edge_case_regressions"
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


@dataclass
class EdgeCaseResult:
    status: str
    detail: str
    output_preview: str


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text).replace("\r", "")


def check_readpdf_relative_image_path() -> tuple[bool, str]:
    from agent_base.tools.tool_file import ReadPDF

    case_dir = TMP_DIR / "readpdf_relative_image"
    case_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = case_dir / "dummy.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%dummy\n")
    image_dir = case_dir / "fake_extracted"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "figure1.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)

    fake_structai = types.ModuleType("structai")
    fake_structai.read_pdf = lambda _: {"text": "", "img_paths": ["fake_extracted/figure1.png"]}
    previous_structai = sys.modules.get("structai")
    sys.modules["structai"] = fake_structai
    try:
        result = ReadPDF().call({"path": str(pdf_path)}, workspace_root=case_dir)
    finally:
        if previous_structai is None:
            sys.modules.pop("structai", None)
        else:
            sys.modules["structai"] = previous_structai

    ok = (
        isinstance(result, str)
        and not result.startswith("[ReadPDF] Error")
        and "source_type: pdf" in result
        and str(image_path.resolve()) in result
    )
    return ok, str(result)


def check_terminal_interrupt_preserves_remainder() -> tuple[bool, str]:
    from agent_base.tools.tool_runtime import TerminalInterrupt, TerminalKill, TerminalRead, TerminalStart, TerminalWrite

    case_dir = TMP_DIR / "terminal_interrupt_remainder"
    case_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[str] = []
    start_result = TerminalStart().call({"cwd": str(case_dir)}, workspace_root=case_dir)
    outputs.append(start_result)
    session_match = re.search(r"session_id: (term_\d+)", start_result)
    if not session_match:
        return False, "\n\n".join(outputs)
    session_id = session_match.group(1)

    try:
        write_result = TerminalWrite().call(
            {
                "session_id": session_id,
                "input": "sleep 0.2; printf 'ABCDEFGHIJ'; sleep 10",
                "yield_time_ms": 0,
            },
            workspace_root=case_dir,
        )
        outputs.append(write_result)
        time.sleep(0.4)
        interrupt_result = TerminalInterrupt().call(
            {
                "session_id": session_id,
                "max_output_chars": 4,
            },
            workspace_root=case_dir,
        )
        outputs.append(interrupt_result)
        read_result = TerminalRead().call(
            {
                "session_id": session_id,
                "yield_time_ms": 200,
                "max_output_chars": 4000,
            },
            workspace_root=case_dir,
        )
        outputs.append(read_result)
    finally:
        outputs.append(TerminalKill().call({"session_id": session_id}, workspace_root=case_dir))

    cleaned_interrupt = strip_ansi(interrupt_result)
    cleaned_read = strip_ansi(read_result)
    ok = "Sent Ctrl-C" in interrupt_result and "EFGHIJ" in cleaned_read and "ABCDEFGHIJ" in (cleaned_interrupt + cleaned_read)
    return ok, "\n\n".join(outputs)


def check_agent_runtime_limit_on_tool_execution() -> tuple[bool, str]:
    from agent_base.react_agent import MultiTurnReactAgent

    case_dir = TMP_DIR / "agent_runtime_limit"
    case_dir.mkdir(parents=True, exist_ok=True)

    class FakeAgent(MultiTurnReactAgent):
        def __init__(self):
            super().__init__(
                function_list=["Bash"],
                llm={
                    "model": "fake-model",
                    "generate_cfg": {
                        "max_input_tokens": 10000,
                        "max_retries": 1,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "presence_penalty": 0.0,
                    },
                },
            )
            self._turn = 0

        def call_llm_api(self, msgs, max_tries=10, runtime_deadline=None):
            self._turn += 1
            if self._turn == 1:
                return {
                    "status": "ok",
                    "finish_reason": "tool_calls",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_fake_bash",
                            "type": "function",
                            "function": {
                                "name": "Bash",
                                "arguments": json.dumps({"command": "sleep 2", "timeout": 30}),
                            },
                        }
                    ],
                }
            return {
                "status": "ok",
                "finish_reason": "stop",
                "content": "done",
                "tool_calls": [],
            }

    previous_runtime = os.environ.get("MAX_AGENT_RUNTIME_SECONDS")
    os.environ["MAX_AGENT_RUNTIME_SECONDS"] = "1"
    try:
        agent = FakeAgent()
        started_at = time.time()
        session = agent._run_session("trigger the slow bash tool", workspace_dir=str(case_dir))
        elapsed = time.time() - started_at
    finally:
        if previous_runtime is None:
            os.environ.pop("MAX_AGENT_RUNTIME_SECONDS", None)
        else:
            os.environ["MAX_AGENT_RUNTIME_SECONDS"] = previous_runtime

    detail = json.dumps(
        {
            "termination": session.get("termination"),
            "result_text": session.get("result_text"),
            "elapsed_seconds": round(elapsed, 3),
        },
        ensure_ascii=False,
        indent=2,
    )
    ok = (
        isinstance(session.get("termination"), str)
        and session["termination"].startswith("agent runtime limit reached")
        and elapsed < 1.8
    )
    return ok, detail


def main() -> int:
    bootstrap()
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    checks = [
        ("ReadPDF relative image path", check_readpdf_relative_image_path),
        ("TerminalInterrupt remainder", check_terminal_interrupt_preserves_remainder),
        ("Agent runtime limit", check_agent_runtime_limit_on_tool_execution),
    ]

    failures: list[str] = []
    outputs: list[str] = []
    for name, func in checks:
        ok, detail = func()
        outputs.append(f"[{name}]\n{detail}")
        if not ok:
            failures.append(name)

    result = EdgeCaseResult(
        status="PASS" if not failures else "FAIL",
        detail="All edge-case regressions passed." if not failures else f"Failed checks: {', '.join(failures)}",
        output_preview=preview("\n\n".join(outputs)),
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
