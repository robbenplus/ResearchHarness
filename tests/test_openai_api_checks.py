#!/usr/bin/env python3

import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_support import TEST_RUNS_DIR, bootstrap, load_trace_records, preview, single_trace_path


TMP_DIR = TEST_RUNS_DIR / "openai_api_checks"
TINY_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


@dataclass
class OpenAIAPICheckResult:
    status: str
    detail: str
    output_preview: str


def main() -> int:
    bootstrap()

    from agent_base.react_agent import MultiTurnReactAgent
    from api.openai_server import (
        build_input_wrapper_messages,
        extract_json_object,
        make_chat_completion_response,
        prepare_openai_input,
    )

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": "researchharness",
        "messages": [
            {"role": "system", "content": "Answer exactly in the requested format."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is the image? Return JSON with key answer."},
                    {"type": "image_url", "image_url": {"url": TINY_PNG_DATA_URL}},
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }

    prepared = prepare_openai_input(payload["messages"], TMP_DIR)
    saved_image = TMP_DIR / "inputs" / "images" / "image_000.png"
    input_wrapper_messages = build_input_wrapper_messages(prepared=prepared, payload=payload)
    parsed_plan = extract_json_object(
        '```json\n{"agent_instruction": "Use the saved image.", "output_contract": "Return JSON.", "wrapper_notes": "ok"}\n```'
    )
    response = make_chat_completion_response(request_id="chatcmpl_test", model="researchharness", content='{"answer":"white"}')

    trace_dir = TMP_DIR / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    class FakeAgent(MultiTurnReactAgent):
        def __init__(self):
            super().__init__(
                function_list=[],
                llm={
                    "model": "fake-vision-model",
                    "generate_cfg": {
                        "max_input_tokens": 10000,
                        "max_retries": 1,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "presence_penalty": 0.0,
                    },
                },
                trace_dir=str(trace_dir),
            )
            self.seen_messages = []

        def call_llm_api(self, msgs, max_tries=10, runtime_deadline=None):
            self.seen_messages = msgs
            return {
                "status": "ok",
                "finish_reason": "stop",
                "content": "Final answer: white",
                "tool_calls": [],
            }

    agent = FakeAgent()
    session = agent._run_session(
        "Solve the image task.",
        workspace_root=str(TMP_DIR / "agent_workspace"),
        initial_content_parts=prepared.initial_content_parts,
    )
    trace_path = single_trace_path(trace_dir)
    rows = load_trace_records(trace_path) if trace_path else []
    first_user_content = agent.seen_messages[1]["content"] if len(agent.seen_messages) > 1 else None
    first_user_trace = rows[1].get("text", "") if len(rows) > 1 else ""

    ok = (
        prepared.image_paths == ["inputs/images/image_000.png"]
        and saved_image.exists()
        and prepared.initial_content_parts
        and input_wrapper_messages[1]["content"].find("response_format") >= 0
        and parsed_plan["output_contract"] == "Return JSON."
        and response["choices"][0]["message"]["content"] == '{"answer":"white"}'
        and session.get("result_text") == "Final answer: white"
        and isinstance(first_user_content, list)
        and any(isinstance(part, dict) and part.get("type") == "image_url" for part in first_user_content)
        and "base64 omitted" in first_user_trace
    )

    result = OpenAIAPICheckResult(
        status="PASS" if ok else "FAIL",
        detail="OpenAI-compatible API helpers and initial multimodal user content are working."
        if ok
        else "OpenAI-compatible API checks failed.",
        output_preview=preview(
            json.dumps(
                {
                    "image_paths": prepared.image_paths,
                    "saved_image_exists": saved_image.exists(),
                    "input_wrapper_messages": input_wrapper_messages,
                    "parsed_plan": parsed_plan,
                    "response": response,
                    "session_result": session.get("result_text"),
                    "first_user_content_type": type(first_user_content).__name__,
                    "first_user_trace": first_user_trace,
                },
                ensure_ascii=False,
                indent=2,
            )
        ),
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
