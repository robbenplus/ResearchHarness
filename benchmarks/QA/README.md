# QA / VQA Benchmarks

This directory documents the lightweight ResearchHarness contract for
question-answering benchmarks, including plain-text QA and multimodal VQA-style
tasks.

The recommended integration is the OpenAI-compatible synchronous API server:

```bash
python3 /abs/path/to/ResearchHarness/serve_openai.py \
  --workspace-root /abs/path/to/rh_api_workspaces \
  --role-prompt-file /abs/path/to/ResearchHarness/benchmarks/QA/role_prompt.md
```

External benchmark runners can then use the regular OpenAI SDK with:

```python
from openai import OpenAI

client = OpenAI(api_key="unused", base_url="http://127.0.0.1:8000/v1")

response = client.chat.completions.create(
    model="researchharness",
    messages=[{"role": "user", "content": "Answer the question."}],
)

answer = response.choices[0].message.content
```

## Multimodal Input

For image benchmarks, send OpenAI-style content parts. The first API version
supports `data:image/...;base64,...` URLs.

```python
response = client.chat.completions.create(
    model="researchharness",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown? Return JSON with key answer."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ],
)
```

The API saves each submitted image inside the request workspace and also passes
the image content to the first ResearchHarness model call when the backend model
supports image parts.

## Scope

- The endpoint is synchronous and returns one final text answer.
- Each request gets a separate workspace subdirectory.
- The API uses an input wrapper, the ResearchHarness agent, and an output
  wrapper so strict benchmark output formats do not destabilize the agent loop.
- Streaming, async run status, artifact download, and remote image fetching are
  intentionally out of scope for this minimal QA contract.
