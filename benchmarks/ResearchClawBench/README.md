# ResearchClawBench

This directory contains the tracked files needed to document how `ResearchHarness`
should be integrated into `ResearchClawBench`.

## Recommended `agents.json` Entry

Use a single direct command that launches the thin top-level ResearchHarness
entrypoint.

```json
{
  "researchharness": {
    "label": "ResearchHarness",
    "icon": "H",
    "logo": "/static/logos/rh.svg",
    "cmd": "python3 /abs/path/to/ResearchHarness/run_agent.py <PROMPT> --workspace-root <WORKSPACE> --role-prompt-file /abs/path/to/ResearchHarness/benchmarks/ResearchClawBench/role_prompt.md --trace-dir <WORKSPACE>"
  }
}
```

## Why This Shape

- `ResearchClawBench` already prepares the workspace, writes `INSTRUCTIONS.md`,
  and isolates hidden checklist data.
- `ResearchHarness` should only execute the agent.
- Benchmark-specific behavior is injected through `role_prompt.md`, not by
  changing `agent_base`.

## Notes

- Replace `/abs/path/to/ResearchHarness/` with the real local checkout path.
- The command should stay one-line and non-interactive.
- Any local batch helpers or ad hoc benchmark scripts should remain untracked
  and live outside the formal integration contract.
