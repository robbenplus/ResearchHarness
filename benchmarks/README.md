# Benchmarks

This folder records benchmark-specific integration documents that live
**outside** `agent_base`.

| Benchmark | Directory | Tracked contract |
| --- | --- | --- |
| ResearchClawBench | `benchmarks/ResearchClawBench/` | `README.md` + `role_prompt.md` + `adapter.py` |

## Notes

- `agent_base/` stays focused on the core runtime.
- Benchmark-specific logic and prompts should live under their own benchmark
  subdirectory.
- Local benchmark helpers may exist for private experimentation, but they do
  not define the formal external integration contract.
