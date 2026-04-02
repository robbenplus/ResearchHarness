import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class PromptAsset:
    name: str
    path: Path
    description: str


PROMPT_ASSETS = {
    "system_base": PromptAsset(
        name="system_base",
        path=PROMPTS_DIR / "system_base.md",
        description="Base general-purpose system prompt for the harness.",
    ),
    "extractor": PromptAsset(
        name="extractor",
        path=PROMPTS_DIR / "extractor.md",
        description="Goal-directed webpage extraction prompt used by WebFetch.",
    ),
}


PROMPT_PLUGINS = {
    "academic_research": PromptAsset(
        name="academic_research",
        path=PROMPTS_DIR / "plugins" / "academic_research.md",
        description="Academic research extension with phase gates, persistent state, and evidence-grounded reporting.",
    ),
}


def _read_prompt_asset(asset: PromptAsset) -> str:
    return asset.path.read_text(encoding="utf-8").strip()


SYSTEM_PROMPT = _read_prompt_asset(PROMPT_ASSETS["system_base"])
EXTRACTOR_PROMPT = _read_prompt_asset(PROMPT_ASSETS["extractor"])


def plugin_names_from_csv(raw: str) -> list[str]:
    names: list[str] = []
    for chunk in raw.replace(";", ",").split(","):
        name = chunk.strip()
        if name:
            names.append(name)
    return names


def dedupe_plugin_names(names: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_name in names:
        name = str(raw_name).strip()
        if not name or name in seen:
            continue
        ordered.append(name)
        seen.add(name)
    return ordered


def resolve_prompt_plugins(names: Iterable[str]) -> list[PromptAsset]:
    resolved: list[PromptAsset] = []
    for name in dedupe_plugin_names(names):
        asset = PROMPT_PLUGINS.get(name)
        if asset is None:
            valid = ", ".join(sorted(PROMPT_PLUGINS))
            raise ValueError(f"Unknown prompt plugin '{name}'. Available plugins: {valid}")
        resolved.append(asset)
    return resolved


def prompt_plugin_blocks(names: Iterable[str]) -> list[str]:
    return [_read_prompt_asset(asset) for asset in resolve_prompt_plugins(names)]


def composed_system_prompt(*, current_date: str, plugin_names: list[str] | None = None) -> str:
    blocks = [SYSTEM_PROMPT.rstrip()]
    for block in prompt_plugin_blocks(plugin_names or []):
        blocks.append(block.rstrip())
    blocks.append(f"Current date: {current_date}")
    return "\n\n".join(blocks)


def configured_prompt_plugins() -> list[str]:
    return dedupe_plugin_names(plugin_names_from_csv(os.environ.get("PROMPT_PLUGINS", "")))


def _show_asset(name: str) -> str:
    asset = PROMPT_ASSETS.get(name)
    if asset is None:
        valid = ", ".join(sorted(PROMPT_ASSETS))
        raise ValueError(f"Unknown prompt asset '{name}'. Available assets: {valid}")
    return _read_prompt_asset(asset)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect prompt assets and prompt plugins.")
    parser.add_argument("--show-system", action="store_true", help="Print the composed system prompt.")
    parser.add_argument("--show-extractor", action="store_true", help="Print the extractor prompt.")
    parser.add_argument("--show-asset", metavar="NAME", help="Print one prompt asset by name.")
    parser.add_argument("--list-assets", action="store_true", help="List registered prompt assets.")
    parser.add_argument("--show-plugin", metavar="NAME", help="Print one registered prompt plugin.")
    parser.add_argument("--list-plugins", action="store_true", help="List registered prompt plugins.")
    parser.add_argument(
        "--with-plugin",
        action="append",
        default=[],
        dest="plugins",
        help="Include a prompt plugin when printing the composed system prompt. May be passed multiple times.",
    )
    args = parser.parse_args(argv)

    if args.list_assets:
        for asset in sorted(PROMPT_ASSETS.values(), key=lambda item: item.name):
            print(f"{asset.name}: {asset.description}")
        return 0

    if args.show_asset:
        print(_show_asset(args.show_asset))
        return 0

    if args.list_plugins:
        for plugin in sorted(PROMPT_PLUGINS.values(), key=lambda item: item.name):
            print(f"{plugin.name}: {plugin.description}")
        return 0

    if args.show_plugin:
        print(_read_prompt_asset(resolve_prompt_plugins([args.show_plugin])[0]))
        return 0

    if args.show_system:
        print(composed_system_prompt(current_date="<DATE>", plugin_names=args.plugins))
        return 0

    if args.show_extractor:
        print(EXTRACTOR_PROMPT)
        return 0

    print(f"prompt_asset_dir={PROMPTS_DIR}")
    print(f"system_prompt_chars={len(composed_system_prompt(current_date='<DATE>', plugin_names=args.plugins))}")
    print(f"extractor_prompt_chars={len(EXTRACTOR_PROMPT)}")
    print(f"prompt_plugin_count={len(PROMPT_PLUGINS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
