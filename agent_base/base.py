from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Sequence


def _normalize_function_list(function_list: Optional[Iterable[str]]) -> Optional[list[str]]:
    if function_list is None:
        return None
    normalized: list[str] = []
    for raw_name in function_list:
        name = str(raw_name).strip()
        if name:
            normalized.append(name)
    return normalized


def agent_role(
    *,
    name: str,
    role_prompt: str = "",
    function_list: Optional[Iterable[str]] = None,
):
    """
    Class decorator used by upper-layer frameworks to declare agent defaults.

    This keeps the lower-layer execution loop generic while allowing subclasses
    to provide role-specific prompt addenda and tool restrictions declaratively.
    """

    def decorator(cls):
        cls.role_name = str(name).strip() or cls.__name__
        cls.default_role_prompt = str(role_prompt).strip()
        cls.default_function_list = _normalize_function_list(function_list)
        return cls

    return decorator


class BaseAgent(ABC):
    """Abstract base class for agents built on top of ResearchHarness."""

    role_name: str = "agent"
    default_role_prompt: str = ""
    default_function_list: Optional[list[str]] = None

    @classmethod
    def resolve_function_list(cls, function_list: Optional[Sequence[str]]) -> Optional[list[str]]:
        if function_list is not None:
            return _normalize_function_list(function_list) or []
        default_tools = getattr(cls, "default_function_list", None)
        if default_tools is None:
            return None
        return list(default_tools)

    @classmethod
    def resolve_role_prompt(cls, role_prompt: Optional[str]) -> str:
        if role_prompt is None:
            role_prompt = getattr(cls, "default_role_prompt", "")
        return str(role_prompt or "").strip()

    def should_accept_plaintext_result(
        self,
        *,
        result_text: str,
        workspace_root: Optional[str],
        messages: Sequence[dict[str, Any]],
    ) -> bool:
        """
        Decide whether a plain assistant text reply with no tool calls is terminal.

        The default behavior preserves the original ResearchHarness semantics:
        any meaningful assistant text without tool calls is accepted as the final
        result. Upper layers may override this hook to require extra completion
        artifacts before termination.
        """

        return True

    def rejected_plaintext_result_message(
        self,
        *,
        result_text: str,
        workspace_root: Optional[str],
        messages: Sequence[dict[str, Any]],
    ) -> str:
        """
        Explain why a plain assistant text reply was not accepted as terminal.

        Returning an empty string falls back to the generic runtime message.
        """

        return ""

    @abstractmethod
    def run(self, prompt: str, workspace_root: Optional[str] = None):
        raise NotImplementedError
