import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional, Union
from urllib.parse import parse_qs, quote, urlencode, urlparse, urlunparse
import xml.etree.ElementTree as ET

import requests
import tiktoken
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from agent_base.provider_compat import apply_sampling_params
from agent_base.prompt import EXTRACTOR_PROMPT
from agent_base.tools.tooling import ToolBase, validate_tool_path, workspace_root
from agent_base.utils import PROJECT_ROOT, env_flag, load_dotenv

DEFAULT_SUMMARY_MODEL_NAME = "gpt-5.4"
DEFAULT_WEBFETCH_TIMEOUT_SECONDS = 600.0
DEFAULT_WEBFETCH_SUMMARY_TEMPERATURE = 0.0
DEFAULT_SCHOLAR_MAX_RESULTS = 10
DEFAULT_DOWNLOADPDF_TIMEOUT_SECONDS = 30
MIN_VALID_PDF_BYTES = 10_000
PAYWALL_DOMAINS = frozenset(
    {
        "sciencedirect.com",
        "springer.com",
        "link.springer.com",
        "wiley.com",
        "onlinelibrary.wiley.com",
        "tandfonline.com",
        "ieeexplore.ieee.org",
        "dl.acm.org",
        "nature.com",
        "science.org",
        "jstor.org",
        "emerald.com",
        "sagepub.com",
        "cambridge.org",
        "oxford.org",
        "oup.com",
    }
)


def search_debug_enabled() -> bool:
    return env_flag("DEBUG_SEARCH")


def scholar_debug_enabled() -> bool:
    return env_flag("DEBUG_SCHOLAR")


def visit_debug_enabled() -> bool:
    return env_flag("DEBUG_VISIT")


def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def _stringify_field(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value).strip()
    return str(value).strip()


def _webfetch_failure(url: str, goal: str, reason: str) -> str:
    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
    useful_information += "Evidence in page: \n" + reason + "\n\n"
    useful_information += "Summary: \n" + "The webpage content could not be processed into the required structured summary." + "\n\n"
    return useful_information


def _parse_extractor_payload(raw) -> tuple[str, str] | None:
    if isinstance(raw, str):
        raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None

    if not isinstance(raw, dict):
        return None

    evidence = _stringify_field(raw.get("evidence"))
    summary = _stringify_field(raw.get("summary"))
    if not evidence or not summary:
        return None
    return evidence, summary


DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
ARXIV_RE = re.compile(
    r"(?:arxiv:\s*|arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)


def _normalize_title(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def _extract_doi(*values: Any) -> str:
    for value in values:
        text = _stringify_field(value)
        if not text:
            continue
        match = DOI_RE.search(text)
        if match:
            return match.group(0).rstrip(").,;").lower()
    return ""


def _extract_arxiv_id(*values: Any) -> str:
    for value in values:
        text = _stringify_field(value)
        if not text:
            continue
        match = ARXIV_RE.search(text)
        if match:
            return match.group(1)
    return ""


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = _stringify_field(value).replace(",", "")
    if not text:
        return None
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        clean = _stringify_field(item)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        output.append(clean)
    return output


def _fingerprint_paper(record: dict[str, Any]) -> str:
    for key in ("doi", "arxiv_id", "s2_id", "openalex_id"):
        value = _stringify_field(record.get(key)).lower()
        if value:
            return f"{key}:{value}"
    return f"title:{_normalize_title(_stringify_field(record.get('title')))}:{record.get('year') or ''}"


def _pdf_candidate(url: str, *, source_type: str, provider: str, confidence: float, license_hint: str = "") -> dict[str, Any]:
    clean_url = _stringify_field(url)
    if not clean_url:
        return {}
    return {
        "url": clean_url,
        "source_type": source_type,
        "provider": provider,
        "confidence": confidence,
        "license_hint": license_hint,
    }


def _merge_pdf_candidates(*candidate_lists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for candidates in candidate_lists:
        for candidate in candidates or []:
            if not isinstance(candidate, dict):
                continue
            url = _stringify_field(candidate.get("url"))
            if not url:
                continue
            current = merged.get(url)
            if current is None or float(candidate.get("confidence", 0) or 0) > float(current.get("confidence", 0) or 0):
                copied = dict(candidate)
                copied["url"] = url
                merged[url] = copied
    return sorted(
        merged.values(),
        key=lambda item: (
            float(item.get("confidence", 0) or 0),
            _stringify_field(item.get("source_type")),
        ),
        reverse=True,
    )


def _merge_paper_records(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key in (
        "title",
        "year",
        "venue",
        "abstract",
        "doi",
        "arxiv_id",
        "s2_id",
        "openalex_id",
        "url",
        "citation_count",
    ):
        if not result.get(key) and incoming.get(key):
            result[key] = incoming[key]
    if incoming.get("authors") and not result.get("authors"):
        result["authors"] = list(incoming["authors"])
    result["pdf_candidates"] = _merge_pdf_candidates(
        list(result.get("pdf_candidates") or []),
        list(incoming.get("pdf_candidates") or []),
    )
    providers = list(result.get("confirmed_by") or [])
    provider = _stringify_field(incoming.get("source_provider"))
    if provider and provider not in providers:
        providers.append(provider)
    result["confirmed_by"] = providers
    return result


def _titles_match(a: str, b: str) -> bool:
    normalized_a = _normalize_title(a)
    normalized_b = _normalize_title(b)
    if not normalized_a or not normalized_b:
        return False
    if normalized_a == normalized_b:
        return True
    shorter, longer = sorted((normalized_a, normalized_b), key=len)
    return len(shorter) >= 20 and shorter in longer


def _host_matches(url: str, domains: frozenset[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _looks_like_html(payload: bytes) -> bool:
    head = payload[:4096].lstrip().lower()
    return (
        head.startswith(b"<!doctype html")
        or head.startswith(b"<html")
        or b"<head" in head[:512]
        or b"<body" in head[:512]
        or b"preparing to download" in head
    )


def _is_pdf_bytes(payload: bytes) -> bool:
    return payload.startswith(b"%PDF")


def _sanitize_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in value.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:80] or "paper"


def _expand_pdf_url(url: str) -> list[str]:
    raw_url = _stringify_field(url)
    if not raw_url:
        return []
    parsed = urlparse(raw_url)
    host = parsed.netloc.lower()
    path = parsed.path
    if host.endswith("arxiv.org"):
        if path.startswith("/abs/"):
            arxiv_id = path.split("/abs/", 1)[1].strip("/")
            return [f"https://arxiv.org/pdf/{arxiv_id}.pdf", raw_url]
        if path.startswith("/pdf/") and not path.endswith(".pdf"):
            return [urlunparse(parsed._replace(path=f"{path}.pdf"))]
    if host.endswith("openreview.net") and "/forum" in path:
        forum_id = parse_qs(parsed.query).get("id", [""])[0]
        if forum_id:
            return [f"https://openreview.net/pdf?id={forum_id}", raw_url]
    return [raw_url]


def _dedupe_urls(urls: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for url in urls:
        for expanded in _expand_pdf_url(url):
            if expanded and expanded not in seen:
                output.append(expanded)
                seen.add(expanded)
    return output


class WebSearch(ToolBase):
    name = "WebSearch"
    description = "Perform Google web searches and return the top results. Accepts multiple complementary queries."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string",
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def google_search_with_serp(self, query: str):
        def contains_chinese_basic(text: str) -> bool:
            return any("\u4E00" <= char <= "\u9FFF" for char in text)

        if contains_chinese_basic(query):
            payload = {
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn",
            }
        else:
            payload = {
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en",
            }
        serper_key = os.getenv("SERPER_KEY_ID", "").strip()
        if not serper_key:
            return "[WebSearch] SERPER_KEY_ID is not set."
        headers = {
            "X-API-KEY": serper_key,
            "Content-Type": "application/json",
        }

        last_error = ""
        res = None
        for i in range(5):
            try:
                res = requests.post(
                    "https://google.serper.dev/search",
                    json=payload,
                    headers=headers,
                    timeout=20,
                )
                res.raise_for_status()
                break
            except requests.RequestException as exc:
                last_error = str(exc)
                if search_debug_enabled():
                    print(exc)
                if i == 4:
                    return f"[WebSearch] Request failed for '{query}': {last_error}"

        if res is None:
            return f"[WebSearch] Request failed for '{query}': {last_error or 'unknown error'}"

        try:
            results = res.json()
        except ValueError as exc:
            return f"[WebSearch] Invalid JSON response for '{query}': {exc}"

        organic_results = results.get("organic")
        if not isinstance(organic_results, list) or not organic_results:
            return f"No results found for '{query}'. Try with a more general query."

        web_snippets = []
        for idx, page in enumerate(organic_results, start=1):
            if not isinstance(page, dict):
                continue
            title = str(page.get("title", "Untitled result"))
            link = str(page.get("link", ""))
            date_published = f"\nDate published: {page['date']}" if "date" in page else ""
            source = f"\nSource: {page['source']}" if "source" in page else ""
            snippet = f"\n{page['snippet']}" if "snippet" in page else ""
            redacted_version = f"{idx}. [{title}]({link}){date_published}{source}\n{snippet}"
            redacted_version = redacted_version.replace("Your browser can't play this video.", "")
            web_snippets.append(redacted_version)

        if not web_snippets:
            return f"No results found for '{query}'. Try with a more general query."

        content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
        return content

    def search_with_serp(self, query: str):
        return self.google_search_with_serp(query)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self.parse_json_args(params)
            query = params["query"]
        except ValueError as exc:
            return f"[WebSearch] {exc}"

        if isinstance(query, list):
            with ThreadPoolExecutor(max_workers=3) as executor:
                responses = list(executor.map(self.search_with_serp, query))
            response = "\n=======\n".join(responses)
        else:
            return "[WebSearch] 'query' must be a list of strings."

        return response


class ScholarSearch(ToolBase):
    name = "ScholarSearch"
    description = "Search academic papers in two layers: Serper finds high-recall clues, then arXiv, Semantic Scholar, and OpenAlex structurally confirm metadata and PDF candidates."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string", "description": "The search query."},
                "minItems": 1,
                "description": "The list of academic search queries. Serper is used to find clues; structured providers confirm results.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum confirmed papers to return per query. Default is 10.",
            },
            "year_from": {
                "type": "integer",
                "description": "Optional lower publication year bound.",
            },
            "year_to": {
                "type": "integer",
                "description": "Optional upper publication year bound.",
            },
            "providers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional structured confirmation providers. Supported: arxiv, semantic_scholar, openalex.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def _serper_scholar_clues(
        self,
        query: str,
        *,
        max_results: int,
        year_from: Optional[int],
        year_to: Optional[int],
    ) -> tuple[list[dict[str, Any]], str]:
        payload: dict[str, Any] = {"q": query, "num": max_results}
        if year_from is not None:
            payload["as_ylo"] = year_from
        if year_to is not None:
            payload["as_yhi"] = year_to
        serper_key = os.getenv("SERPER_KEY_ID", "").strip()
        if not serper_key:
            return [], "[ScholarSearch] SERPER_KEY_ID is not set."
        headers = {
            "X-API-KEY": serper_key,
            "Content-Type": "application/json",
        }
        last_error = ""
        res = None
        for i in range(5):
            try:
                res = requests.post(
                    "https://google.serper.dev/scholar",
                    json=payload,
                    headers=headers,
                    timeout=20,
                )
                res.raise_for_status()
                break
            except requests.RequestException as exc:
                last_error = str(exc)
                if scholar_debug_enabled():
                    print(exc)
                if i == 4:
                    return [], f"[ScholarSearch] Request failed for '{query}': {last_error}"

        if res is None:
            return [], f"[ScholarSearch] Request failed for '{query}': {last_error or 'unknown error'}"

        try:
            results = res.json()
        except ValueError as exc:
            return [], f"[ScholarSearch] Invalid JSON response for '{query}': {exc}"

        organic_results = results.get("organic")
        if not isinstance(organic_results, list) or not organic_results:
            return [], f"No results found for '{query}'. Try with a more general query."

        clues: list[dict[str, Any]] = []
        for page in organic_results[:max_results]:
            if not isinstance(page, dict):
                continue
            title = _stringify_field(page.get("title")) or "Untitled result"
            link = _stringify_field(page.get("link") or page.get("url"))
            pdf_url = _stringify_field(page.get("pdfUrl") or page.get("pdf_url"))
            snippet = _stringify_field(page.get("snippet") or page.get("abstract"))
            publication_info = _stringify_field(page.get("publicationInfo") or page.get("publication_info"))
            year = _coerce_int(page.get("year") or page.get("publication_year"))
            doi = _extract_doi(title, link, pdf_url, snippet, publication_info)
            arxiv_id = _extract_arxiv_id(title, link, pdf_url, snippet, publication_info)
            pdf_candidates: list[dict[str, Any]] = []
            if pdf_url:
                pdf_candidates.append(
                    _pdf_candidate(
                        pdf_url,
                        source_type="serper_pdf_clue",
                        provider="serper",
                        confidence=0.45,
                    )
                )
            clues.append(
                {
                    "title": title,
                    "year": year,
                    "venue": publication_info,
                    "abstract": snippet,
                    "doi": doi,
                    "arxiv_id": arxiv_id,
                    "url": link,
                    "citation_count": _coerce_int(page.get("citedBy") or page.get("cited_by")),
                    "pdf_candidates": [candidate for candidate in pdf_candidates if candidate],
                    "source_provider": "serper",
                }
            )

        return clues, ""

    def _provider_order(self, raw: Any) -> list[str]:
        supported = {"arxiv", "semantic_scholar", "openalex"}
        if not isinstance(raw, list) or not raw:
            return ["arxiv", "semantic_scholar", "openalex"]
        providers = []
        for item in raw:
            provider = _stringify_field(item).lower().replace("-", "_")
            if provider in supported and provider not in providers:
                providers.append(provider)
        return providers or ["arxiv", "semantic_scholar", "openalex"]

    def _record_matches_clue(self, record: dict[str, Any], clue: dict[str, Any]) -> bool:
        if not record:
            return False
        if clue.get("doi") and record.get("doi") and _stringify_field(clue["doi"]).lower() == _stringify_field(record["doi"]).lower():
            return True
        if clue.get("arxiv_id") and record.get("arxiv_id") and _stringify_field(clue["arxiv_id"]).lower() == _stringify_field(record["arxiv_id"]).lower():
            return True
        return _titles_match(_stringify_field(record.get("title")), _stringify_field(clue.get("title")))

    def _confirm_clue(self, clue: dict[str, Any], providers: list[str]) -> Optional[dict[str, Any]]:
        confirmed: Optional[dict[str, Any]] = None
        for provider in providers:
            try:
                if provider == "arxiv":
                    record = self._confirm_with_arxiv(clue)
                elif provider == "semantic_scholar":
                    record = self._confirm_with_semantic_scholar(clue)
                elif provider == "openalex":
                    record = self._confirm_with_openalex(clue)
                else:
                    record = None
            except requests.RequestException as exc:
                if scholar_debug_enabled():
                    print(f"[ScholarSearch] {provider} failed: {exc}")
                record = None
            if record and self._record_matches_clue(record, clue):
                confirmed = record if confirmed is None else _merge_paper_records(confirmed, record)
        if confirmed is None:
            return None
        confirmed = _merge_paper_records(confirmed, clue)
        confirmed["verified"] = True
        return confirmed

    def _confirm_with_arxiv(self, clue: dict[str, Any]) -> Optional[dict[str, Any]]:
        arxiv_id = _stringify_field(clue.get("arxiv_id")).removeprefix("arXiv:").strip()
        if arxiv_id:
            search_query = f"id:{arxiv_id}"
        else:
            title = _stringify_field(clue.get("title"))
            if not title:
                return None
            search_query = f'ti:"{title}"'
        params = urlencode({"search_query": search_query, "start": 0, "max_results": 3})
        response = requests.get(
            f"https://export.arxiv.org/api/query?{params}",
            timeout=20,
            headers={"User-Agent": "research-harness/ScholarSearch"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", namespace):
            record = self._arxiv_entry_to_record(entry, namespace)
            if self._record_matches_clue(record, clue):
                return record
        return None

    def _arxiv_entry_to_record(self, entry: ET.Element, namespace: dict[str, str]) -> dict[str, Any]:
        title = _stringify_field(entry.findtext("atom:title", default="", namespaces=namespace)).replace("\n", " ")
        abstract = _stringify_field(entry.findtext("atom:summary", default="", namespaces=namespace)).replace("\n", " ")
        published = _stringify_field(entry.findtext("atom:published", default="", namespaces=namespace))
        arxiv_url = _stringify_field(entry.findtext("atom:id", default="", namespaces=namespace))
        arxiv_id = arxiv_url.rstrip("/").rsplit("/", 1)[-1]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""
        authors = [
            _stringify_field(author.findtext("atom:name", default="", namespaces=namespace))
            for author in entry.findall("atom:author", namespace)
        ]
        doi = ""
        for link in entry.findall("atom:link", namespace):
            if link.attrib.get("title") == "doi":
                doi = _extract_doi(link.attrib.get("href", ""))
        return {
            "title": title,
            "authors": _dedupe_strings(authors),
            "year": _coerce_int(published[:4]),
            "venue": "arXiv",
            "abstract": abstract,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "url": arxiv_url,
            "pdf_candidates": [
                _pdf_candidate(pdf_url, source_type="arxiv_pdf", provider="arxiv", confidence=0.98, license_hint="open_access")
            ] if pdf_url else [],
            "source_provider": "arxiv",
            "confirmed_by": ["arxiv"],
        }

    def _confirm_with_semantic_scholar(self, clue: dict[str, Any]) -> Optional[dict[str, Any]]:
        fields = "title,authors,year,venue,abstract,externalIds,citationCount,openAccessPdf,url"
        headers: dict[str, str] = {"User-Agent": "research-harness/ScholarSearch"}
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
        if api_key:
            headers["x-api-key"] = api_key
        identifier = ""
        if clue.get("doi"):
            identifier = f"DOI:{clue['doi']}"
        elif clue.get("arxiv_id"):
            identifier = f"ARXIV:{clue['arxiv_id']}"
        if identifier:
            url = f"https://api.semanticscholar.org/graph/v1/paper/{quote(str(identifier), safe=':')}?{urlencode({'fields': fields})}"
            response = requests.get(url, headers=headers, timeout=20)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return self._semantic_scholar_payload_to_record(response.json())

        title = _stringify_field(clue.get("title"))
        if not title:
            return None
        response = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": title, "limit": 3, "fields": fields},
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        for item in response.json().get("data", []) or []:
            record = self._semantic_scholar_payload_to_record(item)
            if self._record_matches_clue(record, clue):
                return record
        return None

    def _semantic_scholar_payload_to_record(self, item: dict[str, Any]) -> dict[str, Any]:
        external = item.get("externalIds") or {}
        oa_pdf = item.get("openAccessPdf") or {}
        pdf_url = _stringify_field(oa_pdf.get("url") if isinstance(oa_pdf, dict) else "")
        return {
            "title": _stringify_field(item.get("title")),
            "authors": _dedupe_strings([author.get("name", "") for author in item.get("authors") or [] if isinstance(author, dict)]),
            "year": _coerce_int(item.get("year")),
            "venue": _stringify_field(item.get("venue")),
            "abstract": _stringify_field(item.get("abstract")),
            "doi": _stringify_field(external.get("DOI")).lower(),
            "arxiv_id": _stringify_field(external.get("ArXiv") or external.get("arXiv")),
            "s2_id": _stringify_field(item.get("paperId")),
            "url": _stringify_field(item.get("url")),
            "citation_count": _coerce_int(item.get("citationCount")),
            "pdf_candidates": [
                _pdf_candidate(pdf_url, source_type="open_access_pdf", provider="semantic_scholar", confidence=0.92, license_hint="open_access")
            ] if pdf_url else [],
            "source_provider": "semantic_scholar",
            "confirmed_by": ["semantic_scholar"],
        }

    def _confirm_with_openalex(self, clue: dict[str, Any]) -> Optional[dict[str, Any]]:
        params: dict[str, str] = {}
        api_key = os.getenv("OPENALEX_API_KEY", "").strip()
        mailto = os.getenv("OPENALEX_MAILTO", "").strip()
        if api_key:
            params["api_key"] = api_key
        if mailto:
            params["mailto"] = mailto
        doi = _stringify_field(clue.get("doi"))
        if doi:
            url = f"https://api.openalex.org/works/doi:{doi}"
            response = requests.get(url, params=params, timeout=20, headers={"User-Agent": "research-harness/ScholarSearch"})
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return self._openalex_payload_to_record(response.json())

        title = _stringify_field(clue.get("title"))
        if not title:
            return None
        params.update({"search": title, "per-page": "3"})
        response = requests.get(
            "https://api.openalex.org/works",
            params=params,
            timeout=20,
            headers={"User-Agent": "research-harness/ScholarSearch"},
        )
        response.raise_for_status()
        for item in response.json().get("results", []) or []:
            record = self._openalex_payload_to_record(item)
            if self._record_matches_clue(record, clue):
                return record
        return None

    def _openalex_payload_to_record(self, item: dict[str, Any]) -> dict[str, Any]:
        primary_location = item.get("primary_location") or {}
        best_location = item.get("best_oa_location") or {}
        source = primary_location.get("source") or {}
        pdf_url = _stringify_field(best_location.get("pdf_url") or primary_location.get("pdf_url"))
        landing = _stringify_field(primary_location.get("landing_page_url") or item.get("doi") or item.get("id"))
        authors = []
        for authorship in item.get("authorships") or []:
            if isinstance(authorship, dict):
                author = authorship.get("author") or {}
                if isinstance(author, dict):
                    authors.append(_stringify_field(author.get("display_name")))
        return {
            "title": _stringify_field(item.get("title")),
            "authors": _dedupe_strings(authors),
            "year": _coerce_int(item.get("publication_year")),
            "venue": _stringify_field(source.get("display_name") if isinstance(source, dict) else ""),
            "abstract": self._openalex_abstract(item),
            "doi": _stringify_field(item.get("doi")).removeprefix("https://doi.org/").lower(),
            "openalex_id": _stringify_field(item.get("id")).rsplit("/", 1)[-1],
            "url": landing,
            "citation_count": _coerce_int(item.get("cited_by_count")),
            "pdf_candidates": [
                _pdf_candidate(pdf_url, source_type="openalex_content", provider="openalex", confidence=0.94, license_hint="open_access")
            ] if pdf_url else [],
            "source_provider": "openalex",
            "confirmed_by": ["openalex"],
        }

    @staticmethod
    def _openalex_abstract(item: dict[str, Any]) -> str:
        inverted = item.get("abstract_inverted_index")
        if not isinstance(inverted, dict):
            return ""
        pairs: list[tuple[int, str]] = []
        for word, positions in inverted.items():
            if not isinstance(positions, list):
                continue
            for position in positions:
                if isinstance(position, int):
                    pairs.append((position, str(word)))
        return " ".join(word for _, word in sorted(pairs))

    def _search_one(
        self,
        query: str,
        *,
        max_results: int,
        year_from: Optional[int],
        year_to: Optional[int],
        providers: list[str],
    ) -> str:
        clues, clue_error = self._serper_scholar_clues(
            query,
            max_results=max_results,
            year_from=year_from,
            year_to=year_to,
        )
        if clue_error and not clues:
            return clue_error

        verified: dict[str, dict[str, Any]] = {}
        unverified: list[dict[str, Any]] = []
        for clue in clues:
            confirmed = self._confirm_clue(clue, providers)
            if confirmed:
                key = _fingerprint_paper(confirmed)
                verified[key] = confirmed if key not in verified else _merge_paper_records(verified[key], confirmed)
            else:
                unverified.append(clue)
            if len(verified) >= max_results:
                break

        return self._format_scholar_response(query, list(verified.values())[:max_results], unverified, clue_error)

    def _format_scholar_response(
        self,
        query: str,
        verified: list[dict[str, Any]],
        unverified: list[dict[str, Any]],
        clue_error: str,
    ) -> str:
        lines = [
            f"A two-layer scholar search for '{query}' found {len(verified)} structurally confirmed result(s).",
            "",
            "## Scholar Results",
        ]
        if not verified:
            lines.append("No structurally verified results were found. Serper clues, if any, are listed below for manual follow-up and should not be treated as confirmed papers.")
        for idx, paper in enumerate(verified, start=1):
            title = _stringify_field(paper.get("title")) or "Untitled result"
            url = _stringify_field(paper.get("url")) or "no available link"
            lines.append(f"{idx}. [{title}]({url})")
            authors = ", ".join(list(paper.get("authors") or [])[:8])
            if authors:
                lines.append(f"authors: {authors}")
            for label, key in (
                ("year", "year"),
                ("venue", "venue"),
                ("doi", "doi"),
                ("arxiv_id", "arxiv_id"),
                ("s2_id", "s2_id"),
                ("openalex_id", "openalex_id"),
                ("citation_count", "citation_count"),
            ):
                value = paper.get(key)
                if value:
                    lines.append(f"{label}: {value}")
            confirmed_by = ", ".join(list(paper.get("confirmed_by") or []))
            if confirmed_by:
                lines.append(f"confirmed_by: {confirmed_by}")
            abstract = _stringify_field(paper.get("abstract"))
            if abstract:
                lines.append(f"abstract: {abstract[:900]}")
            candidates = list(paper.get("pdf_candidates") or [])
            if candidates:
                lines.append("pdf_candidates:")
                for candidate in candidates[:5]:
                    lines.append(
                        "- "
                        + json.dumps(
                            {
                                "url": candidate.get("url"),
                                "source_type": candidate.get("source_type"),
                                "provider": candidate.get("provider"),
                                "confidence": candidate.get("confidence"),
                                "license_hint": candidate.get("license_hint", ""),
                            },
                            ensure_ascii=False,
                        )
                    )
            lines.append("")

        lines.append("## Structured JSON")
        lines.append(json.dumps({"papers": verified, "unverified_clues": unverified[:10]}, ensure_ascii=False, indent=2))
        if clue_error:
            lines.append("")
            lines.append(f"serper_warning: {clue_error}")
        if unverified:
            lines.append("")
            lines.append("## Unverified Serper Clues")
            for idx, clue in enumerate(unverified[:10], start=1):
                title = _stringify_field(clue.get("title")) or "Untitled clue"
                url = _stringify_field(clue.get("url")) or "no available link"
                lines.append(f"{idx}. [{title}]({url})")
                if clue.get("year"):
                    lines.append(f"year: {clue['year']}")
                if clue.get("doi"):
                    lines.append(f"doi_clue: {clue['doi']}")
                if clue.get("arxiv_id"):
                    lines.append(f"arxiv_id_clue: {clue['arxiv_id']}")
                snippet = _stringify_field(clue.get("abstract"))
                if snippet:
                    lines.append(f"snippet: {snippet[:500]}")
        return "\n".join(lines).strip()

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self.parse_json_args(params)
            query = params["query"]
        except ValueError as exc:
            return f"[ScholarSearch] {exc}"

        try:
            max_results = int(params.get("max_results", DEFAULT_SCHOLAR_MAX_RESULTS))
            year_from_raw = params.get("year_from")
            year_to_raw = params.get("year_to")
            year_from = int(year_from_raw) if year_from_raw is not None else None
            year_to = int(year_to_raw) if year_to_raw is not None else None
        except (TypeError, ValueError):
            return "[ScholarSearch] max_results, year_from, and year_to must be integers when provided."
        if max_results <= 0:
            return "[ScholarSearch] max_results must be > 0."
        providers = self._provider_order(params.get("providers"))

        if isinstance(query, list):
            with ThreadPoolExecutor(max_workers=3) as executor:
                response = list(
                    executor.map(
                        lambda item: self._search_one(
                            item,
                            max_results=max_results,
                            year_from=year_from,
                            year_to=year_to,
                            providers=providers,
                        ),
                        query,
                    )
                )
            response = "\n=======\n".join(response)
        else:
            return "[ScholarSearch] 'query' must be a list of strings."
        return response


class DownloadPDF(ToolBase):
    name = "DownloadPDF"
    description = "Download a PDF from trusted/open candidates and validate that the saved file is a real PDF. Rejects HTML, landing pages, tiny files, and paths outside the workspace."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Optional explicit URL to try. Prefer direct PDF or open-access URLs.",
            },
            "title": {
                "type": "string",
                "description": "Optional paper title used for generated filenames.",
            },
            "doi": {
                "type": "string",
                "description": "Optional DOI. Used only for open-access Unpaywall lookup.",
            },
            "arxiv_id": {
                "type": "string",
                "description": "Optional arXiv ID used to construct a direct arXiv PDF URL.",
            },
            "pdf_candidates": {
                "type": "array",
                "items": {"type": ["object", "string"]},
                "description": "Candidate PDF URLs from ScholarSearch. Dict candidates may include url, source_type, provider, and confidence.",
            },
            "output_path": {
                "type": "string",
                "description": "Destination PDF path. Relative paths are resolved from the current workspace.",
            },
            "output_dir": {
                "type": "string",
                "description": "Destination directory. A safe filename will be generated when output_path is omitted.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Whether to overwrite an existing file. Default is false.",
            },
        },
        "required": [],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    def _unpaywall_pdf_url(self, doi: str) -> str:
        clean_doi = _stringify_field(doi).removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        if not clean_doi:
            return ""
        email = os.environ.get("UNPAYWALL_EMAIL", "research-harness@academic-tools.org")
        response = requests.get(
            f"https://api.unpaywall.org/v2/{quote(clean_doi, safe='/')}",
            params={"email": email},
            timeout=10,
            headers={"User-Agent": "research-harness/DownloadPDF"},
        )
        if response.status_code != 200:
            return ""
        try:
            data = response.json()
        except ValueError:
            return ""
        best = data.get("best_oa_location") or {}
        pdf_url = _stringify_field(best.get("url_for_pdf") if isinstance(best, dict) else "")
        if pdf_url:
            return pdf_url
        for location in data.get("oa_locations") or []:
            if isinstance(location, dict):
                pdf_url = _stringify_field(location.get("url_for_pdf"))
                if pdf_url:
                    return pdf_url
        return ""

    def _candidate_urls(self, params: dict[str, Any]) -> list[str]:
        urls: list[str] = []
        for candidate in params.get("pdf_candidates") or []:
            if isinstance(candidate, str):
                urls.append(candidate)
            elif isinstance(candidate, dict):
                source_type = _stringify_field(candidate.get("source_type"))
                provider = _stringify_field(candidate.get("provider"))
                url = _stringify_field(candidate.get("url"))
                if not url:
                    continue
                if source_type in {"arxiv_pdf", "openreview_pdf", "open_access_pdf", "openalex_content", "manual_pdf"} or provider in {"arxiv", "openreview", "semantic_scholar", "openalex"}:
                    urls.append(url)
        explicit_url = _stringify_field(params.get("url"))
        if explicit_url:
            urls.append(explicit_url)
        arxiv_id = _stringify_field(params.get("arxiv_id")).removeprefix("arXiv:").strip()
        if arxiv_id:
            urls.append(f"https://arxiv.org/pdf/{arxiv_id}.pdf")
        doi = _stringify_field(params.get("doi"))
        if doi:
            try:
                unpaywall_url = self._unpaywall_pdf_url(doi)
            except requests.RequestException:
                unpaywall_url = ""
            if unpaywall_url:
                urls.append(unpaywall_url)
        return _dedupe_urls(urls)

    def _target_path(self, params: dict[str, Any], *, base_root: Path) -> Path:
        raw_output_path = _stringify_field(params.get("output_path"))
        if raw_output_path:
            target = validate_tool_path(raw_output_path, "DownloadPDF write access", base_root=base_root)
        else:
            raw_output_dir = _stringify_field(params.get("output_dir"))
            if not raw_output_dir:
                raise ValueError("one of output_path or output_dir is required")
            output_dir = validate_tool_path(raw_output_dir, "DownloadPDF write access", base_root=base_root)
            title = _stringify_field(params.get("title")) or "paper"
            arxiv_id = _stringify_field(params.get("arxiv_id")).removeprefix("arXiv:").strip()
            doi = _stringify_field(params.get("doi"))
            if arxiv_id:
                prefix = _sanitize_filename(arxiv_id)
            elif doi:
                prefix = hashlib.sha256(doi.encode("utf-8")).hexdigest()[:10]
            else:
                prefix = "paper"
            target = output_dir / f"{prefix}_{_sanitize_filename(title)}.pdf"
            target = validate_tool_path(target, "DownloadPDF write access", base_root=base_root)
        if target.suffix.lower() != ".pdf":
            raise ValueError("output_path must end with .pdf")
        return target

    def _format_result(
        self,
        *,
        status: str,
        target: Optional[Path],
        source_url: str = "",
        attempted_urls: Optional[list[str]] = None,
        failure_reason: str = "",
        byte_count: int = 0,
        validated: bool = False,
    ) -> str:
        lines = [
            f"status: {status}",
            f"validated: {str(validated).lower()}",
        ]
        if target is not None:
            lines.append(f"path: {target}")
        if source_url:
            lines.append(f"source_url: {source_url}")
        if byte_count:
            lines.append(f"bytes: {byte_count}")
        if failure_reason:
            lines.append(f"failure_reason: {failure_reason}")
        lines.append("attempted_urls:")
        for url in attempted_urls or []:
            lines.append(f"- {url}")
        return "\n".join(lines)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self.parse_json_args(params)
        except ValueError as exc:
            return f"[DownloadPDF] {exc}"
        base_root = kwargs.get("workspace_root") or workspace_root()
        base_root = Path(base_root).expanduser().resolve()

        try:
            target = self._target_path(params, base_root=base_root)
        except ValueError as exc:
            return f"[DownloadPDF] {exc}"

        overwrite = bool(params.get("overwrite", False))
        if target.exists() and not overwrite:
            return f"[DownloadPDF] File already exists and overwrite is false: {target}"

        candidate_urls = self._candidate_urls(params)
        if not candidate_urls:
            return self._format_result(
                status="failed",
                target=target,
                attempted_urls=[],
                failure_reason="no candidate URLs",
            )

        attempted: list[str] = []
        saw_paywall = False
        last_reason = "all candidate URLs failed"
        for url in candidate_urls:
            attempted.append(url)
            try:
                response = requests.get(
                    url,
                    timeout=DEFAULT_DOWNLOADPDF_TIMEOUT_SECONDS,
                    headers={"User-Agent": "research-harness/DownloadPDF"},
                    allow_redirects=True,
                )
            except requests.RequestException as exc:
                last_reason = f"request failed: {exc}"
                continue

            final_url = getattr(response, "url", url) or url
            status_code = int(getattr(response, "status_code", 0) or 0)
            if status_code in {401, 403} and _host_matches(final_url, PAYWALL_DOMAINS):
                saw_paywall = True
                last_reason = f"paywall or access denied: HTTP {status_code}"
                continue
            if status_code != 200:
                last_reason = f"HTTP {status_code}"
                continue

            payload = getattr(response, "content", b"") or b""
            if _looks_like_html(payload):
                last_reason = "response was HTML, not PDF"
                continue
            if not _is_pdf_bytes(payload):
                last_reason = "response does not start with %PDF"
                continue
            if len(payload) < MIN_VALID_PDF_BYTES:
                last_reason = f"PDF response too small ({len(payload)} bytes)"
                continue

            tmp_path = target.with_name(target.name + ".part")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_bytes(payload)
                written = tmp_path.read_bytes()
                if not _is_pdf_bytes(written) or len(written) < MIN_VALID_PDF_BYTES:
                    tmp_path.unlink(missing_ok=True)
                    last_reason = "written file failed PDF validation"
                    continue
                tmp_path.replace(target)
            except OSError as exc:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                last_reason = f"write failed: {exc}"
                continue

            return self._format_result(
                status="success",
                target=target,
                source_url=final_url,
                attempted_urls=attempted,
                byte_count=len(payload),
                validated=True,
            )

        return self._format_result(
            status="needs_manual" if saw_paywall else "failed",
            target=target,
            attempted_urls=attempted,
            failure_reason=last_reason,
        )


class WebFetch(ToolBase):
    name = "WebFetch"
    description = "Fetch webpage content and return evidence plus a goal-focused summary."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {
                    "type": "string",
                },
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs.",
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s).",
            },
        },
        "required": ["url", "goal"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self._summary_client: Optional[OpenAI] = None
        self._summary_api_base: Optional[str] = None
        self._summary_model_name = os.environ.get(
            "SUMMARY_MODEL_NAME",
            os.environ.get("MODEL_NAME", DEFAULT_SUMMARY_MODEL_NAME),
        )
        self._summary_timeout_seconds = float(
            os.getenv("WEBFETCH_LLM_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", str(DEFAULT_WEBFETCH_TIMEOUT_SECONDS)))
        )
        self._summary_temperature = float(
            os.getenv("WEBFETCH_SUMMARY_TEMPERATURE", str(DEFAULT_WEBFETCH_SUMMARY_TEMPERATURE))
        )

    def _ensure_summary_client(self) -> Optional[OpenAI]:
        if self._summary_client is not None:
            return self._summary_client
        self._summary_api_base = os.environ.get("API_BASE")
        self._summary_model_name = os.environ.get(
            "SUMMARY_MODEL_NAME",
            os.environ.get("MODEL_NAME", DEFAULT_SUMMARY_MODEL_NAME),
        )
        self._summary_timeout_seconds = float(
            os.getenv("WEBFETCH_LLM_TIMEOUT_SECONDS", os.getenv("LLM_TIMEOUT_SECONDS", str(DEFAULT_WEBFETCH_TIMEOUT_SECONDS)))
        )
        self._summary_temperature = float(
            os.getenv("WEBFETCH_SUMMARY_TEMPERATURE", str(DEFAULT_WEBFETCH_SUMMARY_TEMPERATURE))
        )
        if not self._summary_api_base:
            return None
        self._summary_client = OpenAI(
            api_key=os.environ.get("API_KEY", "EMPTY"),
            base_url=self._summary_api_base,
            timeout=self._summary_timeout_seconds,
        )
        return self._summary_client

    @staticmethod
    def _remaining_budget_seconds(runtime_deadline: Optional[float]) -> Optional[float]:
        if runtime_deadline is None:
            return None
        return runtime_deadline - time.time()

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self.parse_json_args(params)
            url = params["url"]
            goal = params["goal"]
        except ValueError as exc:
            return f"[WebFetch] {exc}"
        runtime_deadline = kwargs.get("runtime_deadline")

        start_time = time.time()

        if isinstance(url, str):
            response = self.readpage_jina(url, goal, runtime_deadline=runtime_deadline)
        elif isinstance(url, list):
            response = []
            start_time = time.time()
            for one_url in url:
                remaining = self._remaining_budget_seconds(runtime_deadline)
                if remaining is not None and remaining <= 0:
                    cur_response = _webfetch_failure(
                        url=one_url,
                        goal=goal,
                        reason="Agent runtime limit reached before WebFetch could complete.",
                    )
                    response.append(cur_response)
                    continue
                if time.time() - start_time > 900:
                    cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=one_url, goal=goal)
                    cur_response += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    cur_response += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                else:
                    cur_response = self.readpage_jina(one_url, goal, runtime_deadline=runtime_deadline)
                response.append(cur_response)
            response = "\n=======\n".join(response)
        else:
            return "[WebFetch] 'url' must be a string or a list of strings."

        if visit_debug_enabled():
            print(f"Summary Length {len(response)}")
        return response.strip()

    def call_server(self, msgs, max_retries=2, runtime_deadline: Optional[float] = None):
        client = self._ensure_summary_client()
        if client is None or not self._summary_api_base:
            return "[WebFetch] Summary model error: API_BASE is not set."
        last_error = "unknown summary-model error"
        for attempt in range(max_retries):
            remaining = self._remaining_budget_seconds(runtime_deadline)
            if remaining is not None and remaining <= 0:
                return "[WebFetch] Summary model error: agent runtime limit reached."
            try:
                request_client = (
                    client.with_options(timeout=min(self._summary_timeout_seconds, max(remaining, 0.001)))
                    if remaining is not None
                    else client
                )
                request_kwargs = {
                    "model": self._summary_model_name,
                    "messages": msgs,
                }
                apply_sampling_params(
                    request_kwargs,
                    model_name=self._summary_model_name,
                    temperature=self._summary_temperature,
                )
                chat_response = request_client.chat.completions.create(**request_kwargs)
                content = chat_response.choices[0].message.content
                if content:
                    return content
                last_error = "empty response from summary model"
            except (APIError, APIConnectionError, APITimeoutError) as exc:
                last_error = str(exc)
                if attempt == (max_retries - 1):
                    return f"[WebFetch] Summary model error: {last_error}"

        return f"[WebFetch] Summary model error: {last_error}"

    def jina_readpage(self, url: str, runtime_deadline: Optional[float] = None) -> str:
        max_retries = 3
        timeout = 50
        jina_api_key = os.getenv("JINA_API_KEYS", "").strip()
        if not jina_api_key:
            return "[WebFetch] JINA_API_KEYS is not set."

        last_error = "unknown page-fetch error"
        for attempt in range(max_retries):
            headers = {
                "Authorization": f"Bearer {jina_api_key}",
            }
            try:
                remaining = self._remaining_budget_seconds(runtime_deadline)
                if remaining is not None and remaining <= 0:
                    return "[WebFetch] Failed to read page: agent runtime limit reached."
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=min(timeout, max(remaining, 0.001)) if remaining is not None else timeout,
                )
                if response.status_code == 200:
                    return response.text
                if visit_debug_enabled():
                    print(response.text)
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except requests.RequestException as exc:
                last_error = str(exc)
                remaining = self._remaining_budget_seconds(runtime_deadline)
                if remaining is not None and remaining <= 0:
                    return "[WebFetch] Failed to read page: agent runtime limit reached."
                time.sleep(min(0.5, remaining) if remaining is not None else 0.5)
                if attempt == max_retries - 1:
                    return f"[WebFetch] Failed to read page: {last_error}"

        return f"[WebFetch] Failed to read page: {last_error}"

    def html_readpage_jina(self, url: str, runtime_deadline: Optional[float] = None) -> str:
        max_attempts = 8
        for _ in range(max_attempts):
            remaining = self._remaining_budget_seconds(runtime_deadline)
            if remaining is not None and remaining <= 0:
                return "[WebFetch] Failed to read page: agent runtime limit reached."
            content = self.jina_readpage(url, runtime_deadline=runtime_deadline)
            if content and not content.startswith("[WebFetch] Failed to read page:") and content != "[WebFetch] Empty content." and not content.startswith("[document_parser]"):
                return content
        return "[WebFetch] Failed to read page: exhausted retries"

    def readpage_jina(self, url: str, goal: str, runtime_deadline: Optional[float] = None) -> str:
        summary_page_func = self.call_server
        max_retries = int(os.getenv("VISIT_SERVER_MAX_RETRIES", 1))

        content = self.html_readpage_jina(url, runtime_deadline=runtime_deadline)

        if content and not content.startswith("[WebFetch] Failed to read page:") and content != "[WebFetch] Empty content." and not content.startswith("[document_parser]"):
            content = truncate_to_tokens(content, max_tokens=95000)
            messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
            raw = summary_page_func(messages, max_retries=max_retries, runtime_deadline=runtime_deadline)
            summary_retries = 3
            while len(raw) < 10 and summary_retries >= 0:
                remaining = self._remaining_budget_seconds(runtime_deadline)
                if remaining is not None and remaining <= 0:
                    return _webfetch_failure(
                        url=url,
                        goal=goal,
                        reason="Agent runtime limit reached before WebFetch could complete.",
                    )
                truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                status_msg = (
                    f"[WebFetch] Summary url[{url}] "
                    f"attempt {3 - summary_retries + 1}/3, "
                    f"content length: {len(content)}, "
                    f"truncating to {truncate_length} chars"
                ) if summary_retries > 0 else (
                    f"[WebFetch] Summary url[{url}] failed after 3 attempts, "
                    f"final truncation to 25000 chars"
                )
                if visit_debug_enabled():
                    print(status_msg)
                content = content[:truncate_length]
                extraction_prompt = EXTRACTOR_PROMPT.format(
                    webpage_content=content,
                    goal=goal,
                )
                messages = [{"role": "user", "content": extraction_prompt}]
                raw = summary_page_func(messages, max_retries=max_retries, runtime_deadline=runtime_deadline)
                summary_retries -= 1

            parse_retry_times = 0
            parsed = _parse_extractor_payload(raw)
            while parse_retry_times < 3:
                if parsed is not None:
                    break
                remaining = self._remaining_budget_seconds(runtime_deadline)
                if remaining is not None and remaining <= 0:
                    return _webfetch_failure(
                        url=url,
                        goal=goal,
                        reason="Agent runtime limit reached before WebFetch could complete.",
                    )
                raw = summary_page_func(messages, max_retries=max_retries, runtime_deadline=runtime_deadline)
                parsed = _parse_extractor_payload(raw)
                parse_retry_times += 1

            if parsed is None:
                reason = "The webpage content was fetched, but the summary model did not return the required evidence and summary fields."
                if isinstance(raw, str) and raw.startswith("[WebFetch] Summary model error:"):
                    reason = raw
                useful_information = _webfetch_failure(
                    url=url,
                    goal=goal,
                    reason=reason,
                )
            else:
                evidence, summary = parsed
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + evidence + "\n\n"
                useful_information += "Summary: \n" + summary + "\n\n"

            if len(useful_information) < 10 and summary_retries < 0:
                if visit_debug_enabled():
                    print("[WebFetch] Could not generate valid summary after maximum retries")
                useful_information = "[WebFetch] Failed to read page."

            return useful_information

        return _webfetch_failure(
            url=url,
            goal=goal,
            reason="The provided webpage content could not be accessed. Please check the URL or file format.",
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run web tools directly.")
    subparsers = parser.add_subparsers(dest="tool", required=True)

    search_parser = subparsers.add_parser("search", help="Run WebSearch.")
    search_parser.add_argument("query", nargs="+")

    scholar_parser = subparsers.add_parser("scholar", help="Run ScholarSearch.")
    scholar_parser.add_argument("query", nargs="+")

    download_parser = subparsers.add_parser("download-pdf", help="Run DownloadPDF.")
    download_parser.add_argument("url")
    download_parser.add_argument("output_path")

    fetch_parser = subparsers.add_parser("fetch", help="Run WebFetch.")
    fetch_parser.add_argument("url")
    fetch_parser.add_argument("goal")

    args = parser.parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")

    if args.tool == "search":
        result = WebSearch().call({"query": [" ".join(args.query)]})
    elif args.tool == "scholar":
        result = ScholarSearch().call({"query": [" ".join(args.query)]})
    elif args.tool == "download-pdf":
        result = DownloadPDF().call({"url": args.url, "output_path": args.output_path})
    else:
        result = WebFetch().call({"url": args.url, "goal": args.goal})
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
