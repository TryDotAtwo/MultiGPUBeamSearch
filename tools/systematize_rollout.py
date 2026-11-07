from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


MOLAB_URL_RE = re.compile(r"https?://sb-[a-z0-9-]+\.sb\.molab\.run/?", re.IGNORECASE)
# Do not require word boundaries: terminal output can concatenate the hostname
# with retry/status suffixes, and those still identify the private sandbox.
BARE_MOLAB_HOST_RE = re.compile(r"sb-[a-z0-9-]+\.sb\.molab\.run", re.IGNORECASE)
FLAG_TOKEN_RE = re.compile(r"(?i)(--token\s+)(?:\"[^\"]+\"|'[^']+'|\S+)")
NAMED_SECRET_RE = re.compile(
    r"(?i)((?:auth[_ -]?token|access[_ -]?token|api[_ -]?key|kaggle[_ -]?key|github[_ -]?token)"
    r"\s*[:=]\s*)(?:\"[^\"]+\"|'[^']+'|\S+)"
)
GITHUB_TOKEN_RE = re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{12,})\b")
STANDALONE_HEX_SECRET_RE = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64,}(?![0-9A-Fa-f])")
BEARER_RE = re.compile(r"(?i)(Authorization\s*:\s*Bearer\s+)[A-Za-z0-9._~+/=-]+")

EVIDENCE_RE = re.compile(
    r"(?i)(depth_(?:done|start|sec)|benchmark|tflop|pflop|items/s|throughput|latency|"
    r"vram|memory|oom|overflow|fatal|error|failed|failure|hang|timeout|http\s*410|"
    r"status|complete|solution|replay|capacity|frontier|ring|slot|stream[1-5]|"
    r"micro|shard|accumul|materiali|nvfp4|mxfp|fp16|fp32|relu|silu|tma|dsm|"
    r"mbarrier|cutlass|cuda|molab|marimo|commit|branch|push|kernel version|exit(?:code)?)"
)


@dataclass(frozen=True)
class RolloutEvent:
    source_line: int
    timestamp: str
    kind: str
    text: str
    call_id: str = ""


@dataclass(frozen=True)
class ExtractionResult:
    events: list[RolloutEvent]
    source_lines: int
    skipped_lines: int
    kind_counts: dict[str, int]


def redact_sensitive(text: str) -> str:
    """Remove live session endpoints and credentials before text leaves the source rollout."""
    text = MOLAB_URL_RE.sub("https://<molab-session>/", text)
    text = BARE_MOLAB_HOST_RE.sub("<molab-session>", text)
    text = FLAG_TOKEN_RE.sub(r"\1<redacted>", text)
    text = NAMED_SECRET_RE.sub(r"\1<redacted>", text)
    text = BEARER_RE.sub(r"\1<redacted>", text)
    text = GITHUB_TOKEN_RE.sub("<redacted-github-token>", text)
    text = STANDALONE_HEX_SECRET_RE.sub("<redacted-hex-secret>", text)
    return text


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for item in content:
        if isinstance(item, str):
            chunks.append(item)
        elif isinstance(item, dict):
            value = item.get("text")
            if isinstance(value, str):
                chunks.append(value)
    return "\n".join(chunks)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return repr(value)


def compact_tool_text(text: str, *, max_chars: int = 48_000) -> str:
    text = redact_sensitive(text)
    if len(text) <= max_chars:
        return text

    lines = text.splitlines()
    selected: list[str] = []
    selected.extend(lines[:80])
    for line in lines[80:-80]:
        if EVIDENCE_RE.search(line):
            selected.append(line)
            if len(selected) >= 480:
                break
    selected.extend(lines[-80:])
    body = "\n".join(selected)
    if len(body) > max_chars:
        body = body[: max_chars // 2] + "\n<...evidence truncated...>\n" + body[-max_chars // 2 :]
    return f"<compacted original_chars={len(text)} original_lines={len(lines)}>\n{body}"


def _iter_rollout(path: Path, counters: dict[str, int]) -> Iterable[RolloutEvent]:
    calls: dict[str, str] = {}
    counters["source_lines"] = 0
    counters["skipped"] = 0

    with path.open("r", encoding="utf-8") as handle:
        for source_line, raw in enumerate(handle, 1):
            counters["source_lines"] = source_line
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                counters["skipped"] += 1
                continue

            if row.get("type") != "response_item":
                continue
            payload = row.get("payload") or {}
            payload_type = payload.get("type")
            timestamp = str(row.get("timestamp") or "")

            if payload_type == "message":
                role = str(payload.get("role") or "")
                if role not in {"user", "assistant"}:
                    continue
                text = redact_sensitive(_message_text(payload.get("content")))
                if text.strip():
                    yield RolloutEvent(source_line, timestamp, f"message:{role}", text)
                continue

            if payload_type in {"function_call", "custom_tool_call"}:
                name = str(payload.get("name") or "unknown")
                call_id = str(payload.get("call_id") or payload.get("id") or "")
                calls[call_id] = name
                body = payload.get("arguments", payload.get("input", ""))
                text = compact_tool_text(_stringify(body), max_chars=16_000)
                yield RolloutEvent(source_line, timestamp, f"tool_call:{name}", text, call_id)
                continue

            if payload_type in {"function_call_output", "custom_tool_call_output"}:
                call_id = str(payload.get("call_id") or "")
                name = calls.get(call_id, "unknown")
                text = compact_tool_text(_stringify(payload.get("output", "")))
                yield RolloutEvent(source_line, timestamp, f"tool_output:{name}", text, call_id)


def extract_rollout(path: Path) -> ExtractionResult:
    counters: dict[str, int] = {}
    events = list(_iter_rollout(path, counters))

    counts = Counter(event.kind for event in events)
    return ExtractionResult(
        events,
        counters.get("source_lines", 0),
        counters.get("skipped", 0),
        dict(sorted(counts.items())),
    )


def _write_jsonl(path: Path, events: Iterable[RolloutEvent]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(asdict(event), ensure_ascii=False, sort_keys=True) + "\n")


def write_audit_corpus(source: Path, out_dir: Path) -> ExtractionResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    counters: dict[str, int] = {}
    kind_counts: Counter[str] = Counter()
    message_count = 0
    tool_count = 0

    with (
        (out_dir / "redacted_messages.jsonl").open("w", encoding="utf-8", newline="\n") as messages,
        (out_dir / "redacted_tool_evidence.jsonl").open("w", encoding="utf-8", newline="\n") as tools,
        (out_dir / "event_index.tsv").open("w", encoding="utf-8", newline="\n") as index,
    ):
        index.write("source_line\ttimestamp\tkind\tchars\tsha256\texcerpt\n")
        for event in _iter_rollout(source, counters):
            kind_counts[event.kind] += 1
            serialized = json.dumps(asdict(event), ensure_ascii=False, sort_keys=True) + "\n"
            if event.kind.startswith("message:"):
                messages.write(serialized)
                message_count += 1
            else:
                tools.write(serialized)
                tool_count += 1
            excerpt = " ".join(event.text.split())[:240].replace("\t", " ")
            digest = hashlib.sha256(event.text.encode("utf-8")).hexdigest()
            index.write(
                f"{event.source_line}\t{event.timestamp}\t{event.kind}\t{len(event.text)}\t{digest}\t{excerpt}\n"
            )

    stats = {
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "source_lines": counters.get("source_lines", 0),
        "skipped_invalid_json_lines": counters.get("skipped", 0),
        "events": message_count + tool_count,
        "messages": message_count,
        "tool_events": tool_count,
        "kind_counts": dict(sorted(kind_counts.items())),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ExtractionResult(
        [],
        counters.get("source_lines", 0),
        counters.get("skipped", 0),
        dict(sorted(kind_counts.items())),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a redacted chronological audit corpus from a Codex rollout")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    result = write_audit_corpus(args.source, args.out_dir)
    print(f"ROLLOUT_AUDIT=PASS events={sum(result.kind_counts.values())} source_lines={result.source_lines}")
    print(f"OUTPUT_DIR={args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
