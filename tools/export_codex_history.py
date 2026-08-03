#!/usr/bin/env python3
"""Export local Codex rollouts for this repository into a public-safe archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TARGET_CWD = r"D:\100XH100"
MAX_TOOL_OUTPUT_CHARS = 32_000

SECRET_PATTERNS = [
    (re.compile(r"\b(gAAAAA[A-Za-z0-9_-]{80,}={0,2})\b"), "[REDACTED_OPAQUE_AGENT_PAYLOAD]"),
    (re.compile(r"\b(hf_[A-Za-z0-9_-]{20,})\b"), "[REDACTED_HUGGINGFACE_TOKEN]"),
    (re.compile(r"\b(gh[oprsu]_[A-Za-z0-9_]{20,})\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\b(sk-[A-Za-z0-9_-]{20,})\b"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"\b(AIza[0-9A-Za-z_-]{20,})\b"), "[REDACTED_GOOGLE_KEY]"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s\"']+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)((?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret)\s*[:=]\s*)[^\s,;\"']+"), r"\1[REDACTED]"),
    (re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]"),
]


def redact(text: str, stats: Counter[str]) -> str:
    # Keep the archive valid UTF-8 text even when a tool returned binary bytes.
    text = text.replace("\x00", "[NUL]")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", lambda match: f"[CTRL-{ord(match.group()):02X}]", text)
    for pattern, replacement in SECRET_PATTERNS:
        text, count = pattern.subn(replacement, text)
        if count:
            stats[replacement] += count
    return text


def clipped(text: str) -> tuple[str, int]:
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text, 0
    half = MAX_TOOL_OUTPUT_CHARS // 2
    removed = len(text) - MAX_TOOL_OUTPUT_CHARS
    return (
        text[:half]
        + f"\n\n[... {removed} characters omitted from oversized tool output ...]\n\n"
        + text[-half:],
        removed,
    )


def content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False, indent=2)
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            value = item.get("text") or item.get("input_text") or item.get("output_text")
            if value:
                parts.append(str(value))
            elif item.get("type") not in {"input_image", "image"}:
                parts.append(json.dumps(item, ensure_ascii=False, indent=2))
            else:
                parts.append("[image attachment]")
        else:
            parts.append(str(item))
    return "\n".join(parts)


def fence(text: str) -> str:
    marker = "```"
    while marker in text:
        marker += "`"
    return f"{marker}text\n{text}\n{marker}"


def parse_session(path: Path, secret_stats: Counter[str]) -> tuple[dict, list[dict], Counter[str]] | None:
    meta: dict | None = None
    entries: list[dict] = []
    kinds: Counter[str] = Counter()
    seen_messages: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = record.get("type", "unknown")
            payload = record.get("payload") or {}
            if kind == "session_meta":
                if payload.get("cwd", "").casefold() != TARGET_CWD.casefold():
                    return None
                meta = {
                    "id": payload.get("id") or payload.get("session_id"),
                    "timestamp": payload.get("timestamp") or record.get("timestamp"),
                    "cwd": payload.get("cwd"),
                    "originator": payload.get("originator"),
                    "cli_version": payload.get("cli_version"),
                    "source": payload.get("source"),
                    "model_provider": payload.get("model_provider"),
                    "source_file": str(path),
                }
                continue
            if meta is None:
                continue
            timestamp = record.get("timestamp", "")
            if kind == "response_item":
                item_type = payload.get("type", "unknown")
                if item_type == "message":
                    role = payload.get("role", "unknown")
                    if role not in {"user", "assistant"}:
                        continue
                    text = content_text(payload.get("content", []))
                    key = (timestamp, role, text)
                    if text and key not in seen_messages:
                        entries.append({"timestamp": timestamp, "kind": role, "text": text, "line": line_no})
                        seen_messages.add(key)
                        kinds[role] += 1
                elif item_type in {"function_call", "custom_tool_call"}:
                    name = payload.get("name") or payload.get("tool_name") or item_type
                    args = payload.get("arguments") or payload.get("input") or ""
                    entries.append({"timestamp": timestamp, "kind": "tool_call", "name": name, "text": str(args), "line": line_no})
                    kinds["tool_call"] += 1
                elif item_type in {"function_call_output", "custom_tool_call_output"}:
                    output = payload.get("output") or payload.get("content") or ""
                    text, omitted = clipped(content_text(output))
                    entries.append({"timestamp": timestamp, "kind": "tool_output", "text": text, "omitted": omitted, "line": line_no})
                    kinds["tool_output"] += 1
            elif kind == "event_msg" and payload.get("type") in {"agent_message", "user_message"}:
                role = "assistant" if payload.get("type") == "agent_message" else "user"
                text = str(payload.get("message") or payload.get("text") or "")
                key = (timestamp, role, text)
                if text and key not in seen_messages:
                    entries.append({"timestamp": timestamp, "kind": role, "text": text, "line": line_no})
                    seen_messages.add(key)
                    kinds[role] += 1
    if not meta:
        return None
    for entry in entries:
        entry["text"] = redact(entry["text"], secret_stats)
    return meta, entries, kinds


def render_session(meta: dict, entries: list[dict], kinds: Counter[str]) -> str:
    lines = [
        f"# Codex session {meta['id']}",
        "",
        f"- Started: `{meta['timestamp']}`",
        f"- Working directory: `{meta['cwd']}`",
        f"- Originator: `{meta.get('originator')}`",
        f"- Codex CLI: `{meta.get('cli_version')}`",
        f"- Source log: `{meta['source_file']}`",
        f"- Entries: user={kinds['user']}, assistant={kinds['assistant']}, tool calls={kinds['tool_call']}, tool outputs={kinds['tool_output']}",
        "",
        "This is a chronological export of the user/assistant conversation and tool activity. Internal system/developer instructions and token accounting are intentionally excluded. Secret-shaped values are redacted; oversized tool outputs retain their beginning and end with the omitted character count.",
        "",
    ]
    labels = {"user": "User", "assistant": "Codex", "tool_call": "Tool call", "tool_output": "Tool output"}
    for i, entry in enumerate(entries, 1):
        title = labels[entry["kind"]]
        if entry.get("name"):
            title += f": `{entry['name']}`"
        lines.extend([f"## {i}. {title}", "", f"Time: `{entry['timestamp']}` · source line `{entry['line']}`", ""])
        if entry.get("omitted"):
            lines.extend([f"Oversized output: `{entry['omitted']}` characters omitted from the middle.", ""])
        lines.extend([fence(entry["text"]), ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    sessions_dir = output / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    secret_stats: Counter[str] = Counter()
    records: list[dict] = []
    totals: Counter[str] = Counter()
    total_source_bytes = 0
    for path in sorted(args.sessions_root.rglob("*.jsonl")):
        parsed = parse_session(path, secret_stats)
        if parsed is None:
            continue
        meta, entries, kinds = parsed
        stamp = (meta["timestamp"] or "unknown").replace(":", "-")
        month = stamp[:7] if len(stamp) >= 7 else "unknown"
        source_tag = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:10]
        destination = sessions_dir / month / f"{stamp}-{meta['id']}-{source_tag}.md"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(render_session(meta, entries, kinds), encoding="utf-8", newline="\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record = {**meta, "archive_file": destination.relative_to(output).as_posix(), "source_bytes": path.stat().st_size, "source_sha256": digest, **kinds}
        records.append(record)
        totals.update(kinds)
        total_source_bytes += path.stat().st_size
    records.sort(key=lambda item: item.get("timestamp") or "")
    index_lines = [
        "# Complete Codex work history for D:\\100XH100",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        f"Sessions: **{len(records)}**",
        f"Raw source size: **{total_source_bytes} bytes**",
        f"User entries: **{totals['user']}**",
        f"Codex entries: **{totals['assistant']}**",
        f"Tool calls: **{totals['tool_call']}**",
        f"Tool outputs: **{totals['tool_output']}**",
        "",
        "The archive is ordered by session start time. Every row includes the SHA-256 of its original local JSONL log, allowing exact provenance checks without publishing the private raw log container.",
        "",
        "| Started | Session | User | Codex | Calls | Outputs | Source bytes |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in records:
        index_lines.append(
            f"| {item['timestamp']} | [{item['id']}]({item['archive_file']}) | {item.get('user', 0)} | {item.get('assistant', 0)} | {item.get('tool_call', 0)} | {item.get('tool_output', 0)} | {item['source_bytes']} |"
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8", newline="\n")
    (output / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    redaction_lines = ["# Redaction report", "", "Only secret-shaped values were replaced; no conversational entries were removed.", ""]
    if secret_stats:
        redaction_lines += [f"- `{key}`: {count}" for key, count in sorted(secret_stats.items())]
    else:
        redaction_lines.append("No secret-shaped values were detected.")
    (output / "REDACTIONS.md").write_text("\n".join(redaction_lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"sessions": len(records), "source_bytes": total_source_bytes, "entries": totals, "redactions": secret_stats}, ensure_ascii=False, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
