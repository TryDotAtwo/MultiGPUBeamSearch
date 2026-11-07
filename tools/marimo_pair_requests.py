"""OpenSSL/requests transport for marimo-pair on Windows.

This mirrors execute-code.sh when Windows curl/Schannel or WSL bash is not
usable. Authentication stays in process arguments and is never persisted.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

import requests


def main() -> int:
    # Molab/pip/compiler streams may contain Unicode progress glyphs.  Windows
    # PowerShell commonly exposes cp1251 here; a failed print closes the SSE
    # request and marimo then interrupts the otherwise durable notebook cell.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--code")
    parser.add_argument("--code-file")
    parser.add_argument("--upload-local")
    parser.add_argument("--upload-remote")
    parser.add_argument("--output-log", help="Flush received notebook output to a local UTF-8 log")
    parser.add_argument("--direct", action="store_true", help="Ignore environment proxies for this invocation only; TLS verification remains enabled")
    args = parser.parse_args()
    upload_requested = args.upload_local is not None or args.upload_remote is not None
    if upload_requested:
        if not (args.upload_local and args.upload_remote):
            parser.error("--upload-local and --upload-remote must be provided together")
        if args.code is not None or args.code_file is not None:
            parser.error("upload mode cannot be combined with code mode")
    elif (args.code is None) == (args.code_file is None):
        parser.error("provide exactly one of --code or --code-file")

    code = args.code
    if args.code_file is not None:
        with open(args.code_file, "r", encoding="utf-8") as handle:
            code = handle.read()
    if upload_requested:
        payload = base64.b64encode(Path(args.upload_local).read_bytes()).decode("ascii")
        remote_path = json.dumps(args.upload_remote)
        code = (
            "import base64, pathlib\n"
            f"_p = pathlib.Path({remote_path})\n"
            "_p.parent.mkdir(parents=True, exist_ok=True)\n"
            f"_p.write_bytes(base64.b64decode({payload!r}))\n"
            "print(f'upload_ok path={_p} bytes={_p.stat().st_size}', flush=True)\n"
        )

    base = args.url.rstrip("/")
    headers = {"Authorization": f"Bearer {args.token}"}
    transport = requests.Session()
    transport.trust_env = not args.direct
    sessions_response = transport.get(
        f"{base}/api/sessions", headers=headers, timeout=15
    )
    sessions_response.raise_for_status()
    sessions = sessions_response.json()
    if len(sessions) != 1:
        raise RuntimeError(
            f"expected exactly one active marimo session, found {len(sessions)}"
        )
    session_id = next(iter(sessions))
    execute_headers = {
        **headers,
        "Content-Type": "application/json",
        "Marimo-Session-Id": session_id,
    }
    # Preserve evidence as it arrives, before sandbox loss or tool-output truncation.
    # Only notebook output is logged: never request headers, URL, or credentials.
    log_handle = None
    if args.output_log:
        log_path = Path(args.output_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("x", encoding="utf-8")

    def emit(value: str, *, stderr: bool = False) -> None:
        print(value, end="", file=sys.stderr if stderr else sys.stdout, flush=True)
        if log_handle is not None:
            log_handle.write(value)
            log_handle.flush()

    with transport.post(
        f"{base}/api/kernel/execute",
        headers=execute_headers,
        json={"code": code},
        stream=True,
        timeout=(15, None),
    ) as response:
        response.raise_for_status()
        event = ""
        done = False
        success = False
        for raw_line in response.iter_lines(decode_unicode=True):
            line = raw_line or ""
            if line.startswith("event: "):
                event = line[7:]
                continue
            if not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            if event == "stdout":
                emit(payload.get("data", ""))
            elif event == "stderr":
                emit(payload.get("data", ""), stderr=True)
            elif event == "done":
                done = True
                success = payload.get("success", True)
                output = payload.get("output", {}).get("data")
                if output:
                    emit(str(output) + "\n")
                break
        if not done:
            raise RuntimeError("marimo SSE stream ended without a done event")
        if log_handle is not None:
            log_handle.close()
        return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
