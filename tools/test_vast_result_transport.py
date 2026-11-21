#!/usr/bin/env python3
"""Exercise Vast result retrieval after the remote command revokes its SSH key."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import unittest


FAKE_SSH = r"""#!/usr/bin/env python3
import io
import json
import os
from pathlib import Path
import shlex
import socket
import sys
import tarfile

args = sys.argv[1:]
events_path = Path(os.environ["FAKE_SSH_EVENTS"])
authorized_key = Path(os.environ["FAKE_AUTHORIZED_KEY"])


def event(kind):
    with events_path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(kind) + "\n")


control_path = None
for index, value in enumerate(args[:-1]):
    if value == "-o" and args[index + 1].startswith("ControlPath="):
        control_path = args[index + 1].split("=", 1)[1].replace("%C", "f" * 40)
if control_path is None:
    print("ControlPath was not configured", file=sys.stderr)
    raise SystemExit(90)

if "-O" in args:
    event("close")
    Path(control_path).unlink(missing_ok=True)
    raise SystemExit(0)

if "-MNf" in args:
    event("master")
    if not authorized_key.exists():
        print("one-time key missing before master setup", file=sys.stderr)
        raise SystemExit(91)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(control_path)
    listener.listen(1)
    listener.close()
    raise SystemExit(0)

destination_index = next((i for i, value in enumerate(args) if value.startswith("root@")), None)
if destination_index is None or destination_index + 1 >= len(args):
    print("remote command is missing", file=sys.stderr)
    raise SystemExit(92)
command = args[destination_index + 1]
if not Path(control_path).exists():
    print("control master is not alive", file=sys.stderr)
    raise SystemExit(255)

if command.startswith("tar -C "):
    event("results")
    buffer = io.BytesIO()
    payload = b'{"synthetic":"partial-or-complete-results"}\n'
    info = tarfile.TarInfo("test_results/refactor_20261121/gpu-acceptance/synthetic.json")
    info.size = len(payload)
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))
    sys.stdout.buffer.write(buffer.getvalue())
    raise SystemExit(0)

event("remote")
tokens = shlex.split(command)
if ("987654321" not in tokens
        or "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc" not in command
        or "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd" not in command
        or tokens[-3:] != ["123456789", "2", "test_results/refactor_20261121/gpu-acceptance"]):
    print("the selected instance, private pins, GitHub attempt or result path was not passed to the remote receipt", file=sys.stderr)
    raise SystemExit(94)
if not authorized_key.exists():
    print("the key was revoked before remote work", file=sys.stderr)
    raise SystemExit(93)
authorized_key.unlink()
print("synthetic GPU execution log", flush=True)
raise SystemExit(int(os.environ["FAKE_REMOTE_EXIT"]))
"""


class VastResultTransportTests(unittest.TestCase):
    def run_transport(
        self, remote_exit: int, *, deadline_seconds_from_now: int = 7200
    ) -> tuple[subprocess.CompletedProcess[str], Path, list[str]]:
        temporary = tempfile.TemporaryDirectory(prefix="mgbs-vast-transport-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        fake_ssh = bin_dir / "ssh"
        fake_ssh.write_text(FAKE_SSH, encoding="utf-8")
        fake_ssh.chmod(0o755)

        key_file = root / "ssh-key"
        key_file.write_text("fake-private-key", encoding="utf-8")
        known_hosts = root / "known-hosts"
        known_hosts.write_text("", encoding="utf-8")
        remote_script = root / "remote.sh"
        remote_script.write_text("# fake remote entrypoint\n", encoding="utf-8")
        authorized_key = root / "authorized-key"
        authorized_key.write_text("present", encoding="utf-8")
        events_file = root / "events.jsonl"
        remote_log = root / "remote.log"
        results_archive = root / "results.tar.gz"

        environment = dict(os.environ)
        environment.update({
            "PATH": str(bin_dir) + os.pathsep + environment["PATH"],
            "VAST_SSH_PRIVATE_KEY": str(key_file),
            "VAST_SSH_KNOWN_HOSTS": str(known_hosts),
            "VAST_SSH_HOST": "192.0.2.1",
            "VAST_SSH_PORT": "2200",
            "VAST_SSH_LOG": str(remote_log),
            "VAST_SSH_RESULTS": str(results_archive),
            "VAST_SSH_SCRIPT": str(remote_script),
            "VAST_REMOTE_ROOT": "/workspace/synthetic-run",
            "VAST_REMOTE_RESULTS_RELATIVE": "test_results/refactor_20261121/gpu-acceptance",
            "VAST_REMOTE_RUNTIME_SECONDS": "1200",
            "VAST_BUDGET_DEADLINE_EPOCH": str(int(time.time()) + deadline_seconds_from_now),
            "VAST_STOP_RESERVE_SECONDS": "1200",
            "VAST_RETRIEVAL_MAX_SECONDS": "300",
            "VAST_REMOTE_KEY_BLOB": "ssh-ed25519 fake-public-key",
            "VAST_REMOTE_ARCHIVE": "/workspace/synthetic-run.tar.gz",
            "VAST_REMOTE_ARCHIVE_SHA": "a" * 64,
            "VAST_REMOTE_MANIFEST_SHA": "c" * 64,
            "VAST_REMOTE_CHECKPOINT_SHA": "d" * 64,
            "VAST_SOURCE_SHA": "b" * 40,
            "VAST_INSTANCE_ID": "987654321",
            "VAST_GITHUB_RUN_ID": "123456789",
            "VAST_GITHUB_RUN_ATTEMPT": "2",
            "FAKE_SSH_EVENTS": str(events_file),
            "FAKE_AUTHORIZED_KEY": str(authorized_key),
            "FAKE_REMOTE_EXIT": str(remote_exit),
        })
        helper = Path(__file__).with_name("vast_result_transport.sh")
        completed = subprocess.run(
            ["bash", str(helper)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        events = [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines()]
        return completed, results_archive, events

    def assert_results_archive(self, archive_path: Path) -> None:
        self.assertTrue(archive_path.is_file(), "remote results archive must be downloaded")
        with tarfile.open(archive_path, mode="r:gz") as archive:
            member = archive.extractfile(
                "test_results/refactor_20261121/gpu-acceptance/synthetic.json"
            )
            self.assertIsNotNone(member)
            self.assertEqual(member.read(), b'{"synthetic":"partial-or-complete-results"}\n')

    def run_remote_payload(
        self, worker_exit: int
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path, list[dict[str, str]]]:
        temporary = tempfile.TemporaryDirectory(prefix="mgbs-vast-payload-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source_dir = root / "source"
        worker_dir = source_dir / "tools"
        worker_dir.mkdir(parents=True)
        run_root = root / "remote-run"
        result_dir = run_root / "test_results/refactor_20261121/gpu-acceptance"
        receipt_path = result_dir / "receipt.json"
        worker_events_path = root / "worker-events.jsonl"

        worker = worker_dir / "vast_cnot_gpu_acceptance.sh"
        worker.write_text(textwrap.dedent("""\
            #!/bin/bash
            set -euo pipefail
            receipt="$EXPECTED_RECEIPT_PATH"
            [[ -s "$receipt" ]]
            "$SYSTEM_PYTHON" - "$receipt" "$FAKE_WORKER_EXIT" "$FAKE_WORKER_EVENTS" <<'PY'
            import json, pathlib, sys
            receipt = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
            if receipt.get("gpu_test") != "running":
                raise SystemExit("remote receipt was not written before the build worker started")
            with pathlib.Path(sys.argv[3]).open("a", encoding="utf-8") as output:
                output.write(json.dumps({"phase": "worker_started", "receipt": "running"}) + "\\n")
            PY
            printf 'synthetic GPU execution log\\n'
            printf 'CNOT_CONTROL_COLLECTIVES_OK\\nCNOT_DEVICE_UNSEEN_OK\\n'
            exit "$FAKE_WORKER_EXIT"
            """), encoding="utf-8")
        worker.chmod(0o755)

        bin_dir = root / "bin"
        bin_dir.mkdir()

        def write_stub(name: str, body: str) -> None:
            path = bin_dir / name
            path.write_text(body, encoding="utf-8")
            path.chmod(0o755)

        write_stub("sha256sum", "#!/bin/sh\nprintf '%s  %s\\n' \"$EXPECTED_ARCHIVE_SHA\" \"$1\"\n")
        write_stub("awk", "#!/bin/sh\nIFS=' ' read -r first rest\nprintf '%s\\n' \"$first\"\n")
        write_stub("tar", "#!/bin/sh\nexit 0\n")
        write_stub("mkdir", "#!/bin/sh\nexec /usr/bin/mkdir \"$@\"\n")
        write_stub("rm", "#!/bin/sh\nexec /usr/bin/rm \"$@\"\n")
        write_stub("tee", "#!/bin/sh\nexec /usr/bin/tee \"$@\"\n")
        write_stub("bash", "#!/bin/sh\nexec /bin/bash \"$@\"\n")
        write_stub("python3", "#!/bin/sh\nexec \"$SYSTEM_PYTHON\" \"$@\"\n")

        workflow = Path(__file__).parents[1] / ".github/workflows/vast-cnot-control-collectives.yml"
        source = workflow.read_text(encoding="utf-8")
        marker = 'cat > "${RUNNER_TEMP}/run-cnot-control.sh" <<\'REMOTE\'\n'
        start = source.index(marker) + len(marker)
        end = source.index("\n          REMOTE\n", start)
        payload_lines = source[start:end].splitlines()
        for number, line in enumerate(payload_lines, 1):
            if line and not line.startswith("          "):
                self.fail(f"unexpected indentation in remote payload line {number}")
            if line:
                payload_lines[number - 1] = line[10:]
        payload_path = root / "run-cnot-control.sh"
        payload_path.write_text("\n".join(payload_lines) + "\n", encoding="utf-8")

        environment = {
            "PATH": str(bin_dir),
            "SYSTEM_PYTHON": sys.executable,
            "EXPECTED_ARCHIVE_SHA": "a" * 64,
            "EXPECTED_RECEIPT_PATH": str(receipt_path),
            "FAKE_WORKER_EVENTS": str(worker_events_path),
            "FAKE_WORKER_EXIT": str(worker_exit),
        }
        self.assertNotIn("REMOTE_ROOT", environment)
        completed = subprocess.run(
            [
                "/bin/bash", str(payload_path), str(source_dir), str(run_root),
                "ssh-ed25519 synthetic", str(root / "source.tar.gz"), "a" * 64,
                "b" * 40, "987654321", "c" * 64,
                "92f3ec8e6a6ef831916a9d3ba9bbdae9307c9850d61334916a2b3e690c0020c4",
                "123456789", "2", "test_results/refactor_20261121/gpu-acceptance",
            ],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        events = (
            [json.loads(line) for line in worker_events_path.read_text(encoding="utf-8").splitlines()]
            if worker_events_path.is_file()
            else []
        )
        return completed, receipt_path, result_dir / "cuda-nccl.log", events

    def test_retrieves_results_after_successful_remote_key_revocation(self):
        completed, archive_path, events = self.run_transport(remote_exit=0)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assert_results_archive(archive_path)
        self.assertEqual(events, ["master", "remote", "results", "close"])

    def test_retrieves_partial_results_after_failed_remote_test(self):
        completed, archive_path, events = self.run_transport(remote_exit=17)
        self.assertEqual(completed.returncode, 17, completed.stdout + completed.stderr)
        self.assertIn("synthetic GPU execution log", completed.stdout)
        self.assert_results_archive(archive_path)
        self.assertEqual(events, ["master", "remote", "results", "close"])

    def test_refuses_retrieval_when_the_absolute_deadline_would_consume_stop_reserve(self):
        completed, archive_path, events = self.run_transport(
            remote_exit=0, deadline_seconds_from_now=100
        )
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertFalse(archive_path.exists(), "retrieval must not start without the stop reserve")
        self.assertIn("retrieval cannot start", completed.stderr)
        self.assertEqual(events, ["master", "remote", "close"])

    def test_executes_extracted_remote_payload_with_clean_environment_on_success(self):
        completed, receipt_path, log_path, events = self.run_remote_payload(worker_exit=0)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertTrue(receipt_path.is_file(), "remote receipt must use the supplied run_root")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["gpu_test"], "passed")
        self.assertEqual(receipt["exit_code"], 0)
        self.assertEqual(receipt["worker_exit_code"], 0)
        self.assertEqual(receipt["log_exit_code"], 0)
        self.assertIn("started_at_utc", receipt)
        self.assertIn("finished_at_utc", receipt)
        self.assertTrue(log_path.is_file(), "completed tee log must be in the receipt directory")
        self.assertIn("synthetic GPU execution log", log_path.read_text(encoding="utf-8"))
        self.assertEqual(events, [{"phase": "worker_started", "receipt": "running"}])

    def test_executes_extracted_remote_payload_and_records_controlled_failure(self):
        completed, receipt_path, log_path, events = self.run_remote_payload(worker_exit=17)
        self.assertEqual(completed.returncode, 17, completed.stdout + completed.stderr)
        self.assertTrue(receipt_path.is_file(), "failed remote runs must retain their receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["gpu_test"], "failed")
        self.assertEqual(receipt["exit_code"], 17)
        self.assertEqual(receipt["worker_exit_code"], 17)
        self.assertEqual(receipt["log_exit_code"], 0)
        self.assertTrue(log_path.is_file(), "failed runs must retain the completed tee log")
        self.assertIn("synthetic GPU execution log", log_path.read_text(encoding="utf-8"))
        self.assertEqual(events, [{"phase": "worker_started", "receipt": "running"}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
