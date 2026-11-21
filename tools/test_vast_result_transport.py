#!/usr/bin/env python3
"""Exercise Vast result retrieval after the remote command revokes its SSH key."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import socket
import subprocess
import tarfile
import tempfile
import unittest


FAKE_SSH = r"""#!/usr/bin/env python3
import io
import json
import os
from pathlib import Path
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
if ("987654321" not in command
        or "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc" not in command
        or "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd" not in command
        or "123456789-2" not in command):
    print("the selected Vast instance ID, private hashes or GitHub run attempt was not passed into the remote receipt", file=sys.stderr)
    raise SystemExit(94)
if not authorized_key.exists():
    print("the key was revoked before remote work", file=sys.stderr)
    raise SystemExit(93)
authorized_key.unlink()
print("synthetic GPU execution log", flush=True)
raise SystemExit(int(os.environ["FAKE_REMOTE_EXIT"]))
"""


class VastResultTransportTests(unittest.TestCase):
    def run_transport(self, remote_exit: int) -> tuple[subprocess.CompletedProcess[str], Path, list[str]]:
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
            "VAST_REMOTE_RUNTIME_SECONDS": "1200",
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
