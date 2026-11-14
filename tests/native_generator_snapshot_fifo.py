"""The combined native input must consume one non-seekable generator snapshot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("--state-len", type=int, required=True)
    parser.add_argument("--moves", type=int, required=True)
    parser.add_argument("--tmpdir", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(dir=args.tmpdir) as directory:
        fifo = Path(directory) / "moves.fifo"
        os.mkfifo(fifo)
        generators = {f"m{move}": [(i + move) % args.state_len for i in range(args.state_len)]
                      for move in range(args.moves)}
        payload = (json.dumps(dict(generators=generators)) + "\n").encode()
        errors = []

        def writer():
            try:
                with fifo.open("wb") as stream:
                    stream.write(payload)
            except OSError as error:
                errors.append(error)

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        result = subprocess.run([str(args.binary.resolve()), "--snapshot", str(fifo)],
                                capture_output=True, text=True, timeout=10)
        thread.join(timeout=2)
        assert not thread.is_alive() and not errors, errors
        assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
        assert result.stdout.strip() == "native_generator_snapshot PASS", result.stdout
    print("NATIVE_GENERATOR_ONE_READ_FIFO_OK")


if __name__ == "__main__":
    main()
