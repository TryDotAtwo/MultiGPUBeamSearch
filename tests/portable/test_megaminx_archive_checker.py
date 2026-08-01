import io
from pathlib import Path
import tarfile

import pytest
import zstandard

from test_megaminx_native_release import cuobjdump_for, stage
from tools.build_megaminx_native_release import build_release
from tools.check_megaminx_native_archive import check_archive


def test_checker_rejects_payload_whose_manifest_hash_was_not_updated(tmp_path: Path):
    archive = build_release(stage(tmp_path), tmp_path / "out", 90, cuobjdump_for(90))
    raw = zstandard.ZstdDecompressor().decompress(archive.read_bytes())
    output = io.BytesIO()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as source, tarfile.open(fileobj=output, mode="w") as target:
        for member in source.getmembers():
            data = source.extractfile(member).read()
            if member.name.endswith("/data/test.csv"):
                data += b"corrupt"
                member.size = len(data)
            target.addfile(member, io.BytesIO(data))
    archive.write_bytes(zstandard.ZstdCompressor().compress(output.getvalue()))
    with pytest.raises(ValueError, match="sha256"):
        check_archive(archive, 90, cuobjdump_for(90))
