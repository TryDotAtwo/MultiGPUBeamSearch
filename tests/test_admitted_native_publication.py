"""Consume the actual native publisher contract; never overwrite or synthesize CSV."""
import hashlib
from pathlib import Path
import pytest
from tools.cube4_admitted_launch import validate_native_publication


def test_existing_native_publication_is_consumed_without_recreation(tmp_path):
    raw=b'initial_state_id,path\n954003,-f0.-d0.d2.r0\n'
    native=tmp_path/'submit.csv';copy=tmp_path/'submit_p954003_d12_b4096.csv'
    native.write_bytes(raw);copy.write_bytes(raw)
    assert validate_native_publication(native,copy)==hashlib.sha256(raw).hexdigest()
    assert native.read_bytes()==copy.read_bytes()==raw


@pytest.mark.parametrize('problem',['missing_native','different_copy','symlink_native','symlink_copy','oversized'])
def test_missing_or_corrupt_publication_is_rejected_without_repair(tmp_path,problem):
    native=tmp_path/'submit.csv';copy=tmp_path/'copy.csv'
    raw=b'initial_state_id,path\n954003,f0\n'
    native.write_bytes(raw);copy.write_bytes(raw)
    if problem=='missing_native':native.unlink()
    elif problem=='different_copy':copy.write_bytes(raw.replace(b'954003',b'954004'))
    elif problem=='symlink_native':native.unlink();native.symlink_to(copy)
    elif problem=='symlink_copy':copy.unlink();copy.symlink_to(native)
    else:native.write_bytes(raw+b'\n'*(1024*1024))
    before={p.name:p.read_bytes() for p in (native,copy) if p.exists()}
    with pytest.raises((ValueError,OSError)):
        validate_native_publication(native,copy)
    assert {p.name:p.read_bytes() for p in (native,copy) if p.exists()}==before
    if problem=='missing_native':assert not native.exists()
