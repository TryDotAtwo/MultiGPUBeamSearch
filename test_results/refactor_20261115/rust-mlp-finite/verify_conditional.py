"""Check the isolated ordinary/conditional CUB graph experiment."""
import hashlib
import io
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SHA256 = '05d9e8c2919e8ec62dcf41bfd515f1a9bce49d7cfb395d7202067c6d6a26de7f'
data = (HERE / 'conditional-evidence.tar.gz').read_bytes()
assert hashlib.sha256(data).hexdigest() == SHA256
with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
    entries = archive.getmembers()
    assert len(entries) == len({m.name for m in entries})
    assert all((m.isfile() or m.isdir()) and m.name.startswith('conditional-diagnostic-v2-results')
               and '..' not in Path(m.name).parts for m in entries)
    get = lambda name: archive.extractfile('conditional-diagnostic-v2-results/' + name).read()
    assert get('source/conditional-diagnostic.cu').replace(b'\r\n', b'\n') == (
        HERE / 'conditional-diagnostic.cu').read_bytes().replace(b'\r\n', b'\n')
    for name in ('cuda/radix128.cuh', 'src/types.hpp', 'src/config.hpp'):
        assert get('source/' + name).replace(b'\r\n', b'\n') == (ROOT / name).read_bytes().replace(b'\r\n', b'\n')
    assert get('status.txt').decode().splitlines() == ['COMPILE_START', 'COMPILE_OK',
        'START plain 1', 'DONE plain 1', 'START conditional 1', 'DONE conditional 1',
        'START plain 1025', 'DONE plain 1025', 'START conditional 1025', 'DONE conditional 1025',
        'COMPLETE', 'EXIT_CODE=0']
    for n in (1, 1025):
        for mode in ('plain', 'conditional'):
            stem = f'{mode}-{n}'
            assert get(stem + '-direct.exit').strip() == b'0'
            assert b'DIAGNOSTIC_OK' in get(stem + '-direct.log')
            for tool in ('memcheck', 'synccheck'):
                assert get(f'{stem}-{tool}.exit').strip() == b'0'
                assert b'ERROR SUMMARY: 0 errors' in get(f'{stem}-{tool}.log')
        assert get(f'plain-{n}-racecheck.exit').strip() == b'0'
        assert b'RACECHECK SUMMARY: 0 hazards' in get(f'plain-{n}-racecheck.log')
    assert get('conditional-1-racecheck.exit').strip() == b'86'
    race = get('conditional-1-racecheck.log')
    assert b'marker(unsigned int *)' in race and b'BlockScanWarpScans' in race
    assert b'RACECHECK SUMMARY: 2 hazards' in race and b'DIAGNOSTIC_OK' in race
    assert get('conditional-1025-racecheck.exit').strip() == b'11'
    assert b"process didn't terminate successfully" in get('conditional-1025-racecheck.log')
print('CONDITIONAL_DIAGNOSTIC_VERIFIED: ordinary graph clean; conditional n=1 reported 2 cross-kernel hazards; conditional n=1025 terminated under racecheck')
