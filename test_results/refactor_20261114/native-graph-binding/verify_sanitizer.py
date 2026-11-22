"""Check the memcheck archive against the tested source, binaries and rank PID."""
import hashlib
import json
from pathlib import Path
import re
import tarfile

here = Path(__file__).resolve().parent
archive = here / 'sanitizer-detail-evidence.tar.gz'
sha = lambda b: hashlib.sha256(b).hexdigest()
assert sha(archive.read_bytes()) == '386c2249e3480f5f889484537193a9352d64e799ba3f953b47eb7afd3eda8604'
with tarfile.open(here / 'trace-receipts.tar.gz', 'r:gz') as tar:
    source = json.load(tar.extractfile('source-manifest.json'))
package = json.loads((here / 'package.json').read_text())
prefix = 'beam-results/graph-sanitizer-detail/'
with tarfile.open(archive, 'r:gz') as tar:
    members = tar.getmembers()
    assert all(not member.issym() and not member.islnk() for member in members)
    read = lambda name: tar.extractfile(prefix + name).read()
    assert read('exit.txt').strip() == b'0'
    assert b'GRAPH_BINDING_SANITIZER_SEARCH_OK' in read('driver.log')
    inputs = {}
    for line in read('input-sha256.txt').decode().splitlines():
        digest, path = line.split('  ', 1); inputs[path] = digest
    assert inputs['/workspace/graph-overlay.tar.gz'] == package['inputs']['graph-overlay.tar.gz']
    for name in ('graph-sanitizer-driver.py', 'graph-sanitizer-detail-run.sh'):
        data = tar.extractfile(name).read()
        assert sha(data) == inputs['/workspace/' + name]
        local = 'sanitizer-driver.py' if name == 'graph-sanitizer-driver.py' else 'sanitizer-detail-run.sh'
        assert (here / local).read_bytes().replace(b'\r\n', b'\n') == data.replace(b'\r\n', b'\n')
    for path, digest in inputs.items():
        name = path.removeprefix('/workspace/')
        if name in source['records']:
            assert source['records'][name]['sha256'] == digest
    library = [name for name in inputs if name.endswith('/libbeam_native.so')]
    assert len(library) == 1 and source['records'][library[0].removeprefix('/workspace/')]['sha256'] == inputs[library[0]]
    result = json.loads(read('search/result.json'))
    assert result['world'] == 1 and result['beam'] == 64 and len(result['pids']) == 1
    state = tuple(result['start']); center = tuple(i // 16 for i in range(96))
    for move in result['path']:
        assert type(move) is int and 0 <= move < 24
        permutation = list(range(96)); permutation[move], permutation[move + 24] = permutation[move + 24], permutation[move]
        state = tuple(state[i] for i in permutation)
    assert state == center
    logs = [member for member in members if member.isfile() and member.name.startswith(prefix + 'memcheck-')]
    assert len(logs) == 1
    log = tar.extractfile(logs[0]).read().decode()
    assert re.search(r'Process ID:\s+' + str(result['pids'][0]) + r'\s', log)
    assert 'Target application:' in log and '/_bin/beam-runner 0 4 64' in log
    assert 'LEAK SUMMARY: 0 bytes leaked in 0 allocations' in log
    assert 'ERROR SUMMARY: 0 errors' in log
    assert 'timeout 420 compute-sanitizer --print-session-details --target-processes all --tool memcheck' in read('command.txt').decode()
print('PASS: actual Rust rank PID attached, unfiltered memcheck clean, path replayed, source binaries matched')
