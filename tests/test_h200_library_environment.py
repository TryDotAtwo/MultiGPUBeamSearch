"""Execute actual launcher setup without starting workers or preflight tools."""
import os
from pathlib import Path
import shutil
import subprocess
import pytest


@pytest.mark.parametrize('inherited', ['', '/chosen/nccl:/chosen/cuda'])
def test_launcher_keeps_inherited_library_precedence(tmp_path, inherited):
    bash = shutil.which('bash')
    git_bash = Path('C:/Program Files/Git/bin/bash.exe')
    if os.name == 'nt' and git_bash.is_file():
        bash = str(git_bash)
    if not bash:
        pytest.skip('bash required')
    root = Path(__file__).resolve().parents[1]
    source = (root / 'hpc/h200_dual_large_beam.sh').read_text()
    setup = source[:source.index('if [[ -n ${H200_PROFILE_PATH:')]
    script = tmp_path / 'setup.sh'
    script.write_text(setup + '\nprintf "LIB_RESULT=%s\\n" "$LD_LIBRARY_PATH"\n')
    env = {**os.environ, 'H200_REPO_DIR': root.as_posix(),
           'H200_RUN_ROOT': tmp_path.as_posix(), 'BEAM_WIDTH': '4096',
           'LD_LIBRARY_PATH': inherited}
    result = subprocess.run([bash, str(script)], env=env, capture_output=True,
                            text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    library = next(line.removeprefix('LIB_RESULT=') for line in result.stdout.splitlines()
                   if line.startswith('LIB_RESULT='))
    if inherited:
        assert library.startswith(inherited + ':'), 'launcher replaced caller library paths'
    assert '/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib' in library.split(':')
    assert '/usr/local/cuda/lib64' in library.split(':')
    assert not library.startswith(':')
