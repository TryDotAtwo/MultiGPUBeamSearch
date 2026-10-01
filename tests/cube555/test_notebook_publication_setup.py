import ast
import json
from pathlib import Path

import pytest

from tools.cube555.notebook import build


@pytest.mark.parametrize('source_text', [None, '{invalid json'])
def test_publication_setup_failure_preserves_search_command(tmp_path, monkeypatch, source_text):
    package = tmp_path/'package'
    build('a'*40, package, False)
    notebook = json.loads((package/'cube555-2xt4-blend.ipynb').read_text())
    launcher = next(''.join(cell['source']) for cell in notebook['cells']
                    if 'command += ["--b-micro"' in ''.join(cell['source']))
    publication = next(node for node in ast.parse(launcher).body
                       if isinstance(node, ast.If)
                       and ast.unparse(node.test) == 'PUBLISH_RESULTS and (not SMOKE_TEST)')
    monkeypatch.chdir(tmp_path)
    if source_text is not None:
        (tmp_path/'__notebook__.ipynb').write_text(source_text)
    output = tmp_path/'output'
    output.mkdir()
    command = ['solve', '--beam', '2097152']
    env = dict(Path=Path, json=json, output=output, command=command,
               PUBLISH_RESULTS=True, SMOKE_TEST=False)
    exec(compile(ast.Module(body=[publication], type_ignores=[]), '<publication>', 'exec'), env)
    assert command == ['solve', '--beam', '2097152']
    assert not (output/'publication.json').exists()
    assert 'Publication setup failed' in json.loads((output/'publication_setup_error.json').read_text())['error']
