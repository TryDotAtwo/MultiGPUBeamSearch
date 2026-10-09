"""Execute actual table/device binding branch; not full captured-node acceptance."""
import ast
import json
import os
from pathlib import Path

import pytest


def bind(sm, table):
    source = Path('tools/graph_parameter_layout.py').read_text()
    function = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'validate_non_gemm_parameter_layout')
    branch = next(n for n in function.body if isinstance(n, ast.If)
                  and ast.unparse(n.test) == 'gemm_storage is not None')
    module = ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[]))
    namespace = dict(gemm_storage=table, execution={'device': {'sm': sm}})
    exec(compile(module, '<actual-layout-table-branch>', 'exec'), namespace)
    return namespace['gemm_table']


def observed():
    return json.loads(Path(os.environ['SM80_STORAGE_JSON']).read_text())


def test_actual_binding_accepts_sm86_with_compiled_sm80_table():
    assert len(bind(86, observed())) == 5


@pytest.mark.parametrize('sm', [75, 89, 90, True])
def test_actual_binding_rejects_unmatched_device(sm):
    with pytest.raises(ValueError):
        bind(sm, observed())
