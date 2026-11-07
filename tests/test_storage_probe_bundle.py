import pytest
from tools.storage_probe_bundle import validate_probe_build_binding
import os
import shutil
from pathlib import Path


def fixture(tmp_path,home=None):
    source=tmp_path/'source';source.mkdir()
    build=source/'build';build.mkdir()
    (source/'CMakeLists.txt').write_text('project(test)')
    (build/'CMakeCache.txt').write_text('CMAKE_HOME_DIRECTORY:INTERNAL='+str(home or source)+'\n')
    return source,build


def test_configured_source_binding_accepts_owned_out_of_source_build(tmp_path):
    source,build=fixture(tmp_path)
    assert validate_probe_build_binding(source,build)==(source.resolve(),build.resolve())


def test_cache_from_other_source_rejected(tmp_path):
    foreign=tmp_path/'foreign';foreign.mkdir()
    source,build=fixture(tmp_path,foreign)
    with pytest.raises(ValueError):validate_probe_build_binding(source,build)


def test_duplicate_home_binding_rejected(tmp_path):
    source,build=fixture(tmp_path)
    with (build/'CMakeCache.txt').open('a') as stream:stream.write('CMAKE_HOME_DIRECTORY:INTERNAL='+str(source)+'\n')
    with pytest.raises(ValueError):validate_probe_build_binding(source,build)


def test_in_source_build_rejected(tmp_path):
    source,build=fixture(tmp_path)
    with pytest.raises(ValueError):validate_probe_build_binding(source,source)


def test_real_probe_bundle_binds_independent_public_iterator_table(tmp_path):
    selected=os.environ.get('BEAM_TEST_REAL_PROBE_BUILD')
    if not selected:pytest.skip('explicit authorized remote CMake build required')
    from tools.storage_probe_bundle import build_storage_probes
    source=Path(__file__).resolve().parents[1]
    receipt=build_storage_probes(source,Path(selected),tmp_path,180)
    assert 'gemm_iterator' in receipt['tables'], 'mandatory bundle omitted public iterator geometry'
    assert 'stream1_gemm_iterator_probe' in receipt['binary_sha256']
    assert 'tools/stream1_gemm_iterator_probe.cu' in receipt['source_sha256']
    assert receipt['production_admitted'] is False


def test_real_probe_bundle_binds_independent_mainloop_geometry(tmp_path):
    selected=os.environ.get('BEAM_TEST_REAL_PROBE_BUILD')
    if not selected:pytest.skip('explicit authorized remote CMake build required')
    from tools.storage_probe_bundle import build_storage_probes
    source=Path(__file__).resolve().parents[1]
    receipt=build_storage_probes(source,Path(selected),tmp_path,180)
    for entry in receipt['tables']['gemm_iterator'].values():
        assert set(entry.get('mainloop',{}))=={'A','B'}, 'independent A/B geometry missing'


def test_real_bundle_binds_private_layout_header_identity(tmp_path):
    selected=os.environ.get('BEAM_TEST_REAL_PROBE_BUILD')
    if not selected:pytest.skip('explicit authorized remote CMake build required')
    from tools.storage_probe_bundle import build_storage_probes
    source=Path(__file__).resolve().parents[1]
    receipt=build_storage_probes(source,Path(selected),tmp_path,180)
    expected={
        'predicated_tile_access_iterator.h':'e24899b2f01c7ac9c6cec617c4bb20b17d8dae139bf06bb8ef0f624ffb38b64f',
        'predicated_tile_iterator.h':'894c373193b06e6592aec5403fc585bc6c6bc67a492a81a035f4cb89059e9557',
        'predicated_tile_access_iterator_params.h':'10ee5d4924f625e2c465311a9a821ab1821d020a7ab09ae27b784894ae09a768'}
    assert receipt.get('mainloop_layout_sha256')==expected,'private layout header binding missing'


@pytest.mark.parametrize('mutation',['valid','modified','missing','duplicate_cache'])
def test_pinned_mainloop_layout_source_contract(mutation,tmp_path):
    selected=os.environ.get('BEAM_TEST_MAINLOOP_HEADER_ROOT')
    if not selected:pytest.skip('explicit authorized pinned CUTLASS headers required')
    from tools import storage_probe_bundle
    fn=getattr(storage_probe_bundle,'observe_mainloop_layout',None)
    assert callable(fn),'pinned private layout observer missing'
    header_dir=tmp_path/'cutlass/include/cutlass/transform/threadblock'
    header_dir.mkdir(parents=True)
    names=('predicated_tile_access_iterator.h','predicated_tile_iterator.h',
        'predicated_tile_access_iterator_params.h')
    for name in names:shutil.copyfile(Path(selected)/'include/cutlass/transform/threadblock'/name,header_dir/name)
    build=tmp_path/'build';build.mkdir()
    cache='CUTLASS_DIR:PATH='+str(tmp_path/'cutlass')+'\n'
    if mutation=='duplicate_cache':cache+=cache
    (build/'CMakeCache.txt').write_text(cache)
    if mutation=='modified':
        with (header_dir/names[0]).open('ab') as stream:stream.write(b'\n')
    if mutation=='missing':(header_dir/names[0]).unlink()
    if mutation=='valid':assert set(fn(build))==set(names)
    else:
        with pytest.raises(ValueError):fn(build)
