from dataclasses import replace
from types import SimpleNamespace
import pytest
from cayleypy import CayleyGraph, PermutationGroups
from multigpubeamsearch import dispatch
from multigpubeamsearch.options import NativeOptions, NativeOutcome
from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.backend import parse_terminal


def test_controls_reach_native_options(tmp_path, monkeypatch):
    graph=CayleyGraph(PermutationGroups.lrx(5),device='cpu')
    monkeypatch.setattr(dispatch,'runtime_devices',lambda *a:(0,1))
    monkeypatch.setattr('multigpubeamsearch.bootstrap.prepare_sources',lambda options:options)
    monkeypatch.setattr(dispatch,'prepare_model',lambda *a:SimpleNamespace(manifest={'output_dim':1},artifact_hash='x'))
    monkeypatch.setattr(dispatch,'prepare_runtime',lambda *a,**k:SimpleNamespace(build_metadata={}))
    def run(contract, model, options, *args, **kwargs):
        assert (options.solution_mode,options.collect_extra_depths,options.max_collected_solutions,options.touch_bfs_radius)==('collect',2,7,1)
        return NativeOutcome((2,),1.,16,tmp_path,{'collected_solutions':[{'path':[2]}]})
    monkeypatch.setattr(dispatch,'run_native',run)
    result=dispatch.beam_search(graph,start_state=[1,0,2,3,4],return_path=True,beam_width=16,max_steps=4,
        solution_mode='collect',collect_extra_depths=2,max_collected_solutions=7,bfs_radius=1,
        native_options=NativeOptions(cache_dir=tmp_path))
    assert result.solutions==((2,),) and result.path==[2]


@pytest.mark.parametrize('value',[-1,True,1.2])
def test_invalid_controls_fail_before_work(value):
    with pytest.raises(ValueError): NativeOptions(collect_extra_depths=value)


def test_native_controls_never_silently_fallback():
    with pytest.raises(ValueError,match='cannot silently fall back'):
        dispatch.beam_search(None,backend='auto',solution_mode='collect')


def test_collect_parser_checks_every_path_and_count(tmp_path):
    graph=CayleyGraph(PermutationGroups.lrx(5),device='cpu')
    contract=GraphContract.from_graph(graph,[1,0,2,3,4])
    log=tmp_path/'native.log'
    text=('GLOBAL_BEAM_WIDTH_EFFECTIVE=16\n'
          'solve_bucket_solution=1 puzzle_id=0 depth_index=0 found_depth=1 solution_length=1 owner_rank=0 solution=m2\n'
          'solve_bucket_summary=1 puzzle_id=0 records=1 best_length=1\n'
          'puzzle_solved=1 puzzle_id=0 seconds=1 solution_length=1 solution=m2\n')
    log.write_text(text)
    assert parse_terminal(log,contract,max_steps=3)[2]['collected_solutions'][0]['path']==[2]
    log.write_text(text.replace('records=1','records=2'))
    with pytest.raises(Exception,match='count'):parse_terminal(log,contract,max_steps=3)
    log.write_text(text.replace('owner_rank=0 solution=m2','owner_rank=0 solution=m0'))
    with pytest.raises(Exception,match='replay'):parse_terminal(log,contract,max_steps=3)


def test_reflection_returns_original_frame_and_preserves_collection():
    from multigpubeamsearch.reflection import reflected_search
    from multigpubeamsearch.results import NativeBeamSearchResult
    graph=CayleyGraph(PermutationGroups.lrx(5),device='cpu')
    calls=[]
    def search(kwargs):
        calls.append(kwargs)
        contract=GraphContract.from_graph(graph,kwargs['start_state'])
        assert contract.replay((2,))
        return NativeBeamSearchResult(True,1,[2],{},graph.definition,solutions=((2,),),native_metadata={})
    result=reflected_search(search,graph,dict(start_state=[1,0,2,3,4],return_path=True,reflect_mode='after_original'),'native')
    assert result.solutions==((2,),) and len(calls)==2
    assert result.native_metadata['collected_solutions'][0]['final_orientation']=='reflected'


def test_invalid_reflection_source_fails_before_search():
    from multigpubeamsearch.reflection import reflected_search
    graph=CayleyGraph(PermutationGroups.lrx(5),device='cpu')
    with pytest.raises(ValueError,match='does not solve'):
        reflected_search(lambda _:pytest.fail('must not search'),graph,
            dict(start_state=[1,0,2,3,4],return_path=True,reflect_mode='only',reflection_source=[0]),'native')
