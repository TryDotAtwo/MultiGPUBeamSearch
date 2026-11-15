"""Reject a native graph that differs from the Rust-owned session identity."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from multigpubeamsearch.hamming import write_hamming_artifact
from public_model_snapshot_gpu import graph_fixture, start_state


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('runner', 'library', 'extension', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('_beam_runtime', args.extension)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)

    _, graph, moves = graph_fixture()
    good = graph.to_puzzle_info()
    center_changed = json.loads(json.dumps(good))
    center_changed['central_state'][0], center_changed['central_state'][16] = (
        center_changed['central_state'][16], center_changed['central_state'][0])
    moves_changed = json.loads(json.dumps(good))
    moves_changed['generators']['m0'], moves_changed['generators']['m1'] = (
        moves_changed['generators']['m1'], moves_changed['generators']['m0'])
    for name, value in (('good', good), ('center', center_changed), ('moves', moves_changed)):
        (root / f'{name}.json').write_text(json.dumps(value) + '\n')
    weights = write_hamming_artifact(graph, root)
    start = start_state(graph.center, moves, 3)
    csv = root / 'test.csv'
    csv.write_text('initial_state_id,initial_state\n0,"' + ','.join(map(str, start)) + '"\n')
    records = []
    for case, world in (('moves', 1), ('center', 1), ('target', 1),
                        ('repair', 1), ('rank1-center', 2)):
        directory = root / case
        directory.mkdir()
        wrapper = directory / 'rank.py'
        wrapper.write_text(
            "import os,sys\nfrom pathlib import Path\n"
            "rank=os.environ['RANK'];Path(os.environ['PID_DIR'],f'rank-{rank}.pid').write_text(str(os.getpid()))\n"
            "if os.environ['CASE']=='rank1-center' and rank=='1':"
            "os.environ['BEAM_PUZZLE_INFO_JSON']=os.environ['BAD_CENTER']\n"
            "os.execv(sys.argv[1],sys.argv[1:])\n")
        environment = dict(os.environ, CASE=case, PID_DIR=str(directory),
            BAD_CENTER=str(root / 'center.json'),
            BEAM_EXPECTED_GRAPH_HASH='caller-spoof', BEAM_REQUIRE_LOADED_GRAPH_BINDING='0',
            BEAM_GENERATOR_PATH=str(root / ('moves.json' if case == 'moves' else 'good.json')),
            BEAM_PUZZLE_INFO_JSON=str(root / ('center.json' if case == 'center' else 'good.json')),
            BEAM_TEST_CSV=str(csv), BEAM_WEIGHT_DIR=str(weights),
            BEAM_NATIVE_LIBRARY=str(args.library.resolve()), BEAM_RUNTIME_CONFIG_MODE='auto',
            BEAM_NCCL_ID_FILE=str(directory / 'nccl.bin'), BEAM_SOLVED_NEIGHBORHOOD_RADIUS='0',
            BEAM_STREAM2_SUFFIX_RADIUS='0', BEAM_SOLVE_BUCKET_MODE='0', BEAM_HISTORY_MODE='disk',
            BEAM_HISTORY_RAM_BYTES=str(16 << 20), BEAM_HISTORY_DISK_BYTES=str(1 << 20),
            BEAM_HISTORY_DIR=str(directory / 'history'), BEAM_HISTORY_DISK_PATH=str(directory / 'history'),
            BEAM_STREAM1_EXECUTOR='native_cuda_graph', CUDA_VISIBLE_DEVICES=','.join(map(str, range(world))))
        if case == 'target':
            environment['BEAM_TARGET_STATE_TEXT'] = ','.join(map(str, center_changed['central_state']))
        if case == 'repair':
            environment['BEAM_REPAIR_SOLUTIONS_CSV'] = str(directory / 'unused-repair.csv')
        identity = dict(graph=dict(center=graph.center, generators=graph.generators),
            graph_hash=graph.graph_hash, model_artifact_hash='b' * 64, model_backend='mlp',
            runtime_build=dict(binary_sha256=sha(args.runner.resolve()),
                               native_library_sha256=sha(args.library.resolve()), abi_version=1),
            devices=list(range(world)), beam_width=64, max_steps=4, options={},
            cuda_visible_devices=environment['CUDA_VISIBLE_DEVICES'], cuda_device_order=None)
        launch = dict(command=[sys.executable, str(wrapper), str(args.runner.resolve()), '0', '4', '64'],
            protocol_command=[str(args.runner.resolve()), '--resident-protocol-info'],
            environment=environment, cwd=str(directory))
        owner = runtime.SearchSession()
        began = time.monotonic()
        try:
            try:
                owner.run(json.dumps(identity), json.dumps(launch), list(start), directory, 30., '{}')
            except RuntimeError as error:
                assert 'exited' in str(error), str(error)
                message = str(error)
            else:
                raise AssertionError(f'{case}: mismatched native graph was accepted')
        finally:
            owner.close()
        rank = 1 if case == 'rank1-center' else 0
        log = (directory / 'resident-session' / f'bootstrap-{rank}.log').read_text()
        expected = {'target': 'bound session target differs from central state',
                    'repair': 'bound session cannot use repair mode'}.get(
                        case, 'native loaded graph hash mismatch')
        assert expected in log, (case, log)
        pids = [int(path.read_text()) for path in directory.glob('rank-*.pid')]
        assert len(pids) == world and all(not Path(f'/proc/{pid}').exists() for pid in pids)
        assert owner.closed and owner.pids == []
        assert not (directory / 'native-outcome.json').exists()
        records.append(dict(case=case, world=world, error=message, pids=pids,
                            elapsed_seconds=time.monotonic() - began))
        print(f'LOADED_GRAPH_REJECTED {case} world={world}', flush=True)
    (root / 'results.json').write_text(json.dumps(records, indent=2) + '\n')
    print('NATIVE_LOADED_GRAPH_BINDING_GPU_OK', flush=True)


if __name__ == '__main__':
    main()
