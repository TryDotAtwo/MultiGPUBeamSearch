"""Summarize separate complete-depth timing and diagnostic work evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def read(path):return json.loads(path.read_text())


def tokens(line):return dict(re.findall(r'(\w+)=([^\s]+)',line))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.root;equal=read(root/'global-frontier-equality.json')
    if equal['status']!='pass':raise ValueError('missing exact trajectory equality')
    depths=next(iter(equal['runs'].values()))
    saturated=[int(depth) for depth,value in depths.items() if value['count']==65536]
    if len(saturated)<2:raise ValueError('insufficient saturated depths')
    timing={};work={};identity=None;expected_request=None
    for config in ('00','10','01','11'):
        samples=[]
        for repeat in range(1,4):
            directory=root/f'timing-{config}-{repeat}';request=read(directory/'request.json')
            result=read(directory/'result.json')
            if request['diagnostics'] or request['nsys'] or result['completed_depths']!=6:
                raise ValueError('instrumented or incomplete timing row')
            if identity is None:identity=request['runner_sha256']
            if identity!=request['runner_sha256']:raise ValueError('different timing binaries')
            signature={key:value for key,value in request.items() if key not in ('sort','prefill','repeat','profile')}
            profile={key:value for key,value in request['profile'].items()
                if key not in ('BEAM_STREAM4_BOUNDED_SORT','BEAM_STREAM5_PREFILL_BEFORE_WAIT')}
            signature['profile']=profile
            if expected_request is None:expected_request=signature
            if signature!=expected_request:raise ValueError('different timing workload/profile')
            ranks=[]
            for rank in range(request['world']):
                text=(directory/f'rank{rank}.log').read_text()
                row={int(v['depth_done']):float(v['depth_sec']) for v in
                    (tokens(line) for line in text.splitlines()) if 'depth_done' in v and 'depth_sec' in v}
                if len(row)!=6:raise ValueError('missing complete rank depth timing')
                ranks.append(row)
            samples.append(dict(repeat=repeat,
                complete_depth_seconds=sum(max(row[depth] for row in ranks) for depth in ranks[0]),
                saturated_depth_seconds=sum(max(row[depth] for row in ranks) for depth in saturated)))
        values=[row['saturated_depth_seconds'] for row in samples]
        timing[config]=dict(samples=samples,median_saturated_seconds=statistics.median(values),
                           min_saturated_seconds=min(values),max_saturated_seconds=max(values))
        directory=root/f'diag-{config}';request=read(directory/'request.json');capacity=int(request['profile']['BEAM_SHARD_CAPACITY_CANDIDATES'])
        live=[];inputs=[];bounds=[];upper_bounds=[];final=[]
        for rank in range(request['world']):
            pending={}
            for line in (directory/f'rank{rank}.log').read_text().splitlines():
                value=tokens(line)
                if line.startswith('stream4_histogram_trace') and value.get('phase')=='stream4_launch_pre':
                    slot=int(value['slot'])
                    if slot in pending:raise ValueError('sort slot reused before diagnostic completion')
                    pending[slot]=int(value['clean_count'])+int(value['dirty_count'])
                if line.startswith('stream4_histogram_trace') and value.get('phase')=='stream4_complete_post':
                    count=int(value['clean_count'])
                    incoming=pending.pop(int(value['slot']))
                    if not 0<=count<=incoming<=capacity:raise ValueError('invalid periodic live count')
                    quarter=(capacity+3)//4;half=(capacity+1)//2
                    def selected(n):return quarter if n<=quarter else half if n<=half else capacity
                    bound=capacity if config[0]=='0' else selected(count)
                    upper=capacity if config[0]=='0' else selected(incoming)
                    live.append(count);inputs.append(incoming);bounds.append(bound);upper_bounds.append(upper)
                if line.startswith('final_union_work'):
                    if value['host_snapshot_available']!='1':raise ValueError('unknown final sort bound')
                    final.append(dict(bound=int(value['sort_bound']),capacity=int(value['capacity'])))
            if pending:raise ValueError('unfinished periodic diagnostic jobs')
        if not live or not final:raise ValueError('missing actual periodic/final work diagnostics')
        work[config]=dict(periodic_jobs=len(live),periodic_input_items=sum(inputs),periodic_retained_items=sum(live),
            periodic_selected_bound_lower_items=sum(bounds),periodic_selected_bound_upper_items=sum(upper_bounds),
            periodic_exact_bound_jobs=sum(a==b for a,b in zip(bounds,upper_bounds)),
            periodic_full_capacity_items=len(live)*capacity,
            periodic_bound_scope='interval from observed input and post-dedup counts; compact count before sort is not recorded',
            final_union_jobs=len(final),final_sort_bound_items=sum(row['bound'] for row in final),
            final_capacity_items=sum(row['capacity'] for row in final))
    receipt=dict(status='pass',scope='max-rank complete-depth wall time; independent uninstrumented repeats; no kernel-throughput claim',
        excluded_warmup_repeat=0,saturated_depths=saturated,timing_runner_sha256=identity,timing=timing,diagnostic_work=work,
        frontier_equality_sha256=hashlib.sha256((root/'global-frontier-equality.json').read_bytes()).hexdigest())
    with args.output.open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
