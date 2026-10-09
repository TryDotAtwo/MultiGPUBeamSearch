"""Independent SDK activity replay matching; not parameter values/admission."""
from collections import Counter,defaultdict
import re
import hashlib
import stat
from pathlib import Path

def observe_injection_library(path):
    path = Path(path).resolve(strict=True)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 32*1024*1024:
        raise ValueError('CUPTI injection library must be a bounded regular file')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parser_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

LOCATION=re.compile(r'deviceId (\d+), contextId (\d+), streamId (\d+), graphId (\d+), graphNodeId (\d+)')
KERNEL=re.compile(r'^(?:CONCURRENT_KERNEL|KERNEL) \[ (\d+), (\d+) \].*?"([^"]+)", correlationId (\d+),')
GEOMETRY=re.compile(r'grid \[ (\d+), (\d+), (\d+) \], block \[ (\d+), (\d+), (\d+) \], cluster \[ (\d+), (\d+), (\d+) \], sharedMemory \(static (\d+), dynamic (\d+)\)')

def validate_replay_trace(text,inventory,transfers,*,device_ordinal,replays_per_lane=4):
    if type(device_ordinal) is not int or not 0 <= device_ordinal < 8:raise ValueError('explicit device ordinal 0..7 required')
    if replays_per_lane!=4:raise ValueError('only four canonical raw-score jobs supported')
    if len(text.encode())>32*1024*1024:raise ValueError('trace exceeds bounded size')
    if re.search(r'Dropped\s+[1-9]\d*|CUPTI_ERROR|Fatal error',text):raise ValueError('incomplete/error CUPTI trace')
    if re.search(r'"(?:cuda|cu)Graph(?:Exec[^"\n]*(?:Set|Update)|(?:Kernel|Memcpy|Memset)NodeSetParams|NodeSetParams)[^"\n]*"',text):
        raise ValueError('mutable graph APIs require separate execution proof')
    expected=[];operations={r['lane']:r['operations'] for r in transfers['lanes']}
    for lane in inventory['lanes']:
        pattern=Counter()
        for node in lane['kernels']:
            function=inventory['functions'][node['function_id']]
            pattern[('kernel',function['mangled_name'],tuple(node['grid']),tuple(node['block']),
                function['attributes']['static_shared_bytes'],node['dynamic_shared_bytes'])]+=1
        for op in operations[lane['lane']]:
            if op['kind']=='memset':pattern[('memset',op['bytes'],op['value'])]+=1
            elif op['kind']=='memcpy':pattern[('memcpy','DtoD',op['bytes'])]+=1
            else:raise ValueError('unknown captured transfer kind')
        expected.append(pattern)
    lines=text.splitlines();groups=defaultdict(dict);launches=[];routes={}
    for i,line in enumerate(lines):
        api=re.match(r'^RUNTIME .*?"cudaGraphLaunch_v\d+".*?correlationId (\d+)$',line)
        if api:launches.append(int(api.group(1)))
        kernel=KERNEL.match(line)
        if kernel:
            if i+2>=len(lines):raise ValueError('truncated kernel activity')
            geometry=GEOMETRY.search(lines[i+1]);location=LOCATION.search(lines[i+2])
            if not geometry or not location:raise ValueError('malformed kernel activity')
            start,end,name,correlation=kernel.groups();values=list(map(int,geometry.groups()))
            if int(end)<=int(start) or values[6:9]!=[0,0,0]:raise ValueError('unsupported/incomplete kernel execution')
            event=('kernel',name,tuple(values[:3]),tuple(values[3:6]),values[9],values[10])
        else:
            copy=re.match(r'^MEMCPY "([^"]+)" \[ (\d+), (\d+) \].*?size (\d+), copyCount (\d+).*?correlationId (\d+)$',line)
            fill=re.match(r'^MEMSET \[ (\d+), (\d+) \].*?value (\d+), size (\d+), correlationId (\d+)$',line)
            if not copy and not fill:continue
            if i+1>=len(lines):raise ValueError('truncated transfer activity')
            location=LOCATION.search(lines[i+1])
            if not location:raise ValueError('malformed transfer activity')
            if copy:
                direction,start,end,size,count,correlation=copy.groups()
                if int(count)!=1:raise ValueError('unsupported aggregated transfer activity')
                event=('memcpy',direction,int(size))
            else:
                start,end,value,size,correlation=fill.groups();event=('memset',int(size),int(value))
            if int(end)<=int(start):raise ValueError('incomplete transfer activity')
        device,context,stream,graph,node=map(int,location.groups())
        if graph==0:continue # eager warmup is outside the replay claim
        if device!=device_ordinal or not node:raise ValueError('graph activity device/node mismatch')
        route=(device,context,stream)
        if graph in routes and routes[graph]!=route:
            raise ValueError('executed graph context/stream changed')
        routes[graph]=route
        key=(graph,int(correlation));record=groups[key]
        if node in record:raise ValueError('duplicate executed graph node')
        record[node]=event
    if len(launches)!=len(set(launches)) or len(launches)!=replays_per_lane*len(expected):
        raise ValueError('incomplete/duplicate graph launch API evidence')
    if {correlation for graph,correlation in groups}!=set(launches):raise ValueError('launch/activity correlation disagreement')
    graph_groups=defaultdict(list)
    for (graph,correlation),record in groups.items():graph_groups[graph].append(record)
    if len(graph_groups)!=len(expected):raise ValueError('executed/captured graph count disagreement')
    unmatched=list(expected);total=0
    for graph,records in graph_groups.items():
        if len(records)!=replays_per_lane:raise ValueError('incomplete per-graph replay count')
        nodes=records[0]
        if any(r!=nodes for r in records[1:]):raise ValueError('graph replay node identity/shape changed')
        pattern=Counter(nodes.values())
        for j,wanted in enumerate(unmatched):
            if pattern==wanted:unmatched.pop(j);break
        else:raise ValueError('executed graph differs from captured kernel/transfer profile')
        total+=len(nodes)*len(records)
    if unmatched:raise ValueError('unexecuted captured graph lane')
    return dict(scope='CUPTI_executed_graph_nodes_not_complete_parameter_or_production_admission',
        graph_launches=len(launches),executed_graph_nodes=total,graphs=len(graph_groups),
        immutable_graph_api_observed=True,production_admitted=False)
