import subprocess,json,re
from pathlib import Path
svc=Path(r'C:\Users\Иван Литвак\.codex\worktrees\cube555-blend-2xt4\100XH100\services\cayleypy-results-ingest')
p=subprocess.Popen(['node','node_modules/wrangler/bin/wrangler.js','tail','--config','wrangler.github-staging.jsonc','--format','json'],cwd=svc,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8')
buf='';decoder=json.JSONDecoder()
for line in p.stdout:
 buf+=line
 try:x,end=decoder.raw_decode(buf.lstrip())
 except ValueError:continue
 buf=''
 version=x.get('scriptVersion') or {}; version=version.get('id') if isinstance(version,dict) else None
 summary={'event_timestamp_ms':x.get('eventTimestamp') if isinstance(x.get('eventTimestamp'),(int,float)) else None,'script_version_id':version if isinstance(version,str) and re.fullmatch(r'[0-9a-f-]{36}',version) else None}
 summary.update({k:x.get(k) for k in ['outcome','cpuTime','wallTime','executionModel','entrypoint','truncated']});summary['exceptions']=[e.get('name') for e in x.get('exceptions',[])]; ev=x.get('event') or {}; summary['event_keys']=sorted(ev) if isinstance(ev,dict) else []; summary['event_kind']='alarm' if x.get('executionModel')=='durableObject' and 'scheduledTime' in ev else 'cron' if 'scheduledTime' in ev else 'queue' if 'queue' in ev else 'request' if 'request' in ev else 'rpc' if 'rpcMethod' in ev else 'other'; summary['rpc_method']=ev.get('rpcMethod') if ev.get('rpcMethod') in ['flush','enqueueValidated','fetch','alarm','maintain'] else None; summary['safe_exception_codes']=[e.get('message') for e in x.get('exceptions',[]) if e.get('message') in ['github_writer_retryable','github_temporary_unavailable','github_app_auth_failed','github_app_auth_unavailable']]; summary['safe_http_failures']=[]
 for item in x.get('logs',[]):
  for msg in item.get('message',[]):
   try:d=json.loads(msg)
   except (ValueError,TypeError):continue
   if isinstance(d,dict) and d.get('event')=='github_http_failure':
    safe={k:d[k] for k in ['status','retry-after','x-ratelimit-remaining','x-ratelimit-reset'] if isinstance(d.get(k),int)}
    safe['operation']=d.get('operation') if d.get('operation') in ['auth','request'] else None
    safe['method']=d.get('method') if d.get('method') in ['GET','POST','PATCH','OTHER'] else None
    summary['safe_http_failures'].append(safe)
 line=json.dumps(summary);print(line,flush=True);open(r'D:\100XH100\test_results\cube555_20260930_handoff\downstream_trace.jsonl','a',encoding='utf-8').write(line+'\n')
