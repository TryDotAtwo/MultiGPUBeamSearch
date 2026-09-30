const fs=require('fs'), zlib=require('zlib'), crypto=require('crypto'), cp=require('child_process');
const root='D:/100XH100/test_results/cube555_20260930_handoff';
const repo='C:/Users/Иван Литвак/.codex/worktrees/cube555-blend-2xt4/100XH100';
const items=JSON.parse(zlib.gunzipSync(fs.readFileSync(repo+'/test_results/cube555_retry_2026-09-30/prepared/original.json.gz'))).results;
const snapshot=JSON.parse(fs.readFileSync(root+'/do_live_canary/d1_after.json','utf8').replace(/^\uFEFF/,''));
if(snapshot.some(s=>s.success!==true||s.results.length>=10000)) throw Error('incomplete D1');
const rows=new Map(snapshot.flatMap(s=>s.results).map(r=>[r.idempotency_key,r]));
function canonical(x){if(x===null)return'null';if(typeof x==='number'){if(!Number.isFinite(x))throw Error('invalid number');return Object.is(x,-0)?'0':String(x)}if(typeof x==='string'||typeof x==='boolean')return JSON.stringify(x);if(Array.isArray(x))return'['+x.map(canonical).join(',')+']';return'{'+Object.keys(x).sort().map(k=>JSON.stringify(k)+':'+canonical(x[k])).join(',')+'}'}
const api=p=>JSON.parse(cp.execFileSync('C:/Program Files/GitHub CLI/gh.exe',['api','repos/TryDotAtwo/cayleypy-beam-results/'+p],{encoding:'utf8',maxBuffer:20*1024*1024}));
const segment=x=>x.toLowerCase().replace(/[^a-z0-9._-]+/g,'-').replace(/^-+|-+$/g,'');
let expected=items.map(envelope=>{const row=rows.get(envelope.idempotency_key);if(!row)throw Error('missing original');const path=['results','v1',segment(envelope.competition),segment(envelope.puzzle_type),String(envelope.puzzle_id),envelope.submitted_at.slice(0,10),row.submission_id+'.json'].join('/');const body=Buffer.from(canonical({submission_id:row.submission_id,envelope}));const hash=crypto.createHash('sha1').update(Buffer.from('blob '+body.length+'\0')).update(body).digest('hex');return{semantic_key:envelope.idempotency_key,submission_id:row.submission_id,path,hash}});
let ledger=expected.map(e=>({...e,branches:{}}));
let report={original_count:items.length,branches:{},checked_at:new Date().toISOString(),original_archive_sha256:crypto.createHash('sha256').update(fs.readFileSync(repo+'/test_results/cube555_retry_2026-09-30/prepared/original.json.gz')).digest('hex'),d1_snapshot_sha256:crypto.createHash('sha256').update(fs.readFileSync(root+'/do_live_canary/d1_after.json')).digest('hex')};
for(const branch of ['main','ingest/staging']){
 const ref=api('git/ref/heads/'+branch),head=ref.object.sha;const cache=new Map();
 function directory(path){if(cache.has(path))return cache.get(path);let sha=head;if(path){const parts=path.split('/');const name=parts.pop();const parent=directory(parts.join('/'));const entry=parent.find(e=>e.path===name&&e.type==='tree');if(!entry){cache.set(path,[]);return[]}sha=entry.sha}const t=api('git/trees/'+sha);if(t.truncated)throw Error('truncated tree');cache.set(path,t.tree);return t.tree}
 let present=0,matched=0,mismatched=0;
 for(const e of ledger){const parts=e.path.split('/');const name=parts.pop();const found=directory(parts.join('/')).find(x=>x.path===name&&x.type==='blob');e.branches[branch]={head,actual_blob_sha:found?.sha??null,exact_match:found?.sha===e.hash};if(found){present++;if(found.sha===e.hash)matched++;else mismatched++}}
 report.branches[branch]={head,present,exact_original_blob_matches:matched,mismatched,missing:items.length-present};console.log(branch,report.branches[branch]);
}
const ledgerText=JSON.stringify(ledger,null,2);fs.writeFileSync(root+'/do_live_canary/git_blob_ledger.json',ledgerText);report.ledger_sha256=crypto.createHash('sha256').update(ledgerText).digest('hex');
fs.writeFileSync(root+'/do_live_canary/git_blob_reconciliation.json',JSON.stringify(report,null,2));