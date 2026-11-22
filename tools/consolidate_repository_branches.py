import json, os, urllib.request, urllib.parse, urllib.error
root='https://api.github.com/repos/TryDotAtwo/MultiGPUBeamSearch/'
token=os.environ['GH_TOKEN']
def api(path,method='GET',body=None):
 data=None if body is None else json.dumps(body).encode()
 req=urllib.request.Request(root+path,data=data,method=method,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':'application/json','X-GitHub-Api-Version':'2022-11-28'})
 with urllib.request.urlopen(req,timeout=30) as res:
  raw=res.read();return json.loads(raw) if raw else None
q=lambda s:urllib.parse.quote(s,safe='')
expected=json.load(open('test_results/kaggle_single_branch_release_20261010/branches-before.json'))
# Archive every expected head before deleting any branch.
for b in expected:
 tag='archive/20261010/'+b['name'];sha=b['commit']['sha']
 try:old=api('git/ref/tags/'+q(tag))
 except urllib.error.HTTPError as e:
  if e.code!=404:raise
  old=api('git/refs','POST',{'ref':'refs/tags/'+tag,'sha':sha})
 assert old['object']['sha']==sha,('archive mismatch',tag)
 print('archived',b['name'],sha,flush=True)
# Require exact live inventory; fail rather than delete concurrent work.
live=api('branches?per_page=100')
assert {b['name'] for b in live}=={b['name'] for b in expected},'branch inventory changed'
for b in expected:
 if b['name']=='main':continue
 current=api('git/ref/heads/'+q(b['name']))
 assert current['object']['sha']==b['commit']['sha'],('branch moved',b['name'])
for b in expected:
 if b['name']=='main':continue
 current=api('git/ref/heads/'+q(b['name']))
 assert current['object']['sha']==b['commit']['sha'],('branch moved',b['name'])
 api('git/refs/heads/'+q(b['name']),'DELETE')
 print('deleted',b['name'],flush=True)
remaining=api('branches?per_page=100');assert [b['name'] for b in remaining]==['main'],remaining
print('PASS: only main remains; all26original heads retained as archive tags',flush=True)
