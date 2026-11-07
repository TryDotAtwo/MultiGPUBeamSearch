"""Read official CLI runtime selection; no local tests or cloud mutations."""
import requests
r=requests.get('https://raw.githubusercontent.com/vast-ai/vast-cli/master/vast.py',timeout=20)
r.raise_for_status()
start=r.text.index('def get_runtype')
print(r.text[start:start+2800])
