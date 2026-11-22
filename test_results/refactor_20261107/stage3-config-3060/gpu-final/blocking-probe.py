#!/opt/conda/bin/python
import sys,os,time
from pathlib import Path
root=Path(sys.argv[1]).parent;rank=int(sys.argv[4]);(root/f"rank-{rank}.pid").write_text(str(os.getpid()))
time.sleep(30)
