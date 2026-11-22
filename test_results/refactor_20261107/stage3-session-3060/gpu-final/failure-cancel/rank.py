import os,sys
from pathlib import Path
rank=os.environ['RANK'];Path(os.environ['BEAM_OWNER_PID_DIR'],f'rank-{rank}.pid').write_text(str(os.getpid()))
os.execv(sys.argv[1],sys.argv[1:])
