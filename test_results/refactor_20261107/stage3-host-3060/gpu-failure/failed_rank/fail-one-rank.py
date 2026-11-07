import os,sys
if os.environ['RANK']=='1':os.environ['BEAM_NATIVE_LIBRARY']='/beam-no-such-library.so'
os.execv(sys.argv[1],sys.argv[1:])
