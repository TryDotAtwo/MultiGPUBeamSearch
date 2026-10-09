"""Lossless archival codec for benchmark frontier colors; no model execution."""
import argparse,pathlib,json,hashlib,numpy as np

def decode(packed,destination,parents,expected_sha):
 digest=hashlib.sha256();written=0
 with pathlib.Path(packed).open('rb') as source,pathlib.Path(destination).open('wb') as target:
  while data:=source.read(65536*36):
   if len(data)%36:raise ValueError('invalid packed row')
   rows=len(data)//36;bits=np.unpackbits(np.frombuffer(data,np.uint8).reshape(rows,36),axis=1,bitorder='little').reshape(rows,96,3)
   colors=(bits*np.asarray([1,2,4],dtype=np.uint8)).sum(-1).astype(np.uint8);raw=np.zeros((rows,112),np.uint8);raw[:,:96]=colors
   block=raw.tobytes();target.write(block);digest.update(block);written+=rows
 if written!=parents or digest.hexdigest()!=expected_sha:raise ValueError('decoded frontier hash mismatch')
 return digest.hexdigest()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('packed');p.add_argument('destination');p.add_argument('--parents',type=int,required=True);p.add_argument('--sha256',required=True);a=p.parse_args();print(decode(a.packed,a.destination,a.parents,a.sha256))
