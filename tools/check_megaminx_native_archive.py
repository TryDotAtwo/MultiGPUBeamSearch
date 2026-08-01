"""Independent content and CUDA-image gates for native Megaminx archives."""
from __future__ import annotations
import argparse, io, json, re, subprocess, tarfile, tempfile
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Callable
import zstandard

FORBIDDEN_NAMES={".env","Dockerfile","token.txt","compile.sh"}
FORBIDDEN_SUFFIXES={".cu",".cuh",".cpp",".cc",".o",".a",".ptx"}
FORBIDDEN_CONTENT=re.compile(rb"(?:ghp_[A-Za-z0-9]{20,}|BEGIN (?:RSA|OPENSSH) PRIVATE KEY|CLOUDFLARE_API_TOKEN|[A-Za-z]:\\\\(?:Users|Documents|Downloads)\\\\|/(?:home|Users|mnt)/)",re.I)

def inspect_cuda_image_text(text:str,expected_sm:int)->tuple[int,...]:
    lowered=text.lower()
    if "ptx" in lowered or re.search(r"\bcompute_[0-9]+\b",lowered): raise ValueError("PTX/JIT image is forbidden")
    images=tuple(sorted({int(v) for v in re.findall(r"\bsm_([0-9]+)\b",lowered)}))
    if images!=(expected_sm,): raise ValueError(f"expected only sm_{expected_sm}; found {images}")
    return images

def _members(tar:tarfile.TarFile,prefix:str)->dict[str,tarfile.TarInfo]:
    result={}
    for member in tar.getmembers():
        path=PurePosixPath(member.name)
        if member.name in result: raise ValueError(f"duplicate archive member: {member.name}")
        if member.issym() or member.islnk() or path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0]!=prefix or not member.isfile(): raise ValueError(f"unsafe archive member: {member.name}")
        relative=PurePosixPath(*path.parts[1:])
        if relative.name in FORBIDDEN_NAMES or relative.suffix.lower() in FORBIDDEN_SUFFIXES: raise ValueError(f"forbidden archive member: {relative}")
        result[member.name]=member
    return result

def check_archive(archive:Path,expected_sm:int,cuobjdump:Callable[[Path],str])->dict[str,object]:
    expected_name=f"megaminx-sm{expected_sm}-linux-x86_64.tar.zst"
    if archive.name!=expected_name: raise ValueError(f"archive name must be {expected_name}")
    raw=zstandard.ZstdDecompressor().decompress(archive.read_bytes(),max_output_size=8*1024**3); prefix=expected_name.removesuffix(".tar.zst")
    with tarfile.open(fileobj=io.BytesIO(raw),mode="r:") as tar:
        members=_members(tar,prefix); blobs={name[len(prefix)+1:]:tar.extractfile(member).read() for name,member in members.items()}
    for required in ("MANIFEST.json","SHA256SUMS","bin/production_runner"):
        if required not in blobs: raise ValueError("archive lacks manifest, checksums, or runner")
    if any(FORBIDDEN_CONTENT.search(data) for data in blobs.values()): raise ValueError("forbidden private or secret-like archive content")
    manifest=json.loads(blobs["MANIFEST.json"])
    if manifest.get("native_sm")!=expected_sm or manifest.get("contains_ptx") is not False: raise ValueError("manifest native image declaration mismatch")
    payloads=manifest.get("payloads")
    if not isinstance(payloads,dict) or set(payloads)!=set(blobs)-{"MANIFEST.json","SHA256SUMS"}: raise ValueError("manifest payload allowlist mismatch")
    for name,expected in payloads.items():
        if sha256(blobs[name]).hexdigest()!=expected: raise ValueError(f"sha256 mismatch for {name}")
    checksum_map=dict(line.split("  ",1)[::-1] for line in blobs["SHA256SUMS"].decode("ascii").splitlines())
    expected_checksums={**payloads,"MANIFEST.json":sha256(blobs["MANIFEST.json"]).hexdigest()}
    if checksum_map!=expected_checksums: raise ValueError("SHA256SUMS does not match manifest payloads")
    with tempfile.TemporaryDirectory() as temp:
        runner=Path(temp)/"production_runner"; runner.write_bytes(blobs["bin/production_runner"]); inspect_cuda_image_text(cuobjdump(runner),expected_sm)
    return {"archive":archive.name,"native_sm":expected_sm,"ptx":False,"members":len(members)}

def _cuobjdump(path:Path)->str:
    result=subprocess.run(["cuobjdump","--list-elf","--list-ptx",str(path)],text=True,capture_output=True,check=False)
    if result.returncode!=0: raise ValueError("cuobjdump failed")
    return result.stdout+result.stderr

def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("archive",type=Path); parser.add_argument("--sm",type=int,required=True); args=parser.parse_args(); print(json.dumps(check_archive(args.archive,args.sm,_cuobjdump),sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
