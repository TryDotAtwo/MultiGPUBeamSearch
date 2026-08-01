"""Shared normalization for the Cloudflare Worker v2 hardware contract."""

from __future__ import annotations


def publication_family(gpu_name: str, native_sm: int) -> str:
    upper = gpu_name.upper()
    inferred: tuple[int, str] | None = None
    if "T4" in upper: inferred = (75, "T4")
    if "A100" in upper: inferred = (80, "A100")
    if "RTX 30" in upper or "A10" in upper: inferred = (86, "A10" if "A10" in upper else "Ampere")
    if "L4" in upper: inferred = (89, "L4")
    if "RTX 40" in upper or "L40" in upper: inferred = (89, "Ada")
    if "H100" in upper: inferred = (90, "H100")
    if "BLACKWELL" in upper or "RTX 50" in upper: inferred = (120, "Blackwell")
    if inferred is None or inferred[0] != native_sm:
        raise ValueError(f"GPU name does not match native sm{native_sm}: {gpu_name}")
    return inferred[1]


def validate_publish_hardware(envelope):
    profile = envelope["profile"]
    hardware = envelope["hardware"]
    provenance = envelope["provenance"]
    count, world, sm = hardware.get("accelerator_count"), hardware.get("world_size"), hardware.get("native_sm")
    if hardware.get("platform") != "slurm" or provenance.get("platform") != "slurm": raise ValueError("publication requires SLURM provenance")
    if isinstance(world, bool) or not isinstance(world, int) or not 1 <= world <= 16 or count != world: raise ValueError("hardware world size is invalid")
    names = hardware.get("gpu_names")
    if not isinstance(names, list) or len(names) != world or len(set(names)) != 1: raise ValueError("hardware GPU names are invalid or mixed")
    family = publication_family(names[0], sm)
    if profile.get("gpu_family") != family or profile.get("native_sm") != sm or profile.get("world_size") != world: raise ValueError("profile does not match exact native hardware")
    if profile.get("vram_mib") != hardware.get("vram_mib_per_gpu"): raise ValueError("profile VRAM does not match hardware")
    if profile.get("profile_status") not in {"measured", "bounded_from_measured"}: raise ValueError("profile is not evidence-backed")
    requested, effective, delta = profile.get("requested_beam"), profile.get("effective_beam"), profile.get("alignment_delta")
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (requested, effective, delta)) or effective - requested != delta or delta < 0: raise ValueError("profile beam alignment is invalid")
    if provenance.get("run_id") != envelope.get("run_id"): raise ValueError("provenance run id mismatch")
