# Native NCCL rendezvous sessions

For world_size greater than one, all ranks must receive the same unique
launch identity. Native production runner accepts BEAM_NCCL_RUN_ID, or
TORCHELASTIC_RUN_ID with TORCHELASTIC_RESTART_COUNT from torchrun. Missing,
empty, `none`, newline-containing or excessive identities fail closed.
Never reuse an explicit identity for a new manual launch.

The rendezvous record contains a versioned session/restart header and
exactly one NCCL unique ID. Old sessions, legacy bare IDs, truncated records
and trailing bytes are ignored until a valid matching record is published
or the bounded wait expires. A failed publisher must not be interpreted as
a successful collective initialization.

H200, portable and B300 manual launchers export their unique mktemp run
directory as the session. MEPhI beam_prepare_nccl_file creates a private
directory and nccl.bin per call. Segment repair derives session/file names
from the outer identity plus segment index, even for duplicate repair IDs.
Do not delete rendezvous files while other ranks may still read them.

tests/test_manual_launcher_session.py checks these CPU launch environment
contracts with fake executable/device/model gates. It does not validate
GPU hardware, inference, model inputs, or numerical search quality. Real
two-rank production acceptance is recorded separately under test_results/.
