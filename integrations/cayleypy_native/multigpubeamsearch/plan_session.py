"""Persistent all-rank exact native memory planning with explicit ownership."""
from pathlib import Path
import uuid
from .calibration_session import InferenceSession
from .beam_capacity import CapacityRejected
from .pipeline_profiles import validate_rank_plans


class NativePlanSession:
    def __init__(self, runner, environment, world, directory, *, deadline, puzzle_id=0):
        self.sessions=[]
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
        try:
            env=dict(environment, BEAM_CALIBRATION_PLAN_SESSION='1', BEAM_BENCHMARK_PLAN_ONLY='1',
                     BEAM_NCCL_RUN_ID='plan-session-'+uuid.uuid4().hex,
                     BEAM_NCCL_ID_FILE=str(directory/'nccl-id.bin'))
            for rank in range(world):
                rank_env=dict(env,RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE=str(world))
                self.sessions.append(InferenceSession(
                    [str(runner),str(puzzle_id),'1','1024',str(world),str(rank)],rank_env,
                    directory/f'rank-{rank}.log',deadline=deadline))
            for rank,session in enumerate(self.sessions):
                ready=session.receive()
                if ready.get('ready') is not True or ready.get('rank')!=rank:
                    raise RuntimeError('native planning service did not acknowledge its rank')
        except BaseException:
            self.close()
            raise

    def admit(self, beam, environment):
        for session in self.sessions:
            session.send_request({'beam':beam,'environment':environment})
        rows=[session.receive() for session in self.sessions]
        if any(row.get('capacity_rejected') is True for row in rows):
            raise CapacityRejected('; '.join(row.get('reason','') for row in rows
                                            if row.get('capacity_rejected') is True))
        if any(row.get('admitted') is not True for row in rows):
            raise RuntimeError('native planner returned an invalid admission receipt')
        plans=[row['plan'] for row in rows]
        validate_rank_plans(plans)
        if plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE']<beam:
            raise ValueError('native planner shrank requested frontier')
        return plans

    def close(self):
        for session in self.sessions:
            session.close()
        self.sessions=[]

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
