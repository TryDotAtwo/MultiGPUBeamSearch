"""Persistent all-rank exact native memory planning with explicit ownership."""
from pathlib import Path
import uuid
from .calibration_session import InferenceSession
from .beam_capacity import CapacityRejected
from .pipeline_profiles import validate_rank_plans


class NativePlanSession:
    def __init__(self, runner, environment, world, directory, *, deadline, puzzle_id=0):
        self.sessions=[]
        self.supports_components=False
        self.inference_micro=environment.get('BEAM_ENSEMBLE_INFERENCE_MICRO')
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
            capabilities=[]
            for rank,session in enumerate(self.sessions):
                ready=session.receive()
                if ready.get('ready') is not True or ready.get('rank')!=rank:
                    raise RuntimeError('native planning service did not acknowledge its rank')
                capabilities.append(ready.get('component_protocol')=='persistent-exact-capacity-v1')
            if any(capabilities) and not all(capabilities):raise RuntimeError('mixed native component service cohort')
            self.supports_components=all(capabilities)
        except BaseException:
            self.close()
            raise

    def admit(self, beam, environment):
        environment=self._environment(environment)
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

    def set_deadline(self, deadline):
        """Rebind the retained owner to the downstream phase's bounded budget."""
        import math
        import time
        if not math.isfinite(deadline) or deadline <= time.monotonic():
            raise ValueError('planning phase deadline must be finite and in the future')
        for session in self.sessions:session.deadline=deadline

    def _environment(self,environment):
        environment=dict(environment)
        if 'BEAM_ENSEMBLE_INFERENCE_MICRO' in environment:
            if environment.pop('BEAM_ENSEMBLE_INFERENCE_MICRO')!=self.inference_micro:
                raise ValueError('planner cannot change the frozen model inference microbatch')
        return environment

    def measure_components(self,beam,environment):
        if not self.supports_components:raise RuntimeError('native component session unsupported')
        environment=self._environment(environment)
        for session in self.sessions:
            session.send_request({'beam':beam,'environment':environment,'prepare_component':True})
        ready=[session.receive() for session in self.sessions]
        if any(row.get('prepared') is not True or row.get('rank')!=rank for rank,row in enumerate(ready)):
            for session in self.sessions:session.send_request({'cancel_component':True})
            for session in self.sessions:session.receive()
            raise ValueError('component allocation was not prepared on every rank')
        # Two-phase protocol: no rank enters NCCL until every allocation succeeded.
        for session in self.sessions:session.send_request({'run_component':True})
        return [session.receive() for session in self.sessions]

    def close(self):
        sessions,self.sessions=self.sessions,[]
        failure=None
        for session in sessions:
            try:session.close()
            except Exception as error:
                if failure is None:failure=error
        if failure is not None:raise failure

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
