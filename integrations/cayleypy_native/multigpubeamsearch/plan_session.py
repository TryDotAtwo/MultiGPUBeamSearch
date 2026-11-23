"""Persistent all-rank exact native memory planning with explicit ownership."""
from pathlib import Path
import uuid
from .calibration_session import InferenceSession
from .beam_capacity import CapacityRejected
from .pipeline_profiles import validate_rank_plans


class _PythonPlanSession:
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
        except BaseException as error:
            logs=[Path(session.log.name) for session in self.sessions if hasattr(session,'log')]
            self.close()
            failures=[line.split('=',1)[1] for path in logs for line in path.read_text(errors='replace').splitlines()
                      if line.startswith('production_runner_error=')]
            if failures and all(message=='insufficient blend GPU reserve' for message in failures):
                raise CapacityRejected('native planner initialization rejected measured blend reserve') from error
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


def _native_planning_type():
    """Return the installed Rust planner, or None for a pre-planner wheel."""
    from .errors import NativeUnavailable
    from . import rust_control

    try:
        runtime = rust_control.extension()
    except NativeUnavailable as error:
        if str(error) in (
            "the matching Rust/PyO3 runtime is not installed",
            "the matching Rust/PyO3 runtime could not be loaded",
        ):
            return None, None
        raise
    planner = getattr(runtime, "PlanningSession", None)
    if planner is None:
        return runtime, None
    if not callable(planner):
        raise RuntimeError("installed Rust PlanningSession is not constructible")
    return runtime, planner


class NativePlanSession:
    """Own one all-rank planner through Rust; retain the old-wheel adapter."""

    def __init__(self, runner, environment, world, directory, *, deadline, puzzle_id=0):
        import json
        import math
        import time

        self._native = None
        self._runtime = None
        self._legacy = None
        runtime, planner = _native_planning_type()
        if planner is None:
            self._legacy = _PythonPlanSession(
                runner, environment, world, directory, deadline=deadline, puzzle_id=puzzle_id
            )
            self.supports_components = self._legacy.supports_components
            self.inference_micro = self._legacy.inference_micro
            return

        directory = Path(directory).resolve()
        # ProbeCohort creates this fresh directory as part of Rust-owned launch.
        timeout = deadline - time.monotonic()
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise TimeoutError("persistent native planner deadline expired before launch")
        env = dict(
            environment,
            BEAM_CALIBRATION_PLAN_SESSION="1",
            BEAM_BENCHMARK_PLAN_ONLY="1",
            BEAM_NCCL_RUN_ID="plan-session-" + uuid.uuid4().hex,
            BEAM_NCCL_ID_FILE=str(directory / "nccl-id.bin"),
        )
        command = [str(runner), str(puzzle_id), "1", "1024"]
        self._runtime = runtime
        try:
            self._native = planner(
                command, env, int(world), str(Path.cwd()), str(directory), timeout
            )
        except Exception as error:
            translated = self._translate_capacity_error(error)
            if translated is not None:
                raise translated from error
            raise
        self.supports_components = bool(self._native.supports_components)
        self.inference_micro = environment.get("BEAM_ENSEMBLE_INFERENCE_MICRO")

    def _translate_capacity_error(self, error):
        capacity_error = getattr(self._runtime, "CapacityRejected", None)
        if isinstance(capacity_error, type) and isinstance(error, capacity_error):
            return CapacityRejected(str(error))
        return None

    @property
    def pids(self):
        if self._native is not None:
            return list(self._native.pids)
        if self._legacy is not None:
            return [session.process.pid for session in self._legacy.sessions]
        return []

    def exit_codes(self):
        if self._native is not None:
            return self._native.exit_codes()
        if self._legacy is not None:
            return [session.process.poll() for session in self._legacy.sessions]
        return []

    def admit(self, beam, environment):
        import json

        if self._legacy is not None:
            return self._legacy.admit(beam, environment)
        if self._native is None:
            raise RuntimeError("planner session is closed")
        try:
            rows = json.loads(self._native.admit(beam, dict(environment)))
        except Exception as error:
            translated = self._translate_capacity_error(error)
            if translated is not None:
                raise translated from error
            raise
        if not isinstance(rows, list):
            raise RuntimeError("Rust planner returned a non-list admission")
        return rows

    def set_deadline(self, deadline):
        import math
        import time

        if self._legacy is not None:
            return self._legacy.set_deadline(deadline)
        if self._native is None:
            raise RuntimeError("planner session is closed")
        if not math.isfinite(deadline) or deadline <= time.monotonic():
            raise ValueError("planning phase deadline must be finite and in the future")
        self._native.set_deadline(max(0.001, deadline - time.monotonic()))

    def measure_components(self, beam, environment):
        import json

        if self._legacy is not None:
            return self._legacy.measure_components(beam, environment)
        if self._native is None:
            raise RuntimeError("planner session is closed")
        if not self.supports_components:
            raise RuntimeError("native component session unsupported")
        try:
            rows = json.loads(self._native.measure_components(beam, dict(environment)))
        except Exception as error:
            translated = self._translate_capacity_error(error)
            if translated is not None:
                raise translated from error
            raise
        if not isinstance(rows, list):
            raise RuntimeError("Rust planner returned a non-list component result")
        return rows

    def cancel(self):
        if self._legacy is not None:
            return self._legacy.close()
        if self._native is not None:
            self._native.cancel()

    def close(self):
        if self._legacy is not None:
            legacy, self._legacy = self._legacy, None
            return legacy.close()
        if self._native is not None:
            self._native.close()
            self._native = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
