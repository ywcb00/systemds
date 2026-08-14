# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------
import itertools
import multiprocessing as mp
import multiprocessing.connection as mp_connection
import os
import signal
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

from systemds.scuro.utils.memory_utility import is_cuda_oom

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _resolve_thread_count(num_threads: int) -> int:
    explicit = os.environ.get("OMP_NUM_THREADS")
    if explicit:
        try:
            num_threads = int(explicit)
        except ValueError:
            pass
    return max(1, int(num_threads))


def set_thread_env_before_spawn(num_threads: int) -> None:
    num_threads = _resolve_thread_count(num_threads)
    for var in _THREAD_ENV_VARS:
        os.environ[var] = str(num_threads)


def _worker_initializer(num_threads: int) -> None:
    num_threads = _resolve_thread_count(num_threads)
    for var in _THREAD_ENV_VARS:
        os.environ[var] = str(num_threads)
    try:
        torch.set_num_threads(num_threads)
    except Exception:
        pass


@dataclass
class _Job:
    job_id: int
    kind: str
    payload: tuple
    gpu_id: Optional[int] = None


@dataclass
class _JobResult:
    job_id: int
    ok: bool
    pid: Optional[int]
    value: Any = None
    error: Optional[str] = None
    cuda_oom: bool = False
    worker_died: bool = False


def _worker_main(
    job_q, result_q, dispatch: Dict[str, Callable], num_threads: int
) -> None:
    _worker_initializer(num_threads)
    while True:
        job = job_q.get()
        if job is None:
            return
        try:
            value = dispatch[job.kind](job.payload, job.gpu_id)
            result_q.put(_JobResult(job.job_id, True, os.getpid(), value=value))
        except Exception as e:
            result_q.put(
                _JobResult(
                    job.job_id,
                    False,
                    os.getpid(),
                    error=f"{type(e).__name__}: {e}",
                    cuda_oom=is_cuda_oom(e),
                )
            )


def _describe_worker_death(exitcode: Optional[int]) -> str:
    if exitcode is None:
        return "exit code unknown"
    if exitcode < 0:
        try:
            sig = signal.Signals(-exitcode)
        except ValueError:
            return f"killed by signal {-exitcode}"
        hint = {
            signal.SIGKILL: " (often the OOM killer or an explicit kill -9)",
            signal.SIGSEGV: " (segmentation fault, often a native library crash, e.g. CUDA/BLAS)",
            signal.SIGABRT: " (abort, often a C-level assertion or CUDA error)",
            signal.SIGBUS: " (bus error, often a full /dev/shm or a shared-memory issue)",
        }.get(sig, "")
        return f"killed by signal {sig.name} ({-exitcode}){hint}"
    return f"exited with status {exitcode}"


def create_mp_context():
    ctx_name = os.environ.get("SCURO_MP_CONTEXT", "spawn")
    try:
        return mp.get_context(ctx_name)
    except ValueError:
        return mp.get_context("spawn")


class PersistentWorkerPool:
    def __init__(
        self,
        n_workers: int,
        dispatch: Dict[str, Callable],
        ctx=None,
        threads_per_worker: int = 1,
    ):
        self._ctx = ctx or create_mp_context()
        self._dispatch = dispatch
        self._threads_per_worker = max(1, int(threads_per_worker))
        self._result_q = self._ctx.Queue()
        self._job_counter = itertools.count()
        self._workers: Dict[int, Dict[str, Any]] = {}
        self._idle_pids: List[int] = []
        self._running: Dict[int, tuple] = {}
        for _ in range(max(1, n_workers)):
            self._spawn_worker()

    def _spawn_worker(self) -> None:
        set_thread_env_before_spawn(self._threads_per_worker)
        job_q = self._ctx.Queue()
        p = self._ctx.Process(
            target=_worker_main,
            args=(job_q, self._result_q, self._dispatch, self._threads_per_worker),
            daemon=True,
        )
        p.start()
        self._workers[p.pid] = {"process": p, "job_q": job_q}
        self._idle_pids.append(p.pid)

    @property
    def has_idle_worker(self) -> bool:
        return len(self._idle_pids) > 0

    @property
    def num_in_flight(self) -> int:
        return len(self._running)

    def submit(self, kind: str, payload: tuple, gpu_id: Optional[int] = None) -> int:
        if not self._idle_pids:
            raise RuntimeError("submit() called with no idle worker available")
        job_id = next(self._job_counter)
        job = _Job(job_id, kind, payload, gpu_id)
        pid = self._idle_pids.pop()
        self._running[job_id] = (pid, job)
        self._workers[pid]["job_q"].put(job)
        return job_id

    def wait(self) -> _JobResult:
        while True:
            sentinel_to_pid = {
                w["process"].sentinel: pid for pid, w in self._workers.items()
            }
            ready = mp_connection.wait(
                [self._result_q._reader, *sentinel_to_pid.keys()]
            )
            if self._result_q._reader in ready:
                jr = self._result_q.get()
                entry = self._running.pop(jr.job_id, None)
                if entry is not None:
                    pid, _job = entry
                    if pid in self._workers:
                        self._idle_pids.append(pid)
                return jr
            for r in ready:
                dead_pid = sentinel_to_pid.get(r)
                if dead_pid is None:
                    continue
                result = self._replace_dead_worker(dead_pid)
                if result is not None:
                    return result

    def _replace_dead_worker(self, pid: int) -> Optional[_JobResult]:
        w = self._workers.pop(pid, None)
        if w is None:
            return None
        try:
            if pid in self._idle_pids:
                self._idle_pids.remove(pid)
        except ValueError:
            pass
        try:
            w["process"].join(timeout=1)
        except Exception:
            pass
        exitcode = w["process"].exitcode
        try:
            w["job_q"].close()
            w["job_q"].join_thread()
        except Exception:
            pass

        failed_job_id = None
        for job_id, (running_pid, _job) in self._running.items():
            if running_pid == pid:
                failed_job_id = job_id
                break
        if failed_job_id is not None:
            self._running.pop(failed_job_id, None)

        self._spawn_worker()

        if failed_job_id is None:
            return None
        return _JobResult(
            failed_job_id,
            False,
            pid,
            error=f"worker process died ({_describe_worker_death(exitcode)})",
            worker_died=True,
        )

    def shutdown(self) -> None:
        for w in self._workers.values():
            try:
                w["job_q"].put(None)
            except Exception:
                pass
        for w in self._workers.values():
            try:
                w["process"].join(timeout=5)
                if w["process"].is_alive():
                    w["process"].kill()
                    w["process"].join(timeout=2)
            except Exception:
                pass
        for w in self._workers.values():
            try:
                w["job_q"].close()
                w["job_q"].join_thread()
            except Exception:
                pass
        try:
            self._result_q.close()
            self._result_q.join_thread()
        except Exception:
            pass
        self._workers.clear()
        self._idle_pids.clear()
        self._running.clear()
