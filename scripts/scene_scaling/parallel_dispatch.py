"""Move a queued SS01 arm to another GPU without interrupting active training.

The scientific plan and hashed source/configuration files remain unchanged.
Only the in-memory worker queue is overridden. An existing queue controller
can be suspended while its active training child continues; after that child
finishes, a replacement worker evaluates its final checkpoint.
"""

import argparse
import copy
import fcntl
import os
from pathlib import Path
import signal
import time

from .common import ROOT, read_json, save_json, sha256, verify_package
from .run import worker


class LinuxProcess:
    """Signal a known PID only while its /proc start-time identity matches."""

    def __init__(self, pid):
        self.pid = int(pid)
        self.proc = Path("/proc") / str(self.pid)
        self.start_ticks = self._snapshot()["start_ticks"]

    def _snapshot(self):
        try:
            fields = (self.proc / "stat").read_text().rsplit(")", 1)[1].split()
            return {"state": fields[0], "ppid": int(fields[1]), "start_ticks": int(fields[19])}
        except FileNotFoundError:
            return None

    def status(self):
        state = self._snapshot()
        return state["state"] if state and state["start_ticks"] == self.start_ticks else "exited"

    def is_running(self):
        return self.status() not in ("Z", "exited")

    def cmdline(self):
        return (self.proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")

    def cwd(self):
        return (self.proc / "cwd").resolve(strict=True)

    def ppid(self):
        return self._snapshot()["ppid"]

    def send(self, sig):
        if not self.is_running():
            raise ProcessLookupError(self.pid)
        os.kill(self.pid, sig)

    def suspend(self):
        self.send(signal.SIGSTOP)

    def resume(self):
        self.send(signal.SIGCONT)

    def terminate(self):
        self.send(signal.SIGTERM)

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.is_running():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Controller {self.pid} did not exit")
            time.sleep(.1)


def one_run(package, plan, gpu, run_id):
    assert sum(r["id"] == run_id for r in plan["runs"]) == 1
    runtime_plan = copy.deepcopy(plan)
    runtime_plan["queue"] = {str(gpu): [run_id]}
    worker(package, runtime_plan, gpu, resume=False)


def watch_existing(args, package, plan):
    controller = LinuxProcess(args.controller_pid)
    training = LinuxProcess(args.training_pid)
    for process, command in ((controller, "worker"), (training, "train")):
        cmd = process.cmdline()
        assert "scripts.scene_scaling.run" in cmd and command in cmd
        assert Path(process.cwd()).resolve() == ROOT
    assert training.ppid() == controller.pid
    assert args.run in training.cmdline()
    assert controller.pid != os.getpid()
    assert read_json(package / f"worker_gpu{args.gpu}.json") == {
        "status": "running", "run": args.run, "stage": "train"
    }
    marker = package / "parallel_scratch_handoff.json"
    with (package / "locks/parallel_scratch_handoff.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        suspended = False
        try:
            controller.suspend()  # Signal this controller PID only, never its process group.
            suspended = True
            for _ in range(20):
                if controller.status() == "T":
                    break
                time.sleep(.05)
            assert controller.status() == "T"
            assert training.is_running() and training.status() != "T"
            record = {
                "status": "watching_active_training",
                "run": args.run, "gpu": args.gpu,
                "controller_pid": controller.pid, "training_pid": training.pid,
                "controller_start_ticks": controller.start_ticks,
                "training_start_ticks": training.start_ticks,
                "plan_sha256": sha256(package / "plan.json"),
                "dispatcher_sha256": sha256(Path(__file__)),
                "training_was_not_suspended": True,
            }
            save_json(marker, record)
            print("HANDOFF_READY", record, flush=True)
            while training.is_running():
                time.sleep(10)
            run = next(r for r in plan["runs"] if r["id"] == args.run)
            final = Path(run["new_checkpoint"])
            completion = read_json(final.parent / "completion.json")
            assert completion["status"] == "trained" and completion["global_step"] == 50000
            assert sha256(final) == completion["checkpoint_sha256"]
            assert completion["plan_sha256"] == record["plan_sha256"]
            # The training child has exited. Retire only its old queue controller.
            controller.terminate()
            controller.resume()
            suspended = False
            controller.wait(timeout=15)
            record["status"] = "evaluating_completed_training"
            save_json(marker, record)
            one_run(package, verify_package(package), args.gpu, args.run)
            record["status"] = "complete"
            save_json(marker, record)
        except BaseException as error:
            if suspended and controller.is_running():
                controller.resume()
            save_json(marker, {
                "status": "handoff_failed", "run": args.run,
                "error": repr(error), "old_controller_resumed_if_alive": suspended,
            })
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("worker", "watch-existing"))
    p.add_argument("--package", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--controller-pid", type=int)
    p.add_argument("--training-pid", type=int)
    args = p.parse_args()
    assert Path.cwd().resolve() == ROOT
    package = Path(args.package).resolve()
    plan = verify_package(package)
    if args.command == "watch-existing":
        assert args.controller_pid and args.training_pid
        def interrupted(signum, frame):
            raise InterruptedError(f"Dispatcher received signal {signum}")
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupted)
        watch_existing(args, package, plan)
    else:
        save_json(package / f"dispatch_gpu{args.gpu}.json", {
            "run": args.run, "gpu": args.gpu,
            "plan_sha256": sha256(package / "plan.json"),
            "dispatcher_sha256": sha256(Path(__file__)),
            "queue_override_only": True,
        })
        one_run(package, plan, args.gpu, args.run)


if __name__ == "__main__":
    main()
