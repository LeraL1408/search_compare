from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
import uuid

from .common import ROOT


class Worker:
    def __init__(self, config, fixture=False):
        self.config, self.fixture = config, fixture

    def run(self, request, callback=None):
        timeout = self.config["task_timeout_seconds"]
        worker_path = ROOT / "searchlab" / "worker.py"
        name = "searchlab-" + uuid.uuid4().hex
        if self.fixture:
            command = [sys.executable, "-I", "-S", "-u", str(worker_path)]
        else:
            command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
                       "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "32",
                       "--memory", "256m", "--cpus", "1", "--user", "65534:65534",
                       "-i", "--mount", f"type=bind,source={worker_path},target=/worker.py,readonly",
                       self.config["docker_image"], "python", "-I", "-S", "-u", "/worker.py"]
        # No environment credentials, project mounts, benchmark files or answers.
        env = {k: os.environ[k] for k in ("PATH", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "HOME") if k in os.environ} if not self.fixture else {}
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, env=env, start_new_session=True)
        poller = selectors.DefaultSelector()
        poller.register(process.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout
        try:
            process.stdin.write((json.dumps(request) + "\n").encode())
            process.stdin.flush()
            buffer = b""
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Generated agent exceeded task timeout")
                if not poller.select(min(remaining, 0.5)):
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("Worker exited without a result; check Docker availability")
                buffer += chunk
                if len(buffer) > 1_000_000:
                    raise RuntimeError("Worker output exceeded limit")
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    msg = json.loads(line)
                    if msg["op"] == "result":
                        return msg["value"]
                    if msg["op"] == "error":
                        raise RuntimeError(msg["error"])
                    value = callback(msg) if callback else None
                    if callback is None:
                        raise ValueError("Unexpected worker request")
                    if time.monotonic() > deadline:
                        raise TimeoutError("Task deadline reached during model call")
                    process.stdin.write((json.dumps({"value": value}) + "\n").encode())
                    process.stdin.flush()
        finally:
            poller.close()
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdin.close()
            process.stdout.close()
            if not self.fixture:
                subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=15, check=False, env=env)
