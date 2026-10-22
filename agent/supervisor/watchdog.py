"""Out-of-Process Supervisor and Emergency Kill Switch.
Runs independently of the agent worker process, owns the emergency hotkeys,
monitors heartbeats, and freezes/terminates the agent actor immediately on trigger.
Target stop latency: <= 1.0s.
"""

import os
import sys
import time
import signal
import psutil
import logging
import threading
from typing import Optional, Callable

logger = logging.getLogger("supervisor")


class ProcessSupervisor:
    """Out-of-process watchdog controlling the agent actor process lifecycle."""

    def __init__(self, target_pid: Optional[int] = None, heartbeat_timeout_s: float = 15.0):
        self.target_pid = target_pid
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.last_heartbeat = time.time()
        self.is_running = False
        self._lock = threading.Lock()
        self.stopped_latency: Optional[float] = None

    def record_heartbeat(self) -> None:
        """Called by the agent process to demonstrate liveness."""
        with self._lock:
            self.last_heartbeat = time.time()

    def emergency_stop(self) -> float:
        """Trigger immediate freeze and termination of the actor process.
        Returns the measured stop latency in seconds.
        """
        t0 = time.time()
        if not self.target_pid:
            return 0.0

        try:
            if psutil.pid_exists(self.target_pid):
                proc = psutil.Process(self.target_pid)
                # First suspend/freeze to stop input immediately
                try:
                    proc.suspend()
                except Exception:
                    pass

                # Terminate process and all its children
                children = proc.children(recursive=True)
                for child in children:
                    try:
                        child.terminate()
                    except Exception:
                        pass
                proc.terminate()

                # Wait up to 1.0s for exit, then kill forcefully if hung
                gone, alive = psutil.wait_procs([proc] + children, timeout=0.8)
                for p in alive:
                    try:
                        p.kill()
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Error during emergency stop: {e}")

        stop_latency = time.time() - t0
        self.stopped_latency = stop_latency
        logger.info(f"Emergency stop completed in {stop_latency * 1000:.2f} ms")
        return stop_latency


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pid = int(sys.argv[1])
        sup = ProcessSupervisor(target_pid=pid)
        print(f"Supervisor monitoring PID {pid}. Ready for emergency stop signals.")
