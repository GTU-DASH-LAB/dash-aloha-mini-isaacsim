"""GPU0 load next to every cost figure (g1 tools).

A wall-clock cost measured on a shared card says as much about the other jobs as about ours: the
same hospital nav-camera loop measured 23.4 ms/update with GPU0 at 96% and 46.6 ms at 97% with one
more Kit process on the card (g1/check, 2026-10-05). So every cost these tools report carries, for
the same time window: GPU0 utilisation, how many OTHER Isaac Sim (Kit) processes held GPU0, the
card's used memory, this process's own memory on GPU0 and GPU1, and the 1-minute load average.

`GpuPoll` samples once a second in a daemon thread (nvidia-smi, never from a physics callback).
"""

from __future__ import annotations

import os
import subprocess
import threading
import time


def _kit_like(name: str) -> bool:
    n = name.lower()
    return "isaacsim" in n or "/kit/" in n or n.endswith("/kit")


class GpuPoll(threading.Thread):
    def __init__(self, period: float = 1.0, t0: float | None = None):
        super().__init__(daemon=True)
        self.period = period
        self.t0 = time.time() if t0 is None else t0
        self.samples: list[dict] = []
        self.stop_flag = False
        self.pid = os.getpid()

    def now(self) -> float:
        return time.time() - self.t0

    def read(self) -> dict:
        try:
            apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
                                   "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            cards = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
            idx, util = {}, {}
            for line in cards.strip().splitlines():
                i, u, ut, mu = [c.strip() for c in line.split(",")]
                idx[u] = int(i)
                util[int(i)] = (int(ut), int(mu))
            mine, n0, kit_others = {}, 0, 0
            for line in apps.strip().splitlines():
                parts = [c.strip() for c in line.split(",")]
                if len(parts) < 4:
                    continue
                pid, name, uuid, mib = int(parts[0]), parts[1], parts[2], parts[3]
                gpu = idx.get(uuid)
                if pid == self.pid:
                    mine[gpu] = int(mib) if mib.isdigit() else 0
                elif gpu == 0:
                    n0 += 1
                    kit_others += int(_kit_like(name))
            return {"t": round(self.now(), 1), "mine_gpu0": mine.get(0, 0), "mine_gpu1": mine.get(1, 0),
                    "gpu0_util": util[0][0], "gpu0_used": util[0][1], "other_procs_gpu0": n0,
                    "other_kit_gpu0": kit_others, "load1": round(os.getloadavg()[0], 1)}
        except Exception as e:  # noqa: BLE001
            return {"t": round(self.now(), 1), "error": repr(e)}

    def run(self) -> None:
        while not self.stop_flag:
            self.samples.append(self.read())
            time.sleep(self.period)

    def window(self, t0: float = 0.0, t1: float = 1e12) -> dict:
        s = [x for x in self.samples if "gpu0_util" in x and t0 <= x["t"] <= t1]
        if not s:
            return {"n": 0}
        n = len(s)
        return {"n": n, "gpu0_util_mean": round(sum(x["gpu0_util"] for x in s) / n, 1),
                "gpu0_util_min_max": [min(x["gpu0_util"] for x in s), max(x["gpu0_util"] for x in s)],
                "other_kit_gpu0_min_max": [min(x["other_kit_gpu0"] for x in s), max(x["other_kit_gpu0"] for x in s)],
                "other_procs_gpu0_min_max": [min(x["other_procs_gpu0"] for x in s), max(x["other_procs_gpu0"] for x in s)],
                "gpu0_card_used_mib_max": max(x["gpu0_used"] for x in s),
                "mine_gpu0_max_mib": max(x["mine_gpu0"] for x in s), "mine_gpu1_max_mib": max(x["mine_gpu1"] for x in s),
                "load1_mean": round(sum(x["load1"] for x in s) / n, 1)}
