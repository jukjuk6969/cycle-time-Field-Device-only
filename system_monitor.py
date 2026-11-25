# system_monitor.py
import threading
import time

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    pynvml = None
    _NVML = False


class SystemMonitor:
    """Sampling CPU / RAM / GPU di thread sendiri. UI cukup memanggil snapshot()."""

    def __init__(self, interval: float = 1.0):
        self._interval = interval
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._snap = {}
        self._proc = None
        self._ncpu = 1
        self._gpu = None
        if psutil:
            self._proc = psutil.Process()
            self._ncpu = psutil.cpu_count() or 1
            psutil.cpu_percent(interval=None)      # prime: panggilan pertama selalu 0
            self._proc.cpu_percent(interval=None)
        if _NVML:
            try:
                self._gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._gpu = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sys-monitor")
        self._thread.start()

    def stop(self):
        self._running = False

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._snap)

    def _loop(self):
        while self._running:
            snap = self._sample()
            with self._lock:
                self._snap = snap
            time.sleep(self._interval)

    def _sample(self) -> dict:
        s = {"cpu": None, "ram_pct": None, "ram_used_gb": None, "ram_total_gb": None,
             "proc_cpu": None, "proc_ram_mb": None, "gpu": None}
        if psutil:
            try:
                vm = psutil.virtual_memory()
                s["cpu"] = psutil.cpu_percent(interval=None)
                s["ram_pct"] = vm.percent
                s["ram_used_gb"] = vm.used / 1024 ** 3
                s["ram_total_gb"] = vm.total / 1024 ** 3
                s["proc_cpu"] = min(100.0, self._proc.cpu_percent(interval=None) / self._ncpu)
                s["proc_ram_mb"] = self._proc.memory_info().rss / 1024 ** 2
            except Exception:
                pass
        if self._gpu is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu)
                temp = pynvml.nvmlDeviceGetTemperature(self._gpu, pynvml.NVML_TEMPERATURE_GPU)
                name = pynvml.nvmlDeviceGetName(self._gpu)
                if isinstance(name, bytes):
                    name = name.decode()
                s["gpu"] = {"name": name, "util": util.gpu,
                            "mem_used_mb": mem.used / 1024 ** 2,
                            "mem_total_mb": mem.total / 1024 ** 2,
                            "temp": temp}
            except Exception:
                s["gpu"] = None
        return s


if __name__ == "__main__":
    m = SystemMonitor()
    m.start()
    for _ in range(5):
        time.sleep(1.2)
        print(m.snapshot())
    m.stop()