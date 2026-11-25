import ctypes
import sys
from ctypes import wintypes

_EXEC_SPEED = 0x1          # PROCESS_POWER_THROTTLING_EXECUTION_SPEED
_IGNORE_TIMER = 0x4        # PROCESS_POWER_THROTTLING_IGNORE_TIMER_RESOLUTION


class _PPT(ctypes.Structure):
    _fields_ = [("Version", wintypes.ULONG),
                ("ControlMask", wintypes.ULONG),
                ("StateMask", wintypes.ULONG)]


def disable_throttling(handle=None) -> bool:
    """handle=None -> proses ini. StateMask=0 + ControlMask set = throttling dimatikan."""
    if sys.platform != "win32":
        return False
    try:
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = wintypes.HANDLE
        h = wintypes.HANDLE(int(handle)) if handle else k.GetCurrentProcess()
        s = _PPT(1, _EXEC_SPEED | _IGNORE_TIMER, 0)
        return bool(k.SetProcessInformation(h, 4, ctypes.byref(s), ctypes.sizeof(s)))
    except Exception:
        return False


def timer_1ms():
    if sys.platform == "win32":
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass