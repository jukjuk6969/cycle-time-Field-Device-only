"""
video_sync.py
=============
Sinkron video ke server dalam satu proses:
  1) resend_late : kirim ulang siklus + video yang belum tercatat di server
                   (dibaca dari summary CSV mulai checkpoint, bukan seluruh riwayat)
  2) sync_sizes  : perbarui ukuran (MB) video terbaru lewat /api/video/update-size

Pakai:
    import video_sync
    video_sync.start_background(get_base_url, get_identity)   # tiap 5 menit
    video_sync.sync_all(base_url, line, st, force=True, say=print)   # manual (tombol)
"""
import csv
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from storage_manager import StorageManager, VIDEOS_DIR, MONTH_NAMES

logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
SIZE_CACHE_PATH = BASE / "config" / "size_synced.json"
CHECKPOINT_PATH = BASE / "config" / "resync_state.json"

MIN_AGE_SEC = 10                 # lewati file yang baru ditulis (masih di-finalize)
NOT_FOUND_RETRY_SEC = 1800       # file yang belum ada barisnya: coba lagi tiap 30 menit
BUFFER_MAX_AGE_SEC = 86400 * 3   # jendela cache ukuran
RESYNC_LOOKBACK_DAYS = 150       # sync pertama bila checkpoint belum ada
RESYNC_MARGIN_SEC = 3600
AUTO_MIN_AGE_SEC = 900           # otomatis: jangan sentuh siklus < 15 menit (biar ApiSender selesai)

_run_lock = threading.Lock()
_sm = StorageManager()           # hanya untuk find_video(); thread-nya tidak dijalankan


# ── state JSON ────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _save_json(path: Path, data: dict):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(f"[VideoSync] gagal simpan {path.name}: {e}")


def load_checkpoint(line, st):
    v = _load_json(CHECKPOINT_PATH).get(f"{line}_{st}")
    try:
        return float(v) if v else None
    except (TypeError, ValueError):
        return None


def save_checkpoint(line, st, ts: float):
    d = _load_json(CHECKPOINT_PATH)
    d[f"{line}_{st}"] = ts
    _save_json(CHECKPOINT_PATH, d)


# ── 1) kirim ulang siklus yang terlambat ──────────────────────────────────────
def resend_late(base_url, line, st, min_age=0.0, say=None) -> dict:
    say = say or (lambda t: None)
    res = {"sent": 0, "known": 0, "no_file": 0, "failed": 0}
    now = time.time()

    saved = load_checkpoint(line, st)
    first_run = saved is None
    since = saved or (now - RESYNC_LOOKBACK_DAYS * 86400)
    since_dt = datetime.fromtimestamp(since)
    since_str = since_dt.strftime("%Y-%m-%d %H:%M:%S")
    first_year = since_dt.year
    if first_run and VIDEOS_DIR.is_dir():       # sync pertama: semua tahun, agar anomali lama ikut
        ys = [int(d.name) for d in VIDEOS_DIR.iterdir() if d.is_dir() and d.name.isdigit()]
        first_year = min(ys, default=since_dt.year)

    rows = []
    for year in range(first_year, datetime.now().year + 1):
        p = VIDEOS_DIR / str(year) / f"{year}_summary.csv"
        if not p.is_file():
            continue
        with open(p, newline="", encoding="utf-8") as fh:
            rd = csv.reader(fh)
            header = next(rd, None)
            if not header or "end_timestamp" not in header:
                continue
            ie = header.index("end_timestamp")
            for row in rd:
                if len(row) <= ie:
                    continue
                old = row[ie] < since_str
                if old and not first_run:
                    continue                                  # lama: lewati tanpa parse
                r = dict(zip(header, row))
                if old and r.get("is_anomaly") not in ("1", "True", "true"):
                    continue                                  # normal lama: sudah lewat retensi
                try:
                    if int(r.get("line") or 0) != line or int(r.get("station_id") or 0) != st:
                        continue
                except ValueError:
                    continue
                fn = (r.get("video_filename") or "").strip()
                if not fn:
                    continue
                try:
                    end_ts = datetime.fromisoformat(r["end_timestamp"]).timestamp()
                    start_ts = datetime.fromisoformat(r["start_timestamp"]).timestamp()
                except (ValueError, KeyError, TypeError):
                    continue
                if now - end_ts < min_age:
                    continue
                path = _sm.find_video("anomaly", fn) or _sm.find_video("normal", fn)
                if path is None:
                    res["no_file"] += 1
                    continue
                rows.append((end_ts, start_ts, r, path))
    if not rows:
        return res

    # nama video yang sudah ada di server
    def iso(t):
        return datetime.fromtimestamp(t, tz=timezone.utc).replace(tzinfo=None).isoformat() + "Z"
    t0 = min(x[0] for x in rows) - 3600
    t1 = max(x[0] for x in rows) + 3600
    known, off = set(), 0
    while True:
        q = (f"{base_url}/api/videos?line={line}&station_id={st}"
             f"&from={iso(t0)}&to={iso(t1)}&limit=1000&offset={off}")
        with urllib.request.urlopen(q, timeout=10) as resp:
            page = json.loads(resp.read())
        known.update(v.get("filename") for v in page)
        if len(page) < 1000:
            break
        off += 1000

    todo = sorted((x for x in rows if x[2]["video_filename"] not in known), key=lambda x: x[0])
    res["known"] = len(rows) - len(todo)
    ok_ts = set()

    for i, (end_ts, start_ts, r, path) in enumerate(todo, 1):
        try:
            size_mb = round(path.stat().st_size / 1024 / 1024, 2)
        except OSError:                       # dihapus storage manager di tengah sync
            res["no_file"] += 1
            ok_ts.add(end_ts)
            continue
        atype = (r.get("anomaly_type") or "OK").upper()
        skipped = [{"id": "", "description": d.strip()}
                   for d in (r.get("skipped_expected") or "").split(";") if d.strip()]
        is_anom = r.get("is_anomaly") in ("1", "True", "true")
        genesis = r.get("is_genesis") in ("1", "True", "true")
        payload = {
            "line": line, "station_id": st,
            "cycle_time_sec": float(r["cycle_time_sec"]),
            "operator_id": r.get("operator_id") or None,
            "start_timestamp": start_ts, "end_timestamp": end_ts,
            "is_complete": True, "anomaly_type": atype,
            "label": "GENESIS" if genesis else "", "is_genesis": genesis,
            "is_anomaly": is_anom, "anomaly_label": r.get("anomaly_label") or "",
            "skipped_expected": skipped, "buffered": True,
            "video": {"filename": path.name, "size_mb": size_mb,
                      "is_anomaly": is_anom, "anomaly_type": atype,
                      "label": "GENESIS" if genesis else "", "is_genesis": genesis,
                      "recorded_ts": end_ts},
        }
        req = urllib.request.Request(
            f"{base_url}/api/cycle", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status in (200, 201):
                    res["sent"] += 1
                    ok_ts.add(end_ts)
        except urllib.error.HTTPError as e:
            res["failed"] += 1
            logger.warning(f"[VideoSync] {path.name}: HTTP {e.code}")
            if e.code == 503:
                break
        except Exception as e:
            res["failed"] += 1
            logger.warning(f"[VideoSync] {path.name}: {e}")
            if res["failed"] >= 5 and res["sent"] == 0:
                break
        if i % 5 == 0:
            say(f"Sync video: kirim ulang {i}/{len(todo)}")

    unresolved = [x[0] for x in todo if x[0] not in ok_ts]
    base = min(unresolved) if unresolved else max(x[0] for x in rows)
    save_checkpoint(line, st, base - RESYNC_MARGIN_SEC)
    return res


# ── 2) sinkron ukuran ─────────────────────────────────────────────────────────
def _candidate_files(force: bool, now: float):
    """Non-force: hanya folder bulan yang bisa memuat file <= jendela cache. Force: semua."""
    if force:
        yield from VIDEOS_DIR.rglob("*.mp4")
        return
    seen = set()
    for t in (now - BUFFER_MAX_AGE_SEC, now):
        d = datetime.fromtimestamp(t)
        if (d.year, d.month) in seen:
            continue
        seen.add((d.year, d.month))
        yield from (VIDEOS_DIR / str(d.year) / MONTH_NAMES[d.month - 1]).glob("*/*.mp4")


def post_size(base_url: str, filename: str, size_mb: float, timeout: float = 5.0):
    """Return (status, detail). status: 'ok' | 'not_found' | 'failed'."""
    data = json.dumps({"filename": filename, "size_mb": size_mb}).encode("utf-8")
    req = urllib.request.Request(f"{base_url.rstrip('/')}/api/video/update-size", data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", "ignore")
            try:
                j = json.loads(body)
                if isinstance(j, dict) and j.get("ok") is False:
                    return "not_found", body[:120]
            except Exception:
                pass
            return "ok", f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return ("not_found" if e.code == 404 else "failed"), f"HTTP {e.code}"
    except Exception as e:
        return "failed", str(e)


def sync_sizes(base_url: str, force: bool = False, say=None) -> dict:
    say = say or (lambda t: None)
    res = {"total": 0, "ok": 0, "skipped": 0, "not_found": 0, "failed": 0}
    full = _load_json(SIZE_CACHE_PATH)
    before = dict(full)
    now = time.time()

    entries = []
    for f in _candidate_files(force, now):
        if f.name.startswith("_tmp_"):
            continue
        try:
            st = f.stat()
        except OSError:
            continue
        entries.append((st.st_mtime, st.st_size, f))
    entries.sort(key=lambda e: e[0])
    res["total"] = len(entries)

    alive = {f.name for mtime, _, f in entries if now - mtime <= BUFFER_MAX_AGE_SEC}
    full = {k: v for k, v in full.items() if (k[3:] if k.startswith("nf:") else k) in alive}

    for i, (mtime, size, f) in enumerate(entries, 1):
        age = now - mtime
        if age < MIN_AGE_SEC or (age > BUFFER_MAX_AGE_SEC and not force):
            res["skipped"] += 1
            continue
        size_mb = round(size / 1024 / 1024, 2)
        nf_key = "nf:" + f.name
        if not force:
            if full.get(f.name) == size_mb:
                res["skipped"] += 1
                continue
            nf_ts = full.get(nf_key)
            if nf_ts and now - nf_ts < NOT_FOUND_RETRY_SEC:
                res["skipped"] += 1
                continue

        status, detail = post_size(base_url, f.name, size_mb)
        in_window = age <= BUFFER_MAX_AGE_SEC
        if status == "ok":
            if in_window:
                full[f.name] = size_mb
            full.pop(nf_key, None)
            res["ok"] += 1
        elif status == "not_found":
            res["not_found"] += 1
            if in_window:
                full[nf_key] = now
        else:
            res["failed"] += 1
            logger.warning(f"[VideoSync] ukuran {f.name}: gagal ({detail})")
            if res["failed"] >= 5 and res["ok"] == 0:
                break
        if i % 20 == 0:
            say(f"Sync video: ukuran {i}/{len(entries)}")

    if full != before:
        _save_json(SIZE_CACHE_PATH, full)
    return res


# ── gabungan ──────────────────────────────────────────────────────────────────
def sync_all(base_url: str, line, st, force=False, min_age=0.0, say=None):
    """Kirim ulang yang terlambat dulu, lalu ukuran. Return dict gabungan, atau None bila sedang jalan."""
    if not _run_lock.acquire(blocking=False):
        return None
    try:
        out = {"resend": None, "sizes": None}
        if line and st:
            try:
                out["resend"] = resend_late(base_url, int(line), int(st), min_age, say)
            except Exception as e:
                logger.error(f"[VideoSync] kirim ulang gagal: {e}", exc_info=True)
                out["resend"] = {"error": str(e)}
        out["sizes"] = sync_sizes(base_url, force=force, say=say)
        logger.info(f"[VideoSync] selesai: {out}")
        return out
    finally:
        _run_lock.release()


_bg_started = False


def start_background(get_base_url, get_identity, interval: float = 300.0):
    """get_base_url() -> 'http://host:port' ; get_identity() -> (line, station_id)."""
    global _bg_started
    if _bg_started:
        return
    _bg_started = True

    def loop():
        time.sleep(15)
        while True:
            try:
                line, st = get_identity()
                sync_all(get_base_url(), line, st, min_age=AUTO_MIN_AGE_SEC)
            except Exception as e:
                logger.warning(f"[VideoSync] error: {e}")
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True, name="video-sync").start()