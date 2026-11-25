"""
storage_manager.py
==================
Manajemen storage video — Quarterly Retention Strategy.

Struktur folder:

    videos/
    ├── _tmp/                        (rekaman yang sedang berjalan)
    └── <tahun>/
        ├── <tahun>_summary.csv      (tidak pernah disentuh modul ini)
        └── <bulan>/
            ├── normal/              (retensi quarterly)
            └── anomali/             (permanen, hanya hapus manual)

Aturan:
  - Video normal : record terus berjalan setiap bulan.
  - Setiap awal quarter (April, July, October, January), 3 bulan
    sebelumnya masuk periode BACKUP selama 1 bulan.
  - Setelah 1 bulan backup selesai, pada bulan berikutnya video normal
    dari quarter tersebut mulai dihapus otomatis.
  - Contoh: Jan-Mar -> backup April -> delete mulai May.
  - Video anomali: TIDAK pernah dihapus otomatis. Hapus manual lewat
                   delete_anomaly_video() (path relatif dari VIDEOS_DIR) atau
                   delete_anomaly_by_name() (nama file saja, dipakai API FD).
  - File "_tmp_*" : rekaman yang sedang berjalan ada di TMP_DIR. Sisa rekaman
                    yang terputus (mtime > ORPHAN_TMP_SECONDS) dihapus.
  - File non-mp4 (mis. 2026_summary.csv) tidak pernah disentuh.
  - Video lama (sebelum migrasi) di videos/normal dan videos/anomaly tetap bisa
    diputar dan dihapus lewat find_video(), tetapi tidak ikut retensi otomatis.
    Pindahkan dengan migrate_videos.py.

Konvensi nama file:
    [l<line>_]st<station>_<timestamp>_<ct>[_GENESIS][_LONG|_SHORT_ANOMALY].mp4
"""

import re
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
import csv
import os
import json
import calendar

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
VIDEOS_DIR = BASE_DIR / "videos"
TMP_DIR = VIDEOS_DIR / "_tmp"        # rekaman yang sedang berjalan

NORMAL_DIRNAME = "normal"
ANOMALY_DIRNAME = "anomali"
LEGACY_ANOMALY_DIRNAME = "anomaly"   # nama folder lama (datar) sebelum migrasi

# Nama bulan tetap Inggris dan tidak bergantung locale sistem.
# Ubah ke "01_January", dst. kalau ingin urutan folder kronologis.
MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

RECORD_MONTHS = 3          # satu periode record terdiri dari 3 bulan
BACKUP_WINDOW_MONTHS = 1    # 1 bulan penuh untuk backup sebelum delete
CLEANUP_INTERVAL = 3600     # cek setiap 1 jam
ORPHAN_TMP_SECONDS = 7200   # _tmp_ yang tidak berubah > 2 jam dianggap sisa crash

TMP_PREFIX = "_tmp_"
_GB = 1024 ** 3
_MB = 1024 ** 2

_NAME_RE = re.compile(r"^(?:l(?P<line>\d+)_)?st(?P<station>\d+)_")
_TS_RE = re.compile(r"_(\d{4})(\d{2})\d{2}_\d{6}")     # tahun & bulan dari nama file

# ── Summary CSV ───────────────────────────────────────────────────────────────
SUMMARY_FILENAME_FMT = "{year}_summary.csv"   # ganti ke "{year}-summary.csv" jika perlu
SUMMARY_COLUMNS = [
    "line",
    "station_id",
    "operator_id",
    "cycle_time_sec",
    "start_timestamp",
    "end_timestamp",
    "is_complete",
    "is_anomaly",
    "created_at",
    "cycle_count",
    "video_filename",
    "anomaly_type",
    "is_genesis",
    "anomaly_label",
    "skipped_expected",
]
# ── Konfigurasi retensi (diatur dari main.py) ────────────────────────────────
CONFIG_PATH = BASE_DIR / "config" / "storage_config.json"
RETENTION_MODES = ("quarter", "month", "day")
DEFAULT_CONFIG = {
    "auto_delete": True,   # tombol hapus otomatis ON/OFF
    "mode": "quarter",     # quarter (batch + grace) | month | day
    "grace_months": 1,     # mode quarter: bulan backup setelah quarter selesai
    "keep_months": 3,      # mode month: hapus bila umur file > n bulan
    "keep_days": 90,       # mode day: hapus bila umur file > n hari
}
_cfg_lock = threading.Lock()
_cfg = None


def _validate_config(c: dict) -> dict:
    out = dict(DEFAULT_CONFIG)
    out["auto_delete"] = bool(c.get("auto_delete", out["auto_delete"]))
    if c.get("mode") in RETENTION_MODES:
        out["mode"] = c["mode"]
    for key, lo in (("grace_months", 0), ("keep_months", 1), ("keep_days", 1)):
        try:
            out[key] = max(lo, int(c.get(key, out[key])))
        except (TypeError, ValueError):
            pass
    return out


def get_config() -> dict:
    """Konfigurasi retensi saat ini (salinan). Dimuat dari file sekali."""
    global _cfg
    with _cfg_lock:
        if _cfg is None:
            try:
                raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception:
                raw = {}
            _cfg = _validate_config(raw if isinstance(raw, dict) else {})
        return dict(_cfg)


def set_config(**changes) -> dict:
    """Ubah sebagian konfigurasi, simpan ke file, kembalikan konfigurasi baru.
    Contoh: set_config(auto_delete=False) / set_config(mode="day", keep_days=30)"""
    global _cfg
    merged = {**get_config(), **changes}
    new = _validate_config(merged)
    with _cfg_lock:
        _cfg = new
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(new, indent=2), encoding="utf-8")
        os.replace(tmp, CONFIG_PATH)
    except Exception as e:
        logger.warning(f"[StorageManager] gagal simpan config: {e}")
    logger.info(f"[StorageManager] Config: {new}")
    return dict(new)


def _months_back(dt: datetime, n: int) -> datetime:
    idx = dt.year * 12 + dt.month - 1 - n
    y, m = divmod(idx, 12)
    day = min(dt.day, calendar.monthrange(y, m + 1)[1])
    return dt.replace(year=y, month=m + 1, day=day)
class StorageManager:
    _summary_lock = threading.Lock()
    """
    Background thread cleanup video.
    Hanya menghapus otomatis file di folder <tahun>/<bulan>/normal/.
    Folder anomali dan file CSV tidak pernah disentuh oleh cleanup.
    """

    def __init__(self):
        self._stop_event = threading.Event()
        self._cleanup_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="storage-manager"
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        logger.info(
            f"[StorageManager] Berjalan | "
            f"Record: {RECORD_MONTHS} bulan | "
            f"Backup window: {BACKUP_WINDOW_MONTHS} bulan | "
            f"Delete: mulai bulan setelah backup | "
            f"Anomali: permanen (hapus manual)"
        )

    def stop(self):
        self._stop_event.set()

    # ── Helper path untuk modul perekam ───────────────────────────────────────

    @staticmethod
    def get_save_dir(is_anomaly: bool, when: datetime = None) -> Path:
        """
        Kembalikan folder tujuan simpan video dan buat jika belum ada.
        Contoh: videos/2026/April/normal  atau  videos/2026/April/anomali
        `when` harus waktu yang sama dengan timestamp di nama file, karena
        find_video() membaca tahun/bulan dari nama file.
        """
        when = when or datetime.now()
        kind = ANOMALY_DIRNAME if is_anomaly else NORMAL_DIRNAME
        folder = VIDEOS_DIR / str(when.year) / MONTH_NAMES[when.month - 1] / kind
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    @staticmethod
    def to_relative(path: Path) -> str:
        """Path relatif dari VIDEOS_DIR (pakai '/'), cocok untuk disimpan di DB/CSV."""
        return Path(path).resolve().relative_to(VIDEOS_DIR.resolve()).as_posix()

    # ── Summary CSV tahunan ───────────────────────────────────────────────────

    @staticmethod
    def get_summary_path(year: int) -> Path:
        """
        Path summary CSV tahunan, sejajar dengan folder bulan.
        Contoh: videos/2026/2026_summary.csv
        """
        folder = VIDEOS_DIR / str(year)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / SUMMARY_FILENAME_FMT.format(year=year)

    @staticmethod
    def _to_datetime(value):
        """Terima datetime / epoch (detik) / string ISO / None -> datetime atau None."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return datetime.fromtimestamp(value)          # epoch dari FD
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _csv_value(value):
        """Normalisasi nilai untuk CSV (datetime -> ISO, None -> kosong)."""
        if value is None:
            return ""
        if isinstance(value, datetime):
            return value.isoformat(sep=" ", timespec="seconds")
        return value
    _header_checked: set = set()
    @staticmethod
    def _migrate_summary_header(path: Path):
        """Samakan header file lama (lebih pendek / kolom ganda) dengan SUMMARY_COLUMNS.
        Nilai dipetakan berdasarkan NAMA kolom, jadi aman untuk header lama berkolom ganda."""
        with open(path, "r", newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if not rows or rows[0] == SUMMARY_COLUMNS:
            return
        old = rows[0]
        if not set(old) <= set(SUMMARY_COLUMNS):
            logger.warning(f"[StorageManager] Header {path.name} tidak dikenal, tidak diubah. "
                           f"Ganti nama file itu agar summary baru dibuat.")
            return
        idx = {}
        for i, c in enumerate(old):
            idx.setdefault(c, i)                 # kolom ganda: pakai kemunculan pertama
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(SUMMARY_COLUMNS)
            for r in rows[1:]:
                w.writerow([r[idx[c]] if c in idx and idx[c] < len(r) else ""
                            for c in SUMMARY_COLUMNS])
        os.replace(tmp, path)
        logger.info(f"[StorageManager] Header {path.name} dimigrasi ke {len(SUMMARY_COLUMNS)} kolom")
    def append_summary(self, record: dict, when: datetime = None):
        """
        Tambahkan satu baris siklus ke summary CSV tahunan.
        File dan header dibuat otomatis jika belum ada.

        record: dict dengan key sesuai SUMMARY_COLUMNS, yaitu
                line, station_id, operator_id, cycle_time_sec, start_timestamp,
                end_timestamp, is_complete, is_anomaly, created_at, cycle_count,
                video_filename, anomaly_type, is_genesis.
                Timestamp boleh datetime, epoch (detik), atau string ISO.
                Key yang hilang diisi kosong, key tambahan diabaikan.
                created_at yang kosong diisi end_timestamp.
        when:   menentukan tahun file. Default: end_timestamp, lalu
                start_timestamp, lalu waktu sekarang.

        Catatan: file summary lama dengan header berbeda harus diganti namanya.
        Return Path file summary, atau None jika gagal.
        """
        try:
            when = (
                when
                or self._to_datetime(record.get("end_timestamp"))
                or self._to_datetime(record.get("start_timestamp"))
                or datetime.now()
            )

            row = {col: self._csv_value(record.get(col)) for col in SUMMARY_COLUMNS}
            for col in ("start_timestamp", "end_timestamp", "created_at"):
                dt = self._to_datetime(record.get(col))
                if dt:
                    row[col] = self._csv_value(dt)
            if not row["created_at"]:
                row["created_at"] = row["end_timestamp"] or self._csv_value(datetime.now())

            path = self.get_summary_path(when.year)

            with self._summary_lock:
                if path.exists() and path.stat().st_size > 0 and path not in self._header_checked:
                    self._migrate_summary_header(path)
                    self._header_checked.add(path)
                need_header = not path.exists() or path.stat().st_size == 0
                with open(path, "a", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(
                        fh, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore"
                    )
                    if need_header:
                        writer.writeheader()
                    writer.writerow(row)

            return path
        except Exception as e:
            logger.error(f"[StorageManager] Gagal tulis summary: {e}")
            return None

    @staticmethod
    def normal_retention_expired(ts: float, now: datetime = None) -> bool:
        """True bila video normal bertimestamp `ts` sudah masuk jendela delete quarterly."""
        now = now or datetime.now()
        d = datetime.fromtimestamp(ts)
        q_end = ((d.month - 1) // RECORD_MONTHS + 1) * RECORD_MONTHS
        delete_idx = d.year * 12 + (q_end - 1) + 1 + BACKUP_WINDOW_MONTHS
        return now.year * 12 + (now.month - 1) >= delete_idx

    # ── Scan file ─────────────────────────────────────────────────────────────
    @staticmethod
    def _scan(kind: str) -> list:
        """
        Scan semua mp4 di <tahun>/<bulan>/<kind>/ dan kembalikan list
        (path, mtime, size). stat() hanya dipanggil sekali per file.
        File _tmp_ dilewati.
        """
        result = []
        for f in VIDEOS_DIR.glob(f"*/*/{kind}/*.mp4"):
            if f.name.startswith(TMP_PREFIX):
                continue
            try:
                st = f.stat()
            except OSError:
                continue  # file hilang di tengah scan
            result.append((f, st.st_mtime, st.st_size))
        return result
    # ── Quarterly retention ───────────────────────────────────────────────────
    @staticmethod
    def _delete_period_for(now: datetime = None):
        """
        Tentukan quarter yang sudah melewati 1 bulan backup dan boleh dihapus.

        Pola:
            Jan-Mar  -> Backup April   -> Delete mulai May
            Apr-Jun  -> Backup July    -> Delete mulai August
            Jul-Sep  -> Backup October -> Delete mulai November
            Oct-Dec  -> Backup January -> Delete mulai February

        Return:
            (year, start_month, end_month) jika saat ini bulan delete,
            atau None jika saat ini masih bulan record/backup.
        """
        now = now or datetime.now()

        # Bulan 1, 4, 7, 10 adalah bulan BACKUP. Belum boleh delete.
        # Delete hanya mulai bulan ke-2 setelah quarter selesai:
        # Feb/Mar, May/Jun, Aug/Sep, Nov/Dec.
        if now.month % 3 == 1:
            return None

        # Ambil quarter SEBELUM quarter berjalan.
        # Contoh:
        #   May/June -> Q1 (Jan-Mar)
        #   Aug/Sep  -> Q2 (Apr-Jun)
        #   Nov/Dec  -> Q3 (Jul-Sep)
        #   Feb/Mar  -> Q4 tahun sebelumnya (Oct-Dec)
        current_quarter_start = (
            ((now.month - 1) // RECORD_MONTHS) * RECORD_MONTHS + 1
        )
        quarter_start = current_quarter_start - RECORD_MONTHS
        target_year = now.year

        if quarter_start <= 0:
            quarter_start += 12
            target_year -= 1

        quarter_end = quarter_start + RECORD_MONTHS - 1
        return target_year, quarter_start, quarter_end

    @staticmethod
    def _period_months(year: int, start_month: int, end_month: int) -> set:
        """Buat set (year, month) untuk satu quarter kalender."""
        return {(year, month) for month in range(start_month, end_month + 1)}
    @staticmethod
    def _is_expired(f: Path, mtime: float, now: datetime, cfg: dict) -> bool:
        """Apakah video normal ini sudah waktunya dihapus menurut konfigurasi."""
        mode = cfg["mode"]
        if mode == "day":
            return mtime < (now - timedelta(days=cfg["keep_days"])).timestamp()
        if mode == "month":
            return mtime < _months_back(now, cfg["keep_months"]).timestamp()
        # quarter: videos/<tahun>/<bulan>/normal/file.mp4
        try:
            fy = int(f.parents[2].name)
            fm = MONTH_NAMES.index(f.parents[1].name) + 1
        except (ValueError, IndexError):
            return False
        q_end = ((fm - 1) // RECORD_MONTHS + 1) * RECORD_MONTHS
        return now.year * 12 + now.month - 1 >= fy * 12 + q_end + cfg["grace_months"]
    def run_cleanup(self, now: datetime = None, force: bool = False) -> dict:
        """
        Hapus video normal yang sudah kedaluwarsa menurut get_config().
        auto_delete=False -> tidak ada penghapusan (sisa _tmp_ tetap dibersihkan),
        kecuali force=True (dipakai tombol 'jalankan sekarang').
        Anomali & CSV tidak pernah disentuh.
        """
        with self._cleanup_lock:
            now = now or datetime.now()
            cfg = get_config()
            deleted_count = 0
            deleted_bytes = 0

            if cfg["auto_delete"] or force:
                for f, mtime, size in self._scan(NORMAL_DIRNAME):
                    if not self._is_expired(f, mtime, now, cfg):
                        continue
                    if self._safe_unlink(f, size, f"retensi {cfg['mode']}"):
                        deleted_count += 1
                        deleted_bytes += size

            orphan_count = self._cleanup_orphan_tmp()
            usage = self.get_usage()
            report = {
                "auto_delete": cfg["auto_delete"],
                "mode": cfg["mode"],
                "deleted_count": deleted_count,
                "deleted_mb": round(deleted_bytes / _MB, 2),
                "orphan_tmp_deleted": orphan_count,
                "normal_usage_gb": usage["normal_gb"],
                "anomaly_count": usage["anomaly_count"],
                "anomaly_usage_gb": usage["anomaly_gb"],
            }
            if deleted_count:
                logger.info(f"[StorageManager] Cleanup {cfg['mode']}: {deleted_count} file "
                            f"dihapus ({deleted_bytes / _MB:.1f}MB)")
            return report
    def _cleanup_orphan_tmp(self) -> int:
        """Hapus _tmp_*.mp4 yang tidak berubah > ORPHAN_TMP_SECONDS (sisa rekaman crash)."""
        deleted = 0
        now = time.time()
        candidates = list(TMP_DIR.glob(f"{TMP_PREFIX}*.mp4"))
        candidates += list(VIDEOS_DIR.glob(f"*/*/{NORMAL_DIRNAME}/{TMP_PREFIX}*.mp4"))
        for f in candidates:
            try:
                if now - f.stat().st_mtime > ORPHAN_TMP_SECONDS:
                    f.unlink()
                    deleted += 1
                    logger.warning(f"[StorageManager] Hapus sisa rekaman: {f.name}")
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.error(f"[StorageManager] Gagal hapus {f.name}: {e}")
        return deleted

    @staticmethod
    def _safe_unlink(path: Path, size: int, reason: str) -> bool:
        try:
            path.unlink()
            logger.debug(
                f"[StorageManager] Hapus normal ({reason}): "
                f"{path.name} ({size / _MB:.1f}MB)"
            )
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.error(f"[StorageManager] Gagal hapus {path.name}: {e}")
            return False

    # ── Cari file lewat nama (dipakai API FD) ─────────────────────────────────

    def find_video(self, vtype: str, filename: str):
        """
        Cari file video lewat nama. vtype: 'normal' | 'anomaly' (dari URL server).
        Urutan: folder tahun/bulan dari timestamp di nama file, lalu semua bulan,
        lalu folder lama (datar) untuk video sebelum migrasi.
        Return Path atau None.
        """
        if (not filename.endswith(".mp4") or "/" in filename
                or "\\" in filename or ".." in filename):
            return None
        dirname = ANOMALY_DIRNAME if vtype == "anomaly" else NORMAL_DIRNAME

        m = _TS_RE.search(filename)
        if m and 1 <= int(m.group(2)) <= 12:
            p = (VIDEOS_DIR / m.group(1) / MONTH_NAMES[int(m.group(2)) - 1]
                 / dirname / filename)
            if p.is_file():
                return p

        for p in VIDEOS_DIR.glob(f"*/*/{dirname}/{filename}"):   # nama tak berpola
            if p.is_file():
                return p

        legacy_dir = LEGACY_ANOMALY_DIRNAME if vtype == "anomaly" else NORMAL_DIRNAME
        legacy = VIDEOS_DIR / legacy_dir / filename               # sebelum migrasi
        return legacy if legacy.is_file() else None

    # ── Hapus manual anomali ──────────────────────────────────────────────────

    def delete_anomaly_by_name(self, filename: str):
        """
        Hapus video anomali lewat nama file.
        Return True = terhapus, False = gagal/ditolak, None = tidak ditemukan.
        """
        p = self.find_video("anomaly", filename)
        if p is None:
            return None
        if p.parent.name not in (ANOMALY_DIRNAME, LEGACY_ANOMALY_DIRNAME):
            return False
        try:
            size_mb = p.stat().st_size / _MB
            p.unlink()
            logger.info(f"[StorageManager] Manual delete anomali: {p.name} ({size_mb:.1f}MB)")
            return True
        except Exception as e:
            logger.error(f"[StorageManager] Gagal hapus anomali {p.name}: {e}")
            return False

    def delete_anomaly_video(self, relative_path: str) -> bool:
        """
        Hapus satu video anomali secara manual.

        relative_path: path relatif dari VIDEOS_DIR,
                       contoh "2026/April/anomali/l1_st1_xxx_ANOMALY.mp4".
        Return True jika berhasil, False jika tidak ditemukan, ditolak, atau gagal.
        """
        try:
            base = VIDEOS_DIR.resolve()
            target = (VIDEOS_DIR / relative_path).resolve()

            # Cegah path traversal & pastikan benar-benar file anomali
            if base not in target.parents:
                logger.warning(f"[StorageManager] Ditolak (di luar videos/): {relative_path}")
                return False
            if target.parent.name != ANOMALY_DIRNAME or target.suffix.lower() != ".mp4":
                logger.warning(f"[StorageManager] Ditolak (bukan video anomali): {relative_path}")
                return False
            if not target.is_file():
                logger.warning(f"[StorageManager] File tidak ditemukan: {relative_path}")
                return False

            size_mb = target.stat().st_size / _MB
            target.unlink()
            logger.info(
                f"[StorageManager] Manual delete anomali: {relative_path} ({size_mb:.1f}MB)"
            )
            return True
        except Exception as e:
            logger.error(f"[StorageManager] Gagal hapus anomali {relative_path}: {e}")
            return False
    # ── Hapus manual berdasarkan summary (per bulan) ──────────────────────────

    @staticmethod
    def _iter_summary(year: int):
        """Baris summary CSV satu tahun (dict). Kosong bila file belum ada."""
        p = VIDEOS_DIR / str(year) / SUMMARY_FILENAME_FMT.format(year=year)
        if not p.is_file():
            return
        with open(p, newline="", encoding="utf-8") as fh:
            yield from csv.DictReader(fh)

    def list_summary_months(self) -> list:
        """
        Bulan yang punya baris video di summary, terbaru dulu. Untuk pilihan bulan di UI.
        Return: [{"year": 2026, "month": 4, "month_name": "April", "rows": 1234}, ...]
        """
        counts = {}
        for d in VIDEOS_DIR.glob("[0-9][0-9][0-9][0-9]"):
            if not d.is_dir():
                continue
            y = int(d.name)
            for r in self._iter_summary(y):
                if not (r.get("video_filename") or "").strip():
                    continue
                try:
                    m = int((r.get("end_timestamp") or "")[5:7])
                except ValueError:
                    continue
                if 1 <= m <= 12:
                    counts[(y, m)] = counts.get((y, m), 0) + 1
        return [{"year": y, "month": m, "month_name": MONTH_NAMES[m - 1], "rows": n}
                for (y, m), n in sorted(counts.items(), reverse=True)]

    def list_month_videos(self, year: int, month: int, kind: str = "all",
                          line: int = None, station: int = None) -> list:
        """
        Daftar video dalam satu bulan, dipetakan dari summary CSV (bulan = end_timestamp).
        Hanya baris yang filenya MASIH ADA di disk (yang sudah terhapus tidak muncul).
        kind: "all" | "normal" | "anomaly".
        'path' relatif dari VIDEOS_DIR, langsung bisa dipakai delete_videos().
        """
        prefix = f"{year}-{int(month):02d}"
        base = VIDEOS_DIR.resolve()
        out, seen = [], set()
        for r in self._iter_summary(year):
            ts = r.get("end_timestamp") or ""
            fn = (r.get("video_filename") or "").strip()
            if not fn or fn in seen or not ts.startswith(prefix):
                continue
            try:
                if line and int(r.get("line") or 0) != line:
                    continue
                if station and int(r.get("station_id") or 0) != station:
                    continue
            except ValueError:
                continue
            p = self.find_video("anomaly", fn) or self.find_video("normal", fn)
            if p is None:
                continue
            k = "anomaly" if p.parent.name in (ANOMALY_DIRNAME, LEGACY_ANOMALY_DIRNAME) else "normal"
            if kind != "all" and k != kind:
                continue
            try:
                size = p.stat().st_size
                rel = p.resolve().relative_to(base).as_posix()
            except (OSError, ValueError):
                continue
            seen.add(fn)
            out.append({
                "filename": fn,
                "path": rel,
                "kind": k,
                "end_timestamp": ts,
                "cycle_time_sec": r.get("cycle_time_sec") or "",
                "anomaly_type": r.get("anomaly_type") or "OK",
                "anomaly_label": r.get("anomaly_label") or "",
                "size_mb": round(size / _MB, 2),
                "line": r.get("line") or "",
                "station_id": r.get("station_id") or "",
            })
        out.sort(key=lambda x: x["end_timestamp"], reverse=True)
        return out

    def delete_videos(self, relative_paths) -> dict:
        """
        Hapus video terpilih (normal atau anomali) dari daftar list_month_videos().
        Hanya .mp4 di folder normal/anomali di bawah videos/ yang boleh dihapus;
        path traversal, _tmp_ dan file CSV ditolak. Summary CSV tidak diubah.
        Return: {path: True/False}
        """
        base = VIDEOS_DIR.resolve()
        allowed = (NORMAL_DIRNAME, ANOMALY_DIRNAME, LEGACY_ANOMALY_DIRNAME)
        res = {}
        with self._cleanup_lock:
            for rel in relative_paths:
                ok = False
                try:
                    t = (VIDEOS_DIR / rel).resolve()
                    if (base in t.parents and t.suffix.lower() == ".mp4"
                            and t.parent.name in allowed
                            and not t.name.startswith(TMP_PREFIX) and t.is_file()):
                        size = t.stat().st_size
                        t.unlink()
                        logger.info(f"[StorageManager] Manual delete: {rel} ({size / _MB:.1f}MB)")
                        ok = True
                    else:
                        logger.warning(f"[StorageManager] Ditolak: {rel}")
                except Exception as e:
                    logger.error(f"[StorageManager] Gagal hapus {rel}: {e}")
                res[rel] = ok
        return res
    def delete_anomaly_videos_bulk(self, relative_paths: list) -> dict:
        """Hapus beberapa video anomali sekaligus. Return hasil per path."""
        return {p: self.delete_anomaly_video(p) for p in relative_paths}

    # ── Info & listing ────────────────────────────────────────────────────────

    def get_usage(self) -> dict:
        normal = self._scan(NORMAL_DIRNAME)
        anomaly = self._scan(ANOMALY_DIRNAME)

        normal_bytes = sum(s for _, _, s in normal)
        anomaly_bytes = sum(s for _, _, s in anomaly)

        oldest_normal = None
        if normal:
            oldest_ts = min(m for _, m, _ in normal)
            oldest_normal = datetime.fromtimestamp(oldest_ts).isoformat()

        return {
            "normal_count": len(normal),
            "normal_gb": round(normal_bytes / _GB, 3),
            "record_months": RECORD_MONTHS,
            "backup_window_months": BACKUP_WINDOW_MONTHS,
            "anomaly_count": len(anomaly),
            "anomaly_gb": round(anomaly_bytes / _GB, 3),
            "total_gb": round((normal_bytes + anomaly_bytes) / _GB, 3),
            "oldest_normal_file": oldest_normal,
        }

    def list_videos(
        self,
        anomaly_only: bool = False,
        station_id: int = None,
        line: int = None,
        year: int = None,
        month: str = None,
    ) -> list:
        """
        Daftar video terurut dari yang terbaru.
        Filter opsional: station_id, line, year (2026), month ("April").
        'path' berisi path relatif dari VIDEOS_DIR (dipakai untuk delete/akses).
        """
        kind = ANOMALY_DIRNAME if anomaly_only else NORMAL_DIRNAME
        y = str(year) if year else "*"
        m = month if month else "*"

        entries = []
        for f in VIDEOS_DIR.glob(f"{y}/{m}/{kind}/*.mp4"):
            if f.name.startswith(TMP_PREFIX):
                continue
            if station_id and self._parse_station_id(f.name) != station_id:
                continue
            if line and self._parse_line(f.name) != line:
                continue
            try:
                st = f.stat()
            except OSError:
                continue
            entries.append((f, st))

        entries.sort(key=lambda x: x[1].st_mtime, reverse=True)

        return [
            {
                "filename": f.name,
                "path": f.relative_to(VIDEOS_DIR).as_posix(),
                "year": f.parents[2].name,
                "month": f.parents[1].name,
                "size_mb": round(st.st_size / _MB, 2),
                "created_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                "is_anomaly": anomaly_only,
                "anomaly_type": self._parse_anomaly_type(f.name),
                "is_genesis": "_GENESIS" in f.name,
                "is_expected": "_EXPECTED_" in f.name,
                "line": self._parse_line(f.name),
                "station_id": self._parse_station_id(f.name),
                "can_delete": anomaly_only,  # hanya anomali yang bisa dihapus manual
            }
            for f, st in entries
        ]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_station_id(filename: str) -> int:
        m = _NAME_RE.match(filename)
        return int(m.group("station")) if m else 0

    @staticmethod
    def _parse_line(filename: str) -> int:
        m = _NAME_RE.match(filename)
        return int(m.group("line")) if m and m.group("line") else 0

    @staticmethod
    def _parse_anomaly_type(filename: str) -> str:
        stem = filename[:-4] if filename.endswith(".mp4") else filename
        if not stem.endswith("_ANOMALY"):
            return "OK"
        tokens = stem.split("_")
        if "SHORT" in tokens:
            return "SHORT"
        if "LONG" in tokens:
            return "LONG"
        # hanya EXPECTED terlewat: anomaly_type tetap OK (sama dengan yang dikirim ke server)
        return "OK" if "EXPECTED" in tokens else "LONG"   # nama lama tanpa token = LONG

    def _loop(self):
        # Cleanup langsung saat start, lalu ulangi tiap CLEANUP_INTERVAL
        while not self._stop_event.is_set():
            try:
                self.run_cleanup()
            except Exception as e:
                logger.error(f"[StorageManager] Loop error: {e}")
            self._stop_event.wait(CLEANUP_INTERVAL)