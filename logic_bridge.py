# logic_bridge.py
"""
Penyambung LogicConfig (logic_editor.py) -> Pipeline (plc_logic.py). Tanpa Tk / cv2.

Dipakai oleh logic_editor.py (uji mandiri) dan nanti main.py:
    tester = SequenceTester()
    tester.sync(cfg)                    # True bila ROI/pengaturan berubah (tester di-reset)
    tester.update(dets, t_frame)        # dets = [(x1,y1,x2,y2,class,conf)], t = detik waktu frame
    tester.status / .state / .message / .render() / .assign

cfg dibaca secara "duck typing": field baru RoiDef (mode, min_overlap, priority, group,
dwell, grace) dan LogicConfig (skip_mode, restart_on_input, single, max_det, program)
boleh tidak ada -> dipakai nilai bawaan.
"""
from __future__ import annotations

from plc_engine import (HitRule, SeqSpec, StepSpec, build_sequence_program,
                        point_in_poly, poly_area, poly_bbox)
from plc_logic import (Pipeline, ProgramError, make_config, program_to_text,
                       template_to_program_config)

MODES = ("center", "fraction", "iou")
TAG_MANDATORY = "mandatory"


def _g(o, name, default):
    return getattr(o, name, default)


def _poly(r) -> tuple:
    """Titik polygon ROI ((x, y), ...); () bila ROI berupa kotak biasa."""
    p = _g(r, "poly", None) or ()
    return tuple((float(x), float(y)) for x, y in p)


def _shape(r) -> tuple:
    """Bentuk ROI sebagai polygon (kotak -> 4 sudut). Poly tidak valid (< 3 titik) -> kotak."""
    p = _poly(r)
    if len(p) >= 3:
        return p
    x1, y1, x2, y2 = r.rect
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def _rules(cfg) -> list:
    out = []
    for r in cfg.all_rois():
        p = _poly(r)
        out.append(HitRule(
            name=r.id, rect=(poly_bbox(p) if len(p) >= 3 else tuple(r.rect)),
            poly=(p if len(p) >= 3 else ()),
            classes=set(r.classes), conf=float(r.conf),
            mode=_g(r, "mode", "center"), min_overlap=float(_g(r, "min_overlap", 0.4)),
            priority=int(_g(r, "priority", 0)), group=_g(r, "group", "") or "",
            source=int(_g(r, "source", 0))))
    return out


def _spec(cfg) -> SeqSpec:
    i, o = cfg.input, cfg.output
    return SeqSpec(
        StepSpec(i.id, "expected", "", float(_g(i, "dwell", 0.0)), float(_g(i, "grace", 0.0))),
        StepSpec(o.id, "expected", "", float(_g(o, "dwell", 0.0)), float(_g(o, "grace", 0.0))),
        [StepSpec(r.id, r.tag, r.description, float(_g(r, "dwell", 0.0)), float(_g(r, "grace", 0.0)))
         for r in cfg.sequence],
        skip_mode=_g(cfg, "skip_mode", "immediate"),
        restart_on_input=bool(_g(cfg, "restart_on_input", False)),
        chain=bool(_g(cfg, "chain", True)))


def build_config(cfg) -> dict:
    """LogicConfig -> dict konfigurasi Pipeline (template, atau program teks bila cfg.program diisi)."""
    max_det = int(_g(cfg, "max_det", 0)) or None
    c = make_config(_rules(cfg), spec=_spec(cfg), max_det=max_det, single=bool(_g(cfg, "single", False)))
    prog = (_g(cfg, "program", "") or "").strip()
    if prog:
        c = template_to_program_config(c)          # cycle + messages bawaan
        c["program"] = prog + "\n"
    return c


def template_text(cfg) -> str:
    """Program teks hasil terjemahan template saat ini (titik awal untuk diedit user)."""
    return program_to_text(build_sequence_program(_spec(cfg)))


def _iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
    return inter / max(area(a) + area(b) - inter, 1e-9)


def _roi_overlap(ra, rb) -> float:
    """Tingkat beririsan (IoU) dua ROI. Kotak-kotak: eksak; bila ada polygon: sampling grid."""
    if len(_poly(ra)) < 3 and len(_poly(rb)) < 3:
        return _iou(ra.rect, rb.rect)
    pa, pb = _shape(ra), _shape(rb)
    a, b = poly_bbox(pa), poly_bbox(pb)
    if min(a[2], b[2]) <= max(a[0], b[0]) or min(a[3], b[3]) <= max(a[1], b[1]):
        return 0.0                                   # bounding box saja tidak bertemu
    x1, y1 = min(a[0], b[0]), min(a[1], b[1])
    x2, y2 = max(a[2], b[2]), max(a[3], b[3])
    n = 48
    inter = union = 0
    for i in range(n):
        for j in range(n):
            x = x1 + (i + 0.5) * (x2 - x1) / n
            y = y1 + (j + 0.5) * (y2 - y1) / n
            ia, ib = point_in_poly(x, y, pa), point_in_poly(x, y, pb)
            inter += ia and ib
            union += ia or ib
    return inter / union if union else 0.0


def _ccw(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_cross(a, b, c, d) -> bool:
    d1, d2, d3, d4 = _ccw(a, b, c), _ccw(a, b, d), _ccw(c, d, a), _ccw(c, d, b)
    return d1 * d2 < 0 and d3 * d4 < 0


def poly_self_intersects(p) -> bool:
    """True bila ada dua sisi yang tidak bersebelahan saling berpotongan (bentuk 'dasi')."""
    n = len(p)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue                              # sisi pertama & terakhir bersebelahan
            if _seg_cross(p[i], p[(i + 1) % n], p[j], p[(j + 1) % n]):
                return True
    return False


def check_config(cfg):
    """(errors, warnings) tambahan untuk validate(): parameter, program teks, ROI beririsan."""
    errors, warnings = [], []
    if not (cfg.input and cfg.output):
        return errors, warnings
    for r in cfg.all_rois():
        if _g(r, "mode", "center") not in MODES:
            errors.append(f"ROI '{r.id}': mode harus salah satu {MODES}.")
        if not (0 < float(_g(r, "min_overlap", 0.4)) <= 1):
            errors.append(f"ROI '{r.id}': min_overlap harus 0..1.")
        if float(_g(r, "dwell", 0.0)) < 0 or float(_g(r, "grace", 0.0)) < 0:
            errors.append(f"ROI '{r.id}': dwell/grace tidak boleh negatif.")
        p = _poly(r)
        if _g(r, "poly", None):
            if len(p) < 3:
                errors.append(f"ROI '{r.id}': polygon butuh minimal 3 titik.")
            elif poly_self_intersects(p):
                errors.append(f"ROI '{r.id}': sisi polygon saling berpotongan.")
            elif poly_area(p) <= 0:
                errors.append(f"ROI '{r.id}': luas polygon nol (titik segaris).")
    if _g(cfg, "skip_mode", "immediate") not in ("immediate", "final"):
        errors.append("skip_mode harus 'immediate' atau 'final'.")
    if not errors:
        try:
            warnings += [f"Program: {w}" for w in Pipeline(build_config(cfg)).warnings]
        except (ProgramError, ValueError) as e:
            errors.append(f"Logika/program: {e}")
    rois = cfg.all_rois()
    for a in range(len(rois)):
        for b in range(a + 1, len(rois)):
            ra, rb = rois[a], rois[b]
            if _g(ra, "source", 0) != _g(rb, "source", 0):
                continue                      # koordinat beda frame, tidak sebanding
            ov = _roi_overlap(ra, rb)
            if ov > 0.3:
                warnings.append(f"ROI '{ra.id}' dan '{rb.id}' beririsan {ov:.0%} "
                                f"— pertimbangkan grup eksklusif + prioritas.")
    return errors, warnings


class SequenceTester:
    IDLE, RUNNING, COMPLETE = "IDLE", "RUNNING", "COMPLETE"

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.pipe = None
        self.error = ""
        self.assign = []
        self._sig = None
        self._base = 0
        self._last = None
        if cfg is not None:
            self.sync(cfg)

    # -- util --
    def ready(self) -> bool:
        c = self.cfg
        return bool(c is not None and c.input and c.output and c.sequence)

    @staticmethod
    def _signature(cfg):
        rows = [(r.id, r.role, tuple(r.rect), tuple(sorted(r.classes)), r.conf, r.tag,
                 r.description, _g(r, "mode", "center"), _g(r, "min_overlap", 0.4),
                 _g(r, "priority", 0), _g(r, "group", ""), _g(r, "dwell", 0.0),
                 _g(r, "grace", 0.0), _g(r, "source", 0), _poly(r)) for r in cfg.all_rois()]
                 
        return repr((rows, _g(cfg, "skip_mode", "immediate"), _g(cfg, "restart_on_input", False),
                     _g(cfg, "chain", True),
                     _g(cfg, "single", False), _g(cfg, "max_det", 0), _g(cfg, "program", "")))

    def _rebuild(self):
        if self.pipe is not None:
            self._base += self.pipe.cycles
            if self.pipe.last_cycle is not None:
                self._last = self.pipe.last_cycle
        self.pipe, self.error, self.assign = None, "", []
        if not self.ready():
            return
        try:
            self.pipe = Pipeline(build_config(self.cfg))
            self.pipe.runner.engine.back_tol = 1e-3     # seek mundur 1 frame pun mereset siklus
        except (ProgramError, ValueError) as e:
            self.error = f"Logika/program tidak valid: {e}"

    def reset(self):
        self._rebuild()

    def sync(self, cfg) -> bool:
        """Panggil tiap tick. True bila konfigurasi berubah (tester dibangun ulang)."""
        self.cfg = cfg
        sig = self._signature(cfg)
        if sig == self._sig:
            return False
        self._sig = sig
        self._rebuild()
        return True

    # -- inti --
    def update(self, dets: list, t: float):
        if self.pipe is None:
            self.assign = []
            return
        self.pipe.update(dets, t)
        self.assign = self.pipe.assign

    # -- keluaran --
    @property
    def state(self) -> str:
        return self.pipe.state if self.pipe else self.IDLE

    @property
    def status(self) -> dict:
        return self.pipe.status() if self.pipe else {}

    @property
    def cycles(self) -> int:
        return self._base + (self.pipe.cycles if self.pipe else 0)

    @property
    def last_cycle(self):
        if self.pipe is not None and self.pipe.last_cycle is not None:
            return self.pipe.last_cycle
        return self._last

    @property
    def message(self) -> str:
        if self.error:
            return self.error
        if not self.ready():
            return "Lengkapi ROI Input, Output, dan Sequence dulu"
        return self.pipe.message()

    def _row(self, label, text, rid, st, active=False):
        s = st.get(rid)
        if s and s[0] == "done":
            icon, extra = "✓", (f"{s[1]:.2f}s" if s[1] is not None else "")
        elif s and s[0] == "skip":
            icon, extra = "⤼", "skip process"
        elif active:
            icon, extra = "▶", "menunggu..."
        else:
            icon, extra = "·", ""
        return f"{icon} {label:<5}{text[:24]:<24} {extra}"

    def render(self) -> str:
        if self.error or not self.ready():
            return self.message
        cfg, st = self.cfg, self.status
        run = self.state == self.RUNNING
        active = None
        if run:
            active = next((r.id for r in cfg.sequence if r.id not in st), cfg.output.id)
        last = f"{self.last_cycle:.2f}s" if self.last_cycle is not None else "-"
        lines = [f"Status {self.state}   siklus: {self.cycles}   terakhir: {last}",
                 self.message, "", self._row("IN", "Ambil barang", cfg.input.id, st)]
        for i, r in enumerate(cfg.sequence):
            lines.append(self._row(f"{i + 1}{'M' if r.tag == TAG_MANDATORY else 'E'}",
                                   r.description, r.id, st, active=(active == r.id)))
        lines.append(self._row("OUT", "Taruh barang", cfg.output.id, st, active=(active == cfg.output.id)))
        return "\n".join(lines)


# ── Uji mandiri ────────────────────────────────────────────────────────────

def _selftest():
    from types import SimpleNamespace as NS

    def roi(id, role, x, tag="", desc="", **kw):
        return NS(id=id, role=role, rect=(x, 0, x + 50, 50), classes=set(), conf=0.25,
                  tag=tag, description=desc, **kw)

    class Cfg(NS):
        def all_rois(self):
            return [self.input, *self.sequence, self.output]

    def make(**kw):
        return Cfg(input=roi("in", "input", 0),
                   sequence=[roi("s1", "sequence", 60, "expected", "scan"),
                             roi("s2", "sequence", 120, "mandatory", "pasang", dwell=0.2),
                             roi("s3", "sequence", 180, "expected", "mask")],
                   output=roi("out", "output", 240), **kw)

    def det(name):
        r = {"in": 0, "s1": 60, "s2": 120, "s3": 180, "out": 240}[name]
        return [(r + 20, 20, r + 30, 30, "hand", 0.9)]

    seq = ["", "in", "", "s1", "", "s2", "s2", "s2", "", "s3", "", "out", ""]

    def run(t, frames=seq):
        for i, n in enumerate(frames):
            t.update(det(n) if n else [], round(i * 0.1, 6))
        return t

    # template: urutan penuh
    t = SequenceTester()
    assert not t.ready() and "Lengkapi" in t.message
    assert t.sync(make()) is True and t.sync(make()) is False, "sync mendeteksi perubahan"
    run(t)
    assert t.state == "COMPLETE" and t.cycles == 1 and abs(t.last_cycle - 1.0) < 1e-6
    assert all(v[0] == "done" for v in t.status.values()) and len(t.status) == 5
    assert "urutan OK" in t.message and "COMPLETE" in t.render()

    # ubah pengaturan -> reset, siklus lama tetap terhitung
    cfg = make(skip_mode="final")
    assert t.sync(cfg) is True and t.state == "IDLE" and t.cycles == 1 and t.last_cycle is not None

    # mode program: template diekspor, diedit user, hasil sama + node tambahan
    cfg = make(program=template_text(make()))
    t2 = run(SequenceTester(cfg))
    assert t2.state == "COMPLETE" and abs(t2.last_cycle - 1.0) < 1e-6
    assert t2.status == run(SequenceTester(make())).status

    # program rusak / referensi hilang -> error terbaca, tidak crash
    bad = SequenceTester(make(program="a = FOO(x)"))
    assert "baris 1" in bad.message and bad.pipe is None
    bad.update(det("in"), 0.0)
    errs, _ = check_config(make(program="a = zzz"))
    assert errs and "zzz" in errs[0], errs
    errs, _ = check_config(make(skip_mode="x"))
    assert any("skip_mode" in e for e in errs)

    # ROI beririsan -> peringatan; grup yang sama -> tidak
    c = make()
    c.sequence[1].rect = (70, 0, 120, 50)
    assert any("beririsan" in w for w in check_config(c)[1])
    c.sequence[0].group = c.sequence[1].group = "g"
    assert not any("beririsan" in w for w in check_config(c)[1])

    # satu deteksi bertumpuk: s1 & s2 beririsan, grup + prioritas -> hanya s2 mendapat bit
    c = make(single=True)
    c.sequence[1].rect = (60, 0, 110, 50)           # sama dengan s1
    t3 = SequenceTester(c)
    t3.update([(70, 20, 80, 30, "hand", 0.9)], 0.0)
    assert sum(len(n) for _, n in t3.assign) == 1, t3.assign

    # waktu mundur 1 frame mereset siklus
    t4 = SequenceTester(make())
    for i, n in enumerate(["", "in", "", "s1"]):
        t4.update(det(n) if n else [], round(i * 0.1, 6))
    assert t4.state == "RUNNING"
    t4.update([], 0.25)
    assert t4.state == "IDLE", "mundur 0.05s harus reset"
    # polygon: bentuk L -> box di lekukan tidak mengenai ROI, walau bounding box-nya kena
    L = ((0, 0), (100, 0), (100, 40), (40, 40), (40, 100), (0, 100))
    c = make()
    c.sequence[0].poly = L
    c.sequence[0].rect = (0, 0, 100, 100)
    c.sequence[1].rect = (300, 0, 350, 50)          # jauhkan dari ROI lain
    rules = {r.name: r for r in _rules(c)}
    assert rules["s1"].poly == tuple((float(x), float(y)) for x, y in L)
    assert rules["s1"].rect == (0, 0, 100, 100) and rules["s2"].poly == ()
    from plc_engine import compute_bits
    bits, _ = compute_bits([rules["s1"]], [(70, 70, 90, 90, "hand", 0.9)])
    assert bits == {"s1": False}, bits
    bits, _ = compute_bits([rules["s1"]], [(10, 10, 30, 30, "hand", 0.9)])
    assert bits == {"s1": True}, bits

    # signature berubah bila titik polygon digeser -> tester di-reset
    t5 = SequenceTester(c)
    assert t5.sync(c) is False
    c.sequence[0].poly = L[:-1] + ((0, 90),)
    assert t5.sync(c) is True

    # validasi polygon: <3 titik, segaris, dasi (self-intersect)
    for bad, key in ((((0, 0), (10, 10)), "minimal 3"),
                     (((0, 0), (10, 10), (20, 20)), "luas polygon nol"),
                     (((0, 0), (50, 50), (50, 0), (0, 50)), "berpotongan")):
        c = make()
        c.sequence[0].poly = bad
        errs, _ = check_config(c)
        assert any(key in e for e in errs), (bad, errs)
    c = make()
    c.sequence[0].poly = L
    assert not [e for e in check_config(c)[0] if "polygon" in e]

    # peringatan beririsan memakai bentuk polygon: segitiga diagonal vs kotak di sudut kosong
    c = make()
    c.sequence[0].poly = ((60, 0), (110, 0), (60, 50))          # segitiga kiri-atas dari kotak 60..110
    c.sequence[0].rect = (60, 0, 110, 50)
    c.sequence[1].rect = (95, 30, 145, 80)                        # kena bbox segitiga, tapi di sisi kosong
    assert not any("beririsan" in w and "s1" in w and "s2" in w for w in check_config(c)[1])
    c.sequence[1].rect = (60, 0, 85, 25)                          # benar-benar di dalam segitiga
    assert any("beririsan" in w and "s1" in w and "s2" in w for w in check_config(c)[1])

    print("logic_bridge: semua uji lolos")


if __name__ == "__main__":
    _selftest()