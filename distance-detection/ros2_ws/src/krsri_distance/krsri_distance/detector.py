"""
Pembungkus model YOLO boneka KRSRI — dapat dipakai ulang, tanpa ROS.

Modul ini adalah ekstraksi bersih dari `model-fine-tuned/scripts/raspi_detect.py`,
yang di sana menyatukan capture + inferensi + stabilisasi + rendering dalam satu
loop sehingga tidak bisa di-import oleh node ROS. Di sini deteksi dipisah dari
penggambaran: `detect()` mengembalikan data murni, penggambaran urusan pemanggil.

Tiga perbaikan terhadap versi asli:
  1. `min_hits` kini benar-benar menghitung frame BERTURUT-TURUT. Di versi asli
     `hits` tidak pernah di-reset saat track tidak ter-match, sehingga objek yang
     berkedip tiap 10 frame tetap lolos konfirmasi — padahal justru itu pola
     false-positive yang ingin dibuang.
  2. Nama kelas dibaca dari `model.names`, bukan konstanta global yang bisa
     IndexError kalau bobot model diganti dengan jumlah kelas berbeda.
  3. `metadata.yaml` dibaca dengan `yaml.safe_load`, bukan parser baris manual.
     Ketidakcocokan imgsz melempar exception, bukan `sys.exit()` — supaya node
     ROS bisa melaporkan galatnya alih-alih mati diam-diam.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ── Nilai bawaan (sama dengan raspi_detect.py agar perilaku konsisten) ───────

DEFAULT_IMGSZ = 640
DEFAULT_CONF_THRESHOLD = 0.50
DEFAULT_IOU_THRESHOLD = 0.45

DEFAULT_MIN_HITS = 3  # frame konfirmasi sebelum objek ditampilkan
DEFAULT_MAX_AGE = 15  # buang track yang tak terlihat > N frame
DEFAULT_SMOOTH_ALPHA = 0.5  # EMA box smoothing (1.0 = tanpa smoothing)
DEFAULT_TRACK_IOU = 0.3  # ambang asosiasi deteksi ke track


class ModelSizeMismatchError(RuntimeError):
    """Model NCNN dijalankan pada imgsz berbeda dari ukuran ekspornya."""


@dataclass(frozen=True)
class Detection:
    """Satu objek terdeteksi pada satu frame. bbox dalam piksel (x1, y1, x2, y2)."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    track_id: Optional[int] = None


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-Union dua kotak (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


def _safe_class_name(class_names: Sequence[str], class_id: int) -> str:
    """Nama kelas yang tidak pernah melempar IndexError kalau model diganti."""
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return f"class_{class_id}"


class OnlineStabilizer:
    """
    Stabilizer realtime satu-pass di atas deteksi mentah per-frame.

    Tracker IoU ringan tanpa Kalman — sengaja, agar bebas dari galat numerik
    "Singular matrix" pada sebagian versi numpy dan tetap ringan di ARM.

    Tiga aturan:
      • Konfirmasi min_hits — objek baru muncul hanya setelah terlihat N frame
        berturut-turut, membuang false-positive kedip.
      • Voting kelas berjalan (ditimbang confidence) — label berhenti gonta-ganti.
      • EMA box smoothing — kotak tidak bergetar.
    """

    def __init__(
        self,
        min_hits: int = DEFAULT_MIN_HITS,
        max_age: int = DEFAULT_MAX_AGE,
        alpha: float = DEFAULT_SMOOTH_ALPHA,
        iou_threshold: float = DEFAULT_TRACK_IOU,
    ) -> None:
        self.min_hits = min_hits
        self.max_age = max_age
        self.alpha = alpha
        self.iou_threshold = iou_threshold
        self._tracks: Dict[int, dict] = {}
        self._next_id = 1

    def _associate(self, boxes: np.ndarray) -> Dict[int, int]:
        """Cocokkan indeks deteksi ke track id secara greedy, IoU tertinggi dulu."""
        candidates = []
        for det_idx, box in enumerate(boxes):
            for tid, track in self._tracks.items():
                score = bbox_iou(box, track["box"])
                if score >= self.iou_threshold:
                    candidates.append((score, det_idx, tid))
        candidates.sort(reverse=True)

        used_dets: set = set()
        used_tracks: set = set()
        matches: Dict[int, int] = {}
        for _, det_idx, tid in candidates:
            if det_idx in used_dets or tid in used_tracks:
                continue
            used_dets.add(det_idx)
            used_tracks.add(tid)
            matches[det_idx] = tid
        return matches

    def _spawn(self, box: np.ndarray, conf: float) -> int:
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = {
            "votes": defaultdict(float),
            "hits": 0,
            "box": np.asarray(box, dtype=float).copy(),
            "conf": float(conf),
            "last_seen": -1,
        }
        return tid

    def update(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        frame_no: int,
        class_names: Sequence[str],
    ) -> List[Detection]:
        """Perbarui semua track, kembalikan hanya yang sudah terkonfirmasi."""
        matches = self._associate(boxes)
        seen_this_frame: set = set()

        for det_idx, (box, conf, class_id) in enumerate(
            zip(boxes, confidences, class_ids)
        ):
            tid = matches.get(det_idx)
            if tid is None:
                tid = self._spawn(box, conf)
            track = self._tracks[tid]
            track["votes"][int(class_id)] += float(conf)
            track["hits"] += 1
            track["last_seen"] = frame_no
            track["conf"] = float(conf)
            track["box"] = (
                self.alpha * np.asarray(box, dtype=float)
                + (1.0 - self.alpha) * track["box"]
            )
            seen_this_frame.add(tid)

        # Perbaikan terhadap versi asli: putus rantai konfirmasi untuk track yang
        # tidak terlihat di frame ini, agar min_hits berarti "berturut-turut".
        for tid, track in self._tracks.items():
            if tid not in seen_this_frame:
                track["hits"] = 0

        stale = [
            tid
            for tid, track in self._tracks.items()
            if frame_no - track["last_seen"] > self.max_age
        ]
        for tid in stale:
            del self._tracks[tid]

        confirmed: List[Detection] = []
        for tid, track in self._tracks.items():
            if track["last_seen"] != frame_no or track["hits"] < self.min_hits:
                continue
            voted_id = max(track["votes"], key=track["votes"].get)
            x1, y1, x2, y2 = track["box"].astype(int).tolist()
            confirmed.append(
                Detection(
                    class_id=int(voted_id),
                    class_name=_safe_class_name(class_names, int(voted_id)),
                    confidence=track["conf"],
                    bbox=(x1, y1, x2, y2),
                    track_id=int(tid),
                )
            )
        return confirmed


def read_export_imgsz(model_path: Path) -> Optional[int]:
    """
    Baca ukuran ekspor dari metadata.yaml milik model NCNN.

    None jika bukan folder NCNN atau metadata tidak terbaca — pemanggil
    memperlakukan itu sebagai "tidak bisa dicek", bukan sebagai galat.
    """
    meta = model_path / "metadata.yaml"
    if not model_path.is_dir() or not meta.exists():
        return None
    try:
        import yaml

        with meta.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return None

    imgsz = data.get("imgsz")
    if isinstance(imgsz, (list, tuple)) and imgsz:
        return int(imgsz[0])
    if isinstance(imgsz, int):
        return imgsz
    return None


class BonekaDetector:
    """
    Deteksi boneka dari satu frame BGR. Tidak menggambar apa pun.

    Sengaja tidak tahu-menahu soal ROS maupun jendela OpenCV, sehingga bisa
    dipakai oleh node ROS 2, skrip CLI, atau test — sumber kebenaran tunggal.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        imgsz: int = DEFAULT_IMGSZ,
        conf: float = DEFAULT_CONF_THRESHOLD,
        iou: float = DEFAULT_IOU_THRESHOLD,
        stabilize: bool = True,
        min_hits: int = DEFAULT_MIN_HITS,
        warmup: bool = True,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Model tidak ditemukan: {path}")

        export_imgsz = read_export_imgsz(path)
        if export_imgsz is not None and export_imgsz != imgsz:
            raise ModelSizeMismatchError(
                f"Model NCNN diekspor fixed-shape {export_imgsz}x{export_imgsz} "
                f"tapi dijalankan pada imgsz={imgsz}. NCNN akan mengeluarkan "
                f"ratusan box sampah. Pakai imgsz={export_imgsz}, atau ekspor "
                f"ulang model pada ukuran {imgsz}."
            )

        # Import di sini, bukan di level modul: agar `Detection`/`OnlineStabilizer`
        # tetap bisa di-import (dan di-test) di mesin tanpa ultralytics terpasang.
        from ultralytics import YOLO

        self._model = YOLO(str(path), task="detect")
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self._frame_no = 0
        self._stabilizer = OnlineStabilizer(min_hits=min_hits) if stabilize else None

        names = self._model.names
        self.class_names: List[str] = (
            [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        )

        if warmup:
            self._model.predict(
                np.zeros((imgsz, imgsz, 3), dtype=np.uint8), imgsz=imgsz, verbose=False
            )

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Jalankan inferensi pada satu frame BGR, kembalikan deteksi terkonfirmasi."""
        result = self._model.predict(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )[0]

        raw = result.boxes
        if raw is None or len(raw) == 0:
            boxes = np.empty((0, 4), dtype=float)
            confs = np.empty((0,), dtype=float)
            class_ids = np.empty((0,), dtype=int)
        else:
            boxes = raw.xyxy.cpu().numpy()
            confs = raw.conf.cpu().numpy()
            class_ids = raw.cls.cpu().numpy().astype(int)

        self._frame_no += 1

        if self._stabilizer is not None:
            return self._stabilizer.update(
                boxes, confs, class_ids, self._frame_no, self.class_names
            )

        return [
            Detection(
                class_id=int(class_id),
                class_name=_safe_class_name(self.class_names, int(class_id)),
                confidence=float(conf),
                bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
            )
            for box, conf, class_id in zip(boxes, confs, class_ids)
        ]
