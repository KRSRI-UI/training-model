"""
Estimasi jarak & posisi 3D dari bounding box + citra depth.

Modul ini SENGAJA murni: tidak mengimpor ROS maupun ultralytics, hanya numpy.
Konsekuensinya modul ini bisa dikembangkan dan diuji di laptop mana pun
(termasuk Windows) tanpa Kinect dan tanpa ROS terpasang. Seluruh logika yang
bisa salah secara diam-diam ada di sini, jadi di sinilah test-nya berada.

Alur konseptualnya:
    bbox 2D (dari YOLO) → sampel depth di dalam bbox → jarak Z (meter)
                        → deproyeksi pinhole        → posisi (X, Y, Z)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple

import numpy as np

# ── Konstanta ───────────────────────────────────────────────────────────────

MM_PER_M = 1000.0

#: Fraksi bagian TENGAH bbox yang disampel. Sudut bbox hampir selalu berisi
#: latar belakang (bbox persegi, boneka tidak), jadi menyampel seluruh bbox
#: mencampur jarak objek dengan jarak tembok di belakangnya.
DEFAULT_CENTER_FRACTION = 0.5

#: Ambang minimum proporsi piksel valid di dalam ROI. Di bawah ini kita
#: menolak menebak dan melaporkan NO_DATA — lebih baik jujur tidak tahu
#: daripada mengembalikan angka yang dipakai robot untuk mengambil keputusan.
DEFAULT_MIN_VALID_RATIO = 0.2

#: Jarak di bawah ini dianggap noise, bukan pengukuran. Depth invalid pada
#: Kinect dikodekan sebagai 0, dan 0 tidak boleh ikut ke dalam median.
MIN_MEASURABLE_M = 0.05

#: Jangkauan spesifikasi Kinect v1 (Xbox 360). Di luar rentang ini pembacaan
#: tidak bisa dipercaya meski sensor mengeluarkan angka.
KINECT_MIN_RANGE_M = 0.8
KINECT_MAX_RANGE_M = 4.0

#: Encoding citra depth yang dikenali (mengikuti penamaan sensor_msgs).
ENCODING_MM_UINT16 = "16UC1"
ENCODING_M_FLOAT32 = "32FC1"


class DistanceStatus(str, Enum):
    """Hasil pembacaan depth untuk satu deteksi."""

    OK = "ok"
    TOO_CLOSE = "too_close"
    TOO_FAR = "too_far"
    NO_DATA = "no_data"

    @property
    def is_usable(self) -> bool:
        """True hanya jika angka jaraknya layak dipakai untuk keputusan gerak."""
        return self is DistanceStatus.OK


@dataclass(frozen=True)
class CameraIntrinsics:
    """
    Parameter internal lensa kamera (model pinhole).

    fx, fy : focal length dalam satuan piksel
    cx, cy : titik pusat optik pada citra, dalam piksel

    Nilai ini TIDAK perlu diukur manual — driver kamera mengirimkannya lewat
    topic camera_info. Untuk depth yang sudah teregistrasi, yang dipakai adalah
    intrinsik kamera RGB, karena depth sudah di-warp ke frame RGB.
    """

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_k_matrix(cls, k: Sequence[float]) -> "CameraIntrinsics":
        """Bangun dari matriks K (row-major 3x3) milik sensor_msgs/CameraInfo."""
        if len(k) != 9:
            raise ValueError(f"Matriks K harus 9 elemen, dapat {len(k)}")
        if k[0] == 0.0 or k[4] == 0.0:
            raise ValueError("fx/fy nol — camera_info belum terkalibrasi")
        return cls(fx=float(k[0]), fy=float(k[4]), cx=float(k[2]), cy=float(k[5]))


@dataclass(frozen=True)
class DistanceResult:
    """
    Hasil estimasi untuk satu bounding box.

    distance_m  : jarak (sumbu Z, maju dari kamera). None jika tidak terukur.
    position_m  : (X, Y, Z) dalam optical frame. None jika tidak terukur.
    size_m      : (lebar, tinggi) objek dalam meter, hasil proyeksi balik bbox.
    valid_ratio : proporsi piksel depth yang valid di ROI — indikator keyakinan.
    status      : lihat DistanceStatus.
    """

    status: DistanceStatus
    distance_m: Optional[float] = None
    position_m: Optional[Tuple[float, float, float]] = None
    size_m: Optional[Tuple[float, float]] = None
    valid_ratio: float = 0.0


# ── Fungsi internal ─────────────────────────────────────────────────────────


def _clip_bbox(
    bbox: Sequence[float], width: int, height: int
) -> Optional[Tuple[int, int, int, int]]:
    """Potong bbox agar berada di dalam citra. None jika tidak menyisakan area."""
    x1 = max(0, int(round(bbox[0])))
    y1 = max(0, int(round(bbox[1])))
    x2 = min(width, int(round(bbox[2])))
    y2 = min(height, int(round(bbox[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def center_roi(
    bbox: Sequence[float],
    width: int,
    height: int,
    fraction: float = DEFAULT_CENTER_FRACTION,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Ambil kotak bagian tengah dari bbox, sebesar `fraction` dari tiap sisi.

    Untuk bbox yang sangat kecil, penyusutan bisa membuat ROI kolaps jadi nol
    piksel. Dalam kasus itu kita kembalikan bbox utuh (yang sudah dipotong)
    daripada kehilangan deteksi sama sekali.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction harus di (0, 1], dapat {fraction}")

    clipped = _clip_bbox(bbox, width, height)
    if clipped is None:
        return None

    x1, y1, x2, y2 = clipped
    inset_x = int((x2 - x1) * (1.0 - fraction) / 2.0)
    inset_y = int((y2 - y1) * (1.0 - fraction) / 2.0)
    rx1, ry1 = x1 + inset_x, y1 + inset_y
    rx2, ry2 = x2 - inset_x, y2 - inset_y

    if rx2 <= rx1 or ry2 <= ry1:
        return clipped
    return rx1, ry1, rx2, ry2


def depth_to_meters(depth: np.ndarray, encoding: str) -> np.ndarray:
    """
    Ubah citra depth mentah menjadi float32 meter.

    Salah menebak encoding menyebabkan galat 1000x tanpa gejala lain, jadi
    encoding yang tidak dikenal ditolak keras alih-alih diasumsikan.
    """
    if encoding == ENCODING_MM_UINT16:
        return depth.astype(np.float32) / MM_PER_M
    if encoding == ENCODING_M_FLOAT32:
        return depth.astype(np.float32)
    raise ValueError(
        f"Encoding depth tidak dikenal: {encoding!r}. "
        f"Dukungan: {ENCODING_MM_UINT16} (mm) atau {ENCODING_M_FLOAT32} (meter)."
    )


def deproject(
    u: float, v: float, z_m: float, intr: CameraIntrinsics
) -> Tuple[float, float, float]:
    """
    Ubah piksel (u, v) + jarak z menjadi titik 3D dalam optical frame.

    Konvensi optical frame ROS: X ke kanan, Y ke bawah, Z ke depan (maju).
    Piksel menentukan ARAH sinar; z menentukan seberapa jauh menyusuri sinar itu.
    """
    x = (u - intr.cx) * z_m / intr.fx
    y = (v - intr.cy) * z_m / intr.fy
    return float(x), float(y), float(z_m)


# ── API utama ───────────────────────────────────────────────────────────────


def estimate_distance(
    depth_image: np.ndarray,
    bbox: Sequence[float],
    intrinsics: CameraIntrinsics,
    encoding: str = ENCODING_MM_UINT16,
    center_fraction: float = DEFAULT_CENTER_FRACTION,
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
    min_range_m: float = KINECT_MIN_RANGE_M,
    max_range_m: float = KINECT_MAX_RANGE_M,
) -> DistanceResult:
    """
    Estimasi jarak & posisi 3D satu objek dari bounding box-nya.

    Memakai MEDIAN piksel valid di ROI tengah, bukan piksel tengah tunggal
    (sering rusak) dan bukan rata-rata seluruh bbox (tercampur latar belakang).

    `depth_image` harus SUDAH teregistrasi terhadap citra RGB tempat `bbox`
    dihitung — kalau tidak, ROI akan menunjuk objek yang berbeda.
    """
    if depth_image.ndim != 2:
        raise ValueError(f"depth_image harus 2D, dapat shape {depth_image.shape}")

    height, width = depth_image.shape
    roi_box = center_roi(bbox, width, height, center_fraction)
    if roi_box is None:
        return DistanceResult(status=DistanceStatus.NO_DATA)

    rx1, ry1, rx2, ry2 = roi_box
    roi = depth_to_meters(depth_image[ry1:ry2, rx1:rx2], encoding)

    # Piksel invalid: 0 (Kinect menandai "tidak terukur") dan NaN/inf.
    valid = roi[np.isfinite(roi) & (roi > MIN_MEASURABLE_M)]
    valid_ratio = float(valid.size) / float(roi.size) if roi.size else 0.0

    if valid.size == 0 or valid_ratio < min_valid_ratio:
        return DistanceResult(status=DistanceStatus.NO_DATA, valid_ratio=valid_ratio)

    z_m = float(np.median(valid))

    clipped = _clip_bbox(bbox, width, height)
    if clipped is None:  # tidak mungkin: center_roi sudah lolos
        return DistanceResult(status=DistanceStatus.NO_DATA, valid_ratio=valid_ratio)
    bx1, by1, bx2, by2 = clipped
    position = deproject((bx1 + bx2) / 2.0, (by1 + by2) / 2.0, z_m, intrinsics)
    size = ((bx2 - bx1) * z_m / intrinsics.fx, (by2 - by1) * z_m / intrinsics.fy)

    if z_m < min_range_m:
        status = DistanceStatus.TOO_CLOSE
    elif z_m > max_range_m:
        status = DistanceStatus.TOO_FAR
    else:
        status = DistanceStatus.OK

    return DistanceResult(
        status=status,
        distance_m=z_m,
        position_m=position,
        size_m=size,
        valid_ratio=valid_ratio,
    )
