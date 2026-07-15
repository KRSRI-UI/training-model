#!/usr/bin/env python3
"""
KRSRI Boneka Detection — Raspberry Pi 4 Inference Script
=========================================================
Model  : YOLOv8s NCNN (optimized for ARM)
Target : Raspberry Pi 4 + Pi Camera v2 / USB Camera

Dependencies:
    pip install ultralytics opencv-python

Usage:
    # Kamera realtime:
    python3 raspi_detect.py --source 0

    # Pi Camera:
    python3 raspi_detect.py --source picam

    # Image file:
    python3 raspi_detect.py --source image.jpg

    # Video file:
    python3 raspi_detect.py --source video.mp4
"""

import argparse
import time
import sys
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════
# KONFIGURASI — sesuaikan dengan kondisi lapangan
# ═══════════════════════════════════════════════════════════
MODEL_PATH   = "best_ncnn_model"   # Folder model NCNN
CLASS_NAMES  = ["boneka-asli", "boneka-dummy"]
CLASS_COLORS = {
    0: (0, 255, 0),    # boneka-asli  → Hijau
    1: (0, 0, 255),    # boneka-dummy → Merah
}

# Threshold inferensi
CONF_THRESHOLD = 0.50   # Naikkan jika terlalu banyak false positive
IOU_THRESHOLD  = 0.45   # NMS IoU
IMGSZ          = 640    # HARUS cocok dgn ukuran ekspor NCNN (lihat
                        # best_ncnn_model/metadata.yaml → imgsz).
                        # Model NCNN ini diekspor fixed-shape 640x640.
                        # Menjalankan di ukuran lain (mis. 416) membuat NCNN
                        # mengeluarkan ratusan box sampah (bug fixed-shape).
                        # Untuk berjalan di 320/416 demi kecepatan di Raspi,
                        # ekspor ULANG model di ukuran itu dari notebook.

# ── Stabilisasi realtime (mode --stable) ─────────────────────
# Meredam false-positive kedip & label gonta-ganti yang terlihat di
# pengujian video lapangan. Lihat scripts/track_video_detect.py untuk
# versi offline (2-pass) yang lebih agresif.
MIN_HITS      = 3      # Objek baru ditampilkan setelah terlihat >= N frame
                       # berturut (konfirmasi). Menghapus box kedip sesaat.
                       # Konsekuensi: ada jeda ~N frame sebelum deteksi muncul.
TRACK_MAX_AGE = 15     # Buang state track yang tak terlihat > N frame
SMOOTH_ALPHA  = 0.5    # EMA box smoothing (1.0 = tanpa smoothing)

# ═══════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="KRSRI Boneka Detector")
    parser.add_argument("--source", default="0",
                        help="Input: 0=webcam, picam, image.jpg, video.mp4")
    parser.add_argument("--conf",   type=float, default=CONF_THRESHOLD)
    parser.add_argument("--imgsz",  type=int,   default=IMGSZ)
    parser.add_argument("--save",   action="store_true", help="Simpan output")
    parser.add_argument("--headless", action="store_true",
                        help="Mode tanpa display (untuk robot tanpa monitor)")
    parser.add_argument("--stable", dest="stable", action="store_true", default=True,
                        help="Stabilisasi online: tracking + voting kelas + "
                             "konfirmasi min-hits (default: aktif)")
    parser.add_argument("--raw", dest="stable", action="store_false",
                        help="Matikan stabilisasi (deteksi mentah per-frame, perilaku lama)")
    parser.add_argument("--min-hits", type=int, default=MIN_HITS,
                        help=f"Frame konfirmasi sebelum objek ditampilkan (default: {MIN_HITS})")
    return parser.parse_args()


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


class OnlineStabilizer:
    """
    Stabilizer realtime satu-pass untuk deteksi video/kamera.

    Berisi tracker IoU ringan sendiri (tanpa Kalman/ByteTrack — supaya ringan
    di Raspi dan bebas dari masalah numerik "Singular matrix" pada sebagian
    versi numpy). Menerapkan tiga aturan di atas deteksi mentah per-frame:
      • Konfirmasi (min_hits): objek baru ditampilkan hanya setelah terlihat
        >= min_hits frame berturut → menghapus false-positive kedip.
      • Voting kelas berjalan (ditimbang confidence) → label tidak lagi
        gonta-ganti asli<->dummy antar frame.
      • EMA box smoothing → box tidak bergetar.
    """

    def __init__(self, min_hits=MIN_HITS, max_age=TRACK_MAX_AGE,
                 alpha=SMOOTH_ALPHA, iou_thresh=0.3):
        self.min_hits = min_hits
        self.max_age = max_age
        self.alpha = alpha
        self.iou_thresh = iou_thresh
        self.tracks = {}     # id -> {votes, hits, last_seen, box(ema), conf}
        self._next_id = 1

    def update(self, boxes, confs, clsids, frame_no):
        """Asosiasikan deteksi ke track via IoU, lalu kembalikan yang terkonfirmasi."""
        # ── Asosiasi greedy berdasarkan IoU tertinggi ──
        track_ids = list(self.tracks.keys())
        pairs = []
        for di, box in enumerate(boxes):
            for tid in track_ids:
                v = _iou(box, self.tracks[tid]["box"])
                if v >= self.iou_thresh:
                    pairs.append((v, di, tid))
        pairs.sort(reverse=True)
        used_det, used_trk = set(), set()
        matches = {}
        for v, di, tid in pairs:
            if di in used_det or tid in used_trk:
                continue
            used_det.add(di); used_trk.add(tid); matches[di] = tid

        for di, (box, conf, cls_id) in enumerate(zip(boxes, confs, clsids)):
            tid = matches.get(di)
            if tid is None:                       # deteksi baru → track baru
                tid = self._next_id; self._next_id += 1
                self.tracks[tid] = {"votes": defaultdict(float), "hits": 0,
                                    "box": np.asarray(box, dtype=float).copy(),
                                    "conf": float(conf)}
            tr = self.tracks[tid]
            tr["votes"][int(cls_id)] += float(conf)
            tr["hits"] += 1
            tr["last_seen"] = frame_no
            tr["conf"] = float(conf)
            tr["box"] = self.alpha * np.asarray(box, dtype=float) + (1 - self.alpha) * tr["box"]

        # Buang track lama yang tak terlihat lagi
        for tid in [t for t, v in self.tracks.items()
                    if frame_no - v["last_seen"] > self.max_age]:
            del self.tracks[tid]

        # Keluarkan hanya track terkonfirmasi & terlihat di frame ini
        confirmed = []
        for tid, tr in self.tracks.items():
            if tr["last_seen"] == frame_no and tr["hits"] >= self.min_hits:
                voted_cls = max(tr["votes"], key=tr["votes"].get)
                confirmed.append({
                    "class": CLASS_NAMES[voted_cls], "class_id": voted_cls,
                    "confidence": tr["conf"], "track_id": int(tid),
                    "bbox": tr["box"].astype(int).tolist(),
                })
        return confirmed


def draw_stable(frame, detections):
    """Gambar deteksi yang sudah distabilkan (dengan track id + kelas voting)."""
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cls_id = d["class_id"]
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label_text = f"#{d['track_id']} {d['class']}: {d['confidence']:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


def check_imgsz_matches_export(model_path: str, imgsz: int):
    """
    Model NCNN diekspor fixed-shape. Menjalankan pada ukuran berbeda dari
    ukuran ekspor membuat model mengeluarkan ratusan box sampah. Baca ukuran
    ekspor dari metadata.yaml dan peringatkan keras kalau tidak cocok.
    """
    meta = os.path.join(model_path, "metadata.yaml")
    if not os.path.isdir(model_path) or not os.path.exists(meta):
        return  # bukan folder NCNN dgn metadata → lewati cek
    export_sz = None
    try:
        with open(meta) as f:
            in_imgsz = False
            for line in f:
                if line.startswith("imgsz:"):
                    in_imgsz = True
                    continue
                if in_imgsz:
                    s = line.strip()
                    if s.startswith("- "):
                        export_sz = int(s[2:]); break
                    else:
                        break
    except Exception:
        return
    if export_sz is not None and imgsz != export_sz:
        print("\n" + "!" * 62)
        print(f"[FATAL] Model NCNN ini diekspor fixed-shape {export_sz}x{export_sz},")
        print(f"        tapi Anda menjalankannya di imgsz={imgsz}.")
        print(f"        Ukuran tidak cocok → NCNN mengeluarkan ratusan box SAMPAH.")
        print(f"        Pakai --imgsz {export_sz}, atau ekspor ulang model di {imgsz}.")
        print("!" * 62 + "\n")
        sys.exit(1)


def load_model(model_path: str, imgsz: int) -> YOLO:
    """Load NCNN model dengan validasi."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        print("  Pastikan folder NCNN ada di direktori yang sama.")
        sys.exit(1)

    check_imgsz_matches_export(model_path, imgsz)

    print(f"[INFO] Loading model: {model_path}")
    print(f"[INFO] Image size  : {imgsz}x{imgsz}")
    model = YOLO(model_path, task="detect")

    # Warmup: run sekali untuk inisialisasi
    print("[INFO] Warming up model...")
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=imgsz, verbose=False)
    print("[INFO] Model ready!\n")
    return model


def open_capture(source: str) -> cv2.VideoCapture:
    """Buka input video/camera."""
    if source == "picam":
        # Pi Camera via libcamera (Raspi OS Bullseye+)
        try:
            cap = cv2.VideoCapture("libcamerasrc ! videoconvert ! "
                                   "video/x-raw,format=BGR ! appsink",
                                   cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                raise RuntimeError("libcamera tidak tersedia")
        except Exception:
            # Fallback ke indeks 0
            print("[WARN] libcamera tidak tersedia, pakai /dev/video0")
            cap = cv2.VideoCapture(0)
    elif source.isdigit():
        cap = cv2.VideoCapture(int(source))
        # Resolusi optimal untuk USB cam di Raspi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        # Image atau video file
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Tidak bisa membuka source: {source}")
        sys.exit(1)

    return cap


def draw_detections(frame: np.ndarray, result, conf_threshold: float) -> tuple:
    """
    Gambar bounding box dan label pada frame.
    Return: (annotated_frame, detections_list)
    """
    detections = []

    if result.boxes is None or len(result.boxes) == 0:
        return frame, detections

    boxes  = result.boxes.xyxy.cpu().numpy()
    confs  = result.boxes.conf.cpu().numpy()
    clsids = result.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, clsids):
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = box.astype(int)
        color      = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label      = CLASS_NAMES[cls_id]
        label_text = f"{label}: {conf:.0%}"

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        detections.append({"class": label, "class_id": cls_id, "confidence": float(conf),
                           "bbox": [x1, y1, x2, y2]})

    return frame, detections


def run_inference(model: YOLO, args):
    """Loop inferensi utama."""
    source = args.source
    is_image = source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

    if is_image:
        # ── Inferensi single image ────────────────────────────
        frame = cv2.imread(source)
        if frame is None:
            print(f"[ERROR] Gambar tidak bisa dibaca: {source}")
            return

        result = model.predict(
            frame, imgsz=args.imgsz, conf=args.conf,
            iou=IOU_THRESHOLD, verbose=False
        )[0]

        frame, detections = draw_detections(frame, result, args.conf)
        print(f"\n[RESULT] Deteksi ({len(detections)} objek):")
        for d in detections:
            print(f"  → {d['class']:15s} conf={d['confidence']:.2%}  bbox={d['bbox']}")

        if args.save:
            out_path = "output_" + os.path.basename(source)
            cv2.imwrite(out_path, frame)
            print(f"[INFO] Hasil disimpan: {out_path}")

        if not args.headless:
            cv2.imshow("KRSRI Detection", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # ── Inferensi video / kamera ──────────────────────────────
    cap = open_capture(source)

    # Setup video writer jika save
    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter("output_detection.mp4",
                                  cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # FPS tracking
    fps_counter = 0
    fps_display = 0
    t_start = time.time()

    print("[INFO] Mulai inferensi. Tekan 'q' untuk berhenti.")
    print(f"[INFO] Confidence threshold: {args.conf:.0%}")
    print(f"[INFO] Mode: {'STABLE (tracking+voting+min-hits)' if args.stable else 'RAW (per-frame)'}")

    stabilizer = OnlineStabilizer(min_hits=args.min_hits) if args.stable else None
    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Inferensi (selalu predict per-frame; stabilisasi di atasnya) ──
        t0 = time.perf_counter()
        result = model.predict(
            frame, imgsz=args.imgsz, conf=args.conf, iou=IOU_THRESHOLD,
            verbose=False, stream=False,
        )[0]
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000

        # ── Draw ─────────────────────────────────────────────
        if args.stable:
            b = result.boxes
            if b is not None and len(b) > 0:
                detections = stabilizer.update(
                    b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                    b.cls.cpu().numpy().astype(int), frame_no,
                )
            else:
                detections = stabilizer.update(
                    np.empty((0, 4)), np.array([]), np.array([], dtype=int), frame_no,
                )
            frame = draw_stable(frame, detections)
        else:
            frame, detections = draw_detections(frame, result, args.conf)
        frame_no += 1

        # ── FPS overlay ──────────────────────────────────────
        fps_counter += 1
        if time.time() - t_start >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            t_start = time.time()

        overlay = f"FPS: {fps_display}  |  Infer: {infer_ms:.0f}ms  |  Det: {len(detections)}"
        cv2.putText(frame, overlay, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ── Log ke terminal (hanya jika ada deteksi) ─────────
        if detections:
            det_str = ", ".join(
                f"{d['class']}({d['confidence']:.0%})" for d in detections
            )
            print(f"[DETECT] {det_str}  [{infer_ms:.0f}ms]")

        if writer:
            writer.write(frame)

        if not args.headless:
            cv2.imshow("KRSRI Boneka Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Dihentikan oleh user.")
                break

    cap.release()
    if writer:
        writer.release()
        print("[INFO] Video tersimpan: output_detection.mp4")
    if not args.headless:
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    model = load_model(MODEL_PATH, args.imgsz)

    print("[CONFIG]")
    print(f"  Source    : {args.source}")
    print(f"  Conf thr  : {args.conf:.0%}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Headless  : {args.headless}")
    print()

    run_inference(model, args)


if __name__ == "__main__":
    main()