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
IMGSZ          = 416    # Kecilkan resolusi untuk Raspi (320/416/640)
                        # 320 → paling cepat, 640 → paling akurat

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
    return parser.parse_args()


def load_model(model_path: str, imgsz: int) -> YOLO:
    """Load NCNN model dengan validasi."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        print("  Pastikan folder NCNN ada di direktori yang sama.")
        sys.exit(1)

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Inferensi ────────────────────────────────────────
        t0 = time.perf_counter()
        result = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=IOU_THRESHOLD,
            verbose=False,
            stream=False,
        )[0]
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000

        # ── Draw ─────────────────────────────────────────────
        frame, detections = draw_detections(frame, result, args.conf)

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