#!/usr/bin/env python3
"""
KRSRI Boneka Detection — Video Test Script
=============================================
Tujuan: menguji model di luar 450 foto sesi training (12 Januari 2024),
dengan video uji yang Owen rekam sendiri. Ini cara paling jujur untuk
tahu apakah mAP 99% di valid set juga bertahan di kondisi lain.

Cara pakai (di Colab, setelah training selesai):
    !python test_video_detect.py --video /content/video_test.mp4

Atau dengan opsi custom:
    !python test_video_detect.py \\
        --video /content/video_test.mp4 \\
        --weights /content/runs/phase2_fulltune/weights/best.pt \\
        --conf 0.5 \\
        --output /content/video_test_annotated.mp4

Output: video baru dengan bounding box + label + confidence score
ditempel di setiap frame yang ada deteksinya. Video ASLI tidak diubah
sama sekali — ini membuat salinan baru, bukan menimpa.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════
# KONFIGURASI — disamakan dengan raspi_detect.py (Cell 12)
# supaya hasil video ini konsisten dengan yang nanti dilihat
# Owen saat live di Raspi.
# ═══════════════════════════════════════════════════════════
CLASS_NAMES = ["boneka-asli", "boneka-dummy"]
CLASS_COLORS = {
    0: (0, 255, 0),    # boneka-asli  → Hijau  (BGR, sesuai OpenCV)
    1: (0, 0, 255),    # boneka-dummy → Merah
}

DEFAULT_WEIGHTS = "/content/runs/phase2_fulltune/weights/best.pt"
DEFAULT_CONF = 0.50    # Sama dengan default raspi_detect.py
DEFAULT_IOU = 0.45     # NMS IoU, sama dengan raspi_detect.py
DEFAULT_IMGSZ = 640    # Pakai 640 (bukan 416) — ini di Colab/GPU,
                        # bukan di Raspi, jadi tidak perlu dikecilkan
                        # demi kecepatan. Resolusi lebih tinggi =
                        # box lebih presisi untuk video review.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Jalankan deteksi KRSRI Boneka pada video uji dan simpan hasilnya."
    )
    parser.add_argument("--video", required=True,
                         help="Path ke video .mp4 yang mau diuji")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                         help=f"Path ke model .pt (default: {DEFAULT_WEIGHTS})")
    parser.add_argument("--output", default=None,
                         help="Path output video. Default: <nama_video>_annotated.mp4")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                         help=f"Confidence threshold (default: {DEFAULT_CONF})")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU,
                         help=f"NMS IoU threshold (default: {DEFAULT_IOU})")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                         help=f"Resolusi inferensi (default: {DEFAULT_IMGSZ})")
    parser.add_argument("--show-fps", action="store_true",
                         help="Tampilkan FPS pemrosesan di pojok video")
    parser.add_argument("--log-csv", default=None,
                         help="Simpan log semua deteksi ke file CSV (opsional, untuk analisis lebih lanjut)")
    return parser.parse_args()


def load_model(weights_path: str) -> YOLO:
    """Load model dengan validasi path."""
    if not os.path.exists(weights_path):
        print(f"[ERROR] Model tidak ditemukan: {weights_path}")
        print("  Cek lagi path-nya, atau pakai --weights untuk menunjuk file lain.")
        sys.exit(1)

    print(f"[INFO] Loading model: {weights_path}")
    model = YOLO(weights_path)
    print("[INFO] Model siap.\n")
    return model


def draw_detections(frame: np.ndarray, result, conf_threshold: float) -> tuple:
    """
    Gambar bounding box, label kelas, dan confidence score pada frame.
    Return: (annotated_frame, list_deteksi_untuk_log)
    """
    detections = []

    if result.boxes is None or len(result.boxes) == 0:
        return frame, detections

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clsids = result.boxes.cls.cpu().numpy().astype(int)

    for box, conf, cls_id in zip(boxes, confs, clsids):
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = box.astype(int)
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        label_text = f"{label}: {conf:.1%}"

        # Bounding box — ketebalan 3px supaya jelas terlihat di video,
        # lebih tebal dari raspi_detect.py (2px) karena video biasanya
        # ditonton di layar lebih besar / direview frame-by-frame.
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Background untuk teks label supaya tetap terbaca di background apapun
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        label_y1 = max(y1 - th - baseline - 8, 0)
        cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 10, y1), color, -1)
        cv2.putText(
            frame, label_text, (x1 + 5, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        detections.append({
            "class": label,
            "class_id": int(cls_id),
            "confidence": float(conf),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    return frame, detections


def process_video(model: YOLO, args):
    """Loop utama: baca video frame demi frame, jalankan deteksi, tulis video output."""
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Tidak bisa membuka video: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video input  : {args.video}")
    print(f"[INFO] Resolusi     : {width}x{height} @ {fps:.1f} FPS")
    print(f"[INFO] Total frame  : {total_frames}")
    print(f"[INFO] Conf threshold: {args.conf:.0%}")
    print(f"[INFO] Output       : {args.output}\n")

    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    csv_file = None
    csv_writer = None
    if args.log_csv:
        import csv
        csv_file = open(args.log_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "timestamp_sec", "class", "confidence",
                              "x1", "y1", "x2", "y2"])

    frame_idx = 0
    total_detections = 0
    detections_per_class = {0: 0, 1: 0}
    confidences_per_class = {0: [], 1: []}
    frames_with_detection = 0

    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )[0]

        frame, detections = draw_detections(frame, result, args.conf)

        if detections:
            frames_with_detection += 1
        for d in detections:
            total_detections += 1
            detections_per_class[d["class_id"]] += 1
            confidences_per_class[d["class_id"]].append(d["confidence"])
            if csv_writer:
                ts_sec = frame_idx / fps
                csv_writer.writerow([
                    frame_idx, f"{ts_sec:.2f}", d["class"], f"{d['confidence']:.4f}",
                    *d["bbox"]
                ])

        if args.show_fps:
            elapsed = time.time() - t_start
            processing_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            cv2.putText(
                frame, f"Processing: {processing_fps:.1f} FPS",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            pct = 100 * frame_idx / total_frames if total_frames else 0
            print(f"\r[PROGRESS] Frame {frame_idx}/{total_frames} ({pct:.0f}%)", end="", flush=True)

    cap.release()
    writer.release()
    if csv_file:
        csv_file.close()

    elapsed_total = time.time() - t_start
    print(f"\n\n[INFO] Selesai dalam {elapsed_total:.1f} detik "
          f"({frame_idx/elapsed_total:.1f} FPS rata-rata pemrosesan)")

    # ── Ringkasan hasil ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 RINGKASAN DETEKSI")
    print("=" * 60)
    print(f"Total frame video        : {frame_idx}")
    print(f"Frame dengan deteksi      : {frames_with_detection} "
          f"({100*frames_with_detection/frame_idx:.1f}%)" if frame_idx else "")
    print(f"Total deteksi (semua frame): {total_detections}")

    for cls_id, name in enumerate(CLASS_NAMES):
        count = detections_per_class[cls_id]
        confs = confidences_per_class[cls_id]
        print(f"\n{name}:")
        print(f"  Jumlah deteksi : {count}")
        if confs:
            print(f"  Avg confidence : {np.mean(confs):.1%}")
            print(f"  Min confidence : {np.min(confs):.1%}")
            print(f"  Max confidence : {np.max(confs):.1%}")
        else:
            print("  (tidak ada deteksi untuk kelas ini)")

    if frames_with_detection < frame_idx * 0.5:
        print("\n⚠️  Catatan: kurang dari separuh frame punya deteksi.")
        print("   Ini wajar kalau boneka memang tidak selalu di frame,")
        print("   tapi kalau boneka SEHARUSNYA selalu terlihat dan masih")
        print("   sering tidak terdeteksi, coba turunkan --conf atau cek")
        print("   apakah kondisi video (lighting, sudut, jarak) jauh berbeda")
        print("   dari kondisi 450 foto training (lihat catatan di bawah).")

    print(f"\n✅ Video hasil tersimpan di: {args.output}")
    if args.log_csv:
        print(f"✅ Log CSV tersimpan di    : {args.log_csv}")

    print("\n💡 Yang perlu diperhatikan saat review video:")
    print("   - Apakah confidence stabil tinggi di semua sudut/jarak,")
    print("     atau turun drastis di kondisi tertentu?")
    print("   - Apakah ada false positive (box muncul padahal bukan boneka)?")
    print("   - Apakah boneka-asli vs boneka-dummy pernah tertukar labelnya?")
    print("   Video ini direkam di luar sesi 450 foto training (12 Jan 2024),")
    print("   jadi hasil di sini lebih mewakili kondisi lapangan sungguhan")
    print("   dibanding angka mAP dari valid set manapun di notebook training.")


def main():
    args = parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.video)
        args.output = f"{base}_annotated{ext}"

    if not os.path.exists(args.video):
        print(f"[ERROR] Video tidak ditemukan: {args.video}")
        sys.exit(1)

    model = load_model(args.weights)
    process_video(model, args)


if __name__ == "__main__":
    main()
