#!/usr/bin/env python3
"""
KRSRI Boneka Detection — Video Test dengan Object Tracking + Temporal Voting
============================================================================
Versi lebih tangguh dari `test_video_detect.py`.

`test_video_detect.py` menjalankan deteksi frame-per-frame secara independen,
sehingga hasilnya kedip: box false-positive nongol sekejap di tempat acak,
dan label satu objek bisa gonta-ganti asli<->dummy antar frame.

Skrip ini memperbaiki gejala itu di sisi inferensi (BUKAN mengubah bobot
model — akar masalahnya ada di dataset, lihat README "Keterbatasan").
Empat lapis perbaikan:

  1. OBJECT TRACKING (ByteTrack) — memberi ID konsisten pada tiap objek
     antar frame, jadi kita bisa menilai objek sebagai kesatuan, bukan
     deteksi lepas per frame.
  2. TEMPORAL VOTING — kelas final tiap objek ditentukan lewat voting
     mayoritas (ditimbang confidence) sepanjang hidupnya. Ini menghapus
     label yang gonta-ganti dan objek berlabel ganda.
  3. SUPPRESI DETEKSI KEDIP — track yang hidup lebih pendek dari
     --min-track-len frame dibuang sebagai kemungkinan false positive.
  4. CLASS-AGNOSTIC NMS + BOX SMOOTHING — mencegah dua box beda-kelas
     menumpuk di objek sama, dan meredam getaran posisi box (EMA).

Cara pakai:
    python scripts/track_video_detect.py --video video/uji.mp4 \\
        --weights models/best_yolov8s.pt

Output: video beranotasi yang stabil + CSV log + ringkasan statistik.
Video ASLI tidak diubah.
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# ── Konfigurasi kelas (disamakan dgn test_video_detect.py) ───
CLASS_NAMES = ["boneka-asli", "boneka-dummy"]
CLASS_COLORS = {
    0: (0, 255, 0),    # boneka-asli  -> Hijau (BGR)
    1: (0, 0, 255),    # boneka-dummy -> Merah
}

DEFAULT_WEIGHTS = "models/best_yolov8s.pt"
DEFAULT_CONF = 0.50
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640
DEFAULT_MIN_TRACK_LEN = 5     # track < ini (frame) dibuang sbg FP kedip
DEFAULT_SMOOTH_ALPHA = 0.5    # EMA box smoothing; 1.0 = tanpa smoothing


def parse_args():
    p = argparse.ArgumentParser(
        description="Deteksi KRSRI Boneka pada video dgn tracking + temporal voting (stabil)."
    )
    p.add_argument("--video", required=True, help="Path video .mp4 yang diuji")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS,
                   help=f"Path model .pt (default: {DEFAULT_WEIGHTS})")
    p.add_argument("--output", default=None,
                   help="Path output video (default: <nama>_tracked.mp4)")
    p.add_argument("--conf", type=float, default=DEFAULT_CONF,
                   help=f"Confidence threshold (default: {DEFAULT_CONF})")
    p.add_argument("--iou", type=float, default=DEFAULT_IOU,
                   help=f"NMS IoU threshold (default: {DEFAULT_IOU})")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                   help=f"Resolusi inferensi (default: {DEFAULT_IMGSZ})")
    p.add_argument("--min-track-len", type=int, default=DEFAULT_MIN_TRACK_LEN,
                   help=f"Buang track lebih pendek dari N frame (default: {DEFAULT_MIN_TRACK_LEN})")
    p.add_argument("--smooth-alpha", type=float, default=DEFAULT_SMOOTH_ALPHA,
                   help=f"EMA box smoothing 0-1, 1=off (default: {DEFAULT_SMOOTH_ALPHA})")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="Config tracker Ultralytics (bytetrack.yaml / botsort.yaml)")
    p.add_argument("--log-csv", default=None,
                   help="Simpan log deteksi (setelah voting) ke CSV")
    return p.parse_args()


def load_model(weights_path: str) -> YOLO:
    if not os.path.exists(weights_path):
        print(f"[ERROR] Model tidak ditemukan: {weights_path}")
        sys.exit(1)
    print(f"[INFO] Loading model: {weights_path}")
    model = YOLO(weights_path)
    print("[INFO] Model siap.\n")
    return model


def pass1_collect_tracks(model, args, cap, total_frames):
    """
    PASS 1: jalankan tracking frame-demi-frame, kumpulkan riwayat tiap track.
    Return: dict track_id -> {"frames": {frame_idx: (box, conf, cls)}}
    """
    tracks = defaultdict(lambda: {"frames": {}})
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.track(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            persist=True,
            tracker=args.tracker,
            agnostic_nms=True,     # cegah 2 box beda-kelas menumpuk di objek sama
            verbose=False,
        )[0]

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clsids = result.boxes.cls.cpu().numpy().astype(int)
            ids = result.boxes.id.cpu().numpy().astype(int)
            for box, conf, cls_id, tid in zip(boxes, confs, clsids, ids):
                tracks[tid]["frames"][frame_idx] = (
                    box.astype(float), float(conf), int(cls_id)
                )

        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            pct = 100 * frame_idx / total_frames if total_frames else 0
            print(f"\r[PASS 1/2] Tracking frame {frame_idx}/{total_frames} ({pct:.0f}%)",
                  end="", flush=True)

    dt = time.time() - t0
    print(f"\n[INFO] Pass 1 selesai dalam {dt:.1f} dtk "
          f"({frame_idx/dt:.1f} FPS). {len(tracks)} track mentah.\n")
    return tracks, frame_idx


def resolve_tracks(tracks, min_track_len, smooth_alpha):
    """
    Post-processing: buang track kedip, tetapkan 1 kelas per track lewat
    voting mayoritas ditimbang confidence, dan haluskan box (EMA).
    Return: per_frame dict frame_idx -> list of (box, conf_repr, voted_cls, tid)
    plus statistik.
    """
    kept, dropped = 0, 0
    per_frame = defaultdict(list)

    for tid, data in tracks.items():
        frames = data["frames"]
        length = len(frames)
        if length < min_track_len:
            dropped += 1
            continue
        kept += 1

        # ── Temporal voting: kelas = mayoritas ditimbang confidence ──
        vote = defaultdict(float)
        for _, conf, cls_id in frames.values():
            vote[cls_id] += conf
        voted_cls = max(vote, key=vote.get)

        # confidence representatif = rata-rata conf pada frame yg kelasnya = voted
        matching_confs = [c for _, c, cid in frames.values() if cid == voted_cls]
        conf_repr = float(np.mean(matching_confs)) if matching_confs else 0.0

        # ── Box smoothing (EMA) sepanjang urutan frame track ──
        ema = None
        for fidx in sorted(frames.keys()):
            box, conf, _ = frames[fidx]
            if ema is None:
                ema = box.copy()
            else:
                ema = smooth_alpha * box + (1 - smooth_alpha) * ema
            per_frame[fidx].append(
                (ema.astype(int), conf_repr, voted_cls, tid)
            )

    return per_frame, kept, dropped


def draw_frame(frame, dets):
    """Gambar deteksi (sudah stabil) pada frame. dets: list (box, conf, cls, tid)."""
    for box, conf, cls_id, tid in dets:
        x1, y1, x2, y2 = box
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        label_text = f"#{tid} {label}: {conf:.0%}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y1 = max(y1 - th - baseline - 8, 0)
        cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def pass2_render(args, per_frame, fps, width, height, n_frames):
    """PASS 2: baca ulang video, gambar deteksi stabil, tulis video + CSV."""
    cap = cv2.VideoCapture(args.video)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    csv_file = csv_writer = None
    if args.log_csv:
        import csv
        csv_file = open(args.log_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame", "timestamp_sec", "track_id", "class",
                             "confidence", "x1", "y1", "x2", "y2"])

    frame_idx = 0
    frames_with_det = 0
    per_class = {0: 0, 1: 0}
    conf_per_class = {0: [], 1: []}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = per_frame.get(frame_idx, [])
        if dets:
            frames_with_det += 1
        for box, conf, cls_id, tid in dets:
            per_class[cls_id] += 1
            conf_per_class[cls_id].append(conf)
            if csv_writer:
                x1, y1, x2, y2 = box
                csv_writer.writerow([frame_idx, f"{frame_idx/fps:.2f}", tid,
                                     CLASS_NAMES[cls_id], f"{conf:.4f}",
                                     x1, y1, x2, y2])
        draw_frame(frame, dets)
        writer.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == n_frames:
            pct = 100 * frame_idx / n_frames if n_frames else 0
            print(f"\r[PASS 2/2] Render frame {frame_idx}/{n_frames} ({pct:.0f}%)",
                  end="", flush=True)

    cap.release()
    writer.release()
    if csv_file:
        csv_file.close()
    return frames_with_det, per_class, conf_per_class, frame_idx


def main():
    args = parse_args()
    if args.output is None:
        base, ext = os.path.splitext(args.video)
        args.output = f"{base}_tracked{ext}"
    if not os.path.exists(args.video):
        print(f"[ERROR] Video tidak ditemukan: {args.video}")
        sys.exit(1)

    model = load_model(args.weights)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Tidak bisa membuka video: {args.video}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video input   : {args.video}")
    print(f"[INFO] Resolusi      : {width}x{height} @ {fps:.1f} FPS")
    print(f"[INFO] Total frame   : {total_frames}")
    print(f"[INFO] Conf / IoU    : {args.conf:.0%} / {args.iou}")
    print(f"[INFO] Min track len : {args.min_track_len} frame")
    print(f"[INFO] Tracker       : {args.tracker}")
    print(f"[INFO] Output        : {args.output}\n")

    # PASS 1: kumpulkan track
    tracks, n_frames = pass1_collect_tracks(model, args, cap, total_frames)
    cap.release()

    # POST: voting + suppresi kedip + smoothing
    per_frame, kept, dropped = resolve_tracks(
        tracks, args.min_track_len, args.smooth_alpha)
    print(f"[INFO] Track dipertahankan: {kept} | dibuang (kedip <{args.min_track_len} frame): {dropped}\n")

    # PASS 2: render
    fwd, per_class, conf_pc, frame_idx = pass2_render(
        args, per_frame, fps, width, height, n_frames)

    # ── Ringkasan ──
    print("\n\n" + "=" * 60)
    print("RINGKASAN DETEKSI (setelah tracking + temporal voting)")
    print("=" * 60)
    print(f"Total frame              : {frame_idx}")
    print(f"Frame dengan deteksi     : {fwd} ({100*fwd/frame_idx:.1f}%)")
    print(f"Objek unik (track valid) : {kept}")
    for cls_id, name in enumerate(CLASS_NAMES):
        confs = conf_pc[cls_id]
        print(f"\n{name}:")
        print(f"  Frame-deteksi  : {per_class[cls_id]}")
        if confs:
            print(f"  Avg confidence : {np.mean(confs):.1%}")
    print(f"\nVideo hasil tersimpan di: {args.output}")
    if args.log_csv:
        print(f"Log CSV tersimpan di    : {args.log_csv}")


if __name__ == "__main__":
    main()
