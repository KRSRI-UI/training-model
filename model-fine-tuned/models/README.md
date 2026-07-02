# Trained Models

Model hasil training YOLOv8s pada dataset boneka KRSRI, tersedia dalam tiga format:

| File | Format | Kegunaan |
|---|---|---|
| `best_yolov8s.pt` | PyTorch | Format native Ultralytics — evaluasi, inferensi di GPU, atau basis untuk fine-tuning lanjutan |
| `best_yolov8s.onnx` | ONNX | Format portable untuk inferensi lintas platform/runtime |
| `best_ncnn_model/` | NCNN | Dioptimalkan untuk ARM CPU — digunakan oleh `scripts/raspi_detect.py` untuk deployment di Raspberry Pi 4 |
| `data.yaml` | Config | Konfigurasi kelas (`boneka-asli`, `boneka-dummy`) dan path dataset yang dipakai saat training model ini |

Model dihasilkan melalui training dua fase (frozen backbone, lalu full fine-tune) dari basis pretrained COCO. Proses lengkap — termasuk augmentasi, hyperparameter, dan evaluasi — didokumentasikan di [`notebooks/krsri_boneka_training.ipynb`](../notebooks/krsri_boneka_training.ipynb).

Lihat [README utama](../README.md#hasil) untuk metrik evaluasi, dan bagian [Keterbatasan yang Diketahui](../README.md#keterbatasan-yang-diketahui) untuk konteks penting sebelum menggunakan model ini di luar kondisi dataset training.
