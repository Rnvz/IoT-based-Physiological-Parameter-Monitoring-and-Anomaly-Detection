# Deep Learning (DL) Pipeline

Direktori ini berisi seluruh komponen algoritma **Deep Learning (DL)** berbasis **Dense Autoencoder & Variational Autoencoder (VAE)** untuk deteksi anomali telemetri IoT edge.

## Struktur Direktori
- `src/`: Skrip arsitektur neural network PyTorch, pipeline pelatihan, evaluasi, dan ekspor bobot.
  - `models.py`: Arsitektur DenseAutoencoder dan DenseVAE (< 10.000 parameter, edge-friendly).
  - `train.py`: Loop pelatihan PyTorch dengan early stopping.
  - `evaluate.py`: Perhitungan MSE reconstruction error, ROC-AUC, dan PR-AUC.
  - `export.py`: Ekspor model ke ONNX dan bobot NumPy (`model_weights.npz`).
  - `run_experiment.py`: CLI runner eksperimen perbandingan model.
  - `evaluate_dl_simulator.py`: Evaluasi komparatif multi-skenario simulator.
- `artifacts/`: Artefak model hasil pelatihan, bobot NumPy, file ONNX, dan laporan model card.
  - `dl_v1/`: Model produksi ONNX & NumPy forward.
  - `dl_physio_trained/`: Model PyTorch 5D yang dilatih dengan fitur fisiologis + suhu/kelembaban ruangan.
- `requirements.txt`: Dependensi spesifik deep learning (PyTorch, ONNX).
