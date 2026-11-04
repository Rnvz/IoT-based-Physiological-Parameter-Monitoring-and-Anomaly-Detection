# PANDUAN LINGKUNGAN PELATIHAN DEEP LEARNING (CPU PYTORCH)

> **STATUS**: DOKUMENTASI LINGKUNGAN SEMENTARA (EPHEMERAL / VOLATIL)  
> **Catatan Penting**: Direktori `/tmp` pada sistem Linux umumnya berbasis RAM (`tmpfs`) atau dibersihkan saat sistem me-reboot. Lingkungan virtual ini bersifat volatil dan ditujukan khusus untuk eksperimen evaluasi pelatihan DenseAE tanpa membebani repositori utama maupun project `.venv`.

---

## 1. Spesifikasi Perangkat Lunak & Versi
- **Sistem Operasi**: Linux (x86_64)
- **Python**: `3.14.8`
- **PyTorch**: `2.15.0+cpu`
- **Indeks Rilis PyTorch**: `https://download.pytorch.org/whl/cpu`
- **Paket Terkait**:
  - `torch==2.15.0+cpu`
  - `filelock==4.2.4`
  - `jinja2==3.1.6`
  - `networkx==3.7`
  - `sympy==1.14.0`
  - `typing_extensions==4.16.0`

---

## 2. Langkah Persis Pembuatan Lingkungan Sementara

Jalankan perintah berikut di terminal:

```bash
# 1. Pastikan direktori temporer tersedia
export TMPDIR=/tmp

# 2. Buat virtual environment terisolasi di /tmp
python3 -m venv /tmp/dl_train_venv

# 3. Pasang PyTorch CPU murni langsung dari index resmi PyTorch (timeout 600)
/tmp/dl_train_venv/bin/pip install torch \
  --index-url https://download.pytorch.org/whl/cpu \
  --timeout 600
```

---

## 3. Eksekusi Skrip dengan Mengaitkan `.venv` Project

Untuk mengakses pustaka ilmiah bawaan project (`numpy`, `pandas`, `scikit-learn`, `scipy`, `pytest`) bersamaan dengan `torch` di venv sementara:

```bash
# Menjalankan script evaluasi simulator
PYTHONPATH=.venv/lib/python3.14/site-packages:. /tmp/dl_train_venv/bin/python ml_pipeline/src/evaluate_dl_simulator.py

# Menjalankan test suite
PYTHONPATH=backend:.venv/lib/python3.14/site-packages:. /tmp/dl_train_venv/bin/python -m pytest tests/
```

---

## 4. Karakteristik Volatilitas `/tmp`
1. **/tmp Bersifat Volatil**: Jika perangkat atau container di-reboot, direktori `/tmp/dl_train_venv` akan terhapus.
2. **Reproduksibilitas**: Langkah pembuatan pada Bagian 2 di atas membutuhkan waktu $\approx 30$ detik untuk diunduh dan dipasang kembali jika venv terhapus.
3. **Integritas Repositori Proyek**: File di `.venv` dan repositori Git proyek tetap murni tanpa ada dependensi biner Torch yang berukuran besar.
