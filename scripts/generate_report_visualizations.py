import os
import tempfile
# Set MPLCONFIGDIR ke temporary folder agar tidak memicu permission error di Windows
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Konfigurasi gaya visualisasi profesional
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = False

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_fig1_confusion_matrices():
    """Figure 1: Side-by-side Confusion Matrices."""
    cm_if = np.array([[3562, 339], [1516, 10888]])
    cm_ae = np.array([[3580, 321], [1378, 11026]])

    labels_if = [
        [f"TN\n3,562\n(91.31%)", f"FP\n339\n(8.69%)"],
        [f"FN\n1,516\n(12.22%)", f"TP\n10,888\n(87.78%)"]
    ]
    labels_ae = [
        [f"TN\n3,580\n(91.77%)", f"FP\n321\n(8.23%)"],
        [f"FN\n1,378\n(11.11%)", f"TP\n11,026\n(88.89%)"]
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Subplot 1: Isolation Forest
    sns.heatmap(cm_if, annot=labels_if, fmt='', cmap='Blues', cbar=True,
                ax=axes[0], annot_kws={"size": 12, "weight": "bold"},
                xticklabels=['Normal (0)', 'Anomali (1)'],
                yticklabels=['Normal (0)', 'Anomali (1)'])
    axes[0].set_title("Isolation Forest (Classical ML)\nTest N = 16,305", fontsize=14, weight='bold', pad=12)
    axes[0].set_xlabel("Prediksi Model", fontsize=12, labelpad=8)
    axes[0].set_ylabel("Aktual (Ground Truth)", fontsize=12, labelpad=8)

    # Subplot 2: Dense Autoencoder
    sns.heatmap(cm_ae, annot=labels_ae, fmt='', cmap='Purples', cbar=True,
                ax=axes[1], annot_kws={"size": 12, "weight": "bold"},
                xticklabels=['Normal (0)', 'Anomali (1)'],
                yticklabels=['Normal (0)', 'Anomali (1)'])
    axes[1].set_title("Dense Autoencoder (Deep Learning)\nTest N = 16,305", fontsize=14, weight='bold', pad=12)
    axes[1].set_xlabel("Prediksi Model", fontsize=12, labelpad=8)
    axes[1].set_ylabel("Aktual (Ground Truth)", fontsize=12, labelpad=8)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig1_confusion_matrices.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated: {out_path}")

def generate_fig2_metrics_comparison():
    """Figure 2: Grouped Bar Chart of 6 Key Metrics."""
    metrics = ['ROC-AUC', 'PR-AUC', 'Recall', 'Precision', 'F1-Score', 'Balanced Acc']
    if_scores = [94.51, 96.25, 87.78, 96.98, 92.15, 89.54]
    ae_scores = [94.18, 97.52, 88.89, 97.17, 92.85, 90.33]

    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)

    rects1 = ax.bar(x - width/2, if_scores, width, label='Isolation Forest (Classical ML)',
                    color='#2563eb', edgecolor='#1e40af', alpha=0.9, zorder=3)
    rects2 = ax.bar(x + width/2, ae_scores, width, label='Dense Autoencoder (Deep Learning)',
                    color='#9333ea', edgecolor='#6b21a8', alpha=0.9, zorder=3)

    ax.set_title("Perbandingan Metrik Evaluasi Test Set (N = 16.305)\nIsolation Forest vs Dense Autoencoder",
                 fontsize=15, weight='bold', pad=15)
    ax.set_ylabel("Skor Evaluasi (%)", fontsize=12, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, weight='bold')
    ax.set_ylim(80, 102)
    ax.legend(loc='lower right', fontsize=11, frameon=True, facecolor='white', framealpha=0.95)
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    # Tambahkan angka di atas bar
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, weight='bold', color='#1e40af')

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, weight='bold', color='#6b21a8')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig2_metrics_comparison.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated: {out_path}")

def generate_fig3_domain_breakdown():
    """Figure 3: Breakdown Across gourangomandal (IoMT) vs engrarri21 (IoT Stream)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    metrics = ['Actual FPR', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.35

    # Subplot 1: gourangomandal (IoMT Lab)
    if_gourango = [3.89, 87.16, 92.28]
    ae_gourango = [3.43, 84.03, 90.57]

    b1 = axes[0].bar(x - width/2, if_gourango, width, label='Isolation Forest', color='#3b82f6', edgecolor='#1d4ed8', zorder=3)
    b2 = axes[0].bar(x + width/2, ae_gourango, width, label='Dense Autoencoder', color='#a855f7', edgecolor='#7e22ce', zorder=3)
    axes[0].set_title("Dataset Stasioner: gourangomandal (IoMT)\nTotal: 12.000 sampel (Norm: 3.705, Abn: 8.295)",
                      fontsize=12, weight='bold', pad=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, fontsize=11, weight='bold')
    axes[0].set_ylabel("Persentase (%)", fontsize=11, weight='bold')
    axes[0].set_ylim(0, 110)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    axes[0].legend(loc='upper left', frameon=True)

    for b in b1:
        h = b.get_height()
        axes[0].annotate(f"{h:.1f}%", (b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                         textcoords="offset points", ha='center', va='bottom', fontsize=9, weight='bold')
    for b in b2:
        h = b.get_height()
        axes[0].annotate(f"{h:.1f}%", (b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                         textcoords="offset points", ha='center', va='bottom', fontsize=9, weight='bold')

    # Subplot 2: engrarri21 (Dynamic IoT Stream)
    if_engrarri = [99.49, 89.02, 91.89]
    ae_engrarri = [98.98, 98.71, 97.05]

    b3 = axes[1].bar(x - width/2, if_engrarri, width, label='Isolation Forest', color='#3b82f6', edgecolor='#1d4ed8', zorder=3)
    b4 = axes[1].bar(x + width/2, ae_engrarri, width, label='Dense Autoencoder', color='#a855f7', edgecolor='#7e22ce', zorder=3)
    axes[1].set_title("Dataset Dinamis: engrarri21 (Continuous IoT Stream)\nTotal: 4.305 sampel (Norm: 196, Abn: 4.109)",
                      fontsize=12, weight='bold', pad=10)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics, fontsize=11, weight='bold')
    axes[1].set_ylabel("Persentase (%)", fontsize=11, weight='bold')
    axes[1].set_ylim(0, 115)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    axes[1].legend(loc='lower right', frameon=True)

    for b in b3:
        h = b.get_height()
        axes[1].annotate(f"{h:.1f}%", (b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                         textcoords="offset points", ha='center', va='bottom', fontsize=9, weight='bold')
    for b in b4:
        h = b.get_height()
        axes[1].annotate(f"{h:.1f}%", (b.get_x() + b.get_width()/2, h), xytext=(0, 3),
                         textcoords="offset points", ha='center', va='bottom', fontsize=9, weight='bold')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig3_domain_breakdown.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated: {out_path}")

def generate_fig4_hardware_tradeoff():
    """Figure 4: Hardware Efficiency and Latency Trade-off."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=300)

    # Subplot 1: Model Size (Log Scale)
    models = ['Isolation Forest\n(Tree Ensemble)', 'Dense Autoencoder\n(Neural Weights)']
    sizes_kb = [1247.3, 9.15]  # KB
    colors = ['#f97316', '#10b981']

    bars = axes[0].bar(models, sizes_kb, color=colors, edgecolor='#333333', width=0.5, zorder=3)
    axes[0].set_yscale('log')
    axes[0].set_title("Ukuran Bobot Komputasi Model\n(Skala Logaritmik KB)", fontsize=13, weight='bold', pad=12)
    axes[0].set_ylabel("Ukuran Memori (KB, Log Scale)", fontsize=11, weight='bold')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    axes[0].set_ylim(1, 4000)

    # Garis batas SRAM ESP32 (320 KB)
    axes[0].axhline(y=320, color='#dc2626', linestyle='--', linewidth=2, zorder=4)
    axes[0].text(0.5, 360, "Kapasitas Maksimal SRAM ESP32 (320 KB)", color='#dc2626',
                 fontsize=10.5, weight='bold', ha='center', backgroundcolor='white')

    for bar, val in zip(bars, sizes_kb):
        axes[0].annotate(f"{val:,.2f} KB",
                         xy=(bar.get_x() + bar.get_width() / 2, val),
                         xytext=(0, 6), textcoords="offset points",
                         ha='center', va='bottom', fontsize=11, weight='bold')

    # Subplot 2: Inference Latency
    latencies = [1.50, 0.78]
    bars_lat = axes[1].bar(models, latencies, color=['#3b82f6', '#8b5cf6'], edgecolor='#333333', width=0.5, zorder=3)
    axes[1].set_title("Estimasi Latensi Inferensi Per Sampel\n(CPU Server)", fontsize=13, weight='bold', pad=12)
    axes[1].set_ylabel("Latensi Inferensi (Milidetik, ms)", fontsize=11, weight='bold')
    axes[1].set_ylim(0, 2.0)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    for bar, val in zip(bars_lat, latencies):
        axes[1].annotate(f"{val:.2f} ms",
                         xy=(bar.get_x() + bar.get_width() / 2, val),
                         xytext=(0, 6), textcoords="offset points",
                         ha='center', va='bottom', fontsize=11, weight='bold')

    # Label perbandingan kecepatan
    axes[1].annotate("Autoencoder ~1.9x Lebih Cepat\n(Vectorized SIMD)",
                     xy=(1, 0.78), xytext=(0.5, 1.4),
                     arrowprops=dict(facecolor='#8b5cf6', shrink=0.08, width=1.5, headwidth=8),
                     fontsize=10.5, weight='bold', color='#6d28d9', ha='center', backgroundcolor='white')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'fig4_hardware_tradeoff.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated: {out_path}")

def main():
    print("Membangkitkan visualisasi grafik evaluasi model...")
    generate_fig1_confusion_matrices()
    generate_fig2_metrics_comparison()
    generate_fig3_domain_breakdown()
    generate_fig4_hardware_tradeoff()
    print("Seluruh visualisasi berhasil disimpan di folder:", OUTPUT_DIR)

if __name__ == '__main__':
    main()
