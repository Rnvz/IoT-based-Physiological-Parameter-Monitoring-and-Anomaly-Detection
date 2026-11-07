"""
ml.src.train_physio_synthetic:
Adapter script yang mengarahkan pemrosesan dan pelatihan model ke train_isolation_forest
dengan dataset non-statis (Training_Dataset.csv).
"""

import os
import sys
import argparse
from ml.src.train_isolation_forest import train_and_evaluate

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_dataset = os.path.join(project_root, "Dataset", "Training_Dataset.csv")
    default_model_dir = os.path.join(project_root, "ml", "models")

    parser = argparse.ArgumentParser(description="Proses & Latih Model ML non-statis")
    parser.add_argument("--input", "--dataset", type=str, default=default_dataset, help="Path ke dataset CSV")
    parser.add_argument("--output_data", type=str, default=None, help="Direktori output data (opsional)")
    parser.add_argument("--output_model", type=str, default=default_model_dir, help="Direktori output model")
    parser.add_argument("--n_estimators", type=int, default=100, help="Jumlah estimator")
    parser.add_argument("--contamination", type=float, default=0.01, help="Contamination parameter")
    parser.add_argument("--window", type=int, default=15, help="Ukuran window rolling temporal")

    args = parser.parse_args()
    train_and_evaluate(
        input_csv=args.input,
        output_model_dir=args.output_model,
        n_estimators=args.n_estimators,
        contamination=args.contamination,
        window=args.window
    )
