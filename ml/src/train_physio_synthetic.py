"""
ml.src.train_physio_synthetic:
Adapter script yang mengarahkan pemrosesan dan pelatihan model ke train_isolation_forest.
"""

import os
import sys
import argparse
from ml.src.train_isolation_forest import main as train_isolation_forest_main, train_and_evaluate as _train_and_eval

def train_and_evaluate(
    train_path: str = None,
    val_path: str = None,
    test_path: str = None,
    model_path: str = None,
    report_path: str = None,
    n_estimators: int = 150,
    **kwargs
):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    train_p = train_path or kwargs.get("input_csv") or os.path.join(project_root, "ml", "data", "features_train.csv")
    val_p = val_path or os.path.join(project_root, "ml", "data", "features_val.csv")
    test_p = test_path or os.path.join(project_root, "ml", "data", "features_test.csv")
    model_dir = kwargs.get("output_model_dir")
    if model_dir:
        model_p = os.path.join(model_dir, "isolation_forest_model.joblib")
        report_p = os.path.join(model_dir, "evaluation_report.txt")
    else:
        model_p = model_path or os.path.join(project_root, "ml", "models", "isolation_forest_model.joblib")
        report_p = report_path or os.path.join(project_root, "ml", "models", "evaluation_report.txt")

    return train_isolation_forest_main(
        train_path=train_p,
        val_path=val_p,
        test_path=test_p,
        model_path=model_p,
        report_path=report_p,
        n_estimators=n_estimators
    )

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_train = os.path.join(project_root, "ml", "data", "features_train.csv")
    default_val = os.path.join(project_root, "ml", "data", "features_val.csv")
    default_test = os.path.join(project_root, "ml", "data", "features_test.csv")
    default_model = os.path.join(project_root, "ml", "models", "isolation_forest_model.joblib")
    default_report = os.path.join(project_root, "ml", "models", "evaluation_report.txt")

    parser = argparse.ArgumentParser(description="Proses & Latih Model ML non-statis")
    parser.add_argument("--train", "--input", "--dataset", type=str, default=default_train, help="Path ke dataset CSV")
    parser.add_argument("--val", type=str, default=default_val, help="Path ke val dataset")
    parser.add_argument("--test", type=str, default=default_test, help="Path ke test dataset")
    parser.add_argument("--output_model", "--model", type=str, default=default_model, help="Path output model")
    parser.add_argument("--output_report", "--report", type=str, default=default_report, help="Path output report")
    parser.add_argument("--n_estimators", type=int, default=150, help="Jumlah estimator")

    args = parser.parse_args()
    train_and_evaluate(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        model_path=args.output_model,
        report_path=args.output_report,
        n_estimators=args.n_estimators
    )
