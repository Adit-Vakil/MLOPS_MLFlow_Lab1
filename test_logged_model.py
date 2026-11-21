import argparse
from math import isclose

import mlflow
import mlflow.pyfunc
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("model_uri")
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    recorded_metrics = mlflow.get_run(args.run_id).data.metrics
    digits = load_digits(as_frame=True)
    _, X_test, _, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, stratify=digits.target, random_state=42
    )

    model = mlflow.pyfunc.load_model(args.model_uri)
    predictions = model.predict(X_test)
    observed_metrics = {
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_macro_f1": f1_score(y_test, predictions, average="macro"),
    }
    for metric, observed in observed_metrics.items():
        recorded = recorded_metrics[metric]
        if not isclose(observed, recorded, rel_tol=0, abs_tol=1e-12):
            raise AssertionError(f"{metric}: reloaded model={observed}, logged={recorded}")
        print(f"{metric}: {observed:.4f} matches the logged run")


if __name__ == "__main__":
    main()
