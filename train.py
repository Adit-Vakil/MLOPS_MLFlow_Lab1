import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_model(c):
    return make_pipeline(StandardScaler(), SVC(C=c, kernel="rbf"))


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Handwritten digits classification")

    digits = load_digits(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.2, stratify=digits.target, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores_by_c = {}
    for c in (0.1, 1.0, 10.0):
        with mlflow.start_run(run_name=f"CV SVM C={c}"):
            scores = cross_val_score(
                make_model(c), X_train, y_train, cv=cv, scoring="f1_macro"
            )
            scores_by_c[c] = float(scores.mean())
            mlflow.log_params({"C": c, "kernel": "rbf", "cv_folds": cv.n_splits})
            mlflow.log_metrics(
                {"cv_macro_f1_mean": scores_by_c[c], "cv_macro_f1_std": float(scores.std())}
            )

    best_c = max(scores_by_c, key=scores_by_c.__getitem__)
    mlflow.sklearn.autolog(log_models=False, log_post_training_metrics=False)

    with mlflow.start_run(run_name="Final evaluation") as run:
        model = make_model(best_c)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = {
            "test_accuracy": accuracy_score(y_test, predictions),
            "test_macro_f1": f1_score(y_test, predictions, average="macro"),
        }
        mlflow.log_param("selected_C", best_c)
        mlflow.log_metrics(metrics)

        input_example = X_train.head(5)
        logged_model = mlflow.sklearn.log_model(
            model,
            name="digits-svm",
            input_example=input_example,
            signature=infer_signature(input_example, model.predict(input_example)),
        )

        print(f"Selected C: {best_c}")
        print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Test macro F1: {metrics['test_macro_f1']:.4f}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model URI: {logged_model.model_uri}")


if __name__ == "__main__":
    main()
