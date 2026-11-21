# MLflow Lab 1: Handwritten Digit Classification

This lab uses scikit-learn's built-in Digits dataset and an RBF support vector
classifier. A `StandardScaler` inside the pipeline is fitted separately in each
cross-validation fold. Five-fold stratified cross-validation compares three
values of `C` using mean macro F1 on the training split. The selected
configuration is refitted on the training split and evaluated on a stratified
20% holdout.

## Results

| SVM C | Mean CV macro F1 |
| --- | ---: |
| 0.1 | ~0.93 |
| 1.0 | 0.98121 |
| 10.0 | 0.97982 |

`C=1.0` had the highest mean CV score. The margin over `C=10.0` was about
0.00139. The final model scored 97.50% accuracy and 0.9748 macro F1 on the
held-out set. Reloading the saved model reproduced both logged metrics.

## Train and inspect the runs

From the project root, use Python 3.13, the version used for the verified run.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
```

Training logs three comparison runs and one final evaluation to a local SQLite
database. The final run uses scikit-learn autologging and explicitly logs the
model with an input example and a signature. Use the run ID and model URI
printed by your training run for the commands below.

```bash
python test_logged_model.py "<run-id>" "<model-uri>"
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Open `http://127.0.0.1:5001` and go to **Handwritten digits classification**
→ **Training runs**. Compare `cv_macro_f1_mean` across the CV runs.

## Serve and verify a prediction

From the project root with the virtual environment active, run:

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow models serve -m "<model-uri>" --env-manager local --host 127.0.0.1 --port 5002
```

In a second terminal with the same environment active, send the first
held-out row to the prediction endpoint:

```python
import json
from urllib.request import Request, urlopen

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits(as_frame=True)
_, X_test, _, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, stratify=digits.target, random_state=42
)
payload = {"dataframe_split": X_test.head(1).to_dict(orient="split")}
request = Request(
    "http://127.0.0.1:5002/invocations",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request, timeout=10) as response:
    print(f"HTTP {response.status}: {response.read().decode()}")
print(f"Actual digit: {y_test.iloc[0]}")
```

The verified request returned HTTP 200 with `{"predictions": [5]}`. The actual
digit was `5`. Serving uses the local Python environment and requires no Docker.
