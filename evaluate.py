import pandas as pd
import joblib
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

X_test = sp.load_npz("X_test.npz")
test = pd.read_csv("test.csv")
test["clean"] = test["clean"].fillna("")
y_test = test["label_num"]

files = {
    "Naive Bayes": "naive_bayes.pkl",
    "Logistic Regression": "logistic_regression.pkl",
    "Linear SVM": "linear_svm.pkl",
}

results = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, f) in zip(axes, files.items()):
    model = joblib.load(f)
    pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    })

    print("=" * 50)
    print(name)
    print(classification_report(y_test, pred, target_names=["ham", "spam"]))

    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"]).plot(ax=ax, colorbar=False)
    ax.set_title(name)

plt.tight_layout()
plt.savefig("confusion_matrices.png")
plt.show()

table = pd.DataFrame(results).round(4)
print("\nCOMPARISON TABLE")
print(table.to_string(index=False))
table.to_csv("model_results.csv", index=False)
print("\nSaved confusion_matrices.png and model_results.csv")