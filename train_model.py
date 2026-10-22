import pandas as pd
import joblib
import scipy.sparse as sp
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X_train = sp.load_npz("X_train.npz")
X_test = sp.load_npz("X_test.npz")
y_train = pd.read_csv("train.csv")["label_num"]
y_test = pd.read_csv("test.csv")["label_num"]

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Linear SVM": LinearSVC(class_weight="balanced"),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"{name}: test accuracy = {acc:.4f}")
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, filename)
    print(f"  saved {filename}")