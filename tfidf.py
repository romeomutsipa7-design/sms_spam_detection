import pandas as pd
import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Messages that became empty after cleaning are read as NaN
train["clean"] = train["clean"].fillna("")
test["clean"] = test["clean"].fillna("")

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)

# Learn the vocabulary from TRAINING data only
X_train = vectorizer.fit_transform(train["clean"])
X_test = vectorizer.transform(test["clean"])

print("Train matrix shape:", X_train.shape)
print("Test matrix shape:", X_test.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))

# Highest-weighted terms in the first training message
row = X_train[0].toarray()[0]
terms = vectorizer.get_feature_names_out()
top = row.argsort()[::-1][:5]
print("\nMessage:", train["clean"][0])
for i in top:
    if row[i] > 0:
        print(f"  {terms[i]}: {row[i]:.3f}")

# Save for the next step
joblib.dump(vectorizer, "vectorizer.pkl")
sp.save_npz("X_train.npz", X_train)
sp.save_npz("X_test.npz", X_test)
print("\nSaved vectorizer.pkl, X_train.npz, X_test.npz")