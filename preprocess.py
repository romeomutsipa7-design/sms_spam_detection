import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

# Convert labels to numbers: ham = 0, spam = 1
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)  # links
    text = re.sub(r"\d+", " num ", text)               # numbers
    text = re.sub(r"[^a-z\s]", " ", text)              # punctuation, symbols
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)

df["clean"] = df["message"].apply(clean_text)

# Before / after examples
for i in range(5):
    print("BEFORE:", df["message"][i])
    print("AFTER: ", df["clean"][i])
    print()

# Check for empty messages after cleaning
print("Empty after cleaning:", (df["clean"].str.strip() == "").sum())

# Split BEFORE building features (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label_num"],
    test_size=0.2, random_state=42, stratify=df["label_num"]
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))
print("Spam share in train:", round(y_train.mean(), 3))
print("Spam share in test:", round(y_test.mean(), 3))

# Save for the next steps
pd.DataFrame({"clean": X_train, "label_num": y_train}).to_csv("train.csv", index=False)
pd.DataFrame({"clean": X_test, "label_num": y_test}).to_csv("test.csv", index=False)
print("Saved train.csv and test.csv")