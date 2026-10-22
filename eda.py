import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

# New features
df["length"] = df["message"].str.len()
df["word_count"] = df["message"].str.split().str.len()

print(df.groupby("label")[["length", "word_count"]].mean().round(1))

# Figure 1: class balance and message length
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(x="label", data=df, ax=axes[0])
axes[0].set_title("Spam vs Ham count")
sns.histplot(data=df, x="length", hue="label", bins=40, ax=axes[1])
axes[1].set_title("Message length")
sns.boxplot(x="label", y="word_count", data=df, ax=axes[2])
axes[2].set_title("Word count")
plt.tight_layout()
plt.savefig("eda_plots.png")
plt.show()

# Figure 2: top 10 words in each class
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, lab in zip(axes, ["ham", "spam"]):
    cv = CountVectorizer(stop_words="english")
    counts = cv.fit_transform(df[df["label"] == lab]["message"])
    freq = pd.Series(counts.sum(axis=0).A1, index=cv.get_feature_names_out())
    freq.nlargest(10).sort_values().plot(kind="barh", ax=ax)
    ax.set_title(f"Top words: {lab}")
plt.tight_layout()
plt.savefig("top_words.png")
plt.show()