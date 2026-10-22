import re
import joblib
import tkinter as tk
from tkinter import messagebox
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("logistic_regression.pkl")

# Must be IDENTICAL to the cleaning in preprocess.py
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]
    return " ".join(words)

def check():
    msg = box.get("1.0", tk.END).strip()
    if not msg:
        messagebox.showwarning("Empty", "Please type or paste a message.")
        return
    X = vectorizer.transform([clean_text(msg)])
    prob = model.predict_proba(X)[0][1]
    if prob >= 0.5:
        result.config(text=f"SPAM ({prob:.1%} spam probability)", fg="red")
    else:
        result.config(text=f"NOT SPAM ({1 - prob:.1%} confidence)", fg="green")

def clear():
    box.delete("1.0", tk.END)
    result.config(text="")

root = tk.Tk()
root.title("SMS Spam Detector")
root.geometry("480x320")

tk.Label(root, text="Enter an SMS message:", font=("Arial", 12)).pack(pady=8)
box = tk.Text(root, height=7, width=55, wrap="word")
box.pack(padx=10)

frame = tk.Frame(root)
frame.pack(pady=10)
tk.Button(frame, text="Check message", command=check, width=15).pack(side="left", padx=5)
tk.Button(frame, text="Clear", command=clear, width=10).pack(side="left", padx=5)

result = tk.Label(root, text="", font=("Arial", 14, "bold"))
result.pack(pady=10)

root.mainloop()