
# =========================
# IMPORTS
import os
import re
import tkinter as tk
from tkinter import messagebox

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, fbeta_score

# =========================
DATA_FILE = "spam.csv"

# =========================
def load_dataset():
    if not os.path.exists(DATA_FILE):
        messagebox.showerror("Error", "spam.csv not found!")
        exit()

    df = pd.read_csv(DATA_FILE, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "message"]
    df = df.dropna()

    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df

# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =========================
def build_models(X_train, y_train):
    models = {}

    def tfidf():
        return TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1,2),
            stop_words=None
        )

    def bow():
        return CountVectorizer(
            preprocessor=clean_text,
            ngram_range=(1,2),
            stop_words=None
        )

    def bow_binary():
        return CountVectorizer(
            preprocessor=clean_text,
            ngram_range=(1,2),
            stop_words=None,
            binary=True,
            min_df=2,
            max_features=20000
        )

    models["Naive Bayes"] = Pipeline([
        ("bow", bow()),
        ("clf", MultinomialNB(alpha=0.5, fit_prior=False))
    ])

    models["Logistic Regression"] = Pipeline([
        ("tfidf", tfidf()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
    ])

    models["SVM"] = Pipeline([
        ("tfidf", tfidf()),
        ("clf", LinearSVC(max_iter=5000, class_weight="balanced"))
    ])

    models["Random Forest"] = Pipeline([
        ("bow", bow_binary()),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    models["Decision Tree"] = Pipeline([
        ("tfidf", tfidf()),
        ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=42))
    ])

    for model in models.values():
        model.fit(X_train, y_train)

    return models

# =========================
def _best_threshold_for_spam(y_true, y_prob):
    best_t = 0.5
    best_f1 = -1.0

    for i in range(5, 96):
        t = i / 100
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


def _best_threshold_for_spam_recall(y_true, y_prob):
    best_t = 0.5
    best_f2 = -1.0

    for i in range(5, 96):
        t = i / 100
        y_pred = (y_prob >= t).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_t = t

    return best_t


def _decision_confidence(prob, threshold):
    if prob >= threshold:
        denom = max(1.0 - threshold, 1e-9)
        conf = (prob - threshold) / denom
    else:
        denom = max(threshold, 1e-9)
        conf = (threshold - prob) / denom

    return max(0.0, min(1.0, conf))


def evaluate_models(models, X_train, y_train, X_val, y_val, X_test, y_test):
    results = {}
    thresholds = {}
    val_f1_scores = {}

    print("\n===== VALIDATION RESULTS =====")
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            spam_prob = model.predict_proba(X_val)[:, 1]
            threshold, f1_spam = _best_threshold_for_spam(y_val, spam_prob)

            # Make Random Forest less likely to miss obvious spam.
            if name == "Random Forest":
                threshold = _best_threshold_for_spam_recall(y_val, spam_prob)

            thresholds[name] = threshold
            pred = (spam_prob >= threshold).astype(int)
            val_f1_scores[name] = f1_spam
        else:
            pred = model.predict(X_val)
            val_f1_scores[name] = f1_score(y_val, pred, zero_division=0)

        acc = accuracy_score(y_val, pred)
        print(f"{name} Validation Accuracy: {acc*100:.2f}%")
        print(f"{name} Validation F1(spam): {val_f1_scores[name]*100:.2f}%")
        results[name] = acc

    best_model_name = max(val_f1_scores, key=val_f1_scores.get)
    print(f"\nBest model selected from validation: {best_model_name}")

    # Refit models on Train + Validation before final testing.
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)

    for model in models.values():
        model.fit(X_train_full, y_train_full)

    print("\n===== TEST RESULTS (After Re-training on Train+Validation) =====")
    test_results = {}
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            spam_prob_test = model.predict_proba(X_test)[:, 1]
            threshold = thresholds.get(name, 0.5)
            pred_test = (spam_prob_test >= threshold).astype(int)
        else:
            pred_test = model.predict(X_test)

        test_acc = accuracy_score(y_test, pred_test)
        test_results[name] = test_acc
        print(f"{name} Test Accuracy: {test_acc*100:.2f}%")

    best_model = models[best_model_name]
    if hasattr(best_model, "predict_proba"):
        best_threshold = thresholds.get(best_model_name, 0.5)
        best_prob = best_model.predict_proba(X_test)[:, 1]
        pred_test = (best_prob >= best_threshold).astype(int)
    else:
        pred_test = best_model.predict(X_test)

    print(f"\nBest Model: {best_model_name}")
    print(classification_report(y_test, pred_test))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred_test))

    return results, thresholds, val_f1_scores

# =========================
def create_gui(models, accuracy_scores, thresholds, val_f1_scores):
    root = tk.Tk()
    root.title("Spam Detection Project")
    root.geometry("760x600")
    root.configure(bg="#f5f5f5")

    tk.Label(root,
             text="Spam Detection using 5 Machine Learning Models",
             font=("Arial", 16, "bold"),
             bg="#f5f5f5").pack(pady=10)

    tk.Label(root,
             text="Enter text and click Process",
             font=("Arial", 11),
             bg="#f5f5f5").pack()

    tk.Label(root, text="Input Text:", font=("Arial", 12),
             bg="#f5f5f5").pack(anchor="w", padx=20)

    text_box = tk.Text(root, height=5, width=80, font=("Arial", 11))
    text_box.pack(padx=20, pady=5)

    result_labels = {}

    def create_row(name):
        frame = tk.Frame(root, bg="#f5f5f5")
        frame.pack(fill="x", padx=20, pady=2)

        tk.Label(frame, text=name + ":", width=25,
                 anchor="w", font=("Arial", 11, "bold"),
                 bg="#f5f5f5").pack(side="left")

        label = tk.Label(frame, text="Waiting...",
                         width=40,
                         anchor="w",
                         bg="#e0e0e0",
                         font=("Arial", 11))
        label.pack(side="left", padx=5)

        return label

    for name in models.keys():
        result_labels[name] = create_row(name)

    acc_text = " | ".join([f"{n}: {s*100:.1f}%" for n, s in accuracy_scores.items()])
    tk.Label(root, text=f"Validation Accuracy: {acc_text}",
             font=("Arial", 10),
             fg="blue",
             bg="#f5f5f5").pack(pady=5)

    f1_text = " | ".join([f"{n}: {s*100:.1f}%" for n, s in val_f1_scores.items()])
    tk.Label(root, text=f"Validation F1(spam): {f1_text}",
             font=("Arial", 10),
             fg="blue",
             bg="#f5f5f5").pack(pady=2)

    best_model = max(val_f1_scores, key=val_f1_scores.get)
    tk.Label(root,
             text=f"Best Model: {best_model}",
             font=("Arial", 12, "bold"),
             fg="green",
             bg="#f5f5f5").pack(pady=5)

    def process():
        user_text = text_box.get("1.0", tk.END).strip()

        if not user_text:
            messagebox.showwarning("Warning", "Please enter a message")
            return

        for label in result_labels.values():
            label.config(text="Processing...", fg="black")

        root.update()

        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba([user_text])[0][1])
                # Tree models can output extreme 0/1 probabilities; clipping keeps UI percentages realistic.
                prob = max(0.01, min(0.99, prob))
                threshold = thresholds.get(name, 0.5)
                pred = 1 if prob >= threshold else 0
                confidence = _decision_confidence(prob, threshold)
                if pred == 1:
                    label_text = f"Spam | spam_score={prob*100:.1f}% | conf={confidence*100:.1f}% | th={threshold:.2f}"
                    color = "red"
                else:
                    label_text = f"Not Spam | spam_score={prob*100:.1f}% | conf={confidence*100:.1f}% | th={threshold:.2f}"
                    color = "green"
            else:
                pred = model.predict([user_text])[0]
                if pred == 1:
                    label_text = "Spam 🚨"
                    color = "red"
                else:
                    label_text = "Not Spam ✅"
                    color = "green"

            result_labels[name].config(text=label_text, fg=color)

    tk.Button(root,
              text="Process",
              command=process,
              bg="black",
              fg="white",
              font=("Arial", 12, "bold"),
              width=20).pack(pady=10)

    root.mainloop()

# =========================
def main():
    df = load_dataset()

    # تقسيم احترافي: Train / Validation / Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        df["message"],
        df["label"],
        test_size=0.3,
        random_state=42,
        stratify=df["label"]
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    print("Training...")
    models = build_models(X_train, y_train)

    print("Evaluating...")
    acc, thresholds, val_f1_scores = evaluate_models(models, X_train, y_train, X_val, y_val, X_test, y_test)

    create_gui(models, acc, thresholds, val_f1_scores)

# =========================
if __name__ == "__main__":
    main()