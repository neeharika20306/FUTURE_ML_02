import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("customer_support_tickets.csv")

df = df[["Ticket Description", "Ticket Type", "Ticket Priority"]]

df = df.rename(columns={
    "Ticket Description": "text",
    "Ticket Type": "category",
    "Ticket Priority": "priority"
})

# -------------------------
# 2. Load NLP model
# -------------------------
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation & numbers
    text = re.sub(r'\s+', ' ', text)      # remove extra spaces
    return text.strip()

# -------------------------
# 3. Clean text function
# -------------------------
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

# =========================================================
# 🔵 MODEL 1: CATEGORY CLASSIFICATION
# =========================================================

X_train_text, X_test_text, y_train_cat, y_test_cat = train_test_split(
    df["clean_text"], df["category"],
    test_size=0.2, random_state=42
)

vectorizer_cat = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=5000,
    stop_words='english'
)
X_train_cat = vectorizer_cat.fit_transform(X_train_text)
X_test_cat = vectorizer_cat.transform(X_test_text)

model_category = LogisticRegression(max_iter=1000)
model_category.fit(X_train_cat, y_train_cat)

pred_cat = model_category.predict(X_test_cat)

print("\n===== CATEGORY MODEL =====")
print(classification_report(y_test_cat, pred_cat))

# =========================================================
# 🔴 MODEL 2: PRIORITY CLASSIFICATION
# =========================================================

X_train_text_p, X_test_text_p, y_train_pri, y_test_pri = train_test_split(
    df["clean_text"], df["priority"],
    test_size=0.2, random_state=42
)

vectorizer_pri = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=5000,
    stop_words='english'
)
X_train_pri = vectorizer_pri.fit_transform(X_train_text_p)
X_test_pri = vectorizer_pri.transform(X_test_text_p)

model_priority = LogisticRegression(max_iter=1000)
model_priority.fit(X_train_pri, y_train_pri)

pred_pri = model_priority.predict(X_test_pri)

print("\n===== PRIORITY MODEL =====")
print(classification_report(y_test_pri, pred_pri))

# =========================================================
# 🟢 FINAL PREDICTION FUNCTION
# =========================================================

def predict_ticket(text):
    cleaned = clean_text(text)

    cat_vec = vectorizer_cat.transform([cleaned])
    pri_vec = vectorizer_pri.transform([cleaned])

    category = model_category.predict(cat_vec)[0]
    priority = model_priority.predict(pri_vec)[0]

    return category, priority

# -------------------------
# 4. Test system
# -------------------------
print("\n===== TESTING =====")
print(predict_ticket("My internet is not working and I need urgent help"))
print(predict_ticket("I want refund for my order"))
print(predict_ticket("App keeps crashing frequently"))