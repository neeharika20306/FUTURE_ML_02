# Customer Support Ticket Classification System (ML Task 2)

## 📌 Project Overview
This project is a Machine Learning-based system that automatically classifies customer support tickets into appropriate categories and assigns priority levels based on the ticket description. It helps in improving response time and automating support workflows.

---

## 🎯 Objective
- Classify customer tickets into categories such as Billing, Technical Issue, Refund, etc.
- Predict priority levels such as Low, Medium, High, Critical
- Automate customer support ticket handling using NLP and Machine Learning

---

## 🧠 Technologies Used
- Python
- Pandas
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression

---

## 📂 Dataset
The dataset contains customer support tickets with fields like:
- Ticket Description
- Ticket Type
- Ticket Priority

Source: Kaggle Customer Support Ticket Dataset

---

## ⚙️ Workflow

1. Load and preprocess dataset
2. Clean text data (lowercasing, removing noise)
3. Convert text into numerical features using TF-IDF
4. Train ML models:
   - Category Classification Model
   - Priority Classification Model
5. Evaluate models using classification report
6. Predict category and priority for new tickets

---

## 🧹 Text Preprocessing
- Lowercasing text
- Removing punctuation
- Removing extra spaces
- Lemmatization (optional / simplified in final version)

---

## 🤖 Models Used
- Logistic Regression (for both category and priority prediction)

---

## 📊 Output Example

Input:
"My internet is not working and I need urgent help"

Output:
- Category: Technical Issue
- Priority: High

---

## 📈 Results
The model evaluates performance using:
- Precision
- Recall
- F1-score
- Accuracy

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/your-username/FUTURE_ML_02.git
