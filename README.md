# 🍽️ SmartBite AI
### AI-Powered Fake Review Detection for Food-Tech Platforms

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![ML](https://img.shields.io/badge/ML-scikit--learn-orange)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![XAI](https://img.shields.io/badge/XAI-SHAP-red)
![Streamlit](https://img.shields.io/badge/App-Streamlit-ff4b4b)

> Built as a minor project targeting a real gap in food-tech platforms like Zomato and Swiggy — fake review detection with explainability.

---

## 🚀 Live Demo
🔗 **[Click here to try SmartBite AI](https://smartbite-ai-aobnkthuubrbb9lx8amdxx.streamlit.app/)**

---

## 📌 Problem Statement
Fake and spam reviews on food delivery platforms distort restaurant ratings and mislead customers. Existing platforms lack an intelligent, explainable system to detect suspicious reviews at scale.

## 💡 Solution
SmartBite AI uses NLP and Machine Learning to automatically classify reviews as genuine or fake — and explains **why** using SHAP (Explainable AI).

---

## ✨ Features
- 🔍 **Live fake review detector** — type any review and get instant prediction
- 📊 **SHAP word-level explanations** — see exactly which words triggered the decision
- 📈 **Data insights dashboard** — EDA charts and model metrics
- ⚡ **83.38% accuracy** on balanced test set

---

## 🛠️ Tech Stack
| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| NLP | NLTK, TF-IDF (scikit-learn) |
| ML Model | Logistic Regression |
| Explainability | SHAP |
| Web App | Streamlit |
| Visualization | matplotlib, Plotly |
| Dataset | Yelp Open Dataset (100k reviews) |

---

## 📁 Project Structure

---

## 📊 Model Performance
| Metric | Score |
|---|---|
| Accuracy | 83.38% |
| F1-Score (Fake) | 0.84 |
| Precision (Fake) | 0.82 |
| Recall (Fake) | 0.86 |

---

## 🔍 Key Findings
- Genuine reviews are **55% longer** than suspicious ones on average
- Words like *amazing, best, must* are strong fake indicators
- Words like *however, meal, appetizer* indicate genuine reviews
- Class imbalance handled using **undersampling** technique

---

## 👨‍💻 Author
**Sarthak Bhadke**
MCA Student · Amity University Jaipur

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](YOUR_LINKEDIN_URL)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/sarthakbhadke)