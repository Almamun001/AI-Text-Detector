# 🧠 Detecting AI-Generated Text Using Statistical and Linguistic Features

A machine learning project to classify whether a given text is **AI-generated** or **Human-written**, using **statistical**, **linguistic**, and **TF-IDF**-based features. This solution integrates classical feature engineering with machine learning for high accuracy on a large-scale dataset.

---

## 📌 Table of Contents

* [🚀 Project Overview](#-project-overview)
* [📁 Dataset](#-dataset)
* [🔍 Exploratory Data Analysis](#-exploratory-data-analysis)
* [🧪 Feature Engineering](#-feature-engineering)
* [📊 Feature Selection](#-feature-selection)
* [🤖 Modeling & Evaluation](#-modeling--evaluation)
* [💾 Model Saving](#-model-saving)
* [📈 Results](#-results)
* [🔮 Future Work](#-future-work)
* [📦 Requirements](#-requirements)
* [📎 License](#-license)

---

## 🚀 Project Overview

The main objective of this project is to develop a machine learning pipeline that can accurately distinguish between **human-authored** and **AI-generated** text using:

* **Engineered linguistic/statistical features**
* **TF-IDF text vectorization**
* **Classical machine learning models** (Logistic Regression, Random Forest)

---

## 📁 Dataset

📌 **Source**: [Kaggle – AI vs Human Text Dataset](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text/data)

* Total samples used: **100,000**

  * 50,000 AI-generated
  * 50,000 Human-written
* Format: `.csv` file with `text` and `generated` columns
  (`generated`: 0 = Human, 1 = AI)

---

## 🔍 Exploratory Data Analysis

Performed extensive EDA:

* Checked class balance, duplicates, nulls
* Explored:

  * Character/word counts
  * Class distributions
  * Histogram visualizations per class

---

## 🧪 Feature Engineering

Extracted **14 custom features** from each text sample:

| Type               | Features                                                        |
| ------------------ | --------------------------------------------------------------- |
| Text Statistics    | `char_count`, `word_count`, `avg_word_len`                      |
| Punctuation        | `punct_count`, `exclam_count`, `quest_count`                    |
| Style Metrics      | `digit_ratio`, `upper_ratio`, `stopword_ratio`                  |
| Readability Scores | `flesch_reading_ease`, `flesch_kincaid_grade`                   |
| Vocabulary         | `lexical_diversity`                                             |
| Social Indicators  | `url_count`, `mention_count`, `hashtag_count`, `html_tag_count` |

All features were normalized using **MinMaxScaler**.

---

## 📊 Feature Selection

* Used **ANOVA F-test (`SelectKBest`)** for univariate feature selection.
* Chose top 10 informative features based on F-score.
* Also visualized a **correlation heatmap** to check feature redundancy.

---

## 🤖 Modeling & Evaluation

**Combined Features Used**:

* **TF-IDF** (10,000 features from unigrams & bigrams)
* **Top 10 engineered features**

**Models Trained**:

* ✅ Logistic Regression (baseline)
* ✅ Random Forest Classifier (ensemble)

Each model was evaluated with:

* Classification Report (precision, recall, F1-score)
* Confusion Matrix
* Accuracy and F1 Score comparison

---

## 💾 Model Saving

Saved trained models using `joblib` for future use:

* `random_forest_model.pkl`
* `logistic_regression_model.pkl`

---

## 📈 Results

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| **Random Forest**   | ✅ High   | ✅ Best   |
| Logistic Regression | Good     | Good     |

**Random Forest** showed stronger performance, benefiting from mixed feature types and ensemble structure.

---

## 🔮 Future Work

Planned extensions and enhancements:

* Integrate **transformer models** like BERT for deeper semantic understanding.
* Extend support for **multilingual texts**.
* Add **adversarial AI-generated samples** to improve robustness.
* Apply **explainability tools** like SHAP or LIME.

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

**Main Libraries Used**:

* `pandas`, `numpy`
* `scikit-learn`
* `matplotlib`, `seaborn`
* `joblib`, `re`, `string`

---

## 📎 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Special thanks to:

* Kaggle dataset authors
* Open-source contributors
* Community feedback

---

> 📢 *Have feedback or suggestions? Please open an issue or submit a pull request!*

---
