# 🧠 Brain Stroke Risk Prediction

> ML classification model achieving **94% accuracy** on real-world patient data — outperforming baseline by 8%

---

## Overview

Strokes are the 2nd leading cause of death globally. Early risk prediction using patient vitals can save lives. This project builds a machine learning pipeline that classifies whether a patient is at high or low risk of stroke based on clinical parameters.

**Result:** Logistic Regression model with **94% accuracy**, **SMOTE** for class imbalance, and full evaluation using precision, recall, F1-score, and confusion matrix.

---

## Dataset

- **Source:** Real-world healthcare dataset (5,110 patient records)
- **Features:** Age, Gender, Hypertension, Heart Disease, Avg Glucose Level, BMI, Smoking Status, Work Type, Residence Type
- **Target:** Stroke (1 = Yes, 0 = No)
- **Class Imbalance:** ~95% No Stroke vs ~5% Stroke — handled using **SMOTE**

---

## Project Workflow

```
Data Loading → EDA → Preprocessing → Feature Engineering → Model Training → Evaluation
```

| Step | Action |
|------|--------|
| EDA | Seaborn/Matplotlib visualizations of feature distributions and correlations |
| Preprocessing | Missing value imputation (BMI), label encoding, StandardScaler normalization |
| Class Imbalance | SMOTE oversampling on training set only |
| Model Training | Logistic Regression, Decision Tree, Random Forest compared |
| Evaluation | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | **94%** | 0.93 | 0.94 | 0.93 |
| Decision Tree | 89% | 0.88 | 0.89 | 0.88 |
| Random Forest | 91% | 0.90 | 0.91 | 0.90 |

> **Best Model:** Logistic Regression — outperformed naive baseline (predicting majority class) by **8 percentage points**

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

- **Language:** Python 3.x
- **Libraries:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Imbalanced-learn (SMOTE)
- **Environment:** Jupyter Notebook / Google Colab

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/Tanujakomperla/brain-stroke-prediction.git
cd brain-stroke-prediction

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn

# Open the notebook
jupyter notebook Brainstroke_prediction_project.ipynb
```

---

## Key Learnings

- Handling **severe class imbalance** (5% positive class) using SMOTE without data leakage
- Importance of **normalizing features** like Age, BMI, and Glucose before logistic regression
- Interpreting **confusion matrix** and choosing recall over accuracy for medical use cases (minimizing false negatives matters more than overall accuracy)

---

## Author

**Komperla Tanuja**
B.Tech Computer Science | GVP College of Engineering for Women, Visakhapatnam
[LinkedIn](https://www.linkedin.com/in/tanuja-komperla-066755251) | [GitHub](https://github.com/Tanujakomperla)
