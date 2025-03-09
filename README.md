# 🍷 Wine Classification with Machine Learning

This project applies machine learning techniques to predict whether a wine is red or white based on its chemical composition. Using logistic regression and K-Nearest Neighbors (KNN), the notebook compares model performance, highlights the impact of feature scaling, and tunes hyperparameters using cross-validation.

---

## 📊 Dataset

- **Source**: UCI Machine Learning Repository – [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- Two datasets:
  - `winequality-red.csv` (Red wines)
  - `winequality-white.csv` (White wines)
- Combined into one dataset with a binary target label:  
  `1 = Red`, `0 = White`

---

## 🎯 Objective

> Can we accurately classify a wine as red or white using its measurable chemical features?

---

## 🔬 Features Used

- Acidity levels, residual sugar, chlorides, sulphates
- pH, alcohol content, density
- Free and total sulfur dioxide
- Quality rating (ordinal, treated as a feature)

---

## 🧪 Models & Techniques

- **Logistic Regression** (with and without regularization)
- **K-Nearest Neighbors** (KNN) Classifier
- **StandardScaler** for feature scaling
- **Cross-Validation** (`cross_val_score`)
- **Hyperparameter Tuning** with `GridSearchCV`:
  - `C` for logistic regression
  - `k` (n_neighbors) for KNN

---

## 📈 Results

| Model | Accuracy (Test Set) | CV Score | Notes |
|-------|---------------------|----------|-------|
| Logistic Regression (scaled) | 99.1% | 99.4% | Excellent performance, well-calibrated |
| Penalized Logistic Regression | 99.1% | 99.4% | Similar accuracy, more interpretable coefficients |
| KNN (scaled, tuned k=3) | 99.1% | 99.4% | Slightly more variance, scaling was critical |

- Scaling dramatically improved KNN performance (from ~94% → 99.1%)
- Logistic regression coefficients showed **density**, **residual sugar**, and **sulphates** were key features

---

## 🧰 Tools

- Python (Jupyter Notebook)
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 🧠 What I Learned

- Regularization (L2 penalty) improves generalizability without hurting performance
- Feature scaling is essential for distance-based models like KNN
- Cross-validation gives a more robust view of model accuracy than train/test split alone
- Logistic regression can be highly interpretable even with dense data

---

## 🚀 Future Directions

- Test ensemble models (e.g., Random Forest, Gradient Boosting)
- Explore model interpretability using SHAP or LIME

