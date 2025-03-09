# 🍷 Wine Type Classification with Machine Learning

This project applies machine learning techniques to classify wines as red or white based on their chemical properties. Using logistic regression (with and without regularization) and K-Nearest Neighbors (KNN), the models are evaluated for accuracy and generalization.

---

## 📊 Dataset
- [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- Two CSVs: one for red wine, one for white wine
- Features include acidity, sugar, pH, sulphates, alcohol content, etc.

---

## 🎯 Objective
> Can we accurately classify a wine as red or white based solely on its chemical composition?

---

## 🧪 Models & Techniques Used
- Logistic Regression (standard + penalized)
- K-Nearest Neighbors (with hyperparameter tuning)
- Feature scaling with `StandardScaler`
- Cross-validation (`GridSearchCV`, `cross_val_score`)

---

## 📈 Results
- All models achieved **>99% accuracy** on test data.
- Feature scaling significantly improved KNN performance.
- Logistic Regression with L1 penalty was the most interpretable, highlighting key predictors like **density**, **sugar**, and **sulphates**.

---

## 🧰 Tools
- Python (Jupyter Notebook)
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 💡 Key Takeaways
- Feature scaling is critical for distance-based models.
- Regularized logistic regression improves generalizability.
- Even simple models can perform extremely well on well-behaved data.

